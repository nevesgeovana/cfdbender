"""Bound polynomial NURBS face areas using their coefficients and trim topology.

Each knot rectangle is a Bezier patch. Its vector Jacobian is a polynomial
in Bernstein form, whose coefficients enclose its range. A Taylor polynomial
of its norm has an explicit uniform remainder. Positive Gauss rules integrate
that polynomial exactly; agreement between sampled rules is never evidence.
Green's theorem applies the same polynomial to the actual trim curves.
Unsupported rational curves/surfaces are refused rather than estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import comb, fsum, pi, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BPoly, BSpline, PPoly
from scipy.signal import convolve2d

from cfdbender.geometry._geom_area_domain import (
    NUMERICAL_RTOL,
    _beziers,
    _curves,
    _refuse,
    rectangle_bounds,
)

Array = NDArray[np.float64]
_ROUND = 2e-12
_MAX_DEPTH = 14


def _basis(degree: int, x: Array) -> Array:
    return np.asarray(
        [comb(degree, i) * x**i * (1 - x) ** (degree - i) for i in range(degree + 1)]
    ).T


def _at(coefficients: Array, uv: Array) -> Array:
    return np.asarray(
        np.einsum(
            "ni,nj,ij...->n...",
            _basis(len(coefficients) - 1, uv[:, 0]),
            _basis(coefficients.shape[1] - 1, uv[:, 1]),
            coefficients,
            optimize=False,
        )
    )


def _product(a: Array, b: Array) -> Array:
    p, q = np.array(a.shape) - 1
    r, s = np.array(b.shape) - 1
    wa = np.outer([comb(p, i) for i in range(p + 1)], [comb(q, i) for i in range(q + 1)])
    wb = np.outer([comb(r, i) for i in range(r + 1)], [comb(s, i) for i in range(s + 1)])
    wc = np.outer(
        [comb(p + r, i) for i in range(p + r + 1)],
        [comb(q + s, i) for i in range(q + s + 1)],
    )
    return np.asarray(convolve2d(a * wa, b * wb) / wc)


def _jacobian(net: Array) -> Array:
    du = (len(net) - 1) * np.diff(net, axis=0)
    dv = (net.shape[1] - 1) * np.diff(net, axis=1)
    return np.stack(
        [
            _product(du[..., a], dv[..., b]) - _product(du[..., b], dv[..., a])
            for a, b in ((1, 2), (2, 0), (0, 1))
        ],
        axis=-1,
    )


def _cross_roundoff(net: Array) -> float:
    """Propagate the coefficient reconstruction allowance through both derivatives."""
    p, q = len(net) - 1, net.shape[1] - 1
    du = float(np.max(np.linalg.norm(p * np.diff(net, axis=0), axis=-1)))
    dv = float(np.max(np.linalg.norm(q * np.diff(net, axis=1), axis=-1)))
    position = _ROUND * float(np.max(np.linalg.norm(net, axis=-1)))
    eu, ev = 2 * p * position, 2 * q * position
    return float(eu * dv + ev * du + eu * ev + _ROUND * du * dv * (p + q + 2) ** 2)


def _split(net: Array, axis: int) -> tuple[Array, Array]:
    work = np.moveaxis(net, axis, 0).copy()
    left, right = [work[0].copy()], [work[-1].copy()]
    while len(work) > 1:
        work = (work[:-1] + work[1:]) / 2
        left.append(work[0].copy())
        right.append(work[-1].copy())
    return np.moveaxis(np.array(left), 0, axis), np.moveaxis(np.array(right[::-1]), 0, axis)


@dataclass
class _Polynomial:
    cross: Array
    squared: Array
    centre: float
    coefficients: list[float]
    direction: Array | None
    error: float
    degrees: tuple[int, int]
    factors: tuple[int, ...]

    def at(self, uv: Array) -> Array:
        a, b, c, d = self.factors
        factor = uv[:, 0] ** a * (1 - uv[:, 0]) ** b * uv[:, 1] ** c * (1 - uv[:, 1]) ** d
        if self.direction is not None:
            return np.asarray((_at(self.cross, uv) @ self.direction) * factor)
        x = _at(self.squared, uv) / self.centre - 1
        return np.asarray(
            sqrt(self.centre) * np.polynomial.polynomial.polyval(x, self.coefficients) * factor
        )


def _approximate(net: Array, tolerance: float) -> _Polynomial | None:
    cross = _jacobian(net)
    factors = [0, 0, 0, 0]
    discarded = 0.0
    for axis in (0, 1):
        for end in (0, 1):
            work = np.moveaxis(cross, axis, 0)
            while len(work) > 1:
                residual = float(np.max(np.linalg.norm(work[0 if end == 0 else -1], axis=-1)))
                if residual + discarded > tolerance / 16:
                    break
                discarded += residual
                degree = len(work) - 1
                denominator = np.arange(1, degree + 1) if end == 0 else np.arange(degree, 0, -1)
                work = (work[1:] if end == 0 else work[:-1]) * (degree / denominator)[:, None, None]
                factors[2 * axis + end] += 1
            cross = np.moveaxis(work, 0, axis)
    squared = sum((_product(cross[..., i], cross[..., i]) for i in range(3)), np.zeros((1, 1)))
    magnitude = float(np.max(np.linalg.norm(cross, axis=-1)))
    roundoff = discarded + _cross_roundoff(net)
    roundoff += _ROUND * max(magnitude, 1e-300) * (net.shape[0] + net.shape[1]) ** 2
    mean = cross.mean(axis=(0, 1))
    norm = float(np.linalg.norm(mean))
    if norm:
        direction = mean / norm
        parallel = cross @ direction
        transverse = cross - parallel[..., None] * direction
        # Convexity bounds the perpendicular part everywhere. For nonnegative
        # parallel part, 0 <= norm(C) - dot(C,n) <= norm(C_perpendicular).
        error = float(np.max(np.linalg.norm(transverse, axis=-1)))
        error += max(0.0, -2 * float(parallel.min())) + roundoff
        if error <= tolerance:
            return _Polynomial(
                cross,
                squared,
                0,
                [],
                direction,
                error,
                (len(cross) - 1 + sum(factors[:2]), cross.shape[1] - 1 + sum(factors[2:])),
                tuple(factors),
            )
    padding = _ROUND * float(np.max(np.abs(squared))) * sum(squared.shape) ** 2
    low, high = max(0.0, float(squared.min()) - padding), float(squared.max()) + padding
    if not high > 0 or not np.isfinite(high):
        return None
    centre = (low + high) / 2
    ratio = (high - low) / (high + low)
    if ratio >= 1:
        return None
    coefficients, coefficient = [1.0], 1.0
    for degree in range(1, 13):
        coefficient *= (1.5 - degree) / degree
        coefficients.append(coefficient)
        next_coefficient = coefficient * (0.5 - degree) / (degree + 1)
        # Absolute binomial coefficients decrease. The remaining series is
        # bounded by a geometric tail, uniformly over the coefficient hull.
        error = sqrt(centre) * abs(next_coefficient) * ratio ** (degree + 1) / (1 - ratio)
        error += roundoff
        if error <= tolerance:
            return _Polynomial(
                cross,
                squared,
                centre,
                coefficients,
                None,
                error,
                (
                    (squared.shape[0] - 1) * degree + sum(factors[:2]),
                    (squared.shape[1] - 1) * degree + sum(factors[2:]),
                ),
                tuple(factors),
            )
    return None


def _crossings(curve: Array) -> list[float]:
    cuts = [0.0, 1.0]
    for axis in (0, 1):
        poly = PPoly.from_bernstein_basis(BPoly(curve[:, axis, None], [0.0, 1.0]))
        for value in (0.0, 1.0):
            roots = poly.solve(value, extrapolate=False)
            cuts.extend(float(r) for r in roots if np.isfinite(r) and 0 < r < 1)
    return sorted(set(cuts))


def _curve_split(curve: Array, t: float) -> tuple[Array, Array]:
    left, right = [curve[0]], [curve[-1]]
    work = curve.copy()
    while len(work) > 1:
        work = (1 - t) * work[:-1] + t * work[1:]
        left.append(work[0])
        right.append(work[-1])
    return np.array(left), np.array(right[::-1])


def _domain_enclosure(
    loops: list[list[Array]], knots: tuple[Array, Array]
) -> tuple[Array, float] | None:
    """Bound a nonrectangular loop by a proved strip, without declaring it rectangular."""
    if len(loops) != 1:
        return None
    ends = np.concatenate([np.array([c[0], c[-1]]) for c in loops[0]])
    bounds = np.array([ends.min(axis=0), ends.max(axis=0)])
    for axis, vector in enumerate(knots):
        width = float(np.ptp(vector))
        for side in (0, 1):
            closest = vector[np.argmin(np.abs(vector - bounds[side, axis]))]
            if abs(closest - bounds[side, axis]) <= 1e-6 * width:
                bounds[side, axis] = closest
    widths = bounds[1] - bounds[0]
    if np.any(widths <= 0):
        return None
    normalized = [(c - bounds[0]) / widths for c in loops[0]]
    labels: list[tuple[int, int]] = []
    delta = 0.0
    for curve in normalized:
        options = [
            (float(np.max(np.abs(curve[:, axis] - side))), (axis, side))
            for axis in (0, 1)
            for side in (0, 1)
        ]
        distance, label = min(options)
        delta = max(delta, distance)
        if not labels or labels[-1] != label:
            labels.append(label)
    if delta > 1e-6:
        return None
    if labels[0] == labels[-1]:
        labels.pop()
    if (
        len(labels) != 4
        or len(set(labels)) != 4
        or any(labels[i][0] == labels[(i + 1) % 4][0] for i in range(4))
    ):
        return None
    # The full curve hull must also stay within the expanded rectangle.
    pending = [(c, 0) for c in normalized]
    while pending:
        curve, depth = pending.pop()
        if np.all(curve >= -delta - _ROUND) and np.all(curve <= 1 + delta + _ROUND):
            continue
        if depth >= _MAX_DEPTH:
            return None
        a, b = _curve_split(curve, 0.5)
        pending.extend([(a, depth + 1), (b, depth + 1)])
    return bounds, delta + _ROUND


def _curve_intervals(curve: Array) -> list[tuple[float, float, str]]:
    """Verify clipping intervals by coefficient hulls, including between roots."""
    cuts = _crossings(curve)
    pending = [(a, b, 0) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
    result = []
    padding = _ROUND * max(1.0, float(np.abs(curve).max()))
    while pending:
        a, b, depth = pending.pop()
        left = _curve_split(curve, b)[0]
        part = _curve_split(left, a / b)[1] if a else left
        low, high = part.min(axis=0), part.max(axis=0)
        if high[1] <= padding or low[1] >= 1 - padding or high[0] <= padding:
            continue
        if low[1] >= -padding and high[1] <= 1 + padding:
            if low[0] >= 1 - padding:
                result.append((a, b, "full"))
                continue
            if low[0] >= -padding and high[0] <= 1 + padding:
                result.append((a, b, "partial"))
                continue
        if depth >= _MAX_DEPTH:
            raise _refuse("trim clipping has no coefficient enclosure within the subdivision limit")
        middle = (a + b) / 2
        pending.extend([(a, middle, depth + 1), (middle, b, depth + 1)])
    return result


@lru_cache(maxsize=100)
def _gauss(count: int) -> tuple[Array, Array, float]:
    if not 1 <= count <= 100:
        raise _refuse(f"Gauss order {count} exceeds the verified range 1 to 100")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    if (
        not np.isfinite([nodes, weights]).all()
        or np.any(weights <= 0)
        or np.any(np.abs(nodes) >= 1)
    ):
        raise _refuse("Gauss rule has nonfinite nodes or nonpositive weights")
    # Verify the actual rule's Legendre moments. For a polynomial of degree
    # <= 2*n-1, |coefficient_k| <= (2*k+1)*sup|polynomial| on this interval.
    degree = np.arange(2 * count)
    residual = weights @ np.polynomial.legendre.legvander(nodes, 2 * count - 1) / 2
    residual[0] -= 1
    residual = np.abs(residual) + 16 * np.finfo(float).eps * (degree + 1) ** 2
    allowance = float((2 * degree + 1) @ residual)
    return (nodes + 1) / 2, weights / 2, allowance


def _trim_integral(
    poly: _Polynomial, loops: list[list[Array]], bounds: Array
) -> tuple[float, float]:
    values, errors = [], []
    du, dv = poly.degrees
    snodes, sweights, serror = _gauss(du // 2 + 1)
    maximum = float(np.max(np.linalg.norm(poly.cross, axis=-1))) + poly.error
    for loop in loops:
        contributions = []
        for original in loop:
            curve = (original - bounds[0]) / (bounds[1] - bounds[0])
            degree = len(curve) - 1
            derivative = degree * np.diff(curve, axis=0)
            nodes, weights, terror = _gauss((degree * (du + dv + 2)) // 2 + 1)
            for a, b, mode in _curve_intervals(curve):
                t = a + (b - a) * nodes
                uv = _basis(degree, t) @ curve
                speed = (_basis(degree - 1, t) @ derivative)[:, 1]
                end = np.ones(len(t)) if mode == "full" else uv[:, 0]
                grid = np.stack(
                    [end[:, None] * snodes, np.broadcast_to(uv[:, 1, None], (len(t), len(snodes)))],
                    axis=-1,
                ).reshape(-1, 2)
                flux = poly.at(grid).reshape(len(t), len(snodes)) @ sweights
                contribution = float(weights @ (flux * end * speed) * (b - a))
                contributions.append(contribution)
                # Coefficient hull bounds total variation, including oscillations
                # between quadrature nodes and the root reconstruction allowance.
                variation = float(np.abs(derivative[:, 1]).max()) * (b - a)
                errors.append(poly.error * variation)
                errors.append((serror + terror + serror * terror) * maximum * variation)
        values.append(abs(fsum(contributions)))
        errors.append(_ROUND * fsum(abs(v) for v in contributions))
    return values[0] - fsum(values[1:]), fsum(errors)


def _cell(
    net: Array, bounds: Array, loops: list[list[Array]], budget: float, depth: int = 0
) -> tuple[float, float]:
    variation = sum(
        float(np.abs(np.diff(c[:, 1])).sum()) / (bounds[1, 1] - bounds[0, 1])
        for loop in loops
        for c in loop
    )
    poly = _approximate(net, budget / max(variation, 1.0))
    max_degree = max(len(c) - 1 for loop in loops for c in loop)
    if poly is not None and (
        poly.degrees[0] // 2 + 1 > 100 or (max_degree * (sum(poly.degrees) + 2)) // 2 + 1 > 100
    ):
        poly = None
    if poly is not None:
        value, error = _trim_integral(poly, loops, bounds)
        if error <= budget:
            return value, error
    # A trimmed portion lies between zero and the area of the whole cell.
    # Keep that enclosure until the global error controller refines this cell.
    upper = float(np.mean(np.linalg.norm(_jacobian(net), axis=-1)))
    error = upper / 2 + _cross_roundoff(net)
    error += _ROUND * upper * sum(net.shape[:2]) ** 2
    return upper / 2, error


def _rectangle_cell(net: Array, budget: float, depth: int = 0) -> tuple[float, float]:
    cross = _jacobian(net)
    lower = float(np.linalg.norm(cross.mean(axis=(0, 1))))
    upper = float(np.mean(np.linalg.norm(cross, axis=-1)))
    error = max(0.0, (upper - lower) / 2) + _cross_roundoff(net)
    error += _ROUND * upper * (sum(net.shape[:2])) ** 2
    if error <= budget:
        return (upper + lower) / 2, error
    poly = _approximate(net, budget / 2)
    if poly is not None and max(poly.degrees) // 2 + 1 > 100:
        poly = None
    if poly is not None:
        u, wu, uerror = _gauss(poly.degrees[0] // 2 + 1)
        v, wv, verror = _gauss(poly.degrees[1] // 2 + 1)
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
        value = float(wu @ poly.at(uv).reshape(len(u), len(v)) @ wv)
        maximum = float(np.max(np.linalg.norm(poly.cross, axis=-1))) + poly.error
        quadrature_error = (uerror + verror + uerror * verror) * maximum
        return value, poly.error + quadrature_error + _ROUND * abs(value)
    return (upper + lower) / 2, error


def bounded_area(document: dict[str, Any], scale: float) -> tuple[float, float]:
    """Return area and a coefficient-based absolute error allowance, in square meters.

    ``scale`` allocates the numerical budget only. The result and accumulated
    error must satisfy their own relative tolerance before they can be used.
    """
    surface, loops, closure_error = _curves(document)
    if not scale > 0 or not np.isfinite(scale):
        raise _refuse(f"invalid budget scale {scale}")
    ku, kv = surface.knots
    pu, pv = surface.degrees
    rectangle = rectangle_bounds(loops)
    domain_delta = 0.0
    if rectangle is None:
        enclosure = _domain_enclosure(loops, surface.knots)
        if enclosure is not None:
            rectangle, domain_delta = enclosure
    u0, u1, v0, v1 = surface.domain
    restriction = rectangle if rectangle is not None else np.array([[u0, v0], [u1, v1]])
    net = surface.points.swapaxes(0, 1)
    # Restrict at the exact trim iso-lines before extracting knot rectangles.
    for axis, (knots, degree) in enumerate(((ku, pu), (kv, pv))):
        if restriction is not None:
            spline = BSpline(knots, net, degree, axis=axis)
            for value in restriction[:, axis]:
                multiplicity = int(np.count_nonzero(spline.t == value))
                if multiplicity < degree:
                    spline = spline.insert_knot(value, degree - multiplicity)
            net = np.moveaxis(np.asarray(spline.c), 0, axis)
            if axis == 0:
                ku = np.asarray(spline.t)
            else:
                kv = np.asarray(spline.t)
    ku, net = _beziers(ku, pu, net)
    kv, transposed = _beziers(kv, pv, net.swapaxes(0, 1))
    net = transposed.swapaxes(0, 1)
    us, vs = np.unique(ku), np.unique(kv)
    budget = NUMERICAL_RTOL * scale / ((len(us) - 1) * (len(vs) - 1)) / 4
    cells: list[tuple[float, int, float, Array, Array, int]] = []
    sequence = 0
    jacobian_max = 0.0

    def add(patch: Array, bounds: Array, depth: int) -> None:
        nonlocal sequence
        value, error = (
            _rectangle_cell(patch, budget / 4**depth)
            if rectangle is not None
            else _cell(patch, bounds, loops, budget / 4**depth)
        )
        if not np.isfinite([value, error]).all() or error < 0:
            raise _refuse(f"nonfinite area/error {value}, {error}")
        heappush(cells, (-error, sequence, value, patch, bounds, depth))
        sequence += 1

    for u0, u1 in zip(us[:-1], us[1:], strict=True):
        i = int(np.searchsorted(ku, u0, side="right") - 1)
        for v0, v1 in zip(vs[:-1], vs[1:], strict=True):
            j = int(np.searchsorted(kv, v0, side="right") - 1)
            patch = net[i - pu : i + 1, j - pv : j + 1]
            upper_jacobian = float(np.max(np.linalg.norm(_jacobian(patch), axis=-1)))
            # Domain uncertainty can cross a knot outside the nominal trim
            # rectangle. Its Jacobian bound must include that adjacent patch;
            # polynomial continuation of the inside patch cannot bound it.
            jacobian_max = max(
                jacobian_max, (upper_jacobian + _cross_roundoff(patch)) / ((u1 - u0) * (v1 - v0))
            )
            if (
                u0 < restriction[0, 0]
                or u1 > restriction[1, 0]
                or v0 < restriction[0, 1]
                or v1 > restriction[1, 1]
            ):
                continue
            add(patch, np.array([[u0, v0], [u1, v1]]), 0)
    # Bound pcurve reconstruction and clipping roundoff as a boundary strip.
    # A rectifiable curve's radius-r tube has area <= 2*r*length + pi*r**2.
    # The control polygon bounds its length. Include endpoint closure gaps,
    # and bound polynomial continuation over the entire uncertainty strip.
    uv_scale = max(1.0, max(float(np.abs(c).max()) for loop in loops for c in loop))
    perimeter = sum(
        float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) for loop in loops for c in loop
    )
    uv_error = _ROUND * uv_scale + closure_error
    radius = 4 * uv_error  # Covers both UV coordinates and reconstruction arithmetic.
    minimum_span = min(float(np.min(np.diff(us))), float(np.min(np.diff(vs))))
    continuation = (1 + 2 * radius / minimum_span) ** (2 * pu + 2 * pv)
    tube_area = 2 * radius * perimeter + pi * radius**2 * sum(len(loop) for loop in loops)
    strip_error = jacobian_max * continuation * tube_area
    if domain_delta:
        widths = restriction[1] - restriction[0]
        extrapolation = (1 + 2 * domain_delta * float(widths.max()) / minimum_span) ** (
            2 * pu + 2 * pv
        )
        strip_error += 8 * domain_delta * float(widths.prod()) * jacobian_max * extrapolation
    if not cells:
        raise _refuse("no knot rectangle lies in the face domain")
    for _ in range(10000):
        value = fsum(c[2] for c in cells)
        error = fsum(-c[0] for c in cells) + _ROUND * fsum(abs(c[2]) for c in cells) + strip_error
        if strip_error > NUMERICAL_RTOL * abs(value) / 4:
            raise _refuse(
                f"trim roundoff allowance {strip_error} m2 exceeds "
                f"{NUMERICAL_RTOL * abs(value) / 4} m2"
            )
        if error <= NUMERICAL_RTOL * abs(value) / 2:
            break
        _, _, _, patch, bounds, depth = heappop(cells)
        if depth >= _MAX_DEPTH:
            raise _refuse(
                f"subdivision depth {depth}, error {error} m2 exceeds {NUMERICAL_RTOL * value} m2"
            )
        middle = bounds.mean(axis=0)
        for i, half in enumerate(_split(patch, 0)):
            for j, quarter in enumerate(_split(half, 1)):
                lower = [
                    bounds[0, 0] if i == 0 else middle[0],
                    bounds[0, 1] if j == 0 else middle[1],
                ]
                upper = [
                    middle[0] if i == 0 else bounds[1, 0],
                    middle[1] if j == 0 else bounds[1, 1],
                ]
                add(quarter, np.array([lower, upper]), depth + 1)
    else:
        raise _refuse(
            f"subdivision budget exhausted, error {error} m2 exceeds {NUMERICAL_RTOL * value} m2"
        )
    if not np.isfinite([value, error]).all() or value <= 0 or error > NUMERICAL_RTOL * value:
        raise _refuse(
            f"area {value} m2, error allowance {error} m2 exceeds {NUMERICAL_RTOL * value} m2"
        )
    return value, error

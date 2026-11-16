"""Knot insertion and splitting of the tensor-product B-spline patch.

Pipeline role: pure numerical tools of the geometry package. A skinned surface is built once and
split into patches by inserting a knot to full multiplicity, so the pieces are the same surface
as the whole, not a new fit. Parameter domains are [0, 1] in both directions, as
:func:`~cfdbender.geometry._patch.evaluate_patch` reads them.
"""

from __future__ import annotations

import numpy as np

from cfdbender._errors import InputValueError

from ._patch import Patch

__all__ = [
    "clamped_knots",
    "curve_basis",
    "elevate_curve",
    "fit_curve",
    "full_knots",
    "insert_knot",
    "split_patch",
    "union_knots",
]

_DIRECTIONS = ("u", "v")


def full_knots(knots, multiplicities) -> np.ndarray:
    """Return the knot vector with every distinct knot repeated by its multiplicity."""
    return np.repeat(np.asarray(knots, float), np.asarray(multiplicities, int))


def _insert_once(controls, knots, degree, x):
    """Insert ``x`` once in a curve whose control points run along axis 0 (Boehm)."""
    n = len(controls)
    span = int(np.searchsorted(knots, x, side="right")) - 1
    span = min(max(span, degree), n - 1)
    out = np.empty((n + 1, *controls.shape[1:]))
    out[: span - degree + 1] = controls[: span - degree + 1]
    for i in range(span - degree + 1, span + 1):
        a = (x - knots[i]) / (knots[i + degree] - knots[i])
        out[i] = (1 - a) * controls[i - 1] + a * controls[i]
    out[span + 1 :] = controls[span:]
    return out, np.insert(knots, span + 1, x)


def _along(patch, direction):
    """Return the controls with the chosen direction first, its knots and its degree."""
    if direction == "u":
        return (
            np.asarray(patch.controls, float),
            full_knots(patch.knots_u, patch.multiplicities_u),
            patch.degree_u,
        )
    return (
        np.asarray(patch.controls, float).transpose(1, 0, 2),
        full_knots(patch.knots_v, patch.multiplicities_v),
        patch.degree_v,
    )


def _rebuilt(patch, direction, controls, knots):
    """Return ``patch`` with the chosen direction replaced by ``controls`` and ``knots``."""
    distinct, mult = np.unique(knots, return_counts=True)
    key = {"u": ("knots_u", "multiplicities_u"), "v": ("knots_v", "multiplicities_v")}[direction]
    values = {
        key[0]: tuple(float(k) for k in distinct),
        key[1]: tuple(int(m) for m in mult),
        "controls": controls if direction == "u" else controls.transpose(1, 0, 2),
    }
    fields = {**patch.__dict__, **values}
    return Patch(**fields)


def insert_knot(patch: Patch, direction: str, x: float, times: int = 1) -> Patch:
    """Return the same surface with the knot ``x`` inserted ``times`` more times.

    Parameters
    ----------
    patch : Patch
        Surface to refine.
    direction : {"u", "v"}
        Parametric direction of the insertion.
    x : float
        Parameter strictly inside (0, 1).
    times : int, optional
        Insertions to make; the multiplicity of ``x`` may not exceed the degree.

    Returns
    -------
    Patch
        A surface that evaluates to the same points as ``patch``.

    Raises
    ------
    InputValueError
        When the direction is unknown, ``x`` is not inside (0, 1) or the multiplicity would
        exceed the degree.
    """
    if direction not in _DIRECTIONS:
        raise InputValueError("direction: expected u or v")
    controls, knots, degree = _along(patch, direction)
    if not 0.0 < x < 1.0:
        raise InputValueError("x: the knot must lie strictly inside (0, 1)")
    if times < 0 or np.count_nonzero(knots == x) + times > degree:
        raise InputValueError("times: the multiplicity of a knot may not exceed the degree")
    for _ in range(times):
        controls, knots = _insert_once(controls, knots, degree, x)
    return _rebuilt(patch, direction, controls, knots)


def _clip(patch, direction, bounds, controls, knots):
    """Return a piece of a curve between two clamped knots, its parameter set to [0, 1]."""
    lo, hi = bounds
    return _rebuilt(patch, direction, controls, np.clip((knots - lo) / (hi - lo), 0.0, 1.0))


def split_patch(patch: Patch, direction: str, x: float) -> tuple[Patch, Patch]:
    """Split a patch at the parameter ``x`` of one direction into two patches of the same surface.

    The knot is inserted to full multiplicity and each half is returned with its own parameter
    set to [0, 1], so the two pieces evaluate to the original surface to rounding.

    Parameters
    ----------
    patch : Patch
        Surface to split.
    direction : {"u", "v"}
        Parametric direction to cut across.
    x : float
        Parameter strictly inside (0, 1).

    Returns
    -------
    tuple of Patch
        The piece before ``x`` and the piece after it.

    Raises
    ------
    InputValueError
        As :func:`insert_knot`.

    Examples
    --------
    >>> import numpy as np
    >>> from cfdbender.geometry._patch import interpolate_patch
    >>> grid = np.random.default_rng(1).normal(size=(9, 7, 3))
    >>> low, high = split_patch(interpolate_patch(grid), "u", 0.5)
    >>> bool(np.allclose(low.evaluate(5)[-1], high.evaluate(5)[0]))
    True
    """
    x = _snapped(patch, direction, x)
    refined = _complete(patch, direction, x)
    controls, knots, degree = _along(refined, direction)
    first = int(np.flatnonzero(knots == x)[0])
    left_knots = np.append(knots[: first + degree], x)
    left = _clip(refined, direction, (0.0, x), controls[:first], left_knots)
    right_knots = np.insert(knots[first:], 0, x)
    right = _clip(refined, direction, (x, 1.0), controls[first - 1 :], right_knots)
    return left, right


def _multiplicity(patch, direction, x) -> int:
    """Return how many times ``x`` is already a knot of the direction."""
    _, knots, _ = _along(patch, direction)
    return int(np.count_nonzero(knots == x))


def _complete(patch, direction, x):
    """Raise the multiplicity of an existing knot to the degree."""
    degree = patch.degree_u if direction == "u" else patch.degree_v
    return insert_knot(patch, direction, x, times=degree - _multiplicity(patch, direction, x))


def _snapped(patch, direction, x):
    """Return the knot of the direction that ``x`` equals to rounding, or ``x``."""
    _, knots, _ = _along(patch, direction)
    nearest = knots[int(np.argmin(np.abs(knots - x)))]
    return float(nearest) if 0.0 < nearest < 1.0 and abs(nearest - x) < 1e-12 else x


def clamped_knots(interior, degree: int, multiplicities=None) -> np.ndarray:
    """Return the full clamped knot vector on [0, 1] with the given interior knots.

    Parameters
    ----------
    interior : sequence of float
        Distinct interior knots, strictly increasing inside (0, 1); may be empty.
    degree : int
        Polynomial degree, at least 1.
    multiplicities : sequence of int, optional
        Multiplicity of each interior knot (1 to ``degree``); one each when omitted. A knot of
        multiplicity ``degree`` is a corner, ``degree - 1`` a tangent-continuous join.

    Raises
    ------
    InputValueError
        When the degree is below 1, the knots are not increasing inside (0, 1) or a multiplicity
        lies outside 1 to ``degree``.

    Examples
    --------
    >>> clamped_knots([0.5], 3, [2]).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    """
    knots = np.asarray(interior, float)
    mult = np.ones(len(knots), int) if multiplicities is None else np.asarray(multiplicities, int)
    if degree < 1:
        raise InputValueError("degree: expected at least 1")
    if len(mult) != len(knots):
        raise InputValueError("multiplicities: expected one per interior knot")
    if np.any(knots <= 0.0) or np.any(knots >= 1.0) or np.any(np.diff(knots) <= 0.0):
        raise InputValueError("interior: expected knots strictly increasing inside (0, 1)")
    if np.any(mult < 1) or np.any(mult > degree):
        raise InputValueError("multiplicities: expected values from 1 to the degree")
    return np.concatenate([np.zeros(degree + 1), np.repeat(knots, mult), np.ones(degree + 1)])


def curve_basis(knots, degree: int, parameters) -> np.ndarray:
    """Return the basis matrix ``[parameter, function]`` of a clamped knot vector.

    The last parameter may equal 1; every row sums to 1.
    """
    from scipy.interpolate import BSpline

    return BSpline.design_matrix(
        np.clip(np.asarray(parameters, float), 0.0, 1.0), np.asarray(knots, float), degree
    ).toarray()


def fit_curve(points, parameters, knots, degree: int, end_slopes=None):
    """Fit a B-spline curve on given knots by least squares, ends interpolated.

    Parameters
    ----------
    points : array_like
        Samples, shape ``(m, d)``.
    parameters : array_like
        Parameter of each sample, increasing from 0 to 1.
    knots : array_like
        Full clamped knot vector (see :func:`clamped_knots`).
    degree : int
        Polynomial degree of the curve.
    end_slopes : pair of array_like, optional
        Derivative with respect to the parameter at the first and the last end. When given, the
        curve leaves each end with that slope exactly (degree 2 or more, at least four controls),
        so two curves that meet at a point can share a tangent there.

    Returns
    -------
    controls : numpy.ndarray
        Control points, shape ``(len(knots) - degree - 1, d)``; the first and last equal the
        first and last sample, so curves that meet at a rim meet exactly.
    residual : float
        Largest distance from a sample to the curve.

    Raises
    ------
    InputValueError
        When the samples are fewer than the controls or the parameters do not run from 0 to 1.
    """
    q, t = np.asarray(points, float), np.asarray(parameters, float)
    knots = np.asarray(knots, float)
    basis = curve_basis(knots, degree, t)
    if basis.shape[0] < basis.shape[1]:
        raise InputValueError("points: fewer samples than control points")
    if abs(t[0]) > 1e-12 or abs(t[-1] - 1.0) > 1e-12 or np.any(np.diff(t) <= 0.0):
        raise InputValueError("parameters: expected increasing values from 0 to 1")
    n = basis.shape[1]
    controls = np.zeros((n, q.shape[1]))
    controls[0], controls[-1] = q[0], q[-1]
    fixed = np.zeros(n, bool)
    fixed[[0, -1]] = True
    if end_slopes is not None and degree >= 2 and n >= 4:
        controls[1] = q[0] + knots[degree + 1] / degree * np.asarray(end_slopes[0], float)
        controls[-2] = q[-1] - (1.0 - knots[-degree - 2]) / degree * np.asarray(
            end_slopes[1], float
        )
        fixed[[1, -2]] = True
    if not fixed.all():
        rhs = q - basis[:, fixed] @ controls[fixed]
        controls[~fixed] = np.linalg.lstsq(basis[:, ~fixed], rhs, rcond=None)[0]
    return controls, float(np.max(np.linalg.norm(basis @ controls - q, axis=1)))


def union_knots(first, second) -> np.ndarray:
    """Return the full knot vector of two clamped vectors of one degree, multiplicity the larger.

    Both curves can be refined onto it, so they share one basis.

    Raises
    ------
    InputValueError
        When the vectors do not start and end alike (different degree or domain).
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a[0] != b[0] or a[-1] != b[-1] or np.count_nonzero(a == a[0]) != np.count_nonzero(b == b[0]):
        raise InputValueError("knots: expected clamped vectors of one degree on one domain")
    ka, ma = np.unique(a, return_counts=True)
    kb, mb = np.unique(b, return_counts=True)
    every = np.union1d(ka, kb)
    count = np.zeros(len(every), int)
    for keys, mult in ((ka, ma), (kb, mb)):
        at = np.searchsorted(every, keys)
        count[at] = np.maximum(count[at], mult)
    return np.repeat(every, count)


def elevate_curve(controls, knots, degree: int, times: int = 1):
    """Raise the degree of a curve without changing it.

    Every distinct knot gains ``times`` in multiplicity, so the continuity at each is kept. The
    result is the exact curve in the larger space, found by collocation at its Greville points.

    Returns
    -------
    controls : numpy.ndarray
    knots : numpy.ndarray
        The new full knot vector.
    degree : int
        ``degree + times``.

    Raises
    ------
    InputValueError
        When ``times`` is below 0 or the controls do not match the knots.
    """
    c, k = np.asarray(controls, float), np.asarray(knots, float)
    if times < 0 or len(c) != len(k) - degree - 1:
        raise InputValueError("controls: expected one point per basis function and times >= 0")
    if times == 0:
        return c, k, degree
    new_degree = degree + times
    distinct, mult = np.unique(k, return_counts=True)
    new = np.repeat(distinct, mult + times)
    greville = np.array(
        [new[i + 1 : i + new_degree + 1].mean() for i in range(len(new) - new_degree - 1)]
    )
    values = curve_basis(k, degree, greville) @ c
    return np.linalg.solve(curve_basis(new, new_degree, greville), values), new, new_degree

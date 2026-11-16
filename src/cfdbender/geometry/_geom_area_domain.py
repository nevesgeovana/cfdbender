"""Whole-curve domain certificates from the kernel's serialized NURBS topology."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from cfdbender._errors import InputArtifactError
from cfdbender.geometry._geom_gate import _parse_face, _pieces, _Surface

Array = NDArray[np.float64]
NUMERICAL_RTOL = 1e-5


def _refuse(detail: str) -> InputArtifactError:
    return InputArtifactError(
        f"export area unverified: {detail}. Numerical relative limit {NUMERICAL_RTOL}; "
        "simplify the face or trim definition and rebuild.",
        kind="geometry",
    )


def _beziers(knots: Array, degree: int, points: Array, axis: int = 0) -> tuple[Array, Array]:
    spline = BSpline(knots, points, degree, axis=axis)
    values, counts = np.unique(knots, return_counts=True)
    for value, count in zip(values[1:-1], counts[1:-1], strict=True):
        if count < degree:
            spline = spline.insert_knot(value, degree - count)
    return np.asarray(spline.t), np.asarray(spline.c)


def _curves(document: dict[str, Any]) -> tuple[_Surface, list[list[Array]], float]:
    if int(document["global"][13]) != 6:
        raise _refuse("serialized measurement is not in metres")
    by_id = {e["id"]: e for e in document["entities"]}
    faces = [e for e in by_id.values() if e["type"] == 144]
    if len(faces) != 1:
        raise _refuse(f"measured {len(faces)} serialized faces, limit exactly 1")
    face = _parse_face(faces[0]["parameters"])
    if int(faces[0]["d1"][6]):
        raise _refuse("a transformed trim has no area certificate")
    entity = by_id[face.surface]
    if entity["type"] != 128:
        raise _refuse(f"surface type {entity['type']} has no polynomial area certificate")
    surface = _Surface(entity["parameters"])
    if any(p < 1 or p > 8 for p in surface.degrees):
        raise _refuse(f"surface degrees {surface.degrees} exceed the verified range 1 to 8")
    if not np.all(surface.weights == surface.weights.flat[0]) or surface.weights.flat[0] <= 0:
        raise _refuse("rational surface weights have no polynomial area certificate")
    if not np.isfinite(surface.points).all():
        raise _refuse("surface coefficients are nonfinite")
    expected = tuple(
        value
        for k, p in zip(surface.knots, surface.degrees, strict=True)
        for value in (k[p], k[-p - 1])
    )
    if not (
        expected[0] <= surface.domain[0] < surface.domain[1] <= expected[1]
        and expected[2] <= surface.domain[2] < surface.domain[3] <= expected[3]
    ):
        raise _refuse("surface domain lies outside its knot range")
    transform = int(entity["d1"][6])
    if transform:
        matrix = np.array(
            [float(v.replace("D", "E")) for v in by_id[transform]["parameters"][1:13]]
        )
        matrix = matrix.reshape(3, 4)
        surface.points = surface.points @ matrix[:, :3].T + matrix[:, 3]
    loops = []
    closure_error = 0.0
    for pointer in [face.outer, *face.inner]:
        if not pointer:
            u0, u1, v0, v1 = surface.domain
            corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]])
            loops.append([corners[i : i + 2] for i in range(4)])
            continue
        boundary = by_id[pointer]
        if boundary["type"] != 142:
            raise _refuse("trim boundary is not an explicit kernel pcurve")
        pending = [int(boundary["parameters"][3])]
        while pending:
            member = by_id[pending.pop()]
            if int(member["d1"][6]):
                raise _refuse("a transformed pcurve has no area certificate")
            if member["type"] == 102:
                pending.extend(
                    int(v) for v in member["parameters"][2 : 2 + int(member["parameters"][1])]
                )
        pieces = _pieces(by_id[int(boundary["parameters"][3])], by_id)
        if not pieces:
            raise _refuse("trim curve type has no polynomial area certificate")
        loop = []
        for curve in pieces:
            points = curve.points[:, :2]
            # An iso-line is proved by every coefficient, not sampled points.
            if not np.isfinite(points).all() or not np.all(curve.weights > 0):
                raise _refuse("nonfinite trim coefficients or nonpositive weights")
            if np.any(np.all(points == points[0], axis=0)):
                loop.append(np.array([curve.at(curve.low)[:2], curve.at(curve.high)[:2]]))
                continue
            if not np.all(curve.weights == curve.weights[0]):
                raise _refuse("rational trim weights have no polynomial area certificate")
            if curve.degree < 1 or curve.degree > 12:
                raise _refuse(f"trim degree {curve.degree} exceeds the verified range 1 to 12")
            spline = BSpline(curve.knots, points, curve.degree)
            for value in (curve.low, curve.high):
                count = int(np.count_nonzero(spline.t == value))
                if count < curve.degree:
                    spline = spline.insert_knot(value, curve.degree - count)
            knots, points = _beziers(np.asarray(spline.t), curve.degree, np.asarray(spline.c))
            for a, b in zip(np.unique(knots)[:-1], np.unique(knots)[1:], strict=True):
                if a < curve.low or b > curve.high:
                    continue
                index = int(np.searchsorted(knots, a, side="right") - 1)
                loop.append(points[index - curve.degree : index + 1])
        # The kernel's topological vertex closes numerical pcurve endpoint gaps.
        # Include the connecting segments, rather than discarding their area.
        connected = []
        for i, segment in enumerate(loop):
            connected.append(segment)
            following = loop[(i + 1) % len(loop)]
            if not np.array_equal(segment[-1], following[0]):
                gap = float(np.max(np.abs(segment[-1] - following[0])))
                # Topology proves adjacency, but the pcurves can leave a small
                # numerical gap. Charge its full measured width to the final
                # boundary-strip bound instead of silently snapping endpoints.
                closure_error = max(closure_error, gap)
                connected.append(np.array([segment[-1], following[0]]))
        loops.append(connected)
    if len(loops) > 1:
        # The outer loop encloses every hole in a valid native face. Determine
        # it by an exact polynomial line integral, not the BREP storage order.
        areas = []
        for loop in loops:
            total = Fraction(0)
            for polygon in loop:
                p = len(polygon) - 1
                for i in range(p + 1):
                    for j in range(p):
                        weight = Fraction(comb(p, i) * comb(p - 1, j), comb(2 * p - 1, i + j) * 2)
                        total += (
                            weight
                            * Fraction(float(polygon[i, 0]))
                            * (Fraction(float(polygon[j + 1, 1])) - Fraction(float(polygon[j, 1])))
                        )
            areas.append(abs(total))
        order = sorted(range(len(loops)), key=lambda i: areas[i], reverse=True)
        if areas[order[0]] == areas[order[1]]:
            raise _refuse("native outer loop cannot be distinguished from a hole")
        loops = [loops[i] for i in order]
    return surface, loops, closure_error


def rectangle_bounds(loops: list[list[Array]]) -> Array | None:
    """Prove the boundary is the four iso-lines, using all curve coefficients."""
    if len(loops) != 1:
        return None
    points = np.concatenate(loops[0])
    lower, upper = points.min(axis=0), points.max(axis=0)
    if np.any(upper <= lower):
        return None
    for curve in loops[0]:
        if not any(
            np.all(curve[:, axis] == side) for axis in (0, 1) for side in (lower[axis], upper[axis])
        ):
            return None
    corners = np.array([c[0] for c in loops[0]])
    following = np.roll(corners, -1, axis=0)
    signed = float(np.sum(corners[:, 0] * following[:, 1] - corners[:, 1] * following[:, 0]))
    if abs(signed) != 2 * float(np.prod(upper - lower)):
        return None
    return np.array([lower, upper])

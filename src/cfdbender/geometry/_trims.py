"""Trims of a base surface as loops in its parameter space, and their pure checks.

Pipeline role: pure numerical stage of the geometry package (S7). A face is a base
:class:`Patch` cut by one closed loop of parameter-space curves. The loops built here are
axis-aligned (iso lines of the base), so each piece is an exact degree 1 curve in (u, v) and
no geometry kernel is needed to build, measure or check them. Junction trims, whose pieces are
not iso lines, are added by the assembly stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from cfdbender._errors import InputValueError

from ._bspline import full_knots
from ._patch import Patch

__all__ = [
    "LOOP_GAP",
    "PARTITION_TOLERANCE",
    "Trim",
    "iso_trim",
    "loop_area",
    "trim_areas",
    "trim_checks",
    "trim_record",
]

#: Largest gap between consecutive pieces of a closed loop, in parameter units (design D, S7).
LOOP_GAP = 1e-12
#: Relative difference allowed between the summed face areas of a base and the base area (S7).
PARTITION_TOLERANCE = 1e-9
_GAUSS = np.polynomial.legendre.leggauss(8)


@dataclass(frozen=True)
class Trim:
    """One face: a base surface cut by one closed outer loop in its (u, v) domain.

    Attributes
    ----------
    face : str
        Name of the face in its component, for example ``"upper"`` or ``"te"``.
    base : str
        Name of the base surface the loop lies on, for example ``"skin"``.
    loop : tuple
        The pieces of the loop in order, each a pair of (u, v) points joined by a straight
        line in parameter space (a degree 1 pcurve).
    role : str
        What the face is for (``"upper"``, ``"lower"``, ``"trailing_edge"``, ``"cap"``).
    inner : tuple
        Inner loops, in the form of ``loop``; none is written until the reader probe shows
        that a reader honours them.
    """

    face: str
    base: str
    loop: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    role: str = ""
    inner: tuple = ()

    def to_dict(self) -> dict:
        """Return the plain-data form: lists and numbers only."""
        return {
            "face": self.face,
            "base": self.base,
            "role": self.role,
            "loop": [[list(a), list(b)] for a, b in self.loop],
            "inner": [[[list(a), list(b)] for a, b in loop] for loop in self.inner],
        }


def iso_trim(face, base, u, v, role="") -> Trim:
    """Return the face bounded by the iso lines ``u = (u0, u1)`` and ``v = (v0, v1)``.

    The loop runs counter-clockwise in (u, v): along v0, up u1, back along v1, down u0.
    """
    (u0, u1), (v0, v1) = (float(u[0]), float(u[1])), (float(v[0]), float(v[1]))
    if not (u0 < u1 and v0 < v1):
        raise InputValueError(f"{face}: an iso trim needs u0 < u1 and v0 < v1")
    corners = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return Trim(
        face=face,
        base=base,
        loop=tuple((corners[i], corners[(i + 1) % 4]) for i in range(4)),
        role=role,
    )


def loop_area(trim: Trim) -> float:
    """Return the signed area of the outer loop in parameter space (shoelace formula)."""
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in trim.loop)


def _rectangle(trim):
    """Return ``((u0, u1), (v0, v1))`` of an axis-aligned loop, or None for any other loop."""
    points = [p for piece in trim.loop for p in piece]
    us = sorted({p[0] for p in points})
    vs = sorted({p[1] for p in points})
    if not _is_iso(trim):
        return None
    u0, u1, v0, v1 = us[0], us[-1], vs[0], vs[-1]
    # A side divided at interior points (a cap's rims) still bounds a rectangle: every vertex
    # lies on the bounding box and each piece runs along one of its sides.
    if u0 == u1 or v0 == v1 or any(p[0] not in (u0, u1) and p[1] not in (v0, v1) for p in points):
        return None
    return (u0, u1), (v0, v1)


def _is_iso(trim):
    return all(a[0] == b[0] or a[1] == b[1] for a, b in (*trim.loop, *_inner_pieces(trim)))


def _inner_pieces(trim):
    return [piece for loop in trim.inner for piece in loop]


def _jacobian_area(patch: Patch, u, v):
    """Return |S_u x S_v| on the tensor grid of ``u`` and ``v``."""
    controls = np.asarray(patch.controls, float)
    tu = full_knots(patch.knots_u, patch.multiplicities_u)
    tv = full_knots(patch.knots_v, patch.multiplicities_v)
    du = BSpline(tu, controls, patch.degree_u, axis=0).derivative()
    pu = BSpline(tv, du(u), patch.degree_v, axis=1)(v)
    pv = BSpline(tv, BSpline(tu, controls, patch.degree_u, axis=0)(u), patch.degree_v, axis=1)
    pv = pv.derivative()(v)
    return np.linalg.norm(np.cross(pu, pv), axis=-1)


def _cells(patch, cuts_u, cuts_v):
    edges_u = np.unique(np.r_[patch.knots_u, list(cuts_u), 0.0, 1.0])
    edges_v = np.unique(np.r_[patch.knots_v, list(cuts_v), 0.0, 1.0])
    return edges_u, edges_v


def _cell_areas(patch, edges_u, edges_v):
    """Return the model-space area of every cell of the grid, by 8 point Gauss per cell."""
    nodes, weights = _GAUSS
    half_u, half_v = np.diff(edges_u) / 2, np.diff(edges_v) / 2
    u = ((edges_u[:-1] + edges_u[1:]) / 2)[:, None] + half_u[:, None] * nodes[None, :]
    v = ((edges_v[:-1] + edges_v[1:]) / 2)[:, None] + half_v[:, None] * nodes[None, :]
    jac = _jacobian_area(patch, u.ravel(), v.ravel())
    jac = jac.reshape(len(half_u), len(nodes), len(half_v), len(nodes))
    return np.einsum("injm,n,m->ij", jac, weights, weights) * np.outer(half_u, half_v)


def trim_areas(bases: dict, trims) -> dict:
    """Return the model-space area of every base and of every iso face on it.

    The areas are summed over one grid of cells per base, cut at every knot and every trim
    line, so a set of faces that tiles its base sums to the base area to rounding.

    Returns
    -------
    dict
        ``{"bases": {name: area}, "faces": {face: area}, "covered": {name: cell count array}}``.
    """
    out: dict[str, dict] = {"bases": {}, "faces": {}, "covered": {}}
    for name, patch in bases.items():
        mine = [t for t in trims if t.base == name]
        rects = [_rectangle(t) for t in mine]
        cuts_u = [x for r in rects if r for x in r[0]]
        cuts_v = [x for r in rects if r for x in r[1]]
        edges_u, edges_v = _cells(patch, cuts_u, cuts_v)
        areas = _cell_areas(patch, edges_u, edges_v)
        mid_u, mid_v = (edges_u[:-1] + edges_u[1:]) / 2, (edges_v[:-1] + edges_v[1:]) / 2
        covered = np.zeros(areas.shape, int)
        for trim, rect in zip(mine, rects, strict=True):
            if rect is None:
                continue
            inside = np.outer(
                (mid_u > rect[0][0]) & (mid_u < rect[0][1]),
                (mid_v > rect[1][0]) & (mid_v < rect[1][1]),
            )
            covered += inside
            out["faces"][trim.face] = float(areas[inside].sum())
        out["bases"][name] = float(areas.sum())
        out["covered"][name] = covered
    return out


def _check(check_id, ok, value, note, status_fail="fail"):
    return {"id": check_id, "status": "pass" if ok else status_fail, "value": value, "note": note}


def trim_checks(bases: dict, trims) -> list[dict]:
    """Return the S7 records of the faces of one component.

    ``trim.closed`` (each loop closes within ``LOOP_GAP``), ``trim.in_domain`` (every loop point
    in [0, 1] squared), ``trim.iso_exact`` (every piece axis-aligned), ``trim.partition`` (the
    faces of each base cover its domain once: no cell twice, none left, and the summed face area
    equals the base area within ``PARTITION_TOLERANCE`` relative) and ``trim.outer_only`` (a
    warning for any inner loop).
    """
    trims = list(trims)
    unknown = sorted({t.base for t in trims} - set(bases))
    if unknown:
        raise InputValueError(f"trims: no base surface named {', '.join(unknown)}")
    gap = 0.0
    for trim in trims:
        for loop in (trim.loop, *trim.inner):
            for (_, end), (start, _) in zip(loop, (*loop[1:], loop[0]), strict=True):
                gap = max(gap, abs(end[0] - start[0]), abs(end[1] - start[1]))
    points = np.array([p for t in trims for piece in (*t.loop, *_inner_pieces(t)) for p in piece])
    outside = (
        float(np.max(np.r_[-points.ravel(), points.ravel() - 1.0, 0.0])) if len(points) else 0.0
    )
    skew = sum(1 for t in trims if not _is_iso(t))
    areas = trim_areas(bases, trims)
    worst, holes, twice = 0.0, 0, 0
    for name, base_area in areas["bases"].items():
        # A loop that is not an iso rectangle is not measured here; its base then fails the
        # partition, which this pure stage can only prove for iso faces.
        summed = sum(areas["faces"].get(t.face, 0.0) for t in trims if t.base == name)
        worst = max(worst, abs(summed - base_area) / base_area if base_area > 0 else summed)
        holes += int(np.count_nonzero(areas["covered"][name] == 0))
        twice += int(np.count_nonzero(areas["covered"][name] > 1))
    inner = sum(len(t.inner) for t in trims)
    return [
        _check("trim.closed", gap <= LOOP_GAP, gap, f"largest loop gap, limit {LOOP_GAP}"),
        _check("trim.in_domain", outside == 0.0, outside, "largest excursion outside [0, 1]"),
        _check("trim.iso_exact", skew == 0, skew, "loops with a piece off the iso lines"),
        _check(
            "trim.partition",
            worst <= PARTITION_TOLERANCE and holes == 0 and twice == 0,
            {"relative_area_error": worst, "uncovered_cells": holes, "cells_covered_twice": twice},
            f"faces tile each base once, area within {PARTITION_TOLERANCE} relative",
        ),
        _check("trim.outer_only", inner == 0, inner, "inner loops written", status_fail="warn"),
    ]


def trim_record(trims) -> list[dict]:
    """Return the plain-data form of a list of trims, sorted by base and face."""
    return [t.to_dict() for t in sorted(trims, key=lambda t: (t.base, t.face))]

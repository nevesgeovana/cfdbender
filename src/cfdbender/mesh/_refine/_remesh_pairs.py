"""Quad-dominant remesh: triangles paired into quadrilaterals, and the element report.

A section of :mod:`._remesh`, which imports every name of it that other
modules read, so their import paths hold. Two triangles sharing an edge that
is no curve make a
quadrilateral; the pairs are matched greedily, best quality first, and the
matching is then grown along alternating paths at most
:data:`AUGMENT_DEPTH` pairs deep. :func:`element_report` states what
``refine.json`` reports of a remeshed family's elements.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy

from cfdbender.mesh._refine._geometry import Faces, Points
from cfdbender.mesh._refine._remesh_group import Edge

#: Quad-dominant: a pairing grows the matching along alternating paths this many pairs deep.
AUGMENT_DEPTH = 4


def _unit_normal(p: Points) -> Points:
    """Return the (unnormalised) normal of a triangle given as three rows."""
    return numpy.cross(p[1] - p[0], p[2] - p[0])


def _face_normal(p: Points) -> Points:
    """Return a polygon's (unnormalized) normal: the cross product of its diagonals for four."""
    if len(p) == 4:
        return numpy.cross(p[2] - p[0], p[3] - p[1])
    return _unit_normal(p)


def _quad_of(f0: list[int], f1: list[int], k: Edge) -> list[int] | None:
    """Return the quadrilateral two triangles sharing the edge k make, or None.

    The first triangle runs a, b, c and the second must run b, a, d (the two
    agree in orientation); the quadrilateral is a, d, b, c, which runs every
    remaining edge the way its triangle did.
    """
    a, b = k
    j = f0.index(a)
    if f0[(j + 1) % 3] != b:
        a, b = b, a
        j = f0.index(a)
    c = f0[(j + 2) % 3]
    m = f1.index(b)
    if f1[(m + 1) % 3] != a:
        return None
    return [a, f1[(m + 2) % 3], b, c]


def _matching(pairs: Sequence[tuple[int, int]], quality: Points) -> dict[int, int]:
    """Return a matching of triangles: greedy best quality first, then grown.

    Ties of quality are broken by the pair's face numbers, so the matching is
    deterministic.
    """
    mate: dict[int, int] = {}
    order = sorted(range(len(pairs)), key=lambda c: (-float(quality[c]), pairs[c]))
    for c in order:
        i, j = pairs[c]
        if i not in mate and j not in mate:
            mate[i], mate[j] = j, i
    around: dict[int, list[int]] = defaultdict(list)
    for c in order:
        i, j = pairs[c]
        around[i].append(j)
        around[j].append(i)
    grown = True
    while grown:
        grown = False
        for t in sorted(around):
            if t not in mate and _augment(t, around, mate, AUGMENT_DEPTH, {t}):
                grown = True
    return mate


def _augment(
    t: int, around: Mapping[int, list[int]], mate: dict[int, int], depth: int, seen: set[int]
) -> bool:
    """Grow the matching by an alternating path from the free triangle t; return success.

    The path leaves t by a pair not in the matching, crosses a matched pair,
    and so on, until it reaches a free triangle, at most ``depth`` matched
    pairs deep and never through a triangle twice; it is then flipped, which
    matches one more pair.
    """
    for u in around[t]:
        if u in seen:
            continue
        w = mate.get(u)
        if w is None:
            mate[t], mate[u] = u, t
            return True
        if depth > 0 and w not in seen and _augment(w, around, mate, depth - 1, seen | {u, w}):
            mate[t], mate[u] = u, t
            return True
    return False


def element_report(mode: str, faces: Faces) -> dict[str, Any]:
    """Return what ``refine.json`` states of a remeshed family's elements.

    Parameters
    ----------
    mode : str
        The family's element mode, ``"triangles"`` or ``"quad-dominant"``.
    faces : list of list of int
        Its faces in the level.

    Returns
    -------
    dict
        ``elements`` (the mode), ``quads``, ``triangles`` and ``quad_share``
        (the quadrilaterals over the faces, six decimals).
    """
    quads = sum(len(f) == 4 for f in faces)
    share = quads / len(faces) if faces else 0.0
    return {
        "elements": mode,
        "quads": quads,
        "triangles": len(faces) - quads,
        "quad_share": round(share, 6),
    }

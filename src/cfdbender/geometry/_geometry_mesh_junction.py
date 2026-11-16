"""Strategy 1 junction faces: the trimmed faces a junction curve crosses (delta F.2).

Pipeline role: the kernel half of the junction block step of the strategy 1 mesher. After the
configuration is fragmented, every face of a primary component that touches the transition
component (the wing-fuselage fairing) is one block of the generated layout, each source patch
being a block bounded by the layout lines (body junction stations, azimuth sectors, wing
bays), crossed or bounded by a junction curve. The pure classifier
:func:`~cfdbender.geometry._geometry_mesh_s1.junction_blocks` says whether the curve keeps
the block four-sided; here the face's boundary walk is measured for its corners. A face stays
structured (transfinite between four corners, a run of junction curves as one curved side) or
is meshed with the transition component, quad-dominant with sizes matched on its curves: the
fairing adapts to the fuselage grid, and quad-dominant cells stay inside it (MESH_DESIGN
section 8).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_mesh_occ import curve_endpoints, uv_edges

__all__ = [
    "CORNER_DEG",
    "MIN_CORNER_DEG",
    "block_sides",
    "distribute",
    "face_loop",
    "junction_faces",
    "runs",
]

#: A boundary vertex is a corner when its interior angle is below this (the iges-grid rule).
CORNER_DEG = 160.0
#: A structured junction block refuses a corner sharper than this (a cusp folds the grid).
MIN_CORNER_DEG = 20.0
#: Ids of the composite sides in the count classes, above every kernel curve tag.
VIRTUAL_BASE = 10**8


def face_loop(model: Any, surface: int) -> tuple[int, ...]:
    """Return the closed signed boundary walk of a face with one boundary loop."""
    pending = sorted(t for _, t in model.getBoundary([(2, surface)], oriented=False))
    first = pending.pop(0)
    start, here = curve_endpoints(model, first)
    walk = [first]
    while pending:
        for e in pending:
            a, b = curve_endpoints(model, e)
            if here in (a, b):
                walk.append(e if here == a else -e)
                here = b if here == a else a
                pending.remove(e)
                break
        else:
            raise InputValueError(f"surface {surface}: boundary is not one connected loop")
    if here != start:
        raise InputValueError(f"surface {surface}: boundary loop is not closed")
    return tuple(walk)


def _point(model: Any, edge: int, fraction: float) -> np.ndarray:
    lo, hi = model.getParametrizationBounds(1, abs(edge))
    t = lo[0] + (hi[0] - lo[0]) * (fraction if edge > 0 else 1 - fraction)
    return np.asarray(model.getValue(1, abs(edge), [t]))


def corner_angles(model: Any, loop: tuple[int, ...]) -> list[float]:
    """Return the interior angle, in degrees, at the start vertex of each curve of the walk."""
    angles = []
    for i, edge in enumerate(loop):
        a, b = curve_endpoints(model, edge)
        vertex = np.asarray(model.getValue(0, a if edge > 0 else b, []))
        before = _point(model, loop[i - 1], 0.98) - vertex
        after = _point(model, edge, 0.02) - vertex
        cosine = before @ after / (np.linalg.norm(before) * np.linalg.norm(after))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


def _uv(model: Any, surface: int, edge: int, count: int = 9) -> np.ndarray:
    points = np.array([_point(model, edge, f) for f in np.linspace(0.0, 1.0, count)])
    return np.asarray(model.getParametrization(2, surface, points.ravel())).reshape(-1, 2)


def runs(loop: tuple[int, ...], marked: set[int]) -> list[list[int]]:
    """Return the maximal cyclic runs of consecutive marked curves of the walk."""
    n = len(loop)
    if all(abs(e) in marked for e in loop):
        return [list(loop)]
    start = next(i for i in range(n) if abs(loop[i]) not in marked)
    runs: list[list[int]] = []
    current: list[int] = []
    for k in range(1, n + 1):
        edge = loop[(start + k) % n]
        if abs(edge) in marked:
            current.append(edge)
        elif current:
            runs.append(current)
            current = []
    return runs + ([current] if current else [])


def block_sides(loop: tuple[int, ...], corners: list[int]) -> tuple[tuple[int, ...], ...]:
    n = len(loop)
    return tuple(
        tuple(loop[(corners[k] + j) % n] for j in range((corners[(k + 1) % 4] - corners[k]) % n))
        for k in range(4)
    )


def _measure(model, surface, owner_name, junction_edges) -> dict[str, Any]:
    from cfdbender.geometry._geometry_mesh_s1 import junction_blocks

    loop = face_loop(model, surface)
    angles = corner_angles(model, loop)
    corners = [i for i, a in enumerate(angles) if a < CORNER_DEG]
    lo, hi = model.getParametrizationBounds(2, surface)
    name = f"{owner_name}.f{surface}"
    # Consecutive curves share their end point; keep it once so the run stays strictly monotone.
    curves = [
        np.vstack([_uv(model, surface, e)[1 if k else 0 :] for k, e in enumerate(run)])
        for run in runs(loop, junction_edges)
    ]
    block = junction_blocks({name: {"u": [lo[0], hi[0]], "v": [lo[1], hi[1]]}}, {name: curves})
    row: dict[str, Any] = {
        "surface": int(surface),
        "component": owner_name,
        "curves": len(loop),
        "corners": len(corners),
        "min_corner_deg": round(min(angles[i] for i in corners), 3) if corners else None,
        "block": block[name][0]["kind"],
    }
    if row["block"] == "quad_dominant":
        row["reason"] = "the junction crosses the block twice or is not single valued"
    elif len(corners) != 4:
        row["reason"] = f"{len(corners)} corners under {CORNER_DEG:g} deg, not four"
    elif row["min_corner_deg"] < MIN_CORNER_DEG:
        row["reason"] = f"a corner of {row['min_corner_deg']} deg (cusp) folds the grid"
    else:
        row["sides"] = block_sides(loop, corners)
    return row


def junction_faces(
    model: Any, faces, holder: Mapping[int, str], transition: str
) -> dict[int, dict[str, Any]]:
    """Decide how each primary face touching the transition component is meshed.

    ``faces`` are the faces next to an intersection curve, ``holder`` names each face's
    component and ``transition`` is the component meshed quad-dominant. A face of another
    component is ``structured`` when its block stays four-sided (one curved side at most), it
    has four corners none sharper than :data:`MIN_CORNER_DEG`, and a side of more than one
    curve borders only transition faces (its counts are then free); otherwise it is meshed with
    the transition component (``mesh = "transition"``) and the row says why. Returns one row
    per primary face, with ``sides`` (the four runs of signed curves) when it has more than
    four curves.
    """
    hosted = {s for s in faces if holder[s] != transition}
    absorbed = {s for s in holder if holder[s] == transition}
    junction = {
        t
        for s in absorbed
        for _, t in model.getBoundary([(2, s)], oriented=False)
        if any(holder.get(p) != transition for p in model.getAdjacencies(1, t)[0])
    }
    rows = {s: _measure(model, s, holder[s], junction) for s in sorted(hosted)}
    absorbed |= {s for s, r in rows.items() if "reason" in r}

    def bound(s: int, edge: int) -> bool:
        """Whether the curve also bounds a face meshed structured."""
        return bool({int(p) for p in model.getAdjacencies(1, abs(edge))[0]} - {s} - absorbed)

    while True:
        changed = False
        for s, row in rows.items():
            if s in absorbed:
                continue
            if any(len(side) > 1 and any(bound(s, e) for e in side) for side in row["sides"]):
                row["reason"] = "a side of several curves borders a structured face"
                row.pop("sides")
                absorbed.add(s)
                changed = True
        if not changed:
            break
    for s, row in rows.items():
        row["mesh"] = "transition" if "reason" in row else "structured"
        if row["mesh"] == "structured" and row["curves"] == 4 and _tensor(model, s):
            row.pop("sides")
    return rows


def _tensor(model: Any, surface: int) -> bool:
    """Whether the four curves of a face join at its parameter corners (an untrimmed patch).

    A trimmed face can keep four curves with a corner off the parameter box (a footprint end
    on an azimuth line); it is walked by its measured sides, not from UV (0, 0).
    """
    try:
        uv_edges(model, surface)
    except InputValueError:
        return False
    return True


def distribute(count: int, lengths: list[float]) -> list[int] | None:
    """Split ``count`` cells over curves in proportion to their lengths, one cell at least each.

    Returns ``None`` when there are fewer cells than curves.
    """
    if count < len(lengths):
        return None
    total = float(sum(lengths))
    cells = [max(1, int(round(count * length / total))) for length in lengths]
    while sum(cells) != count:
        step = 1 if sum(cells) < count else -1
        error = [c - count * length / total for c, length in zip(cells, lengths, strict=True)]
        order = np.argsort(error)
        pick = next(int(i) for i in (order if step > 0 else order[::-1]) if cells[i] + step >= 1)
        cells[pick] += step
    return cells

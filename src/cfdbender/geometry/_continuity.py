"""Tangent-plane jump audit across the shared edges of a patch set.

Pipeline role: pure numerical check of the geometry package (stage S9). It finds every edge that
two patches share, measures the turn of the surface across it, and compares the largest turn with
a limit. The measure needs no consistent patch orientation: it is the angle between the
cross-edge tangent of one patch and the reversed cross-edge tangent of the other, taken in the
plane perpendicular to the shared edge, so a smooth join reads 0 and a fold reads up to 180.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from cfdbender._errors import InputValueError

from ._bspline import full_knots
from ._patch import Patch
from ._records import check_entry

__all__ = ["audit_jumps", "bluff_jump_check", "edge_jumps"]

#: Edge names in the order ``Patch.edge_ids`` lists them.
EDGES = ("u0", "u1", "v0", "v1")
_SAMPLES = 17


def _edge_data(patch: Patch, edge: str, count: int = _SAMPLES):
    """Return points and inward cross-edge tangents along one edge of a patch."""
    controls = np.asarray(patch.controls, float)
    u = (patch.knots_u, patch.multiplicities_u, patch.degree_u)
    v = (patch.knots_v, patch.multiplicities_v, patch.degree_v)
    if edge in ("v0", "v1"):
        controls = controls.transpose(1, 0, 2)
        across, along = v, u
    else:
        across, along = u, v
    t_across, t_along = full_knots(*across[:2]), full_knots(*along[:2])
    curve = BSpline(t_across, controls, across[2])
    at = 0.0 if edge.endswith("0") else 1.0
    sign = 1.0 if at == 0.0 else -1.0
    base, slope = curve(at), curve.derivative()(at) * sign
    s = np.linspace(0.0, 1.0, count)
    along_base = BSpline(t_along, base, along[2], axis=0)
    points = along_base(s)
    tangent = along_base.derivative()(s)
    inward = BSpline(t_along, slope, along[2], axis=0)(s)
    return points, tangent, inward


def _unit(vectors):
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norm > 0, norm, 1.0)


def _turn(first, second):
    """Return the angle in degrees between inward tangents, seen perpendicular to the edge.

    At a corner whose two edges are collinear (a straight angle, as at the pole of a cap that
    spans half a turn) the inward tangent runs along the edge and has no part across it, so
    the turn there is undefined rather than a fold; such a sample reads 0. A sample is so only
    when the part across the edge is below 1e-9 of the inward tangent, which no real fold has.
    """
    edge = _unit(first[1])
    perpendicular, defined = [], True
    for inward in (first[2], second[2]):
        flat = inward - np.sum(inward * edge, axis=-1, keepdims=True) * edge
        size = np.linalg.norm(inward, axis=-1)
        defined = defined & (np.linalg.norm(flat, axis=-1) > 1e-9 * np.where(size > 0, size, 1.0))
        perpendicular.append(_unit(flat))
    cosine = np.clip(np.sum(perpendicular[0] * -perpendicular[1], axis=-1), -1.0, 1.0)
    return np.where(defined, np.degrees(np.arccos(cosine)), 0.0)


def edge_jumps(patches, tolerance: float | None = None, count: int = _SAMPLES) -> list[dict]:
    """Return one record per edge that two patches share, with the turn across it.

    Parameters
    ----------
    patches : sequence of Patch
        The surface pieces; an edge belongs to a pair when its sampled points coincide.
    tolerance : float, optional
        Largest distance in meters between the two curves of a shared edge; default
        1e-7 of the bounding diagonal of the set.
    count : int, optional
        Samples along each edge; at least 5.

    Returns
    -------
    list of dict
        ``a``, ``b`` (patch indices), ``edge_a``, ``edge_b``, ``jump_deg`` (largest along the
        edge) and ``kind`` (the two roles joined by ``|``, sorted).

    Raises
    ------
    InputValueError
        When ``count`` is below 5 or ``tolerance`` is not positive.
    """
    if count < 5:
        raise InputValueError("count: at least 5 samples along an edge")
    patches = list(patches)
    if not patches:
        return []
    data = {(i, e): _edge_data(p, e, count) for i, p in enumerate(patches) for e in EDGES}
    keys = list(data)
    everything = np.concatenate([data[k][0] for k in keys])
    diagonal = float(np.linalg.norm(everything.max(axis=0) - everything.min(axis=0)))
    if tolerance is None:
        tolerance = max(1e-7 * diagonal, 1e-12)
    elif not tolerance > 0:
        raise InputValueError("tolerance: expected a positive distance")
    middles = np.array([data[k][0][count // 2] for k in keys])
    tree = cKDTree(middles)
    records = []
    for index, key in enumerate(keys):
        for other in tree.query_ball_point(middles[index], r=tolerance):
            if other <= index or keys[other][0] == key[0]:
                continue
            first, second = data[key], data[keys[other]]
            for candidate in (second, tuple(x[::-1] for x in second)):
                if np.max(np.linalg.norm(first[0] - candidate[0], axis=-1)) <= tolerance:
                    roles = sorted((patches[key[0]].role, patches[keys[other][0]].role))
                    records.append(
                        {
                            "a": key[0],
                            "b": keys[other][0],
                            "edge_a": key[1],
                            "edge_b": keys[other][1],
                            "jump_deg": float(np.max(_turn(first, candidate))),
                            "kind": "|".join(roles),
                        }
                    )
                    break
    return records


def _crease_angles(record, patches, creases):
    """Return the crease angles declared for this edge, and the entries that matched."""
    names = {
        record["kind"],
        patches[record["a"]].edge_ids[EDGES.index(record["edge_a"])]
        if patches[record["a"]].edge_ids
        else "",
        patches[record["b"]].edge_ids[EDGES.index(record["edge_b"])]
        if patches[record["b"]].edge_ids
        else "",
    }
    hits = [i for i, c in enumerate(creases) if c["at"] in names]
    return [float(creases[i]["deg"]) for i in hits], hits


def audit_jumps(patches, limit_deg: float = 1.0, creases=(), tolerance: float | None = None):
    """Audit the tangent-plane jump of every shared edge against ``limit_deg``.

    Parameters
    ----------
    patches : sequence of Patch
        The surface pieces of one component.
    limit_deg : float, optional
        Largest jump allowed on an edge with no declared crease; default 1 degree.
    creases : sequence of dict, optional
        Declared creases, each ``{"at": <edge name or "roleA|roleB">, "deg": <angle>}``. The
        angle is the turn the crease has, not a ceiling: an edge that matches passes when its
        turn lies within ``limit_deg`` of a declared angle, so a declaration cannot hide a smooth
        seam that kinked or a fold larger than the crease.
    tolerance : float, optional
        Shared-edge distance, as :func:`edge_jumps`.

    Returns
    -------
    dict
        ``max_jump_deg`` over edges that no crease covers, ``edges`` (the records, each with an
        ``allowed_deg``: the declared crease angle it is held to, or ``limit_deg``; and a
        ``deviation_deg``: its distance from that crease angle, or its turn), ``failed``
        (indices into ``edges``), ``by_kind`` (largest jump per edge kind), ``unmatched_creases``
        (entries that matched no edge), ``free_edges`` (edges of no pair, for example on a
        symmetry plane) and ``unpaired_edges`` (edge names two patches declare shared whose
        curves do not coincide at equal parameters: unmeasured, so the check fails them).

    Raises
    ------
    InputValueError
        When ``limit_deg`` is negative or a crease lacks ``at`` or ``deg``.
    """
    if limit_deg < 0:
        raise InputValueError("limit_deg: expected a non-negative angle")
    creases = [dict(c) for c in creases]
    for entry in creases:
        if not isinstance(entry.get("at"), str) or not isinstance(entry.get("deg"), int | float):
            raise InputValueError("creases: each entry needs at (a name) and deg (a number)")
    patches = list(patches)
    records = edge_jumps(patches, tolerance)
    used: set[int] = set()
    worst, failed = 0.0, []
    by_kind: dict[str, float] = {}
    for i, record in enumerate(records):
        angles, hits = _crease_angles(record, patches, creases)
        used.update(hits)
        jump = record["jump_deg"]
        if angles:
            # the declared angle nearest the measured turn; the turn must reproduce it
            record["allowed_deg"] = min(angles, key=lambda a: abs(a - jump))
            record["deviation_deg"] = abs(jump - record["allowed_deg"])
        else:
            record["allowed_deg"] = limit_deg
            record["deviation_deg"] = jump
        by_kind[record["kind"]] = max(by_kind.get(record["kind"], 0.0), jump)
        if record["deviation_deg"] > limit_deg:
            failed.append(i)
        if not hits:
            worst = max(worst, record["jump_deg"])
    return {
        "max_jump_deg": worst,
        "edges": records,
        "failed": failed,
        "by_kind": by_kind,
        "unmatched_creases": [c for i, c in enumerate(creases) if i not in used],
        "free_edges": 4 * len(patches) - 2 * len(records),
        "unpaired_edges": _unpaired_names(patches, records),
    }


def _unpaired_names(patches, records):
    """Return the edge names two patches share that the geometric pairing did not join.

    Pairing compares samples at equal parameters, so two patches that meet along an edge with a
    different parameter speed, or that leave a gap there, form no pair and their turn is never
    measured. A name the patches declare shared is the statement that they meet, so it is held to.
    """
    owners: dict[str, list[tuple[int, str]]] = {}
    for index, patch in enumerate(patches):
        for edge, name in zip(EDGES, patch.edge_ids, strict=False):
            owners.setdefault(name, []).append((index, edge))
    paired = {frozenset({(r["a"], r["edge_a"]), (r["b"], r["edge_b"])}) for r in records}
    return sorted(
        name
        for name, sides in owners.items()
        if len(sides) == 2 and sides[0][0] != sides[1][0] and frozenset(sides) not in paired
    )


def bluff_jump_check(result, limit_deg: float = 1.0) -> dict:
    """Return the ``continuity.bluff_jump`` check entry of an :func:`audit_jumps` result.

    A failed edge reports its deviation: its turn, or its distance from its declared crease. An
    edge two patches name as shared but that the audit could not measure fails at 180 degrees,
    the turn nothing bounds.
    """
    worst = 0.0
    where = {}
    unpaired = result.get("unpaired_edges", [])
    if unpaired and not result["failed"]:
        return check_entry(
            "continuity.bluff_jump",
            "failed",
            180.0,
            limit_deg,
            unit="deg",
            where={"edge": unpaired[0]},
            note=f"{len(unpaired)} shared edge names do not coincide: " + ", ".join(unpaired[:5]),
        )
    if result["failed"]:
        record = result["edges"][
            max(result["failed"], key=lambda i: result["edges"][i]["deviation_deg"])
        ]
        worst = record["deviation_deg"]
        where = {k: record[k] for k in ("a", "b", "edge_a", "edge_b", "kind")}
    return check_entry(
        "continuity.bluff_jump",
        "failed" if result["failed"] else "passed",
        worst if result["failed"] else result["max_jump_deg"],
        limit_deg,
        unit="deg",
        where=where,
    )

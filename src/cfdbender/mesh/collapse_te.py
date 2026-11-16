"""Collapse a blunt trailing edge of named blades into one sharp line (OBJ).

A blunt trailing edge is a strip of one cell between two rims. Each pair of rim nodes joined
across the strip is merged at its midpoint, the strip faces leave, the faces that met the strip
lose their repeated node, and the two rim entries of the boundaries file become one line. Every
other node keeps its text and its place, and every other face its place and its node order,
because a solver's results depend on face order.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import (
    GeometryWarning,
    InputValueError,
    NotApplicableWarning,
    warn,
)
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.mesh._refine._obj import (
    ObjMesh,
    mesh_unit,
    obj_text,
    points_from_file,
    read_obj,
    read_sidecar,
    write_te,
)
from cfdbender.mesh._repitch_io import (
    Rewrite,
    attachments,
    morphing_table,
    one_block_per_family,
    read_back,
    read_control,
    read_rotor,
    sidecar_reads_back,
)
from cfdbender.mesh._sidecars import read_boundary_declarations
from cfdbender.mesh._wheel import Faces, parse_axis, wheel_symmetry

__all__ = ["collapse_trailing_edge"]

Edge = tuple[int, int]


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise InputValueError(f"collapse_te: {message}. No file was written.")


def _key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _sides(face: list[int]) -> list[Edge]:
    return [_key(a, b) for a, b in zip(face, face[1:] + face[:1], strict=True)]


@dataclass
class _Plan:
    """One blade to collapse: its strip, its two rims and the entries that named them."""

    blade: str
    route: str
    strip: list[tuple[str, int]]
    rim_a: list[int]
    rim_b: list[int]
    labels: list[str] = field(default_factory=list)
    edge_type: str = "STANDARD"
    unit: str | None = None
    to_file: float = 1.0

    @property
    def pairs(self) -> list[Edge]:
        return list(zip(self.rim_a, self.rim_b, strict=True))


@dataclass
class _Result:
    """The collapsed mesh with what the checks need to compare it with the source."""

    source: ObjMesh
    mesh: ObjMesh
    plans: list[_Plan]
    index: np.ndarray
    moved: dict[int, np.ndarray]
    strip: set[tuple[str, int]]
    kept: list[tuple[str, int]]
    lines: dict[str, np.ndarray]
    collapsed: int
    closure: dict[str, int]
    nothing: dict[str, int]


def _config(path: Path) -> dict[str, Any]:
    data = read_control(path, "collapse_te")
    table = morphing_table(data, "collapse_te", ("blades",))
    _need(isinstance(table, dict), f"{path} needs a [collapse_te] table")
    cfg = dict(table)
    for key in set(cfg) - {"blades", "base"}:
        warn(f"collapse_te.{key} does not apply and is skipped", NotApplicableWarning)
    return cfg


def _names(cfg: dict[str, Any], key: str) -> list[str]:
    value = cfg.get(key, [])
    if key == "blades" and value in ("all", ["all"]):
        return ["all"]
    _need(
        isinstance(value, list) and all(isinstance(n, str) for n in value),
        f"collapse_te.{key} must be a list of family names",
    )
    _need(len(set(value)) == len(value), f"collapse_te.{key} repeats a family")
    return list(value)


class _Index:
    """The edges of the whole mesh with the faces that own them."""

    def __init__(self, mesh: ObjMesh) -> None:
        self.owners: dict[Edge, list[tuple[str, int]]] = {}
        for name, faces in mesh.families.items():
            for i, face in enumerate(faces):
                for e in _sides(face):
                    self.owners.setdefault(e, []).append((name, i))
        self.keys = list(self.owners)
        pairs = np.array(self.keys, dtype=np.int64).reshape(-1, 2)
        self.mids = 0.5 * (mesh.verts[pairs[:, 0]] + mesh.verts[pairs[:, 1]])


def _components(edges: set[Edge]) -> list[tuple[set[int], set[Edge]]]:
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set[int] = set()
    out = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack, nodes = [start], set()
        while stack:
            x = stack.pop()
            if x not in nodes:
                nodes.add(x)
                stack.extend(adj[x])
        seen |= nodes
        out.append((nodes, {e for e in edges if e[0] in nodes}))
    return out


def _path(blade: str, nodes: set[int], edges: set[Edge]) -> list[int]:
    """Order one rim as a polyline from its lower-numbered end, refusing a branch or a loop."""
    degree = Counter(n for e in edges for n in e)
    ends = sorted(n for n, d in degree.items() if d == 1)
    _need(
        max(degree.values()) <= 2 and len(edges) == len(nodes) - 1 and len(ends) == 2,
        f"blade {blade!r}: a rim is not one open polyline (it branches or closes on itself)",
    )
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    line = [ends[0]]
    while len(line) < len(nodes):
        line.append(next(n for n in adj[line[-1]] if n not in line[-2:]))
    return line


def _pair(blade: str, a: list[int], b: list[int], strip_edges: set[Edge]) -> list[int]:
    """Return rim B ordered so that its k-th node pairs with rim A's on a strip edge."""
    _need(
        len(a) == len(b),
        f"blade {blade!r}: the rims hold {len(a)} and {len(b)} nodes, so no node pairs "
        "the two sides; both rims must have the same number of nodes",
    )
    fits = [
        r for r in (b, b[::-1]) if all(_key(x, y) in strip_edges for x, y in zip(a, r, strict=True))
    ]
    _need(
        len(fits) == 1,
        f"blade {blade!r}: "
        + (
            "both orders of the second rim pair with the first"
            if fits
            else "the rim nodes are not joined pair by pair across the strip, so the strip is "
            "more than one cell wide or the rims are not its sides"
        ),
    )
    return fits[0]


def _cells(blade: str, mesh: ObjMesh, strip: list[tuple[str, int]], a: list[int], b: list[int]):
    """Require one quad or two triangles per rung interval, between consecutive rung pairs."""
    place = {n: j for j, n in enumerate(a)} | {n: j for j, n in enumerate(b)}
    cells: dict[int, list[int]] = {}
    for fam, i in strip:
        face = mesh.families[fam][i]
        j = [place.get(n, -9) for n in face]
        _need(
            min(j) >= 0 and max(j) - min(j) == 1,
            f"blade {blade!r}: a strip face is not between two consecutive rungs; the strip is "
            "more than one cell wide",
        )
        cells.setdefault(min(j), []).append(len(face))
    ok = all(cells.get(j) in ([4], [3, 3]) for j in range(len(a) - 1)) and len(cells) == len(a) - 1
    _need(ok, f"blade {blade!r}: each strip interval must be one quad or two triangles")


def _entry_edges(source: Path, decl: Any, index: _Index, unit: str | None):
    """Match each file entry's points to mesh edges: label, type, edge keys, unmatched, points."""
    from scipy.spatial import cKDTree

    tree = cKDTree(index.mids)
    metres = float(METRES_PER_UNIT.get(unit or "METER", 1.0))
    found = []
    for entry in decl.trailing_edges:
        marking = entry.marking
        if marking.route != "file":
            continue
        for one in marking.sets:
            loaded = points_from_file(source, Path(str(one.points_file)))
            assert loaded is not None
            distance, ids = tree.query(loaded.points)
            tol = marking.tolerance / metres
            hit = distance <= tol
            found.append(
                (
                    entry.label,
                    one.edge_type,
                    {index.keys[i] for i in ids[hit].tolist()},
                    int((~hit).sum()),
                    loaded,
                )
            )
    return found


def _owner(plan_edges: set[Edge], index: _Index, family: str) -> int:
    return sum(any(f == family for f, _ in index.owners[e]) for e in plan_edges)


def _rims_only(mesh: ObjMesh, mine: list) -> list:
    """Return the entries of a blade that name rims: those that lie on the sides of its strip.

    The strip is the faces wholly on the nodes the entries declare; an entry with no edge on its
    sides (a flap line on the blade) names no rim and is carried to its new position. An entry
    with only part of its edges there stays a rim, so that a rim one interval short is still
    refused as unequal to the other. When no face lies wholly on those nodes, or no entry lies on
    the strip's sides, every entry is kept and the edge is judged as the entries declare it.
    """
    nodes = {n for m in mine for e in m[2] for n in e}
    sides = {
        e for faces in mesh.families.values() for f in faces if set(f) <= nodes for e in _sides(f)
    }
    rims = [m for m in mine if m[2] & sides]
    return rims or mine


def _points_plan(blade: str, mesh: ObjMesh, mine: list) -> _Plan | None:
    """Plan one blade from its declared rims; None when its edge is sharp."""
    edges = set().union(*[m[2] for m in mine])
    _need(
        all(m[3] == 0 for m in mine),
        f"blade {blade!r}: {sum(m[3] for m in mine)} declared trailing-edge points match no "
        "mesh edge",
    )
    # The first declared rim leads, so its nodes keep their places in the node list; one set
    # naming both rims gives no order and the rim holding the lowest node leads.
    parts = sorted(_components(edges), key=lambda p: (not p[1] & mine[0][2], min(p[0])))
    nodes = set().union(*[p[0] for p in parts])
    strip = [
        (fam, i)
        for fam, faces in mesh.families.items()
        for i, face in enumerate(faces)
        if all(n in nodes for n in face)
    ]
    if len(parts) == 1 and not strip:
        return None
    _need(
        len(parts) == 2,
        f"blade {blade!r}: the declared trailing edge forms {len(parts)} separate rims; a blunt "
        "edge has two",
    )
    a = _path(blade, *parts[0])
    b = _path(blade, *parts[1])
    s_edges = {e for fam, i in strip for e in _sides(mesh.families[fam][i])}
    b = _pair(blade, a, b, s_edges)
    _cells(blade, mesh, strip, a, b)
    types = {m[1] for m in mine}
    _need(len(types) == 1, f"blade {blade!r}: its rims are declared with different edge types")
    first = mine[0][4]
    return _Plan(blade, "boundaries", strip, a, b, [], types.pop(), first.unit, first.to_file)


def _base_plan(blade: str, group: str, mesh: ObjMesh, index: _Index) -> _Plan:
    """Plan one blade from the group holding its strip; the rims are the strip's sides."""
    _need(group != blade, f"blade {blade!r}: base must name the group holding the strip")
    strip = [(group, i) for i in range(len(mesh.families[group]))]
    count = Counter(e for _, i in strip for e in _sides(mesh.families[group][i]))
    boundary = {e for e, c in count.items() if c == 1}
    loop_nodes = {n for e in boundary for n in e}
    strip_nodes = {n for _, i in strip for n in mesh.families[group][i]}
    _need(
        loop_nodes == strip_nodes,
        f"blade {blade!r}: strip {group!r} has nodes inside it, so it is more than one cell wide",
    )
    adj: dict[int, list[int]] = {}
    for x, y in boundary:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    _need(
        bool(adj) and all(len(v) == 2 for v in adj.values()),
        f"blade {blade!r}: strip {group!r} is not one closed band",
    )
    loop = [min(adj)]
    while len(loop) < len(adj):
        loop.append(next(n for n in adj[loop[-1]] if n not in loop[-2:]))
    _need(
        len(loop) == len(adj) and len(loop) % 2 == 0 and len(loop) >= 4,
        f"blade {blade!r}: strip {group!r} boundary is not one loop of an even length",
    )
    size = len(loop)
    k = size // 2 - 1
    s_edges = set(count)
    best = []
    for i in range(k + 1):
        a = [loop[(i + 1 + j) % size] for j in range(k + 1)]
        b = [loop[(i - j) % size] for j in range(k + 1)]
        if not all(_key(x, y) in s_edges for x, y in zip(a, b, strict=True)):
            continue
        rim = {_key(a[j], a[j + 1]) for j in range(k)} | {_key(b[j], b[j + 1]) for j in range(k)}
        if all(any(f == blade for f, _ in index.owners[e]) for e in rim):
            best.append((a, b))
    _need(
        len(best) >= 1,
        f"blade {blade!r}: no pair of strip sides lies against the blade; check base",
    )
    _need(
        len(best) == 1,
        f"blade {blade!r}: strip {group!r} sides cannot be told from its ends; declare the "
        "rims in the boundaries file",
    )
    a, b = best[0]
    _cells(blade, mesh, strip, a, b)
    return _Plan(blade, "base", strip, a, b)


def _all_blades(
    decl: Any, entries: list, mesh: ObjMesh, index: _Index, skipped: dict[str, str]
) -> list[str]:
    """Return the family each points entry lies on; an entry with no points is said skipped."""
    for entry in decl.trailing_edges:
        if entry.marking.route != "file":
            skipped[entry.label] = "marked by detection, so no rims are declared"
            warn(
                f"collapse_te: trailing-edge entry {entry.label!r} is marked by detection, so no "
                "rims are declared, and it is skipped; name its blade and strip with blades and "
                "base",
                NotApplicableWarning,
            )
    # A family held wholly by the rims is a strip, whether or not a base region declares it.
    rim_nodes = {n for m in entries for e in m[2] for n in e}
    strips = {f for f, faces in mesh.families.items() if all(set(x) <= rim_nodes for x in faces)}
    declared = {r.family for r in decl.base_regions} | strips
    blades: list[str] = []
    for m in entries:
        score = {f: _owner(m[2], index, f) for f in mesh.families}
        pool = {f: s for f, s in score.items() if f not in declared and s} or score
        top = max(pool, key=lambda f: pool[f], default=None)
        if top is not None and pool[top] and top not in blades:
            blades.append(top)
    return blades


def _requested(
    cfg: dict[str, Any],
    side: Any,
    decl: Any,
    entries: list,
    *,
    mesh: ObjMesh,
    index: _Index,
    skipped: dict[str, str],
) -> tuple[list[str], list[str], bool]:
    """Return the blades and strip groups the table asks for, and whether it named the blades."""
    blades, bases = _names(cfg, "blades"), _names(cfg, "base")
    explicit = blades != ["all"]
    if not explicit:
        _need(not bases, "base names the strip of an explicit blade; it cannot go with blades=all")
        _need(
            decl is not None,
            "blades = all reads the boundaries file, and none that declares trailing edges sits "
            "beside the mesh. Name the blades and their strips with blades and base",
        )
        blades = _all_blades(decl, entries, mesh, index, skipped)
        _need(
            bool(blades),
            "blades = all found no trailing-edge entry with a points file in the boundaries "
            "file. Name the blades and their strips with blades and base",
        )
    else:
        absent = [n for n in blades + bases if n not in mesh.families]
        _need(not absent, f"families {absent} are absent from the mesh")
    _need(bool(blades), "collapse_te.blades names no family")
    if bases:
        _need(
            len(bases) == len(blades),
            "collapse_te.base needs one strip group for each blade, in the order of blades",
        )
    else:
        _need(
            side is not None,
            "the mesh has no boundaries file, so its trailing-edge rims are not declared; "
            "provide it, or name the strip group of each blade with collapse_te.base",
        )
    return blades, bases, explicit


def _plan_by_points(
    blade: str, mesh: ObjMesh, mine: list, explicit: bool, skipped: dict[str, str]
) -> _Plan | None:
    """Plan a blade from its declared rims, or say why it is skipped and return None."""
    _need(
        bool(mine),
        f"blade {blade!r} has no trailing-edge entry with a points file in the "
        "boundaries file; declare its rims or name its strip group with collapse_te.base",
    )
    try:
        plan = _points_plan(blade, mesh, mine)
    except InputValueError as error:
        if explicit:
            raise
        # The refusal's own words say no file was written; here the others go on.
        why = str(error).removeprefix("collapse_te: ").removesuffix(" No file was written.")
        skipped[blade] = why
        warn(f"collapse_te: {why} Family {blade!r} is skipped", NotApplicableWarning)
        return None
    if plan is None:
        skipped[blade] = "one declared rim with no strip on its nodes: taken as already sharp"
        warn(
            f"collapse_te: blade {blade!r} declares one rim with no strip on its nodes, so it "
            "is taken as already sharp and is skipped. If it is blunt, declare its other rim; "
            "a blade split across families is named by the group holding its strip",
            NotApplicableWarning,
        )
    return plan


def _labelled(plan: _Plan, mine: list, unit: str | None) -> _Plan:
    """Give a plan the labels, the edge type and the units of the entries that name its rims."""
    plan.labels = sorted({m[0] for m in mine})
    if mine:
        plan.edge_type = mine[0][1]
        plan.unit, plan.to_file = mine[0][4].unit, mine[0][4].to_file
    else:
        plan.unit = unit or "METER"
    return plan


def _select(cfg: dict[str, Any], source: Path, mesh: ObjMesh, index: _Index):
    """Return the plans, the sidecar and the blades skipped that the request leads to."""
    side = read_sidecar(source)
    decl = None
    if side is not None and ("trailing_edges" in side[1] or "base_regions" in side[1]):
        decl = read_boundary_declarations(side[0])
    unit = mesh_unit(source)
    entries = _entry_edges(source, decl, index, unit) if decl is not None else []
    skipped: dict[str, str] = {}
    blades, bases, explicit = _requested(
        cfg, side, decl, entries, mesh=mesh, index=index, skipped=skipped
    )
    plans: list[_Plan] = []
    for n, blade in enumerate(blades):
        mine = [m for m in entries if _best(m[2], index, blades) == blade]
        plan: _Plan | None
        if bases:
            plan = _base_plan(blade, bases[n], mesh, index)
            # Only an entry wholly on the strip's sides names a rim; any other marking of the
            # family is no rim and is carried to its new position.
            rims = _rim_edges(plan)
            mine = [m for m in mine if m[2] and m[2] <= rims and not m[3]]
            _first_rim_leads(plan, mine)
        else:
            mine = _rims_only(mesh, mine)
            plan = _plan_by_points(blade, mesh, mine, explicit, skipped)
        if plan is not None:
            plans.append(_labelled(plan, mine, unit))
    return plans, side, skipped, unit


def _first_rim_leads(plan: _Plan, mine: list) -> None:
    """Order a strip's sides as the points route does: the first declared rim leads."""
    if not mine:
        return
    sides = [plan.rim_a, plan.rim_b]
    edges = [{_key(r[j], r[j + 1]) for j in range(len(r) - 1)} for r in sides]
    lead = min((0, 1), key=lambda k: (not edges[k] & mine[0][2], min(sides[k])))
    if lead == 1:
        plan.rim_a, plan.rim_b = plan.rim_b, plan.rim_a


def _best(edges: set[Edge], index: _Index, blades: list[str]) -> str | None:
    score = {b: _owner(edges, index, b) for b in blades}
    top = max(score, key=lambda b: score[b])
    return top if score[top] else None


def _dedupe(face: list[Any]) -> list[Any]:
    """Drop each node that repeats its predecessor, and a last that repeats the first."""
    out: list[Any] = []
    for n in face:
        if not out or n != out[-1]:
            out.append(n)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _pairs_to_merge(old: np.ndarray, plans: list[_Plan]):
    """Return which node each leaving node merges into, and the midpoint of each merged node."""
    merge: dict[int, int] = {}
    moved: dict[int, np.ndarray] = {}
    for plan in plans:
        for a, b in plan.pairs:
            _need(
                a not in merge and b not in merge and a != b and a not in moved and b not in moved,
                f"blade {plan.blade!r}: node {max(a, b)} belongs to more than one rung pair",
            )
            merge[b] = a
            moved[a] = moved[b] = 0.5 * (old[a] + old[b])
    return merge, moved


def _repaired(name: str, i: int, face: list[int], merge: dict[int, int]) -> list[int]:
    """Return a face the merge reaches without its repeated nodes, refusing what cannot close."""
    fixed = _dedupe([merge.get(n, n) for n in face])
    _need(
        len(set(fixed)) == len(fixed),
        f"face {i + 1} of {name!r} repeats a node that is not next to its twin after the merge",
    )
    return fixed


def _tally(counts: dict[str, int], blades: set[str]) -> None:
    for blade in blades:
        counts[blade] = counts.get(blade, 0) + 1


def _build(source: ObjMesh, plans: list[_Plan]) -> _Result:
    """Merge the rung pairs (the first rim's node stays), drop the strips, repair faces."""
    old = source.verts
    merge, moved = _pairs_to_merge(old, plans)
    gone = set(merge)
    index = np.cumsum([0 if i in gone else 1 for i in range(len(old))]) - 1
    for leaving, stays in merge.items():
        index[leaving] = index[stays]
    verts = old[[i for i in range(len(old)) if i not in gone]].copy()
    text = source.vertex_text
    new_text: list[str | None] = (
        [t for i, t in enumerate(text) if i not in gone] if len(text) == len(old) else []
    )
    for stays in set(merge.values()):
        verts[index[stays]] = moved[stays]
        if new_text:
            new_text[index[stays]] = None
    strip = {s for plan in plans for s in plan.strip}
    families: dict[str, list[list[int]]] = {}
    kept: list[tuple[str, int]] = []
    closure: dict[str, int] = {}
    nothing: dict[str, int] = {}
    collapsed = 0
    holder = {n: plan.blade for plan in plans for pair in plan.pairs for n in pair}
    for name, faces in source.families.items():
        rows: list[list[int]] = []
        for i, face in enumerate(faces):
            if (name, i) in strip:
                continue
            near = {holder[n] for n in face if n in holder}
            fixed = _repaired(name, i, face, merge) if near else face
            if len(fixed) < 3:
                collapsed += 1
                _tally(nothing, near)
                continue
            if len(fixed) < len(face):
                _tally(closure, near)
            rows.append([int(index[n]) for n in fixed])
            kept.append((name, i))
        if rows:
            families[name] = rows
    mesh = ObjMesh(verts, families, list(source.header), source.family_tag, new_text)
    lines = {plan.blade: np.array([int(index[a]) for a in plan.rim_a]) for plan in plans}
    return _Result(
        source, mesh, plans, index, moved, strip, kept, lines, collapsed, closure, nothing
    )


def _edge_faces(mesh: ObjMesh) -> dict[Edge, list[str]]:
    out: dict[Edge, list[str]] = {}
    for name, faces in mesh.families.items():
        for face in faces:
            for e in _sides(face):
                out.setdefault(e, []).append(name)
    return out


def _area(points: np.ndarray, face: list[int]) -> float:
    p = points[face]
    return float(0.5 * np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))


def _normal(points: np.ndarray) -> np.ndarray:
    return np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)


def _rows(points: np.ndarray) -> list[tuple[float, ...]]:
    return [tuple(float(x) for x in p) for p in points]


def _open_by_place(mesh: ObjMesh, moved: dict[int, np.ndarray]) -> set[frozenset]:
    """Return the edges owned by one face, as the places of their ends (merged ends moved)."""
    owners = _edge_faces(mesh)
    out = set()
    for e, f in owners.items():
        if len(f) != 1:
            continue
        ends = frozenset(tuple((moved[n] if n in moved else mesh.verts[n]).tolist()) for n in e)
        if len(ends) == 2:
            out.add(ends)
    return out


def _no_crack(src: ObjMesh, moved: dict[int, np.ndarray], twins: set[Edge]) -> None:
    """Refuse a merge that parts two used nodes sharing one place: it opens a crack there."""
    used = {n for faces in src.families.values() for f in faces for n in f}
    for p, q in sorted(twins):
        place_p, place_q = moved.get(p, src.verts[p]), moved.get(q, src.verts[q])
        _need(
            p not in used or q not in used or np.array_equal(place_p, place_q),
            f"source nodes {p + 1} and {q + 1} share one place and the merge moves them apart, "
            "which opens a crack in the mesh; join them in the source first",
        )


def _check_counts(r: _Result) -> dict[str, int]:
    n_strip = len(r.strip)
    src_faces = sum(len(f) for f in r.source.families.values())
    out_faces = sum(len(f) for f in r.mesh.families.values())
    _need(
        out_faces == src_faces - n_strip - r.collapsed,
        f"the output holds {out_faces} faces, not {src_faces} - {n_strip} - {r.collapsed}",
    )
    n_old, n_new = len(r.source.verts), len(r.mesh.verts)
    n_pairs = sum(len(plan.pairs) for plan in r.plans)
    _need(n_new == n_old - n_pairs, f"the output holds {n_new} nodes, not {n_old} - {n_pairs}")
    return {"source": src_faces, "strip": n_strip, "collapsed": r.collapsed}


def _check_near_faces(r: _Result) -> None:
    """Hold the faces the merge reached; a face away from it is held to the source below."""
    touched = set(r.moved)
    written = [(n, f) for n, faces in r.mesh.families.items() for f in faces]
    for (name, i), (family, f) in zip(r.kept, written, strict=True):
        if set(r.source.families[name][i]) & touched:
            _need(len(set(f)) == len(f), f"face {i + 1} of {family!r} repeats a node")
            _need(_area(r.mesh.verts, f) > 0.0, f"face {i + 1} of {family!r} has zero area")


def _check_twins(r: _Result) -> dict[str, int]:
    from scipy.spatial import cKDTree

    source_twins = cKDTree(r.source.verts).query_pairs(0.0)
    before = len(source_twins)
    twins = len(cKDTree(r.mesh.verts).query_pairs(0.0))
    _need(twins <= before, f"{twins - before} node pair(s) share one place after the merge")
    _no_crack(r.source, r.moved, source_twins)
    return {"source": before, "output": twins}


def _check_lines(r: _Result, owners: dict[Edge, list[str]]) -> None:
    for plan in r.plans:
        chain = r.lines[plan.blade]
        _need(
            len(chain) == len(plan.rim_a) and len(set(chain.tolist())) == len(chain),
            f"blade {plan.blade!r}: its trailing-edge line holds {len(set(chain.tolist()))} "
            f"nodes, not the {len(plan.rim_a)} of one rim",
        )
        for x, y in zip(chain[:-1].tolist(), chain[1:].tolist(), strict=True):
            sides = owners.get(_key(x, y), [])
            _need(
                len(sides) == 2,
                f"blade {plan.blade!r}: a trailing-edge edge is owned by {len(sides)} faces, "
                "not two",
            )


def _check_open(r: _Result, owners: dict[Edge, list[str]]) -> dict[str, int]:
    now = _open_by_place(r.mesh, {})
    was = _open_by_place(r.source, r.moved)
    _need(now == was, f"open edges differ from the source: {len(now ^ was)} edge(s) changed")
    over = sum(len(f) > 2 for f in owners.values())
    over0 = sum(len(f) > 2 for f in _edge_faces(r.source).values())
    _need(over <= over0, f"{over - over0} edge(s) now have more than two faces")
    return {"source": len(was), "output": len(now)}


def _check_far_nodes(r: _Result) -> None:
    src, out = r.source, r.mesh
    far = [i for i in range(len(src.verts)) if i not in r.moved]
    far_new = [int(r.index[i]) for i in far]
    _need(
        np.array_equal(out.verts[far_new], src.verts[far]),
        "a node away from the trailing edge differs from the source",
    )
    if out.vertex_text and src.vertex_text:
        _need(
            all(
                out.vertex_text[j] == src.vertex_text[i] for i, j in zip(far, far_new, strict=True)
            ),
            "a node away from the trailing edge is written differently from the source",
        )
    _need(far_new == sorted(far_new), "nodes away from the trailing edge changed order")


def _check_faces(r: _Result) -> None:
    src, old = r.source, r.source.verts
    expected = []
    for name, i in r.kept:
        face = src.families[name][i]
        rows = _rows(np.array([r.moved[n] if n in r.moved else old[n] for n in face]))
        expected.append((name, _dedupe(rows) if set(face) & set(r.moved) else rows))
    actual = [(n, _rows(r.mesh.verts[f])) for n, faces in r.mesh.families.items() for f in faces]
    wanted = [(n, rows) for n, rows in expected if len(rows) >= 3]
    _need(
        len(actual) == len(wanted)
        and all(a[0] == w[0] and a[1] == w[1] for a, w in zip(actual, wanted, strict=True)),
        "a face away from the trailing edge differs from the source in place, family or node order",
    )


def _turned_faces(r: _Result) -> dict[str, Any]:
    """Measure the faces the merge reached against their source normals; count those turned.

    A face whose normal now opposes the source's (cosine at or below zero) is a quality finding:
    it is counted, with the worst cosine and the family, face and place of the worst one, and
    is warned of; nothing refuses here.
    """
    old, new = r.source.verts, r.mesh.verts
    low, worst, count = 1.0, None, 0
    flat = [(n, f) for n, fs in r.mesh.families.items() for f in fs]
    for (name, i), (family, face) in zip(r.kept, flat, strict=False):
        before_f = r.source.families[name][i]
        if not set(before_f) & set(r.moved):
            continue
        a, b = _normal(old[before_f]), _normal(new[face])
        cosine = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
        count += cosine <= 0.0
        if cosine < low:
            low, worst = cosine, [family, i + 1, new[face].mean(axis=0).tolist()]
    return {"count": count, "worst_cosine": low, "worst": worst}


def _turned_warning(turned: dict[str, Any]) -> list[str]:
    """Return the sentence that states the faces the merge turned over, or nothing."""
    if not turned["count"]:
        return []
    family, face, place = turned["worst"]
    return [
        f"{turned['count']} face(s) near the trailing edge are turned over by the merge; the "
        f"worst, face {face} of {family!r} in the source, has cosine "
        f"{turned['worst_cosine']:.4f} against its source normal, at "
        f"[{', '.join(f'{v:.6g}' for v in place)}]"
    ]


def _area_report(r: _Result) -> dict[str, float]:
    src, out = r.source, r.mesh
    area0 = sum(_area(src.verts, f) for fs in src.families.values() for f in fs)
    area_strip = sum(_area(src.verts, src.families[n][i]) for n, i in r.strip)
    area1 = sum(_area(out.verts, f) for fs in out.families.values() for f in fs)
    return {
        "source": area0,
        "strip": area_strip,
        "output": area1,
        "relative_change_against_source_minus_strip": area1 / (area0 - area_strip) - 1,
    }


def _check_midpoints(r: _Result) -> None:
    old, new = r.source.verts, r.mesh.verts
    for plan in r.plans:
        for p, q in plan.pairs:
            _need(
                np.array_equal(new[r.index[p]], 0.5 * (old[p] + old[q])),
                f"merged node of source nodes {p + 1} and {q + 1} is not at their midpoint",
            )


def _verify(r: _Result, rotor: Any = None) -> dict[str, Any]:
    """Refuse any difference from the source the collapse does not account for.

    A face the merge turns over is quality, not a broken mesh: it is warned with its numbers
    and listed under ``warnings``.
    """
    checks: dict[str, Any] = {"faces": _check_counts(r)}
    _check_near_faces(r)
    checks["coincident_node_pairs"] = _check_twins(r)
    owners = _edge_faces(r.mesh)
    _check_lines(r, owners)
    checks["open_edges"] = _check_open(r, owners)
    _check_far_nodes(r)
    _check_faces(r)
    turned = _turned_faces(r)
    checks["smallest_normal_cosine"] = turned["worst_cosine"]
    checks["turned_faces"] = turned
    checks["area"] = _area_report(r)
    _check_midpoints(r)
    checks["periodicity"] = _periodicity(r.source, r.mesh, rotor)
    checks["warnings"] = _turned_warning(turned)
    for sentence in checks["warnings"]:
        warn(f"collapse_te: {sentence}; the mesh is written", GeometryWarning)
    return checks


def _periodicity(src: ObjMesh, out: ObjMesh, rotor: Any) -> Any:
    if rotor is None or len(rotor.get("families_blades", [])) < 2:
        warn(
            "collapse_te: full-wheel periodicity does not apply without a wheel reference; skipped",
            NotApplicableWarning,
        )
        return None
    axis = parse_axis(rotor["axis"])
    origin = np.array([rotor.get(k + "_m", 0.0) for k in ("x", "y", "z")])
    copies = len(rotor["families_blades"])
    found = {}
    for label, mesh in (("source", src), ("output", out)):
        faces = Faces.of([f for fs in mesh.families.values() for f in fs])
        inv = wheel_symmetry(mesh.verts, faces, axis, origin, copies=copies, tol=1e-7)["invariance"]
        found[label] = inv
    whole = found["output"]
    ok = whole["nodes_mapped"] == whole["nodes"] and whole["faces_mapped"] == whole["faces"]
    held = found["source"]["nodes_mapped"] == found["source"]["nodes"]
    _need(ok or not held, "wheel periodicity failed after the collapse")
    if not ok:
        warn("collapse_te: the source wheel is not periodic either", NotApplicableWarning)
    return found["output"]


def _rotor(cfg_path: Path, mesh_path: Path) -> Any:
    """Return the rotor a [repitch] reference of the same control file names, or None."""
    data = read_control(cfg_path, "collapse_te")
    table = data.get("repitch")
    if not isinstance(table, dict) or "reference" not in table:
        return None
    try:
        return read_rotor(dict(table), cfg_path, mesh_path, verb="collapse_te")
    except KeyError as error:
        raise InputValueError(
            f"collapse_te: the rotor of the [repitch] reference states no {error}. "
            "No file was written."
        ) from error


def _undeclarable(data: dict[str, Any]) -> str | None:
    """Return why the boundaries file cannot take a new trailing-edge entry, or None."""
    tes = data.get("trailing_edges", {})
    if "none" in tes:
        return "it states that the mesh has no trailing edges"
    if any(isinstance(t, dict) and "detect" in t for t in tes.values()):
        return "it marks trailing edges by detection"
    return None


def _declare_lines(data: dict[str, Any], plans: list[_Plan], rw: Rewrite) -> None:
    """Add an entry for each blade collapsed with no rim entry, taking a removed region's type."""
    tes = data.setdefault("trailing_edges", {})
    regions = data.get("base_regions", {})
    for plan in plans:
        kind = plan.edge_type
        for family in sorted({f for f, _ in plan.strip}):
            rim = regions.get(family, {}).get("trailing_edges")
            kind = rim["type"] if isinstance(rim, dict) else kind
        label = plan.blade
        while label in tes:
            label += "_collapsed"
        tes[label] = {"file": f"@collapse:{plan.blade}", "type": kind}


def _drop_detect_surfaces(tes: Any, emptied: set[str]) -> None:
    """Take the families the collapse removes off every detection entry's surfaces."""
    for label, table in tes.items() if isinstance(tes, dict) else ():
        detect: Any = table.get("detect") if isinstance(table, dict) else None
        named = detect.get("surfaces", []) if isinstance(detect, dict) else []
        left = [s for s in named if s not in emptied]
        if len(left) != len(named):
            # An empty list means every surface, so the entry cannot be left with none.
            _need(
                bool(left),
                f"trailing-edge entry {label!r} detects only on families the collapse removes; "
                "remove the entry or name another surface",
            )
            detect["surfaces"] = left


def _one_line(tes: dict[str, Any], plan: _Plan, rw: Rewrite) -> None:
    """Replace the rim entries of one blade with one entry at the place of the first."""
    mine = [lab for lab in tes if lab in plan.labels]
    if not mine:
        return
    replaced = {"files", "file", "type"}
    kept = {k: v for k, v in tes[mine[0]].items() if k not in replaced}
    merged: dict[str, Any] = {"file": f"@collapse:{plan.blade}", "type": plan.edge_type, **kept}
    rebuilt = {}
    for lab, table in tes.items():
        if lab == mine[0]:
            rebuilt[lab] = merged
        elif lab not in mine:
            rebuilt[lab] = table
    tes.clear()
    tes.update(rebuilt)


def _drop_regions(regions: dict[str, Any], plans: list[_Plan], emptied: set[str]) -> None:
    """Remove the base regions of what the collapse removes, refusing a merge into one."""
    strips = {f for plan in plans for f, _ in plan.strip}
    hosts = emptied | (strips & {p.blade for p in plans})
    for name in list(regions):
        if name in hosts:
            del regions[name]
    for name, table in regions.items():
        if isinstance(table, dict) and table.get("merge_into") in emptied:
            raise InputValueError(
                f"collapse_te: base region {name!r} merges into {table['merge_into']!r}, "
                "which the collapse removes. No file was written."
            )


def _drop_operations(table: Any, emptied: set[str]) -> None:
    """Skip, with a warning, each import operation on a family the collapse removes."""
    operations = table.get("operations") if isinstance(table, dict) else None
    if not isinstance(operations, list):
        return
    gone, kept = set(emptied), []
    for operation in operations:
        surface = operation.get("surface") if isinstance(operation, dict) else None
        if surface not in gone:
            kept.append(operation)
            continue
        # A rename of a removed family names it anew; an operation on the new name goes too.
        if operation.get("op") == "rename" and isinstance(operation.get("to"), str):
            gone.add(operation["to"])
        warn(
            f"collapse_te: import operation {operation.get('op')!r} names {surface!r}, which "
            "the collapse removes, so it does not apply and is skipped",
            NotApplicableWarning,
        )
    if kept:
        table["operations"] = kept
    else:
        del table["operations"]


def _edit(plans: list[_Plan], result: _Result, rw: Rewrite, declare: bool) -> None:
    """Declare the edit of the boundaries tables: one line per blade, no base for the strip."""
    emptied = {f for f in result.source.families if f not in result.mesh.families}

    def edit(data: dict[str, Any]) -> None:
        if declare:
            _declare_lines(data, [p for p in plans if not p.labels], rw)
        tes = data.get("trailing_edges")
        if isinstance(tes, dict):
            for plan in plans:
                _one_line(tes, plan, rw)
        _drop_detect_surfaces(tes, emptied)
        if isinstance(data.get("base_regions"), dict):
            _drop_regions(data["base_regions"], plans, emptied)
        if isinstance(data.get("boundaries"), list):
            data["boundaries"] = [n for n in data["boundaries"] if n not in emptied]
        _drop_operations(data.get("import"), emptied)

    rw.edit = edit


def collapse_trailing_edge(
    mesh: str | Path, morphing: str | Path, out: str | Path
) -> dict[str, Any]:
    """Collapse the blunt trailing edge of named blades and write only after every check.

    Parameters
    ----------
    mesh : str or Path
        Source OBJ, read only. Any other format is refused.
    morphing : str or Path
        TOML control file with a ``[collapse_te]`` table: ``blades`` (families, or ``"all"``)
        and optionally ``base`` (the group holding each blade's strip).
    out : str or Path
        New OBJ. Existing outputs are refused.

    Returns
    -------
    dict
        The per-blade counts and the checks, also written as ``<out>.collapse.json``.
    """
    path, control, target = Path(mesh), Path(morphing), Path(out)
    record = Path(str(target) + ".collapse.json")
    _need(
        path.suffix.lower() == ".obj",
        f"mesh {path} must be an .obj: the operation works on the OBJ",
    )
    _need(target.suffix.lower() == ".obj", "--out must be an .obj")
    _need(target.resolve() != path.resolve(), "--out must differ from the source")
    _need(not target.exists() and not record.exists(), "output or report exists; choose new --out")
    cfg = _config(control)
    source = read_obj(path)
    one_block_per_family(path, "collapse_te")
    index = _Index(source)
    plans, side, skipped, unit = _select(cfg, path, source, index)
    _need(
        bool(plans),
        "nothing to collapse: "
        + ("; ".join(f"{b}: {why}" for b, why in skipped.items()) or "no blunt trailing edge"),
    )
    if unit is None and any(not plan.labels for plan in plans):
        warn(
            "collapse_te: mesh unit is unstated; a trailing-edge points file with no source entry "
            "to follow says METER over the mesh's own numbers",
            NotApplicableWarning,
        )
    result = _build(source, plans)
    checks = _verify(result, _rotor(control, path))
    payload = obj_text(result.mesh).encode("utf-8")
    read_back(result.mesh, payload, "collapse_te")
    checks["readback_equal"] = True
    rw = Rewrite(
        verb="collapse_te",
        counts=(
            sum(len(f) for f in result.mesh.families.values()),
            [len(f) for f in result.mesh.families.values()],
        ),
    )
    merged_line: dict[str, np.ndarray] = {}
    new_by_old = result.mesh.verts[result.index]
    for plan in plans:
        chain = result.lines[plan.blade]
        pts = result.mesh.verts[chain]
        merged_line[plan.blade] = 0.5 * (pts[:-1] + pts[1:])
        rw.synthetic[f"@collapse:{plan.blade}"] = (
            plan.unit or "METER",
            merged_line[plan.blade],
            plan.to_file,
        )
    rw.dead = {
        e
        for plan in plans
        for fam, i in plan.strip
        for e in _sides(source.families[fam][i])
        if e not in _rim_edges(plan)
    }
    why = None if side is None else _undeclarable(side[1])
    _edit(plans, result, rw, side is not None and why is None)
    extra = attachments(path, target, source, new_by_old, payload, rewrite=rw)
    for plan in [p for p in plans if not p.labels] if side is None or why else []:
        file = target.with_name(f"{target.stem}.{plan.blade}.te.txt")
        extra[file] = _te_bytes(file, plan.unit or "METER", merged_line[plan.blade] * plan.to_file)
        if why:
            warn(
                f"collapse_te: the boundaries file holds no entry for blade {plan.blade!r} and "
                f"cannot take one, because {why}; its trailing edge is in {file.name}, which that "
                "file does not name",
                NotApplicableWarning,
            )
    _sidecar_reads_back(extra, target)
    if side is not None:
        checks["boundaries_file_reads_back"] = True
    for file in extra:
        _need(not file.exists(), f"attachment {file} exists; choose new --out")
    report = _report(result, checks, skipped)
    report["source_sha256"] = file_sha256(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    for file, content in extra.items():
        file.write_bytes(content)
    report["output_sha256"] = file_sha256(target)
    report["attachments"] = sorted(p.name for p in extra)
    _textio.write_text(record, json.dumps(report, indent=2) + "\n")
    return report


def _rim_edges(plan: _Plan) -> set[Edge]:
    return {_key(r[j], r[j + 1]) for r in (plan.rim_a, plan.rim_b) for j in range(len(r) - 1)}


def _sidecar_reads_back(extra: dict[Path, bytes], target: Path) -> None:
    """Refuse a boundaries file that the declaration reader would not take from the output."""
    sidecar_reads_back(extra, target, "collapse_te")


def _te_bytes(file: Path, unit: str, points: np.ndarray) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        probe = write_te(Path(folder) / file.name, unit, points)
        return probe.read_bytes()


def _report(result: _Result, checks: dict[str, Any], skipped: dict[str, str]) -> dict[str, Any]:
    old = result.source.verts
    blades = {}
    for plan in result.plans:
        half = np.array([0.5 * np.linalg.norm(old[a] - old[b]) for a, b in plan.pairs])
        blades[plan.blade] = {
            "route": plan.route,
            "strip_faces_removed": len(plan.strip),
            "closure_faces": result.closure.get(plan.blade, 0),
            "faces_collapsed_to_nothing": result.nothing.get(plan.blade, 0),
            "nodes_merged": len(plan.pairs),
            "trailing_edge_edges": len(plan.pairs) - 1,
            "largest_node_move": float(half.max()),
            "base_thickness_max": float(2 * half.max()),
            "base_thickness_min": float(2 * half.min()),
        }
    return {
        "blades": blades,
        "skipped": skipped,
        "checks": {"passed": not checks["warnings"], **checks},
    }

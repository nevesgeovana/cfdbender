"""Merge two trimmed OBJ along the open boundaries they hold within a threshold.

One part is KEPT and one is ADAPTED, named by the caller. The interface is the
open-boundary chains of the two parts whose nodes lie within the threshold of
the other's chain. Nothing of the kept part changes: its nodes, their text and
order, and its faces and their order are the output's first lines. The adapted
part takes the kept part's nodes along each chain in one of two ways:

- MOVED: both chains carry one node per node within the threshold. The adapted
  nodes are replaced by the kept nodes and no face changes kind.
- CONFORMED: otherwise, a band of :data:`._refine._geometry.BAND_LAYERS` face
  layers of the adapted part is remeshed at factor 1 by
  :func:`._refine._remesh.refine_group`, with the interface rebuilt on the kept
  nodes by :meth:`._refine._remesh.Remesher.conform` and every other node of the
  band projected onto the adapted surface. A band that would hold a whole family
  rebuilds the family, warned, as a refinement does. When the remesh cannot
  hold the kept nodes, an adapted part past one face layer over the kept one being the
  measured case, only the faces on the interface are split onto them, warned.

Every open edge of the two parts that is not on an interface stays open, and the
ones still facing the other part's open boundary are warned with their count. A
threshold that finds no interface warns and writes nothing.

The layer is the workspace row; the refinement package imports nothing of this
module. The remesh, and with it trimesh, scipy and rtree, is imported only when a
chain needs it, so two chains of one node per node need no extra.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import (
    GeometryWarning,
    InputArtifactError,
    InputValueError,
    NotApplicableWarning,
    warn,
)
from cfdbender.mesh._refine import _geometry
from cfdbender.mesh._refine._audit import AUDIT_SUFFIX, MeshAudit, audit_mesh
from cfdbender.mesh._refine._config import QUAD_DOMINANT, TRIANGLES
from cfdbender.mesh._refine._geometry import Faces, Points
from cfdbender.mesh._refine._level import band_of_faces
from cfdbender.mesh._refine._obj import (
    DEFAULT_FAMILY,
    KIND,
    ObjMesh,
    mesh_unit,
    read_obj,
    write_obj,
)
from cfdbender.mesh._sidecars import inventory_sidecar, write_inventory

__all__ = ["MergedMesh", "merge_objs"]

#: The schema version of ``<stem>.merge.json``.
RECORD_SCHEMA = 1
#: The suffix of the merge's record, beside its OBJ.
RECORD_SUFFIX = ".merge.json"
#: The words of the output's statement that declarations are not merged.
DECLARATIONS_NOTE = (
    "the boundary declarations of the two sources are not merged: the output's boundaries "
    "file holds its inventory only, and a declared trailing-edge file is not rewritten"
)
#: An open edge left faces the other part's when the line to the other's nearest open edge
#: stays this close to normal of both edges (the absolute cosine). A boundary running on from
#: a chain's end meets the line along itself, cosine near one; 0.5 is 60 degrees off the edge.
FACING_COSINE = 0.5
#: An open edge further from the other part's open boundary than this fraction of the smaller
#: part's size is no part of a gap between the two, whatever its direction.
FACING_REACH = 0.25
#: The passes that drop an edge whose ends lie nearest two chains at most; a pass only removes
#: edges, and a chain settles in one or two.
NARROW_PASSES = 8
#: An edge whose middle lies this fraction of its length from the other's boundary lies on it:
#: the direction of a line that short is rounding, so it says nothing of running on.
ON_EDGE_FRACTION = 1e-6

Edge = tuple[int, int]


@dataclass(frozen=True)
class MergedMesh:
    """What one merge did.

    Attributes
    ----------
    obj : Path or None
        The merged OBJ; None when no interface was found and nothing was written.
    record : Path or None
        Its ``<stem>.merge.json``.
    boundaries : Path or None
        Its inventory sidecar ``<name>.obj.boundaries.toml``.
    audit : MeshAudit or None
        The audit of the output.
    interface_edges : int
        The edges the two parts now share.
    moved_nodes : int
        The adapted nodes replaced by kept nodes with no change of face.
    band_faces : int
        The adapted faces a remesh replaced.
    triangles_created : int
        The triangles in the output's band beyond the triangles the source band held.
    open_edges_left : int
        Every open edge of the output.
    facing_edges_left : int
        The open edges of the output still facing the other part's open boundary.
    turned_faces : int
        The adapted faces turned against the source face nearest them.
    renamed : dict
        Each adapted family that took a new name, to that name.
    """

    obj: Path | None
    record: Path | None
    boundaries: Path | None
    audit: MeshAudit | None
    interface_edges: int = 0
    moved_nodes: int = 0
    band_faces: int = 0
    triangles_created: int = 0
    open_edges_left: int = 0
    facing_edges_left: int = 0
    turned_faces: int = 0
    renamed: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Chain:
    """One ordered open-boundary chain of a part: its nodes and whether it closes."""

    nodes: tuple[int, ...]
    closed: bool

    @property
    def edges(self) -> list[Edge]:
        """Return the chain's edges, low node first."""
        ends = self.nodes[1:] + (self.nodes[:1] if self.closed else ())
        return [(min(a, b), max(a, b)) for a, b in zip(self.nodes, ends, strict=False)]


@dataclass(frozen=True)
class _Pair:
    """One interface: the kept chain and the adapted chain within the threshold of it."""

    kept: _Chain
    adapted: _Chain


@dataclass
class _Adapted:
    """The adapted part as the output holds it."""

    points: Points
    #: Per family, its faces in the numbering of ``points``, in the output's order.
    families: dict[str, Faces]
    #: Per node of ``points``, the kept node it is, or -1 for a node of its own.
    kept_node: NDArray[numpy.int64]
    #: Per node of ``points``, the source text to write, or None.
    text: list[str | None]
    moved_nodes: int = 0
    band_faces: int = 0
    triangles_created: int = 0
    conformed: list[dict[str, int]] = field(default_factory=list)
    rebuilt: list[str] = field(default_factory=list)
    #: The band could not be remeshed, so the faces on the interface were split.
    split: bool = False


@dataclass(frozen=True)
class _Remeshed:
    """A remesh of the bands: the part with each band a family, band to family, its result."""

    work: ObjMesh
    names: Mapping[str, str]
    done: Any


def _segments(verts: Points, edges: Sequence[Edge]) -> tuple[Points, Points]:
    """Return the two end points of each edge."""
    rows = numpy.asarray(edges, dtype=numpy.int64).reshape(-1, 2)
    return verts[rows[:, 0]], verts[rows[:, 1]]


def _within(
    nodes: Sequence[int], verts: Points, edges: Sequence[Edge], other: Points, threshold: float
) -> dict[int, int]:
    """Return each node within the threshold of an edge, with that edge's index in ``edges``."""
    if not nodes or not edges:
        return {}
    a, b = _segments(other, edges)
    distance, nearest = _geometry.segment_distances(verts[list(nodes)], a, b)
    return {
        node: int(k)
        for node, d, k in zip(nodes, distance, nearest, strict=True)
        if float(d) <= threshold
    }


def _nodes_of(edges: Iterable[Edge]) -> list[int]:
    """Return the sorted nodes of some edges."""
    return sorted({v for e in edges for v in e})


def _walk(edges: Iterable[Edge]) -> list[_Chain]:
    """Return the edges as ordered chains, broken at any node more than two of them meet."""
    around: dict[int, list[int]] = defaultdict(list)
    for a, b in sorted(edges):
        around[a].append(b)
        around[b].append(a)
    used: set[Edge] = set()

    def run(start: int, step: int) -> _Chain:
        nodes = [start]
        while True:
            used.add((min(nodes[-1], step), max(nodes[-1], step)))
            if step == start:
                return _Chain(tuple(nodes), True)
            nodes.append(step)
            ahead = [n for n in around[step] if (min(n, step), max(n, step)) not in used]
            if len(around[step]) != 2 or not ahead:
                return _Chain(tuple(nodes), False)
            step = ahead[0]

    chains: list[_Chain] = []
    for start in [v for v in sorted(around) if len(around[v]) != 2] + sorted(around):
        for step in around[start]:
            if (min(start, step), max(start, step)) not in used:
                chains.append(run(start, step))
    return chains


def _same_chain(
    edges: Sequence[Edge], verts: Points, chains: Sequence[_Chain], other: Points
) -> list[Edge]:
    """Return the edges whose two ends lie nearest ONE chain of the other part."""
    holder = {e: i for i, chain in enumerate(chains) for e in chain.edges}
    listed = list(holder)
    if not listed or not edges:
        return []
    nodes = _nodes_of(edges)
    a, b = _segments(other, listed)
    _, nearest = _geometry.segment_distances(verts[nodes], a, b)
    chain_of = {node: holder[listed[int(k)]] for node, k in zip(nodes, nearest, strict=True)}
    return [e for e in edges if chain_of[e[0]] == chain_of[e[1]]]


def _interface(
    kept: Points,
    kept_open: Sequence[Edge],
    adapted: Points,
    adapted_open: Sequence[Edge],
    threshold: float,
) -> list[_Pair]:
    """Return the chains of the two parts within the threshold of each other.

    An edge is on the interface when both its ends lie within the threshold of the
    other part's open boundary, and nearest one chain of it: a kept edge bridging a
    stretch of the adapted boundary that is out of reach joins nothing. An adapted edge
    must then reach the kept CHAIN, not only the kept boundary: an adapted node that
    touches a stretch of it that joined no chain is not an interface node.
    """
    k_edges = _reaching(kept_open, kept, adapted_open, adapted, threshold)
    k_edges = _without_run_on(k_edges, kept, adapted_open, adapted)
    a_edges = _reaching(adapted_open, adapted, kept_open, kept, threshold)
    a_edges = _without_run_on(a_edges, adapted, kept_open, kept)
    k_edges, a_edges = _narrowed(k_edges, kept, a_edges, adapted)
    # An adapted node can touch the kept boundary where no kept edge joined a chain, and it
    # then has no kept node to take within reach: it stays out, so a chain's end moves by
    # one adapted edge at most. One round: the kept edges stay, as their words say.
    a_edges = _reaching(a_edges, adapted, k_edges, kept, threshold)
    k_edges, a_edges = _narrowed(k_edges, kept, a_edges, adapted)
    k_chains, a_chains = _walk(k_edges), _walk(a_edges)
    return [_Pair(k_chains[i], a_chains[j]) for i, j in _mutual(k_chains, kept, a_chains, adapted)]


def _reaching(
    edges: Sequence[Edge], verts: Points, others: Sequence[Edge], other: Points, threshold: float
) -> list[Edge]:
    """Return the edges whose two ends lie within the threshold of the edges ``others``."""
    near = _within(_nodes_of(edges), verts, others, other, threshold)
    return [e for e in edges if e[0] in near and e[1] in near]


def _without_run_on(
    edges: Sequence[Edge], verts: Points, others: Sequence[Edge], other: Points
) -> list[Edge]:
    """Return the edges left when each open chain loses the end edges that only run on from it.

    A boundary leaving the interface at a chain's end (a symmetry plane, a part's far side)
    is within a threshold wider than its cells, and meets the other part along itself:
    the line from its middle to the other's nearest open edge is near parallel to it
    (:data:`FACING_COSINE`), where an edge across the interface sees the other part abeam.
    An edge inside a chain is never one of them, so only the ends are read.
    """
    if not edges or not others:
        return []
    a, b = _segments(verts, edges)
    c, d = _segments(other, others)
    middle = 0.5 * (a + b)
    distance, _, line = _to_nearest(middle, c, d)
    on_it = distance <= ON_EDGE_FRACTION * numpy.linalg.norm(b - a, axis=1)
    across = on_it | (_cosine(line, b - a) < FACING_COSINE)
    is_across = {e: bool(flag) for e, flag in zip(edges, across, strict=True)}
    left: list[Edge] = []
    for chain in _walk(edges):
        run = chain.edges
        if not chain.closed:
            while run and not is_across[run[0]]:
                run = run[1:]
            while run and not is_across[run[-1]]:
                run = run[:-1]
        left += run
    return sorted(left)


def _narrowed(
    k_edges: list[Edge], kept: Points, a_edges: list[Edge], adapted: Points
) -> tuple[list[Edge], list[Edge]]:
    """Return the edges of each part left when each edge's ends lie nearest one chain."""
    for _ in range(NARROW_PASSES):
        k_next = _same_chain(k_edges, kept, _walk(a_edges), adapted)
        a_next = _same_chain(a_edges, adapted, _walk(k_next), kept)
        settled = len(k_next) == len(k_edges) and len(a_next) == len(a_edges)
        k_edges, a_edges = k_next, a_next
        if settled:
            break
    return k_edges, a_edges


def _mutual(
    k_chains: Sequence[_Chain], kept: Points, a_chains: Sequence[_Chain], adapted: Points
) -> list[tuple[int, int]]:
    """Return the (kept, adapted) chain indices that each take the other as their nearest."""

    def nearest(chains: Sequence[_Chain], verts: Points, others: Sequence[_Chain], at: Points):
        holder = [i for i, chain in enumerate(others) for _ in chain.edges]
        a, b = _segments(at, [e for chain in others for e in chain.edges])
        chosen = []
        for chain in chains:
            _, k = _geometry.segment_distances(verts[list(chain.nodes)], a, b)
            votes = [holder[int(i)] for i in k]
            chosen.append(max(sorted(set(votes)), key=votes.count))
        return chosen

    if not k_chains or not a_chains:
        return []
    k_to_a = nearest(k_chains, kept, a_chains, adapted)
    a_to_k = nearest(a_chains, adapted, k_chains, kept)
    return [(i, j) for i, j in enumerate(k_to_a) if a_to_k[j] == i]


def _one_to_one(pair: _Pair, kept: Points, adapted: Points, threshold: float) -> dict[int, int]:
    """Return adapted node to kept node when the chains match node for node, else nothing.

    Each adapted node takes the nearest kept node of the chain, within the threshold;
    the match must use every kept node once and carry every adapted edge onto a kept edge.
    """
    k_nodes, a_nodes = pair.kept.nodes, pair.adapted.nodes
    if len(k_nodes) != len(a_nodes) or pair.kept.closed != pair.adapted.closed:
        return {}
    where = kept[list(k_nodes)]
    match: dict[int, int] = {}
    for a in a_nodes:
        d = numpy.linalg.norm(where - adapted[a], axis=1)
        j = int(numpy.argmin(d))
        if float(d[j]) > threshold:
            return {}
        match[a] = k_nodes[j]
    if len(set(match.values())) != len(k_nodes):
        return {}
    carried = {(min(match[a], match[b]), max(match[a], match[b])) for a, b in pair.adapted.edges}
    return match if carried == set(pair.kept.edges) else {}


def _all_faces(mesh: ObjMesh) -> Faces:
    """Return the mesh's faces in file order."""
    return [f for faces in mesh.families.values() for f in faces]


def _bands_of(mesh: ObjMesh, seed: set[int]) -> dict[str, set[int]]:
    """Return per family the indices of its faces in the band beside the seed nodes."""
    bands = {
        name: band_of_faces(faces, seed, _geometry.BAND_LAYERS)
        for name, faces in mesh.families.items()
    }
    return {name: band for name, band in bands.items() if band}


def _working_mesh(mesh: ObjMesh, bands: Mapping[str, set[int]]) -> tuple[ObjMesh, dict[str, str]]:
    """Return the adapted part with each band as a family of its own, and band name to family."""
    families: dict[str, Faces] = {}
    names: dict[str, str] = {}
    for name, faces in mesh.families.items():
        band = bands.get(name, set())
        rest = [f for i, f in enumerate(faces) if i not in band]
        if rest:
            families[name] = rest
        if band:
            label = f"{name}\t{len(names)}"
            names[label] = name
            families[label] = [f for i, f in enumerate(faces) if i in band]
    return ObjMesh(mesh.verts, families), names


def _element_mode(faces: Faces) -> str:
    """Return the remesh's element mode: quad-dominant for a band that held a quadrilateral."""
    return QUAD_DOMINANT if any(len(f) > 3 for f in faces) else TRIANGLES


def _triangles(faces: Iterable[Sequence[int]]) -> int:
    """Return how many of the faces are triangles."""
    return sum(1 for f in faces if len(f) == 3)


def _adapt(
    mesh: ObjMesh, kept: Points, pairs: Sequence[_Pair], threshold: float, source: Path
) -> _Adapted:
    """Return the adapted part on the kept nodes: chains moved, the others conformed."""
    moved: dict[int, int] = {}
    conformed: list[_Pair] = []
    for pair in pairs:
        match = _one_to_one(pair, kept, mesh.verts, threshold)
        if match:
            moved.update(match)
        else:
            conformed.append(pair)
    # A band that reaches a moved chain would remesh its nodes, so that chain conforms too.
    bands: dict[str, set[int]] = {}
    while conformed:
        bands = _bands_of(mesh, {v for pair in conformed for v in pair.adapted.nodes})
        touched = {v for name, band in bands.items() for i in band for v in mesh.families[name][i]}
        late = [p for p in pairs if p not in conformed and set(p.adapted.nodes) & touched]
        if not late:
            break
        conformed += late
        for pair in late:
            for a in pair.adapted.nodes:
                moved.pop(a, None)
    if not conformed:
        return _unremeshed(mesh, moved)
    done = _conformed(mesh, kept, conformed, bands, moved=moved)
    if _holds(done, conformed, mesh):
        for family in done.rebuilt:
            warn(
                f"{source} family {family}: the band of {_geometry.BAND_LAYERS} face layers beside "
                f"the interface covers all {len(mesh.families[family])} faces of the family, so "
                "it was rebuilt whole",
                GeometryWarning,
                stacklevel=3,
            )
        return done
    warn(
        f"{source}: the remesh of the band could not take the kept nodes along the interface "
        f"({done.conformed}), so the faces beside it were split onto them instead: the parts "
        "may overlap by more than a face layer, or a chain's end has no kept node beside it",
        GeometryWarning,
        stacklevel=3,
    )
    return _split_faces(mesh, kept, conformed, moved, threshold, source=source)


def _unremeshed(mesh: ObjMesh, moved: Mapping[int, int]) -> _Adapted:
    """Return the adapted part with no band: nodes replaced by kept nodes, faces as they were."""
    return _Adapted(
        mesh.verts,
        {name: [list(f) for f in faces] for name, faces in mesh.families.items()},
        _kept_nodes(len(mesh.verts), moved),
        _text_of(mesh),
        moved_nodes=len(moved),
    )


def _kept_nodes(count: int, moved: Mapping[int, int]) -> NDArray[numpy.int64]:
    """Return per node of the adapted part the kept node it became, or -1."""
    kept_node = numpy.full(count, -1, dtype=numpy.int64)
    for a, k in moved.items():
        kept_node[a] = k
    return kept_node


def _text_of(mesh: ObjMesh) -> list[str | None]:
    """Return each node's source text."""
    if len(mesh.vertex_text) == len(mesh.verts):
        return list(mesh.vertex_text)
    return [None] * len(mesh.verts)


def _conformed(
    mesh: ObjMesh,
    kept: Points,
    pairs: Sequence[_Pair],
    bands: Mapping[str, set[int]],
    *,
    moved: Mapping[int, int],
) -> _Adapted:
    """Return the adapted part with its bands remeshed on the kept nodes of ``pairs``."""
    from cfdbender.mesh._refine._remesh import refine_group

    work, names = _working_mesh(mesh, bands)
    rebuilt = [
        family
        for label, family in names.items()
        if len(work.families[label]) == len(mesh.families[family])
    ]
    conform = [(list(p.adapted.nodes), kept[list(p.kept.nodes)]) for p in pairs]
    done = refine_group(
        work,
        list(names),
        (),
        1.0,
        conform=conform,
        elements={label: _element_mode(work.families[label]) for label in names},
        shared=conform,
        te_edges=set(),
    )
    remeshed = _Remeshed(work, names, done)
    numbering, points, kept_node = _numbered(mesh, remeshed, pairs, kept, moved)
    families, band_faces, created = _band_families(mesh, remeshed, numbering, points)
    return _Adapted(
        points,
        families,
        kept_node,
        _text_of(mesh) + [None] * (len(points) - len(mesh.verts)),
        moved_nodes=len(moved),
        band_faces=band_faces,
        triangles_created=created,
        conformed=list(done.info.get("conformed_interfaces", [])),
        rebuilt=rebuilt,
    )


def _numbered(
    mesh: ObjMesh,
    remeshed: _Remeshed,
    pairs: Sequence[_Pair],
    kept: Points,
    moved: Mapping[int, int],
) -> tuple[NDArray[numpy.int64], Points, NDArray[numpy.int64]]:
    """Return the number of each remeshed node among the source's, and the nodes and their kept.

    A remeshed node at a kept node's exact position is that kept node, and one at
    a source node's exact position is that source node; the others follow the source's.
    """
    done, work = remeshed.done, remeshed.work
    on_kept = {tuple(kept[k].tolist()): k for p in pairs for k in p.kept.nodes}
    band_nodes = {v for label in remeshed.names for f in work.families[label] for v in f}
    source_at = {tuple(mesh.verts[v].tolist()): v for v in sorted(band_nodes)}
    numbering = numpy.empty(len(done.points), dtype=numpy.int64)
    added: list[int] = []
    of_kept: dict[int, int] = {}
    for i, point in enumerate(done.points):
        key = tuple(point.tolist())
        if key not in on_kept and key in source_at:
            numbering[i] = source_at[key]
            continue
        numbering[i] = len(mesh.verts) + len(added)
        if key in on_kept:
            of_kept[len(mesh.verts) + len(added)] = on_kept[key]
        added.append(i)
    points = numpy.vstack([mesh.verts, done.points[added].reshape(-1, 3)])
    kept_node = _kept_nodes(len(points), {**moved, **of_kept})
    return numbering, points, kept_node


def _band_families(
    mesh: ObjMesh,
    remeshed: _Remeshed,
    numbering: NDArray[numpy.int64],
    points: Points,
) -> tuple[dict[str, Faces], int, int]:
    """Return the families in the source's order, each band's faces where the family sits."""
    work, done = remeshed.work, remeshed.done
    labels = {family: label for label, family in remeshed.names.items()}
    families: dict[str, Faces] = {}
    band_faces = created = 0
    for name, faces in mesh.families.items():
        if name not in labels:
            families[name] = [list(f) for f in faces]
            continue
        label = labels[name]
        new = [[int(numbering[v]) for v in f] for f in done.families.get(label, [])]
        rest = [list(f) for f in work.families.get(name, [])]
        families[name] = _geometry.order_like(points, rest + new, mesh.verts, faces)
        band_faces += len(work.families[label])
        created += _triangles(new) - _triangles(work.families[label])
    return families, band_faces, max(created, 0)


def _on_kept_numbers(adapted: _Adapted) -> Faces:
    """Return the adapted faces with a node that became a kept node numbered as ``-1 - kept``."""
    number = [-1 - k if k >= 0 else v for v, k in enumerate(adapted.kept_node.tolist())]
    return [[number[v] for v in f] for faces in adapted.families.values() for f in faces]


def _holds(adapted: _Adapted, pairs: Sequence[_Pair], source: ObjMesh) -> bool:
    """Return whether a remeshed band stands on every kept edge, each by one face, cleanly.

    It holds when every old node left the interface, every edge of a kept chain has
    exactly one adapted face, no edge has more than two and no pair of neighbors in the
    band turned against each other where the source's did not.
    """
    if any(entry["not_removed"] for entry in adapted.conformed):
        return False
    faces = _on_kept_numbers(adapted)
    around = _geometry.edge_faces(faces)
    if any(len(fs) > 2 for fs in around.values()):
        return False
    for pair in pairs:
        for a, b in pair.kept.edges:
            if len(around.get((-1 - b, -1 - a), ())) != 1:
                return False
    return _opposed(faces) <= _opposed(_all_faces(source))


def _opposed(faces: Faces) -> int:
    """Return how many edges two faces run through in one direction."""
    seen: set[tuple[int, int]] = set()
    count = 0
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1], strict=True):
            if (a, b) in seen:
                count += 1
            seen.add((a, b))
    return count


def _alignment(a_points: Points, k_points: Points) -> tuple[float, list[int]]:
    """Return the cheapest monotone match of ``a_points`` to ``k_points`` and each one's match.

    Both sequences are walked once, end to end; a step advances one or both, and the
    cost is the sum of the distances of the pairs it visits. Each ``a`` node takes the
    nearest of the ``k`` nodes it was paired with, which keeps the match monotone.
    """
    gap = numpy.linalg.norm(a_points[:, None, :] - k_points[None, :, :], axis=2)
    rows, columns = gap.shape
    cost = numpy.full((rows + 1, columns + 1), numpy.inf)
    cost[0, 0] = 0.0
    for i in range(rows):
        for j in range(columns):
            cost[i + 1, j + 1] = gap[i, j] + min(cost[i, j], cost[i, j + 1], cost[i + 1, j])
    paired: list[list[int]] = [[] for _ in range(rows)]
    i, j = rows, columns
    while i and j:
        paired[i - 1].append(j - 1)
        steps = (
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        )
        _, i, j = min(steps, key=lambda step: step[0])
    match = [min(row, key=lambda c: (gap[n, c], c)) for n, row in enumerate(paired)]
    match[0], match[-1] = 0, columns - 1
    return float(cost[rows, columns]), match


def _sequences(pair: _Pair, kept: Points, adapted: Points) -> list[tuple[list[int], list[int]]]:
    """Return the ways of walking one chain against the other, end to end."""
    a, k = list(pair.adapted.nodes), list(pair.kept.nodes)

    def nearest(point: Points, nodes: Sequence[int], at: Points) -> int:
        return int(numpy.argmin(numpy.linalg.norm(at[list(nodes)] - point, axis=1)))

    def arcs(nodes: list[int], end: int) -> list[list[int]]:
        """Return the two ways round a loop from its first node to the node at ``end``."""
        back = nodes[:1] + nodes[:0:-1]
        if not end:
            return [nodes + nodes[:1], back + nodes[:1]]
        return [nodes[: end + 1], back[: len(nodes) - end + 1]]

    if pair.kept.closed and pair.adapted.closed:
        start = nearest(adapted[a[0]], k, kept)
        k = k[start:] + k[:start]
        return [(a + a[:1], k + k[:1]), (a + a[:1], k[:1] + k[:0:-1] + k[:1])]
    if pair.kept.closed:
        start = nearest(adapted[a[0]], k, kept)
        k = k[start:] + k[:start]
        return [(a, arc) for arc in arcs(k, nearest(adapted[a[-1]], k, kept))]
    if pair.adapted.closed:
        start = nearest(kept[k[0]], a, adapted)
        a = a[start:] + a[:start]
        return [(arc, k) for arc in arcs(a, nearest(kept[k[-1]], a, adapted))]
    return [(a, k), (a, k[::-1])]


def _split_edges(
    pairs: Sequence[_Pair], kept: Points, adapted: Points, rows: Mapping[int, int]
) -> tuple[dict[int, int], dict[Edge, list[int]]]:
    """Return each chain node's kept row, and the kept rows each directed chain edge now holds."""
    onto: dict[int, int] = {}
    held: dict[Edge, list[int]] = {}
    for pair in pairs:
        walks = [
            (*_alignment(adapted[a], kept[k]), a, k) for a, k in _sequences(pair, kept, adapted)
        ]
        _, match, a_walk, k_walk = min(walks, key=lambda walk: walk[0])
        for n, node in enumerate(a_walk):
            onto[node] = rows[k_walk[match[n]]]
        for n in range(len(a_walk) - 1):
            held[(a_walk[n], a_walk[n + 1])] = [
                rows[k_walk[c]] for c in range(match[n] + 1, match[n + 1])
            ]
    return onto, held


def _split_face(f: list[int], onto: Mapping[int, int], held: Mapping[Edge, list[int]]) -> list[int]:
    """Return a face with its chain nodes on kept rows and its chain edges through the new ones."""
    out: list[int] = []
    for u, v in zip(f, f[1:] + f[:1], strict=True):
        node = onto.get(u, u)
        if not out or out[-1] != node:
            out.append(node)
        out += held.get((u, v), held.get((v, u), [])[::-1])
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _warn_far_moves(
    mesh: ObjMesh, points: Points, onto: Mapping[int, int], threshold: float, source: Path
) -> None:
    """Warn the chain nodes the split moved past the threshold, each to its kept node."""
    moves = numpy.linalg.norm(mesh.verts[list(onto)] - points[list(onto.values())], axis=1)
    far = int((moves > threshold).sum())
    if far:
        warn(
            f"{source}: {far} of {len(moves)} interface node(s) moved further than the threshold "
            f"{threshold:g} to reach a kept node, the furthest by {float(moves.max()):g}: the "
            "threshold measures a node against the kept edges and the kept nodes lie further apart",
            GeometryWarning,
            stacklevel=4,
        )


def _split_faces(
    mesh: ObjMesh,
    kept: Points,
    pairs: Sequence[_Pair],
    moved: Mapping[int, int],
    threshold: float,
    *,
    source: Path,
) -> _Adapted:
    """Return the adapted part with only the faces on the interface split onto the kept nodes.

    The smallest correct change when a band cannot be remeshed: each adapted chain node
    takes one kept node, in chain order, and the kept nodes between two of them are
    inserted in the face on that edge, which is cut into triangles. The faces keep
    their places, and a face that lost an edge and took no node keeps its nodes.
    """
    from cfdbender.mesh._refine._remesh_group import triangulate_polygon

    nodes = sorted({k for p in pairs for k in p.kept.nodes})
    rows = {k: len(mesh.verts) + n for n, k in enumerate(nodes)}
    points = numpy.vstack([mesh.verts, kept[nodes].reshape(-1, 3)])
    onto, held = _split_edges(pairs, kept, mesh.verts, rows)
    _warn_far_moves(mesh, points, onto, threshold, source)
    families: dict[str, Faces] = {}
    band_faces = created = 0
    for name, faces in mesh.families.items():
        families[name] = []
        for f in faces:
            new = _split_face(list(f), onto, held)
            if new == list(f):
                families[name].append(list(f))
                continue
            band_faces += 1
            if len(new) < 3:
                continue
            parts = [new] if len(new) <= len(f) else triangulate_polygon(points, new)
            families[name] += parts
            created += _triangles(parts) - _triangles([f])
    return _Adapted(
        points,
        families,
        _kept_nodes(len(points), {**moved, **{row: k for k, row in rows.items()}}),
        _text_of(mesh) + [None] * len(nodes),
        moved_nodes=len(moved),
        band_faces=band_faces,
        triangles_created=max(created, 0),
        split=True,
    )


def _output_mesh(kept: ObjMesh, adapted: _Adapted, renamed: Mapping[str, str]) -> ObjMesh:
    """Return the merged mesh, the kept part first and untouched.

    Only the adapted nodes a face still uses follow the kept ones, in their order; an
    adapted node that became a kept node takes that node's number, so it is that node.
    """
    used = {v for faces in adapted.families.values() for f in faces for v in f}
    number = [-1] * len(adapted.points)
    rows: list[int] = []
    for v in range(len(adapted.points)):
        if adapted.kept_node[v] >= 0:
            number[v] = int(adapted.kept_node[v])
        elif v in used:
            number[v] = len(kept.verts) + len(rows)
            rows.append(v)
    families = {name: [list(f) for f in faces] for name, faces in kept.families.items()}
    for name, faces in adapted.families.items():
        families[renamed.get(name, name)] = [[number[v] for v in f] for f in faces]
    has_text = len(kept.vertex_text) == len(kept.verts)
    known: list[str | None] = list(kept.vertex_text) if has_text else [None] * len(kept.verts)
    return ObjMesh(
        numpy.vstack([kept.verts, adapted.points[rows].reshape(-1, 3)]),
        families,
        list(kept.header),
        kept.family_tag,
        known + [adapted.text[v] for v in rows],
    )


def _renames(kept: ObjMesh, adapted: ObjMesh, source: Path) -> dict[str, str]:
    """Return the new name of each adapted family a kept family already holds, warned."""
    taken = set(kept.families) | set(adapted.families)
    renamed: dict[str, str] = {}
    for name in adapted.families:
        if name not in kept.families:
            continue
        number = 2
        while f"{name}_{number}" in taken:
            number += 1
        renamed[name] = f"{name}_{number}"
        taken.add(renamed[name])
        warn(
            f"{source} family {name}: the kept part holds a family of that name, so this one "
            f"is written as {renamed[name]}",
            GeometryWarning,
            stacklevel=4,
        )
    return renamed


def _warn_units(kept: Path, adapted: Path) -> None:
    """Warn two parts whose boundaries files state different length units."""
    try:
        units = mesh_unit(kept), mesh_unit(adapted)
    except InputArtifactError:
        return
    if all(units) and units[0] != units[1]:
        warn(
            f"geometry merge: {kept} states units {units[0]} and {adapted} states units "
            f"{units[1]}. The coordinates and the threshold are merged as they stand, so the "
            "output mixes the two; scale one part to the other's unit and merge again",
            GeometryWarning,
            stacklevel=3,
        )


def _warn_scattered(path: Path) -> None:
    """Warn each family whose faces sit in more than one place in the file.

    A mesh holds a family's faces together, so such a family is written together and the
    part's face order is not the file's, which is what the solver's results depend on.
    """
    current, last = DEFAULT_FAMILY, None
    seen: set[str] = set()
    scattered: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tag, *rest = line.split(None, 1) or [""]
        if tag in ("g", "o"):
            current = rest[0].strip() if rest and rest[0].strip() else DEFAULT_FAMILY
        elif tag == "f" and current != last:
            if current in seen and current not in scattered:
                scattered.append(current)
            seen.add(current)
            last = current
    for name in scattered:
        warn(
            f"{path} family {name}: its faces sit in more than one place in the file. They are "
            "written together, so this part's face order in the output is not the file's",
            GeometryWarning,
            stacklevel=3,
        )


def _facing_edges(mesh: ObjMesh, kept_nodes: int, reach: float) -> int:
    """Return the open edges of either part facing an open edge of the other.

    An edge left open on one part counts when the line from its middle to the nearest
    open edge of the other part is no longer than ``reach`` and stays near normal to
    both edges (:data:`FACING_COSINE`). The boundaries that run on from an interface's
    end and leave it (a symmetry plane, a part's far side) meet that line along
    themselves, so they do not count. Neither does an edge whose own face lies between
    it and the other part, further than that face's depth: a part's far boundary can be
    near the other's, and it is no gap.
    """
    faces = _all_faces(mesh)
    around = _geometry.edge_faces(faces)
    holder = {e: faces[fs[0]] for e, fs in around.items() if len(fs) == 1}
    open_edges = sorted(holder)
    kept = [e for e in open_edges if e[1] < kept_nodes]
    other = [e for e in open_edges if e[1] >= kept_nodes]
    return _facing_one_way(mesh.verts, kept, other, reach, holder) + _facing_one_way(
        mesh.verts, other, kept, reach, holder
    )


def _facing_one_way(
    verts: Points,
    edges: Sequence[Edge],
    others: Sequence[Edge],
    reach: float,
    holder: Mapping[Edge, list[int]],
) -> int:
    """Return how many edges see an edge of ``others`` across a line near normal to both."""
    if not edges or not others:
        return 0
    a, b = _segments(verts, edges)
    c, d = _segments(verts, others)
    middle = 0.5 * (a + b)
    distance, along, line = _to_nearest(middle, c, d)
    # In the surface and out of the part: from the middle of the edge's face to the edge.
    outward = middle - numpy.array([verts[holder[e]].mean(axis=0) for e in edges])
    depth = numpy.linalg.norm(outward, axis=1)
    toward = (line * outward).sum(axis=1) / numpy.maximum(distance * depth, 1e-300)
    # A line shorter than that is a small overlap across the cut, which is a gap like another.
    outside = (toward > -FACING_COSINE) | (distance <= depth)
    facing = (
        (distance > 0.0)
        & (distance <= reach)
        & (_cosine(line, b - a) < FACING_COSINE)
        & (_cosine(line, along) < FACING_COSINE)
        & outside
    )
    return int(facing.sum())


def _to_nearest(middle: Points, c: Points, d: Points) -> tuple[Points, Points, Points]:
    """Return per point the distance to the nearest segment c-d, that segment and the line to it."""
    distance, nearest = _geometry.segment_distances(middle, c, d)
    along = d[nearest] - c[nearest]
    t = ((middle - c[nearest]) * along).sum(axis=1) / numpy.maximum(
        (along * along).sum(axis=1), 1e-300
    )
    return distance, along, c[nearest] + numpy.clip(t, 0.0, 1.0)[:, None] * along - middle


def _cosine(line: Points, edge: Points) -> NDArray[numpy.float64]:
    """Return the absolute cosine between each line and edge, zero for a line of no length."""
    lengths = numpy.linalg.norm(line, axis=1) * numpy.linalg.norm(edge, axis=1)
    return numpy.abs((line * edge).sum(axis=1)) / numpy.maximum(lengths, 1e-300)


def _turned_faces(source: ObjMesh, adapted: _Adapted, kept: Points) -> int:
    """Return how many adapted faces turned against the source face nearest them."""
    points = adapted.points.copy()
    moved = adapted.kept_node >= 0
    points[moved] = kept[adapted.kept_node[moved]]
    turned = 0
    for name, faces in adapted.families.items():
        if not faces:
            continue
        normals, centres = _geometry.normals_and_centroids(points, faces)
        before, at = _geometry.normals_and_centroids(source.verts, source.families[name])
        _, nearest = _geometry.NearestIndex(at).query(centres)
        turned += int(((normals * before[nearest]).sum(axis=1) < 0.0).sum())
    return turned


def _refuse(reason: str) -> InputArtifactError:
    return InputArtifactError(f"geometry merge: {reason}. Nothing was written.", kind=KIND)


def _which_adapted(first: Path, second: Path, adapt: Path) -> tuple[Path, Path]:
    """Return (kept, adapted) from the two parts and the one ``adapt`` names."""
    resolved = adapt.resolve()
    hits = [p for p in (first, second) if p.resolve() == resolved] or [
        p for p in (first, second) if p.name == adapt.name
    ]
    if len(hits) != 1:
        raise _refuse(
            f"adapt (CLI: --adapt) {adapt} names neither {first} nor {second}; name one of the two"
        )
    if adapt.parent != Path(".") and hits[0].resolve() != resolved:
        warn(
            f"adapt (CLI: --adapt) {adapt} is not the path of the part of that name, {hits[0]}; "
            "that part is the one adapted. Name that part's own path or its bare file name to "
            "select it with no warning",
            GeometryWarning,
            stacklevel=3,
        )
    return (second, first) if hits[0] == first else (first, second)


def _check_call(first: Path, second: Path, threshold: float, out: Path, *, overwrite: bool) -> None:
    """Refuse what no merge can answer: no threshold, one part twice, a source as output."""
    if not math.isfinite(threshold) or threshold < 0.0:
        raise InputValueError(
            f"geometry merge: threshold (CLI: --threshold) {threshold} is not a length of zero "
            "or more; state the largest distance across the interface. Nothing was written."
        )
    if first.resolve() == second.resolve():
        raise _refuse(f"{first} is named as both parts; name two OBJ files")
    if out.resolve() in (first.resolve(), second.resolve()):
        raise _refuse(f"out (CLI: --out) {out} is one of the two parts; name another file")
    written = (
        out,
        out.with_name(out.stem + RECORD_SUFFIX),
        out.with_name(out.stem + AUDIT_SUFFIX),
        inventory_sidecar(out, "obj"),
    )
    clash = [path.name for path in written if path.exists()]
    if clash and not overwrite:
        raise _refuse(
            f"{', '.join(clash)} exist(s) beside {out}; pass overwrite (CLI: --overwrite) to "
            "replace"
        )


def _of_the_parts_kind(out: Path) -> Path:
    """Return the output named as an OBJ; another kind is warned, as the wheel verbs do."""
    if out.suffix.lower() == ".obj":
        return out
    if out.name in ("", ".."):
        raise _refuse(f"out (CLI: --out) {out} names no file; name an OBJ file")
    target = out.with_suffix(".obj")
    warn(
        f"out (CLI: --out) {out.name} is not the parts' kind (.obj); the output is {target.name}. "
        "Name an .obj file to choose its name.",
        NotApplicableWarning,
        stacklevel=3,
    )
    return target


@dataclass(frozen=True)
class _Call:
    """What the call named."""

    kept: Path
    adapted: Path
    out: Path
    threshold: float


def merge_objs(
    first: str | Path,
    second: str | Path,
    *,
    threshold: float,
    adapt: str | Path,
    out: str | Path,
    overwrite: bool = False,
    say: Callable[[str], None] | None = None,
) -> MergedMesh:
    """Merge two trimmed OBJ along their open boundaries within a threshold.

    The command is ``pyfs-workspace geometry merge``; ``docs/geometry.md`` states the
    method and its limits.

    Parameters
    ----------
    first, second : str or Path
        The two parts.
    threshold : float
        The largest distance, in the OBJ's own length unit, between an interface node and
        the other part's open boundary.
    adapt : str or Path
        The part that is adapted, one of the two, by its path or its file name; a path
        elsewhere with a part's file name selects that part and is warned. The other is
        kept unchanged.
    out : str or Path
        The merged OBJ; ``<out stem>.merge.json``, ``<out stem>.audit.json`` and its
        inventory ``<out>.boundaries.toml`` are written beside it. A name of another kind is
        warned and takes ``.obj``.
    overwrite : bool
        Replace an existing output.
    say : callable, optional
        Receives one line of progress per stage.

    Returns
    -------
    MergedMesh
        What was written and measured. ``obj`` is None, with a warning, when no interface
        lies within the threshold: nothing was written.

    Raises
    ------
    InputArtifactError
        A part that is not a readable OBJ, ``adapt`` naming neither part, or an output
        that is a source or exists without ``overwrite``.
    InputValueError
        A threshold that is not a length of zero or more.
    """
    tell = say or (lambda _line: None)
    one, two, target = Path(first), Path(second), _of_the_parts_kind(Path(out))
    _check_call(one, two, threshold, target, overwrite=overwrite)
    kept_path, adapted_path = _which_adapted(one, two, Path(adapt))
    kept, adapted = read_obj(kept_path), read_obj(adapted_path)
    tell(f"kept {kept_path}, adapted {adapted_path}")
    _warn_units(kept_path, adapted_path)
    _warn_scattered(kept_path)
    _warn_scattered(adapted_path)
    pairs = _interface(
        kept.verts,
        _geometry.boundary_edges(_all_faces(kept)),
        adapted.verts,
        _geometry.boundary_edges(_all_faces(adapted)),
        threshold,
    )
    if not pairs:
        warn(
            f"geometry merge: no open boundary of {kept_path} lies within {threshold:g} of an "
            f"open boundary of {adapted_path}; nothing was written, so state a larger threshold",
            GeometryWarning,
            stacklevel=2,
        )
        return MergedMesh(None, None, None, None)
    tell(f"{len(pairs)} interface chain(s) within {threshold:g}")
    renamed = _renames(kept, adapted, adapted_path)
    done = _adapt(adapted, kept.verts, pairs, threshold, adapted_path)
    merged = _output_mesh(kept, done, renamed)
    turned = _turned_faces(adapted, done, kept.verts)
    call = _Call(kept_path, adapted_path, target, threshold)
    return _publish(
        merged, kept, done, call, pairs=pairs, renamed=renamed, turned=turned, tell=tell
    )


def _publish(
    merged: ObjMesh,
    kept: ObjMesh,
    done: _Adapted,
    call: _Call,
    *,
    pairs: Sequence[_Pair],
    renamed: Mapping[str, str],
    turned: int,
    tell: Callable[[str], None],
) -> MergedMesh:
    """Write the OBJ, its inventory, its audit and its record; warn what the output holds."""
    around = _geometry.edge_faces(_all_faces(merged))
    interface = sum(1 for p in pairs for e in p.kept.edges if len(around.get(e, ())) == 2)
    open_left = sum(1 for fs in around.values() if len(fs) == 1)
    size = min(kept.size, ObjMesh(merged.verts, {"adapted": _adapted_faces(merged, kept)}).size)
    facing = _facing_edges(merged, len(kept.verts), FACING_REACH * size)
    note = (
        f"merged: kept {call.kept.name}, adapted {call.adapted.name}, threshold {call.threshold:g}"
    )
    call.out.parent.mkdir(parents=True, exist_ok=True)
    write_obj(merged, call.out, note)
    side = _inventory(call.out)
    audit = audit_mesh(call.out)
    record = call.out.with_name(call.out.stem + RECORD_SUFFIX)
    _textio.write_json(
        record,
        {
            "schema_version": RECORD_SCHEMA,
            "threshold": call.threshold,
            "kept": str(call.kept),
            "kept_sha256": file_sha256(call.kept),
            "adapted": str(call.adapted),
            "adapted_sha256": file_sha256(call.adapted),
            "output": str(call.out),
            "interface_chains": len(pairs),
            "interface_edges": interface,
            "moved_nodes": done.moved_nodes,
            "band_faces": done.band_faces,
            "triangles_created": done.triangles_created,
            "open_edges_left": open_left,
            "open_edges_facing": facing,
            "turned_faces": turned,
            "degenerate_faces": len(audit.degenerate_faces),
            "non_manifold_edges": sum(1 for fs in around.values() if len(fs) > 2),
            "renamed_families": dict(renamed),
            "rebuilt_families": list(done.rebuilt),
            "faces_split": done.split,
            "conformed_interfaces": list(done.conformed),
            "declarations_merged": False,
            "audit_passed": audit.passed,
        },
    )
    tell(DECLARATIONS_NOTE)
    _warn_left(call, facing, turned, len(audit.degenerate_faces))
    return MergedMesh(
        call.out,
        record,
        side,
        audit,
        interface_edges=interface,
        moved_nodes=done.moved_nodes,
        band_faces=done.band_faces,
        triangles_created=done.triangles_created,
        open_edges_left=open_left,
        facing_edges_left=facing,
        turned_faces=turned,
        renamed=dict(renamed),
    )


def _adapted_faces(merged: ObjMesh, kept: ObjMesh) -> Faces:
    """Return the output's faces that follow the kept part's."""
    return _all_faces(merged)[len(_all_faces(kept)) :]


def _warn_left(call: _Call, facing: int, turned: int, degenerate: int) -> None:
    """Warn what the merge left: open edges facing the other part, turned or degenerate faces."""
    if facing:
        warn(
            f"geometry merge: {facing} open edge(s) of {call.out} still face the other part's "
            f"open boundary, further from it than the threshold {call.threshold:g}, and were "
            "left open; state a larger threshold to merge them",
            GeometryWarning,
            stacklevel=4,
        )
    if turned:
        warn(
            f"geometry merge: {turned} face(s) of the adapted part in {call.out} turned against "
            "the source face nearest them, which an adapted part overlapping the kept one by "
            "more than a face layer does; trim it back and merge again",
            GeometryWarning,
            stacklevel=4,
        )
    if degenerate:
        warn(
            f"geometry merge: {degenerate} degenerate face(s) in {call.out}, each listed under "
            "degenerate_faces in its audit; inspect them or re-cut the parts near them",
            GeometryWarning,
            stacklevel=4,
        )


def _inventory(out: Path) -> Path:
    """Write the merged OBJ's inventory, replacing any earlier one of the same output."""
    inventory_sidecar(out, "obj").unlink(missing_ok=True)
    return write_inventory(out)

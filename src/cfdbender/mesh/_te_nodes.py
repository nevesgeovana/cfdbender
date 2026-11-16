"""Trailing-edge points written from a node list or from an OBJ's own edges.

Three routes lead to a trailing-edge points file here:

* from an ordered node list a mesh-preparation tool saved from a node set
  (:func:`write_trailing_edge_points_from_nodes`), the supplied order being the chain;
* from the same list in any node order (:func:`write_trailing_edge_midnodes`), the nodes being
  put in order along their lines, and per family of an OBJ
  (:func:`write_trailing_edge_midnodes_per_family`);
* from an OBJ alone or with a node list (:func:`write_trailing_edges_identified`), where the
  trailing edge of a family is its rearmost line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

from cfdbender._errors import GeometryWarning, InputArtifactError, warn
from cfdbender.mesh._te_points import (
    QUAD_FAMILY_SHARE,
    quad_families,
    write_node_listing,
    write_trailing_edge_points,
)

__all__ = [
    "MIDNODE_GAP",
    "NEIGHBOUR_REACH",
    "write_trailing_edge_midnodes",
    "write_trailing_edge_midnodes_per_family",
    "write_trailing_edge_points_from_nodes",
    "write_trailing_edges_identified",
]

_KIND = "trailing_edges"

#: A step this many times longer than a node's distance to its nearest node starts a
#: new line in a node list: along a line the next node is about one spacing away.
MIDNODE_GAP = 5.0

#: A listed node is joined to each other node closer than this many times its distance
#: to its nearest node: along a line the two neighbours are about one spacing away, and a
#: corner where two lines of different spacing meet still joins both.
NEIGHBOUR_REACH = 2.5


def _ansa_nodes(source: Path) -> NDArray[numpy.float64]:
    """Read five-column SET rows in listed order, with physical line diagnostics."""
    points = []
    seen: set[int] = set()
    number = 1
    for number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = [field.strip() for field in raw.split(",")]
        reason = "expected five fields: Id, Name, X, Y, Z"
        if len(fields) == 5:
            try:
                node = int(fields[0])
                xyz = [float(value) for value in fields[2:]]
            except ValueError:
                reason = "expected an integer node id and three finite coordinates"
            else:
                reason = "expected three finite coordinates"
                if node in seen:
                    reason = f"repeated node id {node}; list each node once"
                elif all(math.isfinite(value) for value in xyz):
                    seen.add(node)
                    points.append(xyz)
                    continue
        raise InputArtifactError(f"{source}: line {number}: {reason}", kind=_KIND)
    if len(points) < 2:
        raise InputArtifactError(
            f"{source}: line {number}: fewer than two nodes; list at least two ordered nodes",
            kind=_KIND,
        )
    return numpy.asarray(points, dtype=float)


def write_trailing_edge_points_from_nodes(
    source: str | Path, path: str | Path, *, unit: str = "METER", overwrite: bool = False
) -> Path:
    """Write edge mid-points from an ordered node-window export.

    Parameters
    ----------
    source : str or pathlib.Path
        Comma-separated Id, Name, X, Y, Z rows; blank and comment lines are ignored.
    path : str or pathlib.Path
        Destination points file. No mesh is needed.
    unit : str
        Length unit of the coordinates, preserved in the output.
    overwrite : bool
        Allow replacement of an existing destination.

    Returns
    -------
    pathlib.Path
        The written points file, with one midpoint per neighbouring pair in listed order.

    Raises
    ------
    InputArtifactError
        For fewer than two nodes, malformed rows, repeated ids or non-finite coordinates,
        naming the source line; also for an invalid unit or an existing output.

    Notes
    -----
    Coordinates retain the input reference frame. Nodes are neither sorted nor joined
    by proximity: the supplied order defines the trailing-edge chain.
    """
    nodes = _ansa_nodes(Path(source))
    return write_trailing_edge_points(
        path, nodes[:-1] / 2 + nodes[1:] / 2, unit=unit, overwrite=overwrite
    )


def _warn_empty(what: str) -> None:
    """Warn that a verb found nothing to write and wrote nothing (``accept_empty``)."""
    warn(f"{what}; nothing is written (accept_empty, CLI --accept-empty)", GeometryWarning, 3)


def write_trailing_edge_midnodes(
    source: str | Path,
    path: str | Path,
    *,
    unit: str = "METER",
    overwrite: bool = False,
    accept_empty: bool = False,
) -> Path | None:
    """Write the edge mid-points of a node-window export, in any node order.

    The nodes are put in order along their lines: from an end, each step goes to the nearest
    node left, and a step longer than :data:`MIDNODE_GAP` times the node's distance to its
    nearest node starts a new line, so one list may hold the trailing edges of every
    component. Each neighbouring pair gives one mid-point. No mesh is needed. A closed line
    loses the mid-point of its closing edge. Nodes at one position (two ids) are read once,
    with a warning.

    Parameters
    ----------
    source : str or pathlib.Path
        Comma-separated Id, Name, X, Y, Z rows; blank and comment lines are ignored.
    path : str or pathlib.Path
        Destination points file.
    unit : str
        Length unit of the coordinates, written on the first line.
    overwrite : bool
        Allow replacement of an existing destination.
    accept_empty : bool
        A list with fewer than two nodes, or whose nodes give no mid-point, is warned and
        nothing is written (None is returned); by default it is refused.

    Returns
    -------
    pathlib.Path or None
        The written points file; None when ``accept_empty`` wrote nothing.

    Raises
    ------
    InputArtifactError
        For fewer than two nodes, malformed rows, repeated ids or non-finite coordinates,
        naming the source line; also for an invalid unit or an existing output.
    """
    try:
        nodes = _ansa_nodes(Path(source))
    except InputArtifactError as error:
        if not accept_empty or "fewer than two nodes" not in str(error):
            raise
        _warn_empty(str(error))
        return None
    # Two ids at one position (an open trailing edge's upper and lower node) are one
    # node of the line: a twin's zero distance would otherwise be every node's spacing.
    _, first = numpy.unique(nodes, axis=0, return_index=True)
    if len(first) < len(nodes):
        warn(
            f"{source}: {len(nodes) - len(first)} node(s) repeat the position of another "
            "and are read once",
            GeometryWarning,
            stacklevel=2,
        )
        nodes = nodes[numpy.sort(first)]
    lines = _node_lines(nodes)
    alone = sum(len(line) == 1 for line in lines)
    if alone:
        warn(
            f"{source}: {alone} node(s) are far from every other and give no mid-point; "
            "list one continuous line of nodes per trailing edge",
            GeometryWarning,
            stacklevel=2,
        )
    mids = [
        (nodes[a] + nodes[b]) / 2.0 for line in lines for a, b in zip(line, line[1:], strict=False)
    ]
    if not mids and accept_empty:
        _warn_empty(f"{source}: no two nodes are near enough to give a mid-point")
        return None
    return write_trailing_edge_points(
        path, numpy.asarray(mids).reshape(-1, 3), unit=unit, overwrite=overwrite
    )


def write_trailing_edge_midnodes_per_family(
    source: str | Path,
    mesh: str | Path,
    path: str | Path,
    *,
    unit: str = "METER",
    overwrite: bool = False,
    accept_empty: bool = False,
) -> list[Path]:
    """Write one points file per family of a mesh from a node list.

    Each listed node is matched to the mesh vertex within a quarter of that vertex's
    shortest edge, and every edge of a family joining two listed nodes gives its mid-point,
    so the points are the mesh's own. The files are named ``<path stem>.<family>.te.txt``
    beside ``path``; a family no edge of the list lies on gets none, and a node that matches
    no vertex is warned and skipped.

    Parameters
    ----------
    source : str or pathlib.Path
        The node list saved from the node set (Id, Name, X, Y, Z), in any order.
    mesh : str or pathlib.Path
        The OBJ the nodes belong to, in the same unit as the list.
    path : str or pathlib.Path
        The points file the per-family names are built from.
    unit : str
        Length unit of the coordinates, written on each file's first line.
    overwrite : bool
        Allow replacement of existing destinations.
    accept_empty : bool
        A list with no node read, or that joins no edge of the mesh, is warned and nothing is
        written (an empty list is returned); by default it is refused.

    Returns
    -------
    list of pathlib.Path
        The written points files, in the mesh's family order.

    Raises
    ------
    InputArtifactError
        For a list with no node read, a list that joins no edge of the mesh, an unreadable
        mesh, an invalid unit or an existing output.
    """
    from cfdbender.mesh._refine._obj import (
        midpoints,
        perimeter_edges,
        read_node_list,
        read_obj,
    )

    nodes = read_node_list(source)
    if not nodes.ids:
        no_node = f"{source}: no node is read; save the node window (Id, Name, X, Y, Z)"
        if accept_empty:
            _warn_empty(no_node)
            return []
        raise InputArtifactError(no_node, kind=_KIND)
    obj = read_obj(Path(mesh))
    found, unmatched = perimeter_edges(obj.verts, obj.families, nodes)
    if unmatched:
        shown = ", ".join(str(i) for i in unmatched[:5])
        warn(
            f"{source}: {len(unmatched)} of {len(nodes.ids)} node(s) match no vertex of {mesh} "
            f"and are skipped (ids {shown}{' and more' if len(unmatched) > 5 else ''}); the "
            "list and the OBJ must be saved from the same model, in the same unit",
            GeometryWarning,
            stacklevel=2,
        )
    target = Path(path)
    stem = target.name[: -len(".te.txt")] if target.name.endswith(".te.txt") else target.stem
    written = []
    for name, edges in found.items():
        if not edges:
            continue
        mids = midpoints(obj.verts, edges)
        written.append(
            write_trailing_edge_points(
                target.with_name(f"{stem}.{name}.te.txt"), mids, unit=unit, overwrite=overwrite
            )
        )
    if not written:
        no_edge = f"{source}: no edge of {mesh} joins two listed nodes; nothing was written"
        if accept_empty:
            _warn_empty(no_edge)
            return []
        raise InputArtifactError(no_edge, kind=_KIND)
    return written


def write_trailing_edges_identified(
    mesh: str | Path,
    source: str | Path | None = None,
    families: Sequence[str] | None = None,
    *,
    aft: Sequence[float] = (1.0, 0.0, 0.0),
    max_angle: float = 45.0,
    unit: str | None = None,
    overwrite: bool = False,
    accept_empty: bool = False,
    all_families: bool = False,
) -> list[Path]:
    """Find the trailing edges of an OBJ and write their points files.

    The trailing edge of a family is its rearmost line: the edges its surface lies forward
    of (``aft`` the direction from the nose to the tail), taken among

    - with ``source``, the lines of its listed nodes (a node-window export, any order, every
      perimeter in one file), each node joined to its near neighbours
      (:data:`NEIGHBOUR_REACH`): an edge whose two ends lie on those segments is on a line, so
      the OBJ may be a refined level whose nodes are not the listed ones;
    - without it, the boundaries of the family cut along its sharp edges, an edge kept when
      its tangent makes ``max_angle`` degrees or less with the plane across ``aft``, the yz
      plane for +x.

    Beside the mesh, ``families`` given, each family's file is ``<mesh stem>.<family>.te.txt``;
    else the trailing edges of the quad families (every family with ``all_families``) are one
    file, ``<mesh stem>.te.txt``. Without ``source``, the nodes found are also written as
    ``<mesh stem>.nodes.txt`` in the node-list format, the list a refine file's
    ``[[perimeter]]`` reads.

    Parameters
    ----------
    mesh : str or pathlib.Path
        The OBJ, with its families (merged components, for a refined level).
    source : str or pathlib.Path, optional
        The node list saved from the node set (Id, Name, X, Y, Z), in the mesh's unit.
    families : sequence of str, optional
        The families, one file each; a name the OBJ does not hold, or one on which no
        trailing edge is found, is warned and skipped. Default: every quad family, in one
        file; the others are warned and skipped.
    aft : sequence of float
        The aft direction, from the nose to the tail (default +x).
    max_angle : float
        Without ``source``, the largest angle in degrees between a trailing-edge edge and the
        yz plane (default 45).
    unit : str, optional
        The coordinates' unit; else the ``[import] units`` of the mesh's boundaries file,
        else METER.
    overwrite : bool
        Replace existing files.
    accept_empty : bool
        A node list with fewer than two nodes, or no trailing edge found on any family, is
        warned and nothing is written (an empty list is returned); by default it is refused.
    all_families : bool
        Without ``families``, map every family, the ones that are not quad families included.

    Returns
    -------
    list of pathlib.Path
        The points files written in the order of ``families``, then the node list.

    Raises
    ------
    InputArtifactError
        For a node list with fewer than two nodes, an unreadable mesh, an invalid unit, an
        existing file, or no trailing edge found.
    """
    from cfdbender.mesh._refine._obj import aft_direction, mesh_unit, read_obj

    target = Path(mesh)
    obj = read_obj(target)
    names = _families_to_map(target, obj.families, families, all_families)
    for name in names:
        if name not in obj.families:
            warn(
                f"{target}: holds no family {name!r} (it holds {', '.join(obj.families)}); "
                "it is skipped",
                GeometryWarning,
                stacklevel=2,
            )
    chosen = {n: obj.families[n] for n in names if n in obj.families}
    try:
        found = _identified_edges(obj, chosen, source, aft_direction(aft, str(target)), max_angle)
    except InputArtifactError as error:
        if not accept_empty or "fewer than two nodes" not in str(error):
            raise
        _warn_empty(str(error))
        return []
    for name, edges in found.items():
        if not edges and families is not None:
            warn(
                f"{target}: no trailing edge was found on family {name!r} (no line its surface "
                "lies forward of); it is skipped",
                GeometryWarning,
                stacklevel=2,
            )
    found = {n: e for n, e in found.items() if e}
    if not found:
        none = (
            f"{target}: no trailing edge was found on {', '.join(names) or 'any family'}; "
            "nothing was written"
        )
        if accept_empty:
            _warn_empty(none)
            return []
        raise InputArtifactError(none, kind=_KIND)
    points, listing = _identified_targets(target, found, families, source)
    existing = [f for f in [*points, *([listing] if listing else [])] if f.exists()]
    if existing and not overwrite:
        raise InputArtifactError(
            f"{existing[0]}: exists; pass overwrite (CLI: --overwrite) to replace it. "
            "Nothing was written.",
            kind=_KIND,
        )
    return _write_identified(
        obj.verts, found, points=points, listing=listing, unit=unit or mesh_unit(target) or "METER"
    )


def _families_to_map(
    target: Path, held: dict[str, Any], families: Sequence[str] | None, all_families: bool
) -> list[str]:
    """Return the families identify-te maps: the named ones, else the quad ones, else all."""
    if families is not None:
        return list(families)
    if all_families:
        return list(held)
    quads = quad_families(held)
    empty = [name for name, faces in held.items() if not faces]
    if empty:
        warn(f"{target}: {', '.join(empty)} holds no face; it is skipped", GeometryWarning, 3)
    others = [name for name in held if name not in quads and name not in empty]
    if others:
        verb = "is not a quad family" if len(others) == 1 else "are not quad families"
        warn(
            f"{target}: {', '.join(others)} {verb} "
            f"(quadrilaterals under {QUAD_FAMILY_SHARE:.0%} of the faces); left out of the "
            "trailing edges and the node list. Pass all_families (CLI: --all-families) to map "
            "them too.",
            GeometryWarning,
            stacklevel=3,
        )
    return quads


def _identified_edges(
    obj: Any,
    chosen: dict[str, Any],
    source: str | Path | None,
    aft: tuple[float, float, float],
    max_angle: float,
) -> dict[str, set[tuple[int, int]]]:
    """Return each family's trailing-edge edges: among a node list's lines, else from the mesh."""
    from cfdbender.mesh._refine._obj import (
        perimeter_edges_near,
        read_node_list,
        rearmost_edges,
        rearmost_perimeter,
    )
    from cfdbender.mesh._refine._prepare import rearmost_lines

    if source is None:
        return {n: rearmost_lines(obj.verts, f, aft, max_angle) for n, f in chosen.items()}
    nodes = read_node_list(source)
    if len(nodes.ids) < 2:
        raise InputArtifactError(
            f"{source}: fewer than two nodes are read; save the node window (Id, Name, X, Y, Z)",
            kind=_KIND,
        )
    pairs = _neighbour_pairs(nodes.points)
    a = nodes.points[[i for i, _ in pairs]].reshape(-1, 3)
    b = nodes.points[[j for _, j in pairs]].reshape(-1, 3)
    near = perimeter_edges_near(obj.verts, chosen, a, b)
    return {
        name: rearmost_edges(
            obj.verts, faces, rearmost_perimeter(obj.verts, faces, near[name], aft), aft
        )
        for name, faces in chosen.items()
    }


def _identified_targets(
    target: Path,
    found: dict[str, set[tuple[int, int]]],
    families: Sequence[str] | None,
    source: str | Path | None,
) -> tuple[dict[Path, set[tuple[int, int]]], Path | None]:
    """Return the points files and the node list identify-te writes."""
    if families is None:
        every = set().union(*found.values())
        points = {target.with_name(f"{target.stem}.te.txt"): every}
    else:
        points = {target.with_name(f"{target.stem}.{n}.te.txt"): e for n, e in found.items()}
    listing = None if source is not None else target.with_name(f"{target.stem}.nodes.txt")
    return points, listing


def _write_identified(
    verts: NDArray[numpy.float64],
    found: dict[str, set[tuple[int, int]]],
    *,
    points: dict[Path, set[tuple[int, int]]],
    listing: Path | None,
    unit: str,
) -> list[Path]:
    """Write the points files, then the node list."""
    from cfdbender.mesh._refine._obj import midpoints, write_te

    written = [write_te(path, unit, midpoints(verts, edges)) for path, edges in points.items()]
    if listing is not None:
        written.append(write_node_listing(listing, verts, found))
    return written


def _neighbour_pairs(nodes: NDArray[numpy.float64]) -> list[tuple[int, int]]:
    """Return each pair of listed nodes closer than NEIGHBOUR_REACH times either's spacing."""
    pairs: set[tuple[int, int]] = set()
    count = len(nodes)
    for start in range(0, count, 512):
        block = numpy.linalg.norm(nodes[start : start + 512, None] - nodes[None], axis=2)
        rows = numpy.arange(len(block))
        block[rows, start + rows] = numpy.inf
        spacing = block.min(axis=1)
        for r, c in zip(*numpy.nonzero(block <= NEIGHBOUR_REACH * spacing[:, None]), strict=True):
            i, j = int(start + r), int(c)
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _node_lines(nodes: NDArray[numpy.float64]) -> list[list[int]]:
    """Order nodes into lines: grow each line both ways to the nearest node left.

    A step longer than :data:`MIDNODE_GAP` times the node's spacing ends that direction; the
    line then grows from its other end, so a line started in its middle is still walked whole.
    """
    count = len(nodes)
    spacing = numpy.empty(count)
    for start in range(0, count, 512):
        block = numpy.linalg.norm(nodes[start : start + 512, None] - nodes[None], axis=2)
        block[numpy.arange(len(block)), numpy.arange(start, start + len(block))] = numpy.inf
        spacing[start : start + 512] = block.min(axis=1)
    left = numpy.ones(count, dtype=bool)

    def grow(line: list[int]) -> None:
        while left.any():
            rest = numpy.flatnonzero(left)
            distance = numpy.linalg.norm(nodes[rest] - nodes[line[-1]], axis=1)
            nearest = int(numpy.argmin(distance))
            if distance[nearest] > MIDNODE_GAP * spacing[line[-1]]:
                return
            line.append(int(rest[nearest]))
            left[line[-1]] = False

    lines: list[list[int]] = []
    while left.any():
        rest = numpy.flatnonzero(left)
        centre = nodes[rest].mean(axis=0)
        line = [int(rest[numpy.argmax(numpy.linalg.norm(nodes[rest] - centre, axis=1))])]
        left[line[0]] = False
        grow(line)
        line.reverse()
        grow(line)
        lines.append(line)
    return lines

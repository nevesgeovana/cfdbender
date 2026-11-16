"""The OBJ and trailing-edge points files a refinement reads and writes.

A mesh is one vertex array and an ordered mapping from family name to its
faces, each face a list of 0-based vertex indices. Families are the OBJ
groups (``g``) or objects (``o``) in file order; the comment lines before the
first vertex are kept and written back. Every write goes through the
package's one text route, so a level holds no carriage return.
"""

from __future__ import annotations

import copy
import math
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._decimal import plain_decimal
from cfdbender._digest import file_sha256
from cfdbender._errors import GeometryWarning, InputArtifactError, warn
from cfdbender._lengths import METRES_PER_UNIT, scale
from cfdbender.mesh._refine._geometry import (
    RIDGE_DEGREES,
    NearestIndex,
    dihedral_degrees,
    edge_faces,
    segment_distances,
)
from cfdbender.mesh._sidecars import (
    BOUNDARY_FACES_KEY,
    MESH_FACES_KEY,
    MESH_SHA256_KEY,
    TRAILING_EDGES_TABLE,
    inventory_sidecar,
    obj_boundary_names,
    obj_face_counts,
    read_trailing_edges,
    toml_value,
)

Faces = list[list[int]]

#: The family a face belongs to when it comes before any ``g`` or ``o`` line.
DEFAULT_FAMILY = "default"


def output_stems(names: Sequence[str]) -> dict[str, str]:
    """Reserve level and family filenames, renaming collisions in source order."""
    suffixes = (".te.txt", ".fs.te.txt", ".nodes.txt", ".fs.nodes.txt")

    def paths(name: str) -> set[str]:
        return {f".{name}{suffix}".casefold() for suffix in suffixes}

    occupied = {
        ".obj",
        ".te.txt",
        ".nodes.te.txt",
        ".nodes.txt",
        ".fs.nodes.txt",
        ".refine.json",
        ".audit.json",
        ".inspect.pdf",
        ".refine-governors.pdf",
        ".obj.boundaries.toml",
    }
    original = set().union(*(paths(name) for name in names))
    stems = {}
    for name in names:
        stem = name
        if paths(stem) & occupied:
            stem = name + "_family"
            number = 2
            while paths(stem) & (occupied | original):
                stem = f"{name}_family_{number}"
                number += 1
            warn(
                f"refine family {name}: output stem {name!r} collides with another level "
                f"output; using {stem!r} for its files",
                GeometryWarning,
                stacklevel=3,
            )
        stems[name] = stem
        occupied.update(paths(stem))
    return stems


def restore_node_names(paths: Sequence[Path], stems: dict[str, str]) -> None:
    """Keep original family labels in node lists written under safe output stems."""
    renamed = {stem: name for name, stem in stems.items() if name != stem}
    if not renamed:
        return
    for path in paths:
        if not path.name.endswith(".nodes.txt") or path.name.endswith(".fs.nodes.txt"):
            continue
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            node, separator, rest = line.partition(", ")
            for stem, name in renamed.items():
                if separator and rest.startswith(stem + ", "):
                    line = node + separator + name + rest[len(stem) :]
                    break
            lines.append(line)
        _textio.write_text(path, "".join(lines))


@dataclass
class ObjMesh:
    """One OBJ: vertices, families in file order, the header comments and the family keyword."""

    verts: NDArray[numpy.float64]
    families: dict[str, Faces]
    header: list[str] = field(default_factory=list)
    family_tag: str = "g"
    #: Per vertex, the coordinate text to write (the source's ``v`` words, as read),
    #: or None to write the number itself; empty to write every vertex's number.
    vertex_text: list[str | None] = field(default_factory=list)

    def family_vertices(self, name: str) -> NDArray[numpy.int64]:
        """Return the sorted indices of the vertices the family's faces use."""
        faces = self.families[name]
        return numpy.unique(numpy.fromiter((v for f in faces for v in f), dtype=numpy.int64))

    @property
    def size(self) -> float:
        """Return the diagonal of the bounding box of the used vertices (the SIZE of)."""
        used = numpy.unique(
            numpy.fromiter(
                (v for faces in self.families.values() for f in faces for v in f), dtype=numpy.int64
            )
        )
        if not used.size:
            return 0.0
        box = self.verts[used]
        return float(numpy.linalg.norm(box.max(axis=0) - box.min(axis=0)))


#: The artifact kind a refusal of this package names.
KIND = "mesh"


def _refuse(path: Path, reason: str) -> InputArtifactError:
    """Return the refusal of a file: its name, the reason, the remedy, and nothing written."""
    return InputArtifactError(
        f"{path}: {reason}. Fix the file and run again. Nothing was written.", kind=KIND
    )


def _refuse_at(where: str, what: str) -> InputArtifactError:
    """Return the refusal of one line of an OBJ."""
    return InputArtifactError(
        f"{where} is {what}. Fix the file and run again. Nothing was written.", kind=KIND
    )


def read_obj(path: str | Path) -> ObjMesh:
    """Read an OBJ: ``v`` and ``f`` lines, ``g`` or ``o`` families, header comments.

    Texture and normal indices of a face (``f 1/2/3``) are dropped; a negative
    index counts back from the last vertex read. Every other statement
    (``usemtl``, ``mtllib``, ``s``, ``vn``, ``vt``) is read past and not
    carried: a mesh written from this one holds only vertices and faces by family.
    A vertex coordinate that is
    not a finite number, a face index that is not an integer, and a face with
    fewer than three vertices or naming a vertex that does not exist, are
    refused naming the file and the line. A path that is not a file is
    refused in the words :func:`cfdbender.mesh._refine.audit_mesh` uses,
    so the refinement and the audit refuse a missing mesh alike.
    """
    source = Path(path)
    if not source.is_file():
        raise InputArtifactError(
            f"{source}: no such OBJ file; name an existing OBJ and run again. Nothing was written.",
            kind=KIND,
        )
    mesh = ObjMesh(numpy.zeros((0, 3)), {})
    verts: list[tuple[float, float, float]] = []
    current = DEFAULT_FAMILY
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise _refuse(
            source,
            f"not UTF-8 text (byte {error.start}); a replacement character would merge distinct "
            "family names, so save the OBJ as UTF-8",
        ) from error
    for number, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if s.startswith("#") and not verts:
            mesh.header.append(s)
        elif s and not s.startswith("#"):
            current = _statement(f"{source}: line {number}", s, verts, mesh, current)
            if len(verts) > len(mesh.vertex_text):
                mesh.vertex_text.append(" ".join(s.split()[1:4]))
    if not mesh.families:
        raise _refuse(source, "the file holds no face")
    mesh.verts = numpy.asarray(verts, dtype=float).reshape(-1, 3)
    return mesh


def _statement(
    where: str, s: str, verts: list[tuple[float, float, float]], mesh: ObjMesh, current: str
) -> str:
    """Read one ``v``, ``f``, ``g`` or ``o`` statement into the mesh; return the current family.

    The keyword is separated from its arguments by any whitespace, a tab included.
    """
    tag, *more = s.split(None, 1)
    rest = more[0] if more else ""
    if tag == "v":
        words = rest.split()
        if len(words) < 3:
            raise _refuse_at(where, "a vertex with fewer than three numbers")
        x, y, z = (_coordinate(where, word) for word in words[:3])
        verts.append((x, y, z))
    elif tag == "f":
        idx = [_index(where, tok.split("/")[0]) for tok in rest.split()]
        face = [i - 1 if i > 0 else len(verts) + i for i in idx]
        if len(face) < 3 or any(not 0 <= v < len(verts) for v in face):
            raise _refuse_at(where, "a face with fewer than three vertices or a missing vertex")
        mesh.families.setdefault(current, []).append(face)
    elif tag in ("g", "o"):
        mesh.family_tag = tag
        return rest.strip() or DEFAULT_FAMILY
    return current


def _coordinate(where: str, word: str) -> float:
    """Return a vertex coordinate, refusing a word that is not a finite number."""
    try:
        value = float(word)
    except ValueError:
        raise _refuse_at(where, f"a vertex whose coordinate {word!r} is not a number") from None
    if not math.isfinite(value):
        raise _refuse_at(where, f"a vertex whose coordinate {word!r} is not finite")
    return value


def _index(where: str, word: str) -> int:
    """Return a face's vertex index, refusing a word that is not an integer."""
    try:
        return int(word)
    except ValueError:
        raise _refuse_at(where, f"a face whose vertex index {word!r} is not an integer") from None


def coordinates(point: NDArray[numpy.float64]) -> str:
    """Return a point as written: each coordinate the plain decimal that reads back to it.

    The shortest round-trip digits and never an exponent
    (:func:`cfdbender._decimal.plain_decimal`), so a node written and
    read back is the node computed, at any scale.
    """
    return " ".join(plain_decimal(float(x)) for x in point)


def face_blocks(path: str | Path) -> list[tuple[str, int]]:
    """Return an OBJ's face blocks in file order, each as its family and its face count.

    A family that returns after another has one entry per run of faces, so a text
    written with these blocks (``obj_text(..., blocks=...)``) keeps the source's face order.
    """
    blocks: list[tuple[str, int]] = []
    current = DEFAULT_FAMILY
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = line.split(None, 1)
        if words and words[0] in ("g", "o"):
            current = (words[1].strip() if len(words) > 1 else "") or DEFAULT_FAMILY
        elif words and words[0] == "f":
            if blocks and blocks[-1][0] == current:
                blocks[-1] = (current, blocks[-1][1] + 1)
            else:
                blocks.append((current, 1))
    return blocks


def obj_text(
    mesh: ObjMesh, note: str | None = None, blocks: Sequence[tuple[str, int]] | None = None
) -> str:
    """Return the OBJ text of a mesh, families in their order.

    A vertex whose source text is known (``vertex_text``) is written as the
    source wrote it; every other is written by :func:`coordinates`. With ``blocks``
    (:func:`face_blocks` of the source) the faces are written in the source's order,
    a family that returns after another in a block of its own.
    """
    lines = list(mesh.header)
    if note:
        lines.append(f"# {note}")
    known = mesh.vertex_text if len(mesh.vertex_text) == len(mesh.verts) else []
    lines += [
        f"v {known[i] if known and known[i] is not None else coordinates(p)}"
        for i, p in enumerate(mesh.verts)
    ]
    if blocks is None:
        blocks = [(name, len(faces)) for name, faces in mesh.families.items()]
    taken = dict.fromkeys(mesh.families, 0)
    for name, count in blocks:
        lines.append(f"{mesh.family_tag} {name}")
        faces = mesh.families[name][taken[name] : taken[name] + count]
        taken[name] += count
        lines += ["f " + " ".join(str(v + 1) for v in f) for f in faces]
    return "\n".join(lines) + "\n"


def write_obj(mesh: ObjMesh, path: str | Path, note: str | None = None) -> Path:
    """Write a mesh as an OBJ through the package's text route and return the path."""
    target = Path(path)
    _textio.write_text(target, obj_text(mesh, note))
    return target


def compact(
    verts: NDArray[numpy.float64], families: dict[str, Faces]
) -> tuple[NDArray[numpy.float64], dict[str, Faces]]:
    """Drop unreferenced vertices and renumber, keeping the order of first use."""
    order: dict[int, int] = {}
    for faces in families.values():
        for f in faces:
            for v in f:
                if v not in order:
                    order[v] = len(order)
    keep = numpy.fromiter(order.keys(), dtype=numpy.int64)
    return verts[keep], {n: [[order[v] for v in f] for f in fs] for n, fs in families.items()}


#: The suffix of a mesh's boundaries file, beside it.
SIDECAR_SUFFIX = ".boundaries.toml"
#: The suffix of a mesh's trailing-edge points file when its boundaries file names none.
TE_SUFFIX = ".te.txt"


def read_sidecar(obj: str | Path) -> tuple[Path, dict[str, Any]] | None:
    """Return a mesh's boundaries file and its tables, or None when it has none.

    Raises
    ------
    InputArtifactError
        The file is not TOML; nothing was written.
    """
    sidecar = inventory_sidecar(obj)
    if not sidecar.is_file():
        return None
    try:
        return sidecar, tomllib.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise InputArtifactError(
            f"{sidecar}: not TOML ({error}). Fix the file and run again. Nothing was written.",
            kind=KIND,
        ) from error


@dataclass(frozen=True)
class TrailingRoute:
    """What a mesh's boundaries file states of its trailing edge, ``[trailing_edges]``.

    ``file`` is the points file it names, beside it, whether or not that
    file exists; ``detect`` is whether it routes the trailing edge to
    detection (``detect = "auto"`` or a ``detect`` table), and ``surfaces``
    the surfaces it names for detection, empty for every surface.
    ``sidecar`` is the boundaries file and ``named`` the points file's name
    as it is written there. ``none`` is whether it states ``none = true``,
    that the mesh has no trailing edge to route: the mesh then has no points
    file, not even a ``<stem>.te.txt`` beside it.
    """

    file: Path | None = None
    sidecar: Path | None = None
    named: str | None = None
    detect: bool = False
    surfaces: tuple[str, ...] = ()
    none: bool = False

    def detects(self, family: str) -> bool:
        """Return whether the boundaries file routes the family's trailing edge to detection."""
        return self.detect and (not self.surfaces or family in self.surfaces)


def te_route(obj: str | Path) -> TrailingRoute:
    """Return what a mesh's boundaries file states of its trailing edge.

    The standalone audit's reader of the boundaries file's
    ``[trailing_edges]``. Refinement selects its points in the refine file. A table that routes the
    trailing edge to detection is read by the workflow's own reader,
    :func:`cfdbender.mesh._sidecars.read_trailing_edges`, so a
    ``detect`` the workflow refuses is refused here with the same words. A
    mesh without a boundaries file, or whose file has no such table, states
    nothing.

    Raises
    ------
    InputArtifactError
        The boundaries file is not TOML, or its ``[trailing_edges]`` routes
        the trailing edge to detection in a shape the workflow refuses;
        nothing was written.
    """
    read = read_sidecar(obj)
    table = None if read is None else read[1].get(TRAILING_EDGES_TABLE)
    if read is None or not isinstance(table, dict):
        return TrailingRoute()
    named = table.get("file")
    surfaces: tuple[str, ...] = ()
    if "detect" in table:
        try:
            marking = read_trailing_edges(read[0], table)
        except InputArtifactError as error:
            raise InputArtifactError(f"{error} Nothing was written.", kind=KIND) from error
        surfaces = () if marking is None else marking.detect_surfaces
    return TrailingRoute(
        file=read[0].parent / named if isinstance(named, str) else None,
        sidecar=read[0],
        named=named if isinstance(named, str) else None,
        detect="detect" in table,
        surfaces=surfaces,
        none=table.get("none") is True,
    )


def te_file(obj: str | Path) -> Path | None:
    """Return a mesh's trailing-edge points file, or None when it has none.

    It is the file the boundaries file names under ``[trailing_edges] file``,
    or else ``<stem>.te.txt`` beside the mesh. The standalone audit
    reads it here; refinement instead uses its own explicit points input. A boundaries
    file that states ``[trailing_edges] none = true`` has no points file, and
    a ``<stem>.te.txt`` left beside the mesh is not read.

    Raises
    ------
    InputArtifactError
        The boundaries file is not TOML, or names a points file that does not
        exist; nothing was written.
    """
    route = te_route(obj)
    if route.none:
        return None
    if route.file is not None:
        if not route.file.is_file():
            raise InputArtifactError(
                f"{route.sidecar}: names the trailing-edge file {route.named}, which does not "
                "exist. Write the points file or remove the key and run again. Nothing was "
                "written.",
                kind=KIND,
            )
        return route.file
    default = Path(obj).with_name(Path(obj).stem + TE_SUFFIX)
    return default if default.is_file() else None


def mesh_unit(obj: str | Path) -> str | None:
    """Return the length unit the boundaries file states for the mesh, ``[import] units``.

    The unit a raw mesh is written in is stated there and never assumed
    (:class:`cfdbender.mesh._sidecar_models.MeshImport`); None when it is not stated.
    """
    read = read_sidecar(obj)
    table = None if read is None else read[1].get("import")
    units = table.get("units") if isinstance(table, dict) else None
    return units.strip().upper() if isinstance(units, str) and units.strip() else None


@dataclass(frozen=True)
class TrailingPoints:
    """A mesh's trailing-edge points: the file's unit line and the points in the mesh's unit.

    ``to_file`` multiplies a length in the mesh's unit into the file's unit,
    so a points file written back keeps the source file's unit line.
    """

    unit: str
    points: NDArray[numpy.float64]
    to_file: float


def te_points(obj: str | Path) -> TrailingPoints | None:
    """Read a mesh's trailing-edge points and convert them to the mesh's unit.

    The points file states its unit on its first line; the mesh's unit is the
    ``[import] units`` of its boundaries file. When both are stated and differ
    the points are converted (:func:`cfdbender._lengths.scale`), as the
    package converts a points file for a run; when the mesh states none, the
    points are read in the mesh's unit.

    Raises
    ------
    InputArtifactError
        The points file, or the boundaries file that names it, is refused
        (:func:`te_file`, :func:`read_te`), or the two units cannot be
        converted (a unit that names no scale); nothing was written.
    """
    file = te_file(obj)
    if file is None:
        return None
    return points_from_file(obj, file)


def points_from_file(obj: str | Path, file: Path | None) -> TrailingPoints | None:
    """Read only the explicitly selected points file, converting to the mesh unit."""
    if file is None:
        return None
    unit, points = read_te(file)
    target = mesh_unit(obj)
    if target is None or unit.strip().upper() == target:
        return TrailingPoints(unit, points, 1.0)
    into, back = scale(unit.strip().upper(), target), scale(target, unit.strip().upper())
    if into is None or back is None:
        raise InputArtifactError(
            f"{file}: its points are in {unit} and the mesh is in {target} (the [import] units "
            "of its boundaries file), and one of the two names no length scale, so the points "
            f"cannot be converted. Give both in one of {', '.join(METRES_PER_UNIT)} and run "
            "again. Nothing was written.",
            kind=KIND,
        )
    return TrailingPoints(unit, points * into, back)


def read_te(path: str | Path) -> tuple[str, NDArray[numpy.float64]]:
    """Read a trailing-edge points file: a unit line, then one ``x,y,z`` row per point.

    Every row holds exactly three finite numbers; any other row is refused
    naming the file and its line, so no row is ever read as part of another
    point. The first row must be a unit the package converts, one of
    :data:`cfdbender._lengths.METRES_PER_UNIT` (case and surrounding
    blanks ignored); any other first row is refused naming the file and the
    units it accepts, before any point is read, so a file that lacks its unit
    line never loses its first point silently.
    """
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise _refuse(source, f"not readable UTF-8 text ({error})") from error
    rows = [(n, r.strip()) for n, r in enumerate(lines, start=1) if r.strip()]
    if not rows:
        raise _refuse(source, "the trailing-edge points file is empty")
    if rows[0][1].upper() not in METRES_PER_UNIT:
        raise _refuse(
            source,
            f"line {rows[0][0]} {rows[0][1]!r} is not a length unit; the first row of a "
            f"trailing-edge points file is its unit, one of {', '.join(METRES_PER_UNIT)}",
        )
    points = [_te_row(source, number, row) for number, row in rows[1:]]
    return rows[0][1], numpy.array(points, dtype=float).reshape(-1, 3)


def _te_row(source: Path, number: int, row: str) -> list[float]:
    """Return one trailing-edge row as its three coordinates, refusing any other row."""
    words = [w.strip() for w in row.split(",")]
    said = None
    if len(words) != 3:
        said = f"holds {len(words)} number(s)"
    else:
        try:
            values = [float(w) for w in words]
        except ValueError:
            said = "holds a word that is not a number"
        else:
            if all(math.isfinite(v) for v in values):
                return values
            said = "holds a number that is not finite"
    raise _refuse(source, f"line {number} {row!r} {said}; a trailing-edge row is x,y,z")


def write_te(path: str | Path, unit: str, midpoints: NDArray[numpy.float64]) -> Path:
    """Write a trailing-edge points file under the given unit line and return the path."""
    target = Path(path)
    rows = [unit] + [coordinates(p).replace(" ", ",") for p in midpoints]
    _textio.write_lines(target, rows)
    return target


def edge_midpoints(
    verts: NDArray[numpy.float64], faces: Faces
) -> tuple[NDArray[numpy.int64], NDArray[numpy.float64]]:
    """Return every undirected edge of the faces, sorted, and its mid-point."""
    edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1], strict=True)}
    e = numpy.array(sorted(edges), dtype=numpy.int64).reshape(-1, 2)
    return e, 0.5 * (verts[e[:, 0]] + verts[e[:, 1]])


def midpoints(verts: NDArray[numpy.float64], edges: set[tuple[int, int]]) -> NDArray[numpy.float64]:
    """Return the mid-point of each edge, the edges sorted, shape (n, 3)."""
    pairs = numpy.array(sorted(edges), dtype=numpy.int64).reshape(-1, 2)
    return 0.5 * (verts[pairs[:, 0]] + verts[pairs[:, 1]])


#: A listed node is the mesh vertex closer to it than this fraction of the vertex's
#: shortest edge: ANSA writes a node's position rounded, so the match is made within the
#: local edge length, never within a fraction of the whole mesh.
NODE_EDGE_FRACTION = 0.25


@dataclass(frozen=True)
class NodeList:
    """The nodes of an ANSA SET node-window export: their ids and positions, in file order."""

    path: Path
    ids: list[int]
    points: NDArray[numpy.float64]


def read_node_list(path: str | Path) -> NodeList:
    """Read the ASCII node list ANSA saves from a SET (``#  Id, Name, X , Y, Z``).

    The header line names the columns, and Id, X, Y and Z are read by name;
    without a header the five columns are Id, Name, X, Y, Z. Blank lines and
    other comment lines are skipped. A row that cannot be read, and a node id
    listed before, is warned and skipped; the caller decides on an empty list.
    """
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as error:
        raise _refuse(source, f"the node list is not readable ({error})") from error
    columns = {"ID": 0, "X": 2, "Y": 3, "Z": 4}
    ids: list[int] = []
    points: list[list[float]] = []
    skipped: list[int] = []
    for number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        if text.startswith("#"):
            names = [c.strip().upper() for c in text.lstrip("#").split(",")]
            if {"ID", "X", "Y", "Z"} <= set(names):
                columns = {k: names.index(k) for k in ("ID", "X", "Y", "Z")}
            continue
        words = [w.strip() for w in text.split(",")]
        try:
            node = int(words[columns["ID"]])
            xyz = [float(words[columns[k]]) for k in ("X", "Y", "Z")]
        except (ValueError, IndexError):
            skipped.append(number)
            continue
        if node in ids or not all(math.isfinite(v) for v in xyz):
            skipped.append(number)
            continue
        ids.append(node)
        points.append(xyz)
    if skipped:
        shown = ", ".join(str(n) for n in skipped[:5])
        warn(
            f"{source}: {len(skipped)} line(s) are skipped (not Id, Name, X, Y, Z, or a node "
            f"listed before): line(s) {shown}{' and more' if len(skipped) > 5 else ''}",
            GeometryWarning,
            stacklevel=2,
        )
    return NodeList(source, ids, numpy.array(points, dtype=float).reshape(-1, 3))


def perimeter_edges(
    verts: NDArray[numpy.float64], families: dict[str, Faces], nodes: NodeList
) -> tuple[dict[str, set[tuple[int, int]]], list[int]]:
    """Return, per family, the edges whose two ends are listed nodes, and the ids matched to none.

    A mesh vertex is a listed node when it lies closer to one than
    :data:`NODE_EDGE_FRACTION` of its shortest edge, so two vertices at one
    position (an open trailing edge) are both listed. The order of the nodes
    does not matter, and one list may hold the perimeters of several families.
    """
    every = [f for fs in families.values() for f in fs]
    edges, _ = edge_midpoints(verts, every)
    shortest = numpy.full(len(verts), numpy.inf)
    if len(edges):
        length = numpy.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        numpy.minimum.at(shortest, edges[:, 0], length)
        numpy.minimum.at(shortest, edges[:, 1], length)
    used = numpy.flatnonzero(numpy.isfinite(shortest))
    listed: set[int] = set()
    if len(used):
        dist, _ = NearestIndex(nodes.points).query(verts[used])
        listed = {int(v) for v in used[dist < NODE_EDGE_FRACTION * shortest[used]]}
        near, index = NearestIndex(verts[used]).query(nodes.points)
        hit = near < NODE_EDGE_FRACTION * shortest[used[index]]
    else:
        hit = numpy.zeros(len(nodes.ids), dtype=bool)
    unmatched = [nodes.ids[i] for i in numpy.flatnonzero(~hit)]
    out = {}
    for name, faces in families.items():
        fam, _ = edge_midpoints(verts, faces)
        out[name] = {(int(a), int(b)) for a, b in fam if int(a) in listed and int(b) in listed}
    return out, unmatched


#: The places a value can be closed: a quote of a string, the bracket of an array, or
#: the last digit of a number (an inventory's face count).
_VALUE_ENDS = re.compile(r"[\"'\]]|\d(?=[\s,#}\]]|\Z)")


def rewrite_value(
    side: Path,
    text: str,
    key: str,
    old: object,
    wanted: dict[str, Any],
    *,
    value: object | None = None,
) -> str:
    """Return ``text`` with the value of ``key`` replaced so that it parses to ``wanted``.

    Every spelling of the key (bare, quoted, the last part of a dotted key, a
    member of an inline table) is a candidate; the value after it is the
    shortest text that parses to ``old``; the first replacement whose whole
    file parses to ``wanted`` is the one. The shared TOML writer escapes
    control characters, including DEL, in replacement strings.
    """
    new = toml_value(_value_of(wanted, key) if value is None else value)
    spelled = rf"(?<![\w\-\"'])(?:{key}|\"{key}\"|'{key}')\s*=[ \t]*"
    for match in re.finditer(spelled, text):
        start = match.end()
        for end in (m.end() for m in _VALUE_ENDS.finditer(text, start)):
            try:
                found = tomllib.loads("v = " + text[start:end] + "\n")["v"]
            except tomllib.TOMLDecodeError:
                continue
            if found != old:
                continue
            candidate = text[:start] + new + text[end:]
            try:
                if tomllib.loads(candidate) == wanted:
                    return candidate
            except tomllib.TOMLDecodeError:
                pass
            break
    raise _refuse(
        side,
        f"the value of {key} cannot be rewritten for the level; state it as "
        f"{key} = {toml_value(old)} and run again",
    )


def _value_of(data: dict[str, Any], key: str) -> object:
    """Return the value a rewrite writes: the ``boundaries`` list or the points file name."""
    return data[key] if key == "boundaries" else data["trailing_edges"][key]


def level_inventory(side: Path, text: str, obj: Path) -> str:
    """Return a level's boundaries file with the face counts and digest of its own OBJ.

    A source boundaries file that states ``mesh_faces``, ``boundary_faces`` or
    ``mesh_sha256`` states them of the SOURCE mesh; the level's file states
    each of those the source's states, counted in the level's OBJ as written
    (``boundary_faces`` in the level's ``boundaries`` order, after any
    ``[components]`` merge) and its sha256 of those bytes. A file stating none
    is returned as it is. Each value is rewritten in place
    (:func:`rewrite_value`), every declaration byte kept. The inventory header
    comment ("the OBJ's sha256 was ...") names the level OBJ's digest too.
    A digest appearing in an authored declaration stays unchanged.

    Raises
    ------
    InputArtifactError
        A value the rewrite cannot place, or a boundary the level's OBJ does
        not hold; nothing was written.
    """
    data = tomllib.loads(text)
    sha = file_sha256(obj)
    keys = [k for k in (MESH_FACES_KEY, BOUNDARY_FACES_KEY, MESH_SHA256_KEY) if k in data]
    if keys:
        text = _counted_inventory(side, text, obj, data, keys, sha=sha)
    return _SOURCE_SHA.sub(lambda match: match.group(0).replace(match.group(1), sha), text)


#: The header line the package writes with an OBJ's inventory, naming the OBJ's sha256.
_SOURCE_SHA = re.compile(r"(?m)^# the OBJ's sha256 was ([0-9a-f]{64})")


def _counted_inventory(
    side: Path, text: str, obj: Path, data: dict[str, Any], keys: list[str], *, sha: str
) -> str:
    """Return the file with each inventory key the source states rewritten to the level's."""
    held: dict[str, int] = {}
    for name, faces in zip(obj_boundary_names(obj), obj_face_counts(obj), strict=True):
        held[name] = held.get(name, 0) + faces
    listed = data.get("boundaries", list(held))
    missing = [n for n in listed if n not in held]
    if missing:
        raise _refuse(side, f"its boundaries {missing} are not groups of the level's OBJ")
    values: dict[str, object] = {
        MESH_FACES_KEY: sum(held.values()),
        BOUNDARY_FACES_KEY: [held[n] for n in listed],
        MESH_SHA256_KEY: sha,
    }
    for key in keys:
        wanted = {**data, key: values[key]}
        text = rewrite_value(side, text, key, data[key], wanted, value=values[key])
        data = wanted
    return text


def rename_table_key(side: Path, text: str, old: str, new: str, wanted: dict[str, Any]) -> str:
    """Replace only the family key whose parsed result equals the intended declarations."""
    spellings = [toml_value(old), "'" + old + "'", old]
    for spelling in spellings:
        for match in re.finditer(re.escape(spelling), text):
            candidate = text[: match.start()] + toml_value(new) + text[match.end() :]
            try:
                if tomllib.loads(candidate) == wanted:
                    return candidate
            except tomllib.TOMLDecodeError:
                pass
    raise _refuse(side, f"cannot rename base_regions family {old!r}; use a quoted table key")


#: The aft direction a trailing edge is found by when none is stated: +x, the axis a
#: fuselage runs along from the nose to the tail in an aircraft's body coordinates.
AFT = (1.0, 0.0, 0.0)


def aft_direction(aft: Sequence[float], where: str) -> tuple[float, float, float]:
    """Return a stated aft direction, or +x (:data:`AFT`) with a warning.

    One that is not three finite numbers, not all zero, is warned and +x taken,
    as a refine file's ``[[perimeter]] aft`` is.
    """
    try:
        x, y, z = (float(c) for c in aft)
    except (TypeError, ValueError):
        x = y = z = math.nan
    if not all(math.isfinite(c) for c in (x, y, z)) or not any((x, y, z)):
        warn(
            f"{where}: aft {tuple(aft)} is not a direction (three finite numbers, not all "
            "zero); +x is taken",
            GeometryWarning,
            stacklevel=3,
        )
        return AFT
    return (x, y, z)


#: A listed edge is a trailing edge when the family's surface lies forward of it: the unit
#: vector from the edge into its faces points against the aft direction with a cosine
#: below this (within 60 degrees of forward), so a trailing edge swept up to 60 degrees is
#: one, and a tip, a root or a leading edge, whose surface lies beside or behind it, is not.
REARMOST_COSINE = -0.5

#: A listed edge held by two faces that lie on one side of it, the sum of their two unit
#: inward vectors longer than this (the faces within about 83 degrees of each other), is
#: a fold of the surface: a sharp trailing edge, or a sharp leading edge.
FOLD_SIZE = 1.5

#: A fold is a trailing edge when its faces lie forward of it within about 78 degrees of
#: forward. The cone is wider than :data:`REARMOST_COSINE` because a fold points one way
#: only: a twisted blade's chord lies near the rotation plane at the tip (20 degrees from
#: it at a typical tip pitch), and a leading-edge fold points aft whatever the twist.
FOLD_COSINE = -0.2


def rearmost_edges(
    verts: NDArray[numpy.float64],
    faces: Faces,
    edges: set[tuple[int, int]],
    aft: tuple[float, float, float] = AFT,
) -> set[tuple[int, int]]:
    """Return the edges the faces lie forward of: a family's trailing edge among its lines.

    For each edge, the vector from its mid-point to the centroid of each face
    that holds it, less its part along the edge, is summed over those faces;
    the edge is kept when that inward direction points forward
    (:data:`REARMOST_COSINE`), or, for a fold of the surface
    (:data:`FOLD_SIZE`), within the wider cone of :data:`FOLD_COSINE`. An edge
    two faces hold on opposite sides of a leading edge points aft and is left
    out; a sharp trailing edge held by an upper and a lower face points forward
    on both.
    """
    direction = numpy.asarray(aft, dtype=float)
    direction = direction / numpy.linalg.norm(direction)
    out = set()
    for edge, (held, size, cosine) in _inward(verts, faces, edges, direction).items():
        # Faces on both sides of an interior edge cancel: only a clear inward direction counts.
        if size > 0.5 and cosine < REARMOST_COSINE:
            out.add(edge)
        elif held == 2 and size > FOLD_SIZE and cosine < FOLD_COSINE:
            out.add(edge)
    return out


def folded_edges(
    verts: NDArray[numpy.float64], faces: Faces, edges: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Return the given edges two of ``faces`` hold on one side of them (:data:`FOLD_SIZE`)."""
    found = _inward(verts, faces, edges, numpy.asarray(AFT, dtype=float))
    return {e for e, (held, size, _) in found.items() if held == 2 and size > FOLD_SIZE}


def vertex_components(faces: Faces) -> dict[int, int]:
    """Return each vertex's component: faces sharing a vertex are one component."""
    return _pieces_of_vertices(faces)


def _inward(
    verts: NDArray[numpy.float64],
    faces: Faces,
    edges: set[tuple[int, int]],
    direction: NDArray[numpy.float64],
) -> dict[tuple[int, int], tuple[int, float, float]]:
    """Return per held edge: its face count, its summed inward vector's size and cosine."""
    holders: dict[tuple[int, int], list[int]] = {e: [] for e in edges}
    for i, face in enumerate(faces):
        for a, b in zip(face, [*face[1:], face[0]], strict=True):
            key = (min(a, b), max(a, b))
            if key in holders:
                holders[key].append(i)
    out: dict[tuple[int, int], tuple[int, float, float]] = {}
    for (a, b), held in holders.items():
        if not held:
            continue
        along = verts[b] - verts[a]
        along = along / max(float(numpy.linalg.norm(along)), 1e-300)
        mid = 0.5 * (verts[a] + verts[b])
        inward = numpy.zeros(3)
        for i in held:
            reach = verts[faces[i]].mean(axis=0) - mid
            reach = reach - (reach @ along) * along
            size = float(numpy.linalg.norm(reach))
            if size > 0.0:
                inward += reach / size
        size = float(numpy.linalg.norm(inward))
        cosine = float(inward @ direction) / size if size > 0.0 else 0.0
        out[(a, b)] = (len(held), size, cosine)
    return out


def rearmost_perimeter(
    verts: NDArray[numpy.float64],
    faces: Faces,
    edges: set[tuple[int, int]],
    aft: tuple[float, float, float] = AFT,
) -> set[tuple[int, int]]:
    """Return the given edges on the one perimeter per connected piece that reaches furthest aft.

    A perimeter is a loop of the piece's own boundary and sharp edges
    (:data:`._geometry.RIDGE_DEGREES`), so a trailing edge listed in two parts
    (a node left out) is one perimeter, and a hole's loop, or a root apart from
    the trailing edge, is another. A given edge on no such loop forms its own
    chain. Of a piece's perimeters, only the one whose given edges reach
    furthest along ``aft`` is kept, so the trailing-edge nodes of a piece come
    from a single perimeter.
    """
    if not edges:
        return set()
    direction = numpy.asarray(aft, dtype=float)
    piece = _pieces_of_vertices(faces)
    held = edge_faces(faces)
    sharp = dihedral_degrees(verts, faces)
    outline = {e for e, fs in held.items() if len(fs) == 1 or sharp.get(e, 0.0) > RIDGE_DEGREES}
    loop_of: dict[tuple[int, int], int] = {}
    for number, loop in enumerate(_chains(outline)):
        for e in loop:
            loop_of[e] = number
    groups: dict[object, set[tuple[int, int]]] = {}
    for number, chain in enumerate(_chains(edges)):
        for e in chain:
            groups.setdefault(loop_of.get(e, ("chain", number)), set()).add(e)
    best: dict[int, tuple[float, set[tuple[int, int]]]] = {}
    for group in groups.values():
        nodes = sorted({v for e in group for v in e})
        holder = next((piece[v] for v in nodes if v in piece), -1)
        reach = float((verts[nodes] @ direction).max())
        if holder not in best or reach > best[holder][0]:
            best[holder] = (reach, group)
    return set().union(*(group for _, group in best.values()))


def _chains(edges: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Return the edges joined end to end, each chain once."""
    at: dict[int, list[tuple[int, int]]] = {}
    for e in edges:
        for v in e:
            at.setdefault(v, []).append(e)
    out: list[set[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(edges):
        if start in seen:
            continue
        stack, chain = [start], {start}
        while stack:
            for v in stack.pop():
                for e in at[v]:
                    if e not in chain:
                        chain.add(e)
                        stack.append(e)
        seen |= chain
        out.append(chain)
    return out


def _pieces_of_vertices(faces: Faces) -> dict[int, int]:
    """Return each vertex's piece: faces sharing a vertex are one piece."""
    parent: dict[int, int] = {}

    def root(v: int) -> int:
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for face in faces:
        first = root(face[0])
        for v in face[1:]:
            other = root(v)
            if other != first:
                parent[other] = first
    return {v: root(v) for v in parent}


def perimeter_edges_near(
    verts: NDArray[numpy.float64],
    families: dict[str, Faces],
    a: NDArray[numpy.float64],
    b: NDArray[numpy.float64],
) -> dict[str, set[tuple[int, int]]]:
    """Return, per family, the edges whose two ends lie on the polyline of segments a-b.

    An end lies on it within :data:`NODE_EDGE_FRACTION` of the edge's own
    length, so a mesh refined or coarsened from the one the nodes were listed
    on, whose nodes are no longer the listed ones, is read along the same line.
    """
    out: dict[str, set[tuple[int, int]]] = {}
    for name, faces in families.items():
        edges, _ = edge_midpoints(verts, faces)
        if not len(edges) or not len(a):
            out[name] = set()
            continue
        length = numpy.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        da, _ = segment_distances(verts[edges[:, 0]], a, b)
        db, _ = segment_distances(verts[edges[:, 1]], a, b)
        near = numpy.maximum(da, db) < NODE_EDGE_FRACTION * length
        out[name] = {(int(x), int(y)) for x, y in edges[near]}
    return out


def base_component_names(
    side: Path, text: str, data: dict[str, Any], holder: dict[str, str]
) -> str:
    """Rename base family keys and merge targets without reserializing declarations."""
    bases = data.get("base_regions", {})
    if not holder or not isinstance(bases, dict):
        return text
    for name in list(bases):
        entry = bases[name]
        if not isinstance(entry, dict):
            continue
        target = entry.get("merge_into")
        if isinstance(target, str) and holder.get(target, target) != target:
            changed = copy.deepcopy(data)
            changed["base_regions"][name]["merge_into"] = holder[target]
            text = rewrite_value(side, text, "merge_into", target, changed, value=holder[target])
            data = changed
        renamed = holder.get(name, name)
        if renamed == name:
            continue
        if renamed in data["base_regions"]:
            raise InputArtifactError(
                f"{side}: components combine two base declarations as {renamed!r}. "
                "Nothing was written.",
                kind=KIND,
            )
        changed = copy.deepcopy(data)
        changed["base_regions"][renamed] = changed["base_regions"].pop(name)
        text = rename_table_key(side, text, name, renamed, changed)
        data = changed
    return text


def stale_declaration_message(declared: str, folder: Path) -> str:
    """Return the warning of a preserved points file that cannot mark the refined level.

    The level folder holds a points file under its own name only when the level has a listed
    or found trailing edge, so the warning names that file only when it is there.
    """
    level_file = f"{folder.name}{TE_SUFFIX}"
    action = (
        f"review {level_file} and point the declaration's file at the desired one"
        if (folder / level_file).is_file()
        else f"this level holds no {level_file}, so write it with "
        f"cfd-mesh identify-te {folder.name}.obj in the level's folder and point "
        "the declaration's file at it"
    )
    return (
        f"{declared}: the preserved trailing-edge declaration does not match the refined mesh; "
        f"{action} in {folder.name}.obj.boundaries.toml."
    )

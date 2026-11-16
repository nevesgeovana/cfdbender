"""Degenerate geometries derived from a blade mesh: the thin blade.

A thin-blade study replaces a blade by its mean surface, a sheet with no
thickness. This module builds
it from the blade's own surface mesh and writes it beside the source, with
the boundary inventory the raw-mesh route reads, so the command
``cfd-mesh degenerate --kind thin-blade`` has one library
function to call: :func:`derive_thin_blade`.

WHAT IS COMPUTED, in the mesh's own frame and length unit:

1. THE SPAN. The span direction is the principal axis of the blade's
   vertices (the direction of their largest spread), snapped to a global
   axis when it lies within :data:`SNAP_DEGREES` of one, so a blade modelled
   along an axis is cut in its own modelling planes. Its sense goes from the
   root to the tip, the ROOT being the end nearer the origin of the mesh's
   coordinates, where a rotor's blade meets its spinner; a blade whose two
   ends lie equally far from the origin is refused, because which end is the
   root cannot be told.
2. THE STATIONS. The root station is the blade's lowest span coordinate
   moved outward by the required root offset, so the sheet does
   not cross the spinner. A mesh built in rings (its vertices on planes of
   constant span) is cut at its own rings above the root station; any other
   mesh at :data:`SPAN_STATIONS` evenly spaced stations, the last one
   :data:`TIP_MARGIN` of the span below the tip, where a closed tip cap
   leaves no chord.
3. THE SECTIONS. Each station's plane cuts the surface in one closed loop,
   the vertices an export writes once per side (an unwelded trailing edge)
   merged first, so such an edge is joined. Its two ends are the two loop
   points farthest apart (the leading and the trailing edge, in either
   order), and the two halves of the loop between them are the blade's two
   sides. A blunt trailing edge, a short base across the chord with a corner
   at each end, ends the chord at the base's midpoint, each side at its own
   corner.
4. THE MEAN SURFACE. Each side is projected on its chord, resampled at the
   cosine stations ``(1 - cos(pi k / n)) / 2`` of :data:`CHORD_PANELS`
   panels, and the thin blade's point at each station is the mean of the two
   sides there: the surface midway between them, from the root
   to the tip, with the solid section's leading and trailing edge. The root
   row runs from the leading edge, the end where the two sides stand farther
   apart, and every row above it runs the way the row below it runs.

WHAT IS WRITTEN. ``<stem>_thin_blade.obj`` beside the source, one group named
as the source's boundary, in triangles; and its ``<stem>_thin_blade.boundaries.toml``
naming that group, stating the root offset used and, where the source states
it, the ``[import]`` table: the OBJ
sidecar's unit and the mesh operations that act on the blade (on every surface,
or on the blade by its name or a name an earlier rename gave it), because the
thin blade lies in the blade's own frame and the same operations place it; an
operation naming another body is left out and named in a comment. The source
is only read.

WHAT IS REFUSED, by name and with the reason, before anything is written: a file that is not an
OBJ, one that cannot
be read, one holding more than one boundary when ``boundary`` names none, a
boundary the file does not hold or holds twice, an OBJ whose sidecar states a
CAD or CCS conversion, a root offset that is not a positive length shorter
than the blade, a blade whose root cannot be told from its tip, a section
that is not one closed loop (an open surface, such as a sheet already without
thickness, or a plane that cuts another closed body in the same group), and
a section whose two sides cannot be separated. An existing output is refused
unless ``overwrite`` is given.

WHICH BLADE. A file that holds the blade with a spinner, a nacelle or other
bodies is read with ``boundary`` naming the blade's boundary: only
its faces are taken, so the other bodies take no part in the span, the stations
or the sections, and the output's stem carries the boundary's name.

WHAT IT IS NOT. The derived geometry is a modelling choice of the user; this
module makes no claim about the solver's behaviour on it, which a run
confirms. No trailing edge is marked on the sheet: the sidecar's
``[trailing_edges]`` table is the user's to write, as for any raw mesh.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.mesh._demoted import InputTypeError
from cfdbender.mesh._sidecar_models import EVERY_SURFACE, MeshOperation
from cfdbender.mesh._sidecars import inventory_sidecar, read_mesh_import

__all__ = [
    "CHORD_PANELS",
    "THIN_BLADE_KIND",
    "ThinBlade",
    "derive_thin_blade",
    "thin_blade_path",
]

#: The supported kind of degenerate geometry.
THIN_BLADE_KIND = "thin-blade"

#: The suffixes read.
READ_SUFFIXES = (".obj",)

#: What the output's stem adds to the source's.
THIN_BLADE_STEM = "_thin_blade"

#: The chordwise panels of the thin sheet, at cosine stations.
CHORD_PANELS = 40

#: The spanwise stations of a mesh that is not built in rings.
SPAN_STATIONS = 41

#: The fraction of the span below the tip where a mesh not built in rings
#: has its last station, because a closed tip cap leaves no chord at the tip.
TIP_MARGIN = 0.005

#: A principal axis this close to a global axis (degrees) is taken as it.
SNAP_DEGREES = 5.0

#: The fewest vertices a plane of constant span holds to count as a ring.
RING_MIN_VERTICES = 8

#: The share of the vertices that must lie on rings for the mesh to be cut there.
RING_SHARE = 0.9

#: Lengths below this fraction of the span are round-off.
SPAN_TOLERANCE = 1e-9

#: The longest base of a blunt trailing edge, as a fraction of the chord.
BLUNT_FRACTION = 0.05

#: The turn of the loop, in degrees, at each corner of a blunt trailing edge's base.
BASE_TURN_DEGREES = 60.0

#: The share of the span at each end whose vertices say how far that end lies
#: from the origin, for telling the root from the tip.
END_BAND = 0.05

_KIND = "geometry"

Points = NDArray[numpy.float64]
Faces = NDArray[numpy.int64]


@dataclass(frozen=True)
class ThinBlade:
    """What :func:`derive_thin_blade` wrote.

    Attributes
    ----------
    mesh : Path
        The thin blade, an OBJ beside the source.
    sidecar : Path
        Its boundary inventory, ``<stem>.boundaries.toml`` beside it.
    boundary : str
        The one boundary (OBJ group) of the thin blade, named as the source's.
    root_offset : float
        The offset the root was moved outward by, in the mesh's length unit.
    span_direction : tuple of float
        The unit vector from the blade's root to its tip.
    root_span, tip_span : float
        The span coordinates (along ``span_direction``, from the origin) of
        the first and the last station of the thin blade.
    stations : int
        The spanwise sections the sheet passes through.
    unit : str or None
        The length unit written in the sidecar's ``[import]`` table, None
        when the source does not state one.
    """

    mesh: Path
    sidecar: Path
    boundary: str
    root_offset: float
    span_direction: tuple[float, float, float]
    root_span: float
    tip_span: float
    stations: int
    unit: str | None


def thin_blade_path(geometry: str | Path, boundary: str | None = None) -> Path:
    """Return the path the thin blade of ``geometry`` is written at.

    Parameters
    ----------
    geometry : str or Path
        The source blade mesh.
    boundary : str, optional
        The boundary the blade was selected by; its name, with
        every character that is not a letter, a digit or a hyphen turned
        into an underscore, joins the stem so two blades of one file do
        not share an output.

    Returns
    -------
    Path
        ``<stem>_thin_blade.obj`` beside it, or ``<stem>_<boundary>_thin_blade.obj``
        with a boundary, whether or not it exists.
    """
    path = Path(geometry)
    part = "_" + re.sub(r"[^0-9A-Za-z-]", "_", boundary) if boundary else ""
    return path.with_name(path.stem + part + THIN_BLADE_STEM + ".obj")


def _refuse(source: Path, reason: str) -> InputArtifactError:
    """Return the refusal of ``source``: its name, the reason, and that nothing was written."""
    return InputArtifactError(
        f"{source}: no thin blade is derived from it: {reason}. Nothing was written.",
        kind=_KIND,
    )


# --- reading the blade -------------------------------------------------------------


@dataclass(frozen=True)
class _Blade:
    vertices: Points
    faces: Faces
    boundary: str
    unit: str | None
    #: The source sidecar's mesh operations, carried to the thin blade's.
    operations: tuple[MeshOperation, ...] = ()


def _not_one_blade(source: Path, kind: str, names: Sequence[str]) -> InputArtifactError:
    """Return the refusal of a file holding several boundaries or groups, naming the way out."""
    return _refuse(
        source,
        f"it holds {len(names)} {kind} ({', '.join(names)}), and a thin blade is derived "
        "from a file holding one blade alone, or from the one boundary named with --boundary",
    )


def _no_such_boundary(source: Path, boundary: str, names: Sequence[str]) -> InputArtifactError:
    """Return the refusal of a ``boundary`` the file does not hold, or holds under two names."""
    if names.count(boundary) > 1:
        return _refuse(
            source,
            f"--boundary {boundary!r} names {names.count(boundary)} of its boundaries, so it "
            "selects none; rename one in the geometry",
        )
    return _refuse(
        source,
        f"it holds no boundary named {boundary!r} (--boundary); it holds "
        f"{', '.join(names) or 'none'}",
    )


def _sub_mesh(vertices: Points, faces: Faces) -> tuple[Points, Faces]:
    """Return the faces with only the vertices they use, renumbered in their own order."""
    used, inverse = numpy.unique(faces.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(numpy.int64)


def _check_boundary_name(source: Path, boundary: str, names: Sequence[str]) -> None:
    """Refuse an empty ``boundary`` and one whose output name another boundary would share."""
    if boundary == "":
        raise _refuse(
            source,
            "boundary (CLI: --boundary) is empty; it names the one boundary that is the blade",
        )
    mine = thin_blade_path(source, boundary)
    for other in names:
        if other != boundary and thin_blade_path(source, other) == mine:
            raise _refuse(
                source,
                f"boundary (CLI: --boundary) {boundary!r} would write the same output as its "
                f"boundary {other!r}, a character of a name that is not a letter, a digit or a "
                "hyphen being written as an underscore; rename one of the two boundaries",
            )


def _obj_face(words: Sequence[str], count: int, where: str) -> list[int]:
    """Return the 0-based vertex indices of one OBJ face statement."""
    indices = []
    for word in words:
        index = int(word.split("/", 1)[0])
        indices.append(index - 1 if index > 0 else count + index)
    if len(indices) < 3 or not all(0 <= i < count for i in indices):
        raise InputValueError(
            f"{where} names a vertex outside the {count} read before it; correct the vertex index"
        )
    return indices


def _obj_statements(text: str) -> tuple[list[list[float]], dict[str, list[list[int]]]]:
    """Return the vertices and the triangles by group of an OBJ's text."""
    vertices: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = ""
    for number, line in enumerate(text.splitlines(), start=1):
        statement = line.strip().split(maxsplit=1)
        if len(statement) == 2 and statement[0] == "o":
            current = statement[1]
            continue
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        if words[0] == "v":
            if len(words) < 4:
                raise InputValueError(
                    f"line {number} states a vertex of {len(words) - 1} coordinates, not three;"
                    f" write x y z"
                )
            vertices.append([float(word) for word in words[1:4]])
        elif words[0] == "g" and len(words) > 1:
            current = words[1]
        elif words[0] == "f":
            face = _obj_face(words[1:], len(vertices), f"line {number}")
            fan = [[face[0], face[k], face[k + 1]] for k in range(1, len(face) - 1)]
            groups.setdefault(current, []).extend(fan)
    return vertices, groups


def _read_obj(source: Path, boundary: str | None) -> _Blade:
    """Read an OBJ holding one group of faces, or the one named, refusing by name."""
    try:
        vertices, groups = _obj_statements(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as error:
        raise _refuse(source, f"it cannot be read as an OBJ ({error})") from error
    names = [name or "(no group)" for name in groups]
    if boundary is None and len(groups) > 1:
        raise _not_one_blade(source, "groups of faces", names)
    if boundary is not None:
        _check_boundary_name(source, boundary, names)
    if boundary is not None and boundary not in groups:
        raise _no_such_boundary(source, boundary, names)
    sidecar = inventory_sidecar(source)
    stated = read_mesh_import(sidecar) if sidecar.is_file() else None
    if stated is not None and (stated.cad is not None or stated.ccs is not None):
        table = "[import.cad]" if stated.cad is not None else "[import.ccs]"
        raise _refuse(
            source,
            f"its sidecar {sidecar.name} states {table}, a conversion of a CAD or CCS "
            "file, which a mesh read as an OBJ does not take",
        )
    name, faces = (boundary, groups[boundary]) if boundary else next(iter(groups.items()), ("", []))
    points = numpy.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = numpy.asarray(faces, dtype=numpy.int64).reshape(-1, 3)
    if len(triangles):
        # The implicit single group is cut to its own vertices as a named one is, so the
        # axis and the station range never see a vertex no face uses.
        points, triangles = _sub_mesh(points, triangles)
    return _Blade(
        points,
        triangles,
        name or source.stem,
        stated.units if stated is not None else None,
        stated.operations if stated is not None else (),
    )


def _welded(blade: _Blade) -> _Blade:
    """Return the blade with coincident vertices merged, in their first order.

    An export that writes a vertex once per side (an unwelded trailing edge)
    leaves the surface open along that edge; merging the copies closes it.
    The faces a merge collapses to a line are dropped. A mesh with no
    coincident vertices is returned with its vertices and faces unchanged.
    """
    _, first, inverse = numpy.unique(blade.vertices, axis=0, return_index=True, return_inverse=True)
    if len(first) == len(blade.vertices):
        return blade
    order = numpy.argsort(first)
    rank = numpy.empty_like(order)
    rank[order] = numpy.arange(len(order))
    faces = rank[inverse.reshape(-1)][blade.faces]
    kept = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    )
    return _Blade(
        blade.vertices[first[order]],
        faces[kept],
        blade.boundary,
        blade.unit,
        blade.operations,
    )


def _read_blade(source: Path, boundary: str | None = None) -> _Blade:
    """Read the blade mesh ``source`` names, refusing what cannot be read."""
    if not source.is_file():
        raise _refuse(source, "it is not a file")
    if source.suffix.lower() not in READ_SUFFIXES:
        raise _refuse(
            source,
            f"a blade mesh is read from an OBJ (.obj), "
            f"and this file is a {source.suffix or 'file without a suffix'}",
        )
    blade = _read_obj(source, boundary)
    if len(blade.faces) < 4 or len(blade.vertices) < 4:
        raise _refuse(source, "it holds no surface mesh of a blade (fewer than four faces)")
    if not bool(numpy.isfinite(blade.vertices).all()):
        raise _refuse(source, "a vertex coordinate is not a finite number")
    return _welded(blade)


# --- the span ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    """The span direction and two in-plane directions, orthonormal."""

    span: Points
    across: Points
    normal: Points

    def local(self, points: Points) -> Points:
        """Return (across, normal, span) coordinates of ``points``."""
        return numpy.stack([points @ self.across, points @ self.normal, points @ self.span], -1)

    def glob(self, local: Points) -> Points:
        """Return the global points of (across, normal, span) coordinates."""
        return (
            local[..., 0:1] * self.across
            + local[..., 1:2] * self.normal
            + local[..., 2:3] * self.span
        )


def _principal_axis(vertices: Points) -> Points:
    """Return the unit direction of the vertices' largest spread, snapped to a near axis."""
    centred = vertices - vertices.mean(axis=0)
    _, vectors = numpy.linalg.eigh(centred.T @ centred)
    axis = vectors[:, -1]
    for k in range(3):
        if abs(axis[k]) >= math.cos(math.radians(SNAP_DEGREES)):
            snapped = numpy.zeros(3)
            snapped[k] = math.copysign(1.0, axis[k])
            return snapped
    return axis / numpy.linalg.norm(axis)


def _span_frame(source: Path, vertices: Points) -> _Frame:
    """Return the blade's frame, its span from the root (the end nearer the origin) to the tip."""
    span = _principal_axis(vertices)
    s = vertices @ span
    low, high = float(s.min()), float(s.max())
    length = high - low
    if length <= 0.0:
        raise _refuse(source, "its vertices have no extent, so it has no span")
    band = END_BAND * length
    near_low = float(numpy.linalg.norm(vertices[s <= low + band], axis=1).mean())
    near_high = float(numpy.linalg.norm(vertices[s >= high - band], axis=1).mean())
    if abs(near_low - near_high) <= SPAN_TOLERANCE * length:
        raise _refuse(
            source,
            "its two ends lie equally far from the origin of its coordinates, so its root "
            "(the end nearer the origin, where a blade meets its spinner) cannot be told "
            "from its tip",
        )
    if near_low > near_high:
        span = -span
    seed = numpy.zeros(3)
    seed[int(numpy.argmin(numpy.abs(span)))] = 1.0
    across = seed - (seed @ span) * span
    across /= numpy.linalg.norm(across)
    return _Frame(span, across, numpy.cross(span, across))


# --- the stations ------------------------------------------------------------------


def _levels(s: Points, tolerance: float) -> list[tuple[float, int]]:
    """Return the distinct span levels of the vertices, each with its vertex count."""
    order = numpy.sort(s)
    breaks = numpy.nonzero(numpy.diff(order) > tolerance)[0] + 1
    return [(float(group.mean()), len(group)) for group in numpy.split(order, breaks)]


def _stations(source: Path, s: Points, root_offset: float) -> Points:
    """Return the span coordinates of the sections, the root moved by ``root_offset``."""
    low, high = float(s.min()), float(s.max())
    tolerance = SPAN_TOLERANCE * (high - low)
    full = [(level, count) for level, count in _levels(s, tolerance) if count >= RING_MIN_VERTICES]
    rings = [level for level, _ in full]
    on_rings = sum(count for _, count in full)
    structured = len(rings) >= 3 and on_rings >= RING_SHARE * len(s)
    top = max(rings) if structured else high - TIP_MARGIN * (high - low)
    root = low + root_offset
    if root >= top - tolerance:
        raise _refuse(
            source,
            f"the root offset {root_offset!r} reaches the blade's last section, "
            f"{top - low!r} from its root; the offset is a length shorter than the blade",
        )
    if structured:
        inner = [level for level in rings if level > root + 1e3 * tolerance]
        if not inner:
            raise _refuse(
                source,
                f"the root offset {root_offset!r} leaves no section beyond the root: every "
                f"section lies within {root - low!r} of the blade's root; the offset is a "
                "length shorter than the blade",
            )
        return numpy.asarray([root, *inner])
    return numpy.linspace(root, top, SPAN_STATIONS)


# --- one section -------------------------------------------------------------------


def _crossings(
    faces: Faces, above: NDArray[numpy.bool_]
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Return the edges the plane crosses, each with the crossed edges it is joined to."""
    joined: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cut = faces[above[faces].any(axis=1) & ~above[faces].all(axis=1)]
    for a, b, c in cut.tolist():
        keys = [(min(i, j), max(i, j)) for i, j in ((a, b), (b, c), (c, a)) if above[i] != above[j]]
        joined.setdefault(keys[0], []).append(keys[1])
        joined.setdefault(keys[1], []).append(keys[0])
    return joined


def _loops(
    joined: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[list[tuple[int, int]]] | None:
    """Return the closed chains of crossed edges, None when one is open or branches."""
    if any(len(nexts) != 2 for nexts in joined.values()):
        return None
    seen: set[tuple[int, int]] = set()
    loops = []
    for start in joined:
        if start in seen:
            continue
        loop, previous, current = [start], None, start
        seen.add(start)
        while True:
            following = [n for n in joined[current] if n != previous] or joined[current][:1]
            if following[0] == start or following[0] in seen:
                break
            previous, current = current, following[0]
            loop.append(current)
            seen.add(current)
        loops.append(loop)
    return loops


def _section(local: Points, faces: Faces, station: float, tolerance: float) -> list[Points] | None:
    """Return every closed loop the plane of ``station`` cuts; None if one is open or none cut."""
    d = local[:, 2] - station
    d[numpy.abs(d) <= tolerance] = 0.0
    loops = _loops(_crossings(faces, d >= 0.0))
    if not loops:
        return None
    sections = []
    for loop in loops:
        i = numpy.asarray([edge[0] for edge in loop])
        j = numpy.asarray([edge[1] for edge in loop])
        t = d[i] / (d[i] - d[j])  # a crossed edge has one end on each side, so never 0 / 0
        points = local[i, :2] + t[:, None] * (local[j, :2] - local[i, :2])
        keep = numpy.linalg.norm(points - numpy.roll(points, 1, axis=0), axis=1) > tolerance
        sections.append(points[keep] if keep.any() else points[:1])
    return sections


def _ends(points: Points) -> tuple[int, int]:
    """Return the indices of the two loop points farthest apart (the chord's ends)."""
    best, a, b = -1.0, 0, 0
    batch = max(1, min(512, 131072 // max(1, len(points))))
    for start in range(0, len(points), batch):
        block = points[start : start + batch]
        distance = numpy.linalg.norm(block[:, None, :] - points[None, :, :], axis=2)
        row, column = numpy.unravel_index(int(numpy.argmax(distance)), distance.shape)
        if float(distance[row, column]) > best:
            best, a, b = float(distance[row, column]), start + int(row), int(column)
    # The end farther from the first point comes first, as the chord has always been oriented.
    if numpy.linalg.norm(points[b] - points[0]) > numpy.linalg.norm(points[a] - points[0]):
        a, b = b, a
    return a, b


def _resampled(side: Points, start: Points, chord: Points, stations: Points) -> Points:
    """Return one side resampled at the chord fractions ``stations``.

    The fractions are clipped to [0, 1] before they are made non-decreasing. This prevents a point
    beyond the chord from producing
    accumulated fractions above the forced last 1.0, outside numpy.interp's
    defined ordering.
    """
    xi = numpy.maximum.accumulate(
        numpy.clip((side - start) @ chord / float(chord @ chord), 0.0, 1.0)
    )
    xi[0], xi[-1] = 0.0, 1.0
    return numpy.stack(
        [numpy.interp(stations, xi, side[:, 0]), numpy.interp(stations, xi, side[:, 1])], 1
    )


def _turn(incoming: Points, outgoing: Points) -> float:
    """Return the angle, in degrees, a loop turns through from one step to the next."""
    sizes = float(numpy.linalg.norm(incoming) * numpy.linalg.norm(outgoing))
    if sizes == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(incoming @ outgoing) / sizes))))


def _base_corner(points: Points, end: int, other: int) -> tuple[int, int] | None:
    """Return the far corner of a blunt edge at ``end`` and the loop direction to it, or None.

    A BLUNT TRAILING EDGE is a short base across the chord whose
    two corners each turn the loop through at least :data:`BASE_TURN_DEGREES`:
    the side runs into one corner, the base crosses the chord, and the other
    side leaves the far corner back along the chord. A sharp edge's steps run
    along the chord, and a rounded leading edge turns a little at each point,
    so neither is read as a base.
    """
    n = len(points)
    axis = points[end] - points[other]
    length = float(numpy.linalg.norm(axis))
    unit = axis / length
    for step in (1, -1):
        incoming = points[end] - points[(end - step) % n]
        corner, crossing = end, points[(end + step) % n] - points[end]
        if _turn(incoming, crossing) < BASE_TURN_DEGREES:
            continue
        while True:
            following = (corner + step) % n
            across = points[following] - points[corner]
            size = float(numpy.linalg.norm(across))
            if (
                following == other
                or size == 0.0
                or float(numpy.linalg.norm(points[following] - points[end]))
                > BLUNT_FRACTION * length
                or abs(float(across @ unit)) > 0.5 * size
                or (corner != end and _turn(crossing, across) >= BASE_TURN_DEGREES)
            ):
                break
            corner, crossing = following, across
        if corner == end:
            continue
        leaving = points[(corner + step) % n] - points[corner]
        if _turn(crossing, leaving) >= BASE_TURN_DEGREES:
            return corner, step
    return None


def _mean_line(points: Points, panels: int) -> Points | None:
    """Return the mean line of one closed section, None when its two sides cannot be told.

    It runs from the leading edge, the end where the two sides stand farther
    apart, and a blunt trailing edge ends it at the midpoint of
    its base, each side stopping at its own corner.
    """
    n = len(points)
    if n < 6:
        return None
    a, b = _ends(points)
    base = _base_corner(points, b, a)
    if base is None and (base := _base_corner(points, a, b)) is not None:
        a, b = b, a
    far_forward = far_backward = b
    end = points[b]
    if base is not None:
        corner, step = base
        if step == 1:
            far_backward = corner
        else:
            far_forward = corner
        end = 0.5 * (points[b] + points[corner])
    forward = points[[(a + k) % n for k in range((far_forward - a) % n + 1)]]
    backward = points[[(a - k) % n for k in range((a - far_backward) % n + 1)]]
    if len(forward) < 3 or len(backward) < 3:
        return None
    stations = 0.5 * (1.0 - numpy.cos(numpy.pi * numpy.arange(panels + 1) / panels))
    chord = end - points[a]
    first = _resampled(forward, points[a], chord, stations)
    second = _resampled(backward, points[a], chord, stations)
    line = 0.5 * (first + second)
    thickness = numpy.linalg.norm(first - second, axis=1)
    if thickness[stations >= 0.75].mean() > thickness[stations <= 0.25].mean():
        line = line[::-1]
    return line


# --- the sheet ---------------------------------------------------------------------


def _mean_surface(source: Path, blade: _Blade, frame: _Frame, stations: Points) -> Points:
    """Return the thin blade's grid, (stations, CHORD_PANELS + 1, 3), refusing a section."""
    local = frame.local(blade.vertices)
    tolerance = SPAN_TOLERANCE * float(numpy.ptp(local[:, 2]))
    rows: list[Points] = []
    for station in stations:
        where = f"the section at span {float(station)!r} (along {frame.span.round(6).tolist()})"
        loops = _section(local, blade.faces, float(station), tolerance)
        if loops is None:
            raise _refuse(
                source,
                f"{where} is not one closed loop: the surface is open there (a sheet already "
                "without thickness, or two sides not joined along an edge), so its two "
                "sides cannot be separated",
            )
        if len(loops) > 1:
            raise _refuse(
                source,
                f"{where} is not one closed loop: its plane cuts {len(loops)} closed loops, "
                "so the group holds another closed body beside the blade and the blade's "
                "two sides cannot be separated from it",
            )
        loop = loops[0]
        line = _mean_line(loop, CHORD_PANELS)
        if line is None:
            raise _refuse(
                source,
                f"{where} holds {len(loop)} points, too few for its two sides to be separated",
            )
        # EVERY ROW RUNS THE WAY THE ONE BELOW IT RUNS: a row from the
        # other edge joined each leading edge to a trailing edge and folded the sheet.
        # Compare chord directions, independent of the section's spanwise sweep.
        if rows and numpy.dot(line[-1] - line[0], rows[-1][-1, :2] - rows[-1][0, :2]) < 0:
            line = line[::-1]
        rows.append(numpy.column_stack([line, numpy.full(len(line), station)]))
    return frame.glob(numpy.asarray(rows))


def _triangles(stations: int, panels: int) -> Iterable[tuple[int, int, int]]:
    """Yield the sheet's triangles as 1-based OBJ indices, two per grid quad."""
    for k in range(stations - 1):
        for j in range(panels):
            a, b = k * (panels + 1) + j + 1, k * (panels + 1) + j + 2
            c, d = b + panels + 1, a + panels + 1
            yield a, b, c
            yield a, c, d


def _obj_text(source: Path, grid: Points, boundary: str, root_offset: float) -> str:
    """Return the thin blade's OBJ text, LF line ends."""
    stations, columns = grid.shape[0], grid.shape[1]
    lines = [
        f"# Thin blade of {source.name}: the mean surface of its blade, the root moved",
        f"# outward along the span by {root_offset!r} (cfdbender).",
        f"o {boundary}",
        # repr is the shortest text that reads back as the same double, so distinct stations
        # stay distinct at a large coordinate origin.
        *(f"v {x!r} {y!r} {z!r}" for x, y, z in grid.reshape(-1, 3).tolist()),
        *(f"f {a} {b} {c}" for a, b, c in _triangles(stations, columns - 1)),
    ]
    return "\n".join(lines) + "\n"


def _sidecar_text(source: Path, mesh: Path, blade: _Blade, root_offset: float) -> str:
    """Return the thin blade's boundary inventory, the sidecar the raw-mesh route reads."""
    lines = [
        f"# Boundary inventory of {mesh.name}, the thin blade cfdbender derived from",
        f"# {source.name} (sha256 {file_sha256(source)}) by `pyfs-matrix degenerate`.",
        f"# root_offset = {root_offset!r}, in the mesh's own length unit: the root of the",
        "# thin blade is moved outward along the span by it, off the spinner.",
        "# Add the [trailing_edges] table beneath, as for any raw mesh (docs/mesh-inputs.md).",
        f"boundaries = [{json.dumps(blade.boundary, ensure_ascii=False)}]",
    ]
    if blade.unit is None:
        lines.append(
            f"# {source.name} states no length unit; write [import] units beneath this list."
        )
        return "\n".join(lines) + "\n"
    kept, dropped = _blade_operations(blade)
    lines += [
        "",
        f"# The [import] table is {source.name}'s: the thin blade lies in the blade's own",
        "# frame, so the same unit and mesh operations place it as they place the blade.",
        *(
            f"# Left out: operation {number} ({operation.op} of {operation.surface!r}), "
            "which names a surface the thin blade does not hold."
            for number, operation in dropped
        ),
        "[import]",
        f"units = {json.dumps(blade.unit, ensure_ascii=False)}",
    ]
    for operation in kept:
        lines += ["", "[[import.operations]]"]
        for key, value in operation.model_dump(exclude_none=True).items():
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _blade_operations(
    blade: _Blade,
) -> tuple[list[MeshOperation], list[tuple[int, MeshOperation]]]:
    """Split the source's mesh operations into those the thin blade takes and those it does not.

    A blade cut out of a file holding other
    bodies carried every operation of the source, and one naming a surface the
    thin blade lacks (a mirror of the spinner, say) refused the thin blade's
    own plan. An operation is kept when it acts on every surface or on the
    blade, by its own name or by the name an earlier kept rename gave it.

    Returns
    -------
    tuple
        The kept operations in order, and the left-out ones with their
        1-based position in the source's list.
    """
    name = blade.boundary
    kept: list[MeshOperation] = []
    dropped: list[tuple[int, MeshOperation]] = []
    for number, operation in enumerate(blade.operations, start=1):
        if operation.surface not in (EVERY_SURFACE, name):
            dropped.append((number, operation))
            continue
        kept.append(operation)
        if operation.op == "rename" and operation.to is not None:
            name = operation.to
    return kept, dropped


def _toml_value(value: object) -> str:
    """Return a mesh operation's value as TOML: a string, a number or a list of numbers."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(float(value))
    raise InputTypeError(f"a mesh operation holds {value!r}, which the sidecar does not write")


def _checked_offset(source: Path, root_offset: float) -> float:
    """Return the root offset as a float, refusing one that is not a positive length."""
    if isinstance(root_offset, (bool, str, bytes)):
        raise _refuse(source, f"the root offset {root_offset!r} is not a number")
    try:
        offset = float(root_offset)
    except (TypeError, ValueError, OverflowError) as error:
        raise _refuse(source, f"the root offset {root_offset!r} is not a number") from error
    if not math.isfinite(offset) or offset <= 0.0:
        raise _refuse(
            source,
            f"the root offset {root_offset!r} is not a positive length; the root is moved "
            "outward by it so the thin blade does not cross the spinner",
        )
    return offset


def derive_thin_blade(
    geometry: str | Path,
    *,
    root_offset: float,
    overwrite: bool = False,
    boundary: str | None = None,
    out_dir: str | Path | None = None,
) -> ThinBlade:
    """Derive the thin blade of a blade mesh and write it beside the source.

    Parameters
    ----------
    geometry : str or Path
        The blade mesh: an OBJ holding one
        blade alone.
    root_offset : float
        How far the root is moved outward along the span, in the mesh's own
        length unit, so the thin blade does not cross the spinner. Required,
        positive and shorter than the blade.
    overwrite : bool
        Rewrite an existing thin blade and its sidecar. Without it an
        existing one is refused.
    boundary : str, optional
        The name of the one boundary (OBJ group) that is the blade, for a
        file that holds the blade with a spinner, a nacelle or other bodies. Only its faces are
        read, and the output is written
        under the name :func:`thin_blade_path` gives for it. Without it a
        file holding more than one boundary is refused.
    out_dir : str or Path, optional
        Folder for the derived OBJ and inventory, created when needed. Default: beside
        the source.

    Returns
    -------
    ThinBlade
        The paths written and the frame and stations used.

    Raises
    ------
    InputArtifactError
        Naming the source and the reason, with nothing written: a file that
        is not an OBJ or cannot be read, one holding
        more than one boundary, an OBJ whose sidecar states a CAD or CCS
        conversion, a root offset that is not a positive length shorter
        than the blade, a blade whose root cannot be told from its tip, a
        section that is not one closed loop (open, or one of several) or
        whose two sides cannot be separated, and an existing output without
        ``overwrite``.
    """
    source = Path(geometry)
    mesh = thin_blade_path(source, boundary)
    if out_dir is not None:
        mesh = Path(out_dir) / mesh.name
    sidecar = inventory_sidecar(mesh)
    for existing in (mesh, sidecar):
        if existing.exists() and not overwrite:
            raise _refuse(source, f"{existing} already exists; pass overwrite to rewrite it")
    offset = _checked_offset(source, root_offset)
    blade = _read_blade(source, boundary)
    frame = _span_frame(source, blade.vertices)
    stations = _stations(source, blade.vertices @ frame.span, offset)
    grid = _mean_surface(source, blade, frame, stations)
    mesh_text = _obj_text(source, grid, blade.boundary, offset)
    sidecar_text = _sidecar_text(source, mesh, blade, offset)
    mesh.parent.mkdir(parents=True, exist_ok=True)
    _textio.write_text(mesh, mesh_text)
    _textio.write_text(sidecar, sidecar_text)
    return ThinBlade(
        mesh=mesh,
        sidecar=sidecar,
        boundary=blade.boundary,
        root_offset=offset,
        span_direction=(
            float(frame.span[0]) + 0.0,
            float(frame.span[1]) + 0.0,
            float(frame.span[2]) + 0.0,
        ),
        root_span=float(stations[0]),
        tip_span=float(stations[-1]),
        stations=len(stations),
        unit=blade.unit,
    )

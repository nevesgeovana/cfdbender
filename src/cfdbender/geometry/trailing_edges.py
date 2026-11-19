"""Trailing-edge extraction from a blade surface mesh.

The rotor axis and hub define the spanwise and tangential directions used to identify the aft
end of a section. Returned points are selected mesh vertices and mesh-edge midpoints,
expressed in the mesh's own reference frame and length units.

It turns a surface mesh the user already has into the trailing-edge points file a
geometry names: the mid-point of every mesh edge on the trailing edge, under a unit line
(:func:`cfdbender._points.write_trailing_edge_points`).

MID-POINTS, BECAUSE A POINT OF THE FILE IS MATCHED TO AN EDGE BY ITS MID-POINT. A list of the
edges' end vertices matches only the edges shorter than twice the matching tolerance (an end
vertex lies half an edge length from the mid-point), which on a mesh of ordinary edges is
none. So :func:`trailing_edge_midpoints` finds the EDGES: the sections below give one aftmost
vertex each, consecutive ones are joined by the shortest path along mesh edges that are the
aft ridge of both their faces, and each edge of that path contributes its mid-point.

Extraction uses the section geometry rather than a dihedral-angle criterion. A strongly
twisted blade can have a trailing edge that is not a crease.

WHAT IS COMPUTED INSTEAD. Given a surface mesh, a rotor axis and a hub
position:

1. the SPANWISE coordinate of a vertex is its distance from the rotor
   axis, measured radially from the hub and perpendicular to the axis;
2. the vertices are binned into chordwise SECTIONS by that coordinate;
3. within each section the CHORDWISE direction is taken from the
   section's own extent, never from a global axis, which is what makes
   the answer survive twist;
4. the trailing-edge node of the section is the AFTMOST vertex along
   that direction, and the node list is those vertices ordered by span;
5. for the mid-points, an interior mesh edge is a trailing-edge
   CANDIDATE when the far vertex of each of its two faces lies forward of
   it, forward meaning against the aft chord of its section with the
   edge's own direction removed. That is a question about which side of
   the edge the surface is on, in the rotor frame, and no angle is
   compared with a threshold;
6. consecutive sections' aftmost vertices are joined by the shortest
   path through candidate edges, the chain is extended past the first
   and last section along the one outward candidate edge while there is
   exactly one, and the chain's edges give the mid-points.

It needs vertices, faces and a section plane. It needs no adjacency
angle, no watertightness test and no proximity query, so it runs on a
base install with none of this package's optional extras
(````).

WHICH END OF THE CHORD IS AFT is decided by the rotor frame and not
guessed. A section's extent gives a LINE, which has two ends and no
sign; the sense comes from the tangential direction
``axis x radial``, and aft is the end AGAINST it. So the ``axis`` vector
carries the rotation sense by the right-hand rule, and a rotor turning
the other way is described by passing the axis the other way round. The
sign is resolved once, on the section whose chord is most nearly
tangential (the least twisted one, ordinarily near the tip), and carried
to the other sections by continuity, so a strongly twisted root section
whose chord is nearly axial never has to decide the question on its own.

IT REFUSES RATHER THAN GUESSES. With no axis and hub there is no
spanwise direction and no chord direction, so the extraction refuses
naming both. Guessing a global axis is exactly how a twisted blade gets
the wrong edge silently, which is the failure this capability exists to
remove.
"""

from __future__ import annotations

import heapq
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from cfdbender._errors import InputArtifactError
from cfdbender._points import write_trailing_edge_points
from cfdbender.mesh.objio import read_mesh

__all__ = [
    "DEFAULT_SECTIONS",
    "MINIMUM_SECTION_VERTICES",
    "TrailingEdge",
    "extract_trailing_edge",
    "trailing_edge_midpoints",
    "write_trailing_edge_node_file",
]

#: Chordwise sections cut along the span by default. A blade's trailing
#: edge is a curve and this is how finely it is sampled, so the number is
#: a modeling choice rather than a tuning constant: twenty nodes
#: describe the edge of an ordinary rotor blade without asking the mesh
#: for more resolution than a coarse one has. Raise it for a highly
#: swept edge, lower it for a coarse mesh.
DEFAULT_SECTIONS = 20

#: Vertices a section must hold for its own extent to define a chord
#: direction. Two points define a line but not a section, and one
#: defines nothing; three is the smallest number that can disagree,
#: which is what makes the direction a measurement rather than a
#: restatement of the input.
MINIMUM_SECTION_VERTICES = 3

#: How nearly tangential the best-aligned section's chord must be for the
#: aft sense to be readable at all, as the cosine between the chord
#: direction and the tangential direction. Well below any real blade: a
#: chord within 89.94 degrees of the tangential direction clears it. It
#: exists to catch the case where the mesh, the axis and the hub
#: together describe something that is not a rotor blade, not to police
#: twist.
_MINIMUM_TANGENTIAL_ALIGNMENT = 1.0e-3

#: The kind every refusal of this module carries, so a caller can tell
#: an extraction refusal from an input-library one without parsing text.
_KIND = "trailing_edges"


@dataclass(frozen=True)
class TrailingEdge:
    """One extracted trailing edge, and what it was extracted with.

    Coordinates are in the MESH's own length units throughout. This
    class never learns what that unit is, because the mesh file does not
    carry one; the unit is declared when the points file is written
    (:func:`write_trailing_edge_node_file`).

    Attributes
    ----------
    nodes : numpy.ndarray
        The trailing-edge node coordinates, shape (n, 3), in the mesh's
        reference frame and length units, ordered by increasing span.
        One node per section.
    span : numpy.ndarray
        Spanwise coordinate of each node, shape (n,), in mesh length
        units: the distance from the rotor axis, measured perpendicular
        to it. Increasing, by construction.
    axis : numpy.ndarray
        The rotor axis the extraction used, shape (3,), normalized to
        unit length. Its SENSE carries the rotation sense by the
        right-hand rule; see the module docstring.
    hub : numpy.ndarray
        The hub position the extraction used, shape (3,), in mesh length
        units and the mesh's reference frame.
    sections : int
        Number of chordwise sections the span was cut into, which is
        also ``len(nodes)``.
    """

    nodes: numpy.ndarray
    span: numpy.ndarray
    axis: numpy.ndarray
    hub: numpy.ndarray
    sections: int


def _three_vector(value: Any, name: str) -> numpy.ndarray:
    """Read one three-component vector, or refuse naming what it is for."""
    try:
        array = numpy.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise InputArtifactError(
            f"the {name} could not be read as three numbers: {error}. It is a "
            "sequence of three coordinates in the mesh's own reference frame, for "
            "example (0.0, 0.0, 1.0)",
            kind=_KIND,
        ) from error
    if array.shape != (3,):
        raise InputArtifactError(
            f"the {name} has {array.size} components and a direction in space has "
            "three. It is given in the mesh's own reference frame, in the order x, "
            "y, z",
            kind=_KIND,
        )
    if not bool(numpy.isfinite(array).all()):
        raise InputArtifactError(
            f"the {name} carries a component that is not a finite number, so it "
            "names no direction and no position in the mesh frame",
            kind=_KIND,
        )
    return array


def _unit_axis(value: Any) -> numpy.ndarray:
    """Read the rotor axis and normalize it, or refuse."""
    array = _three_vector(value, "rotor axis")
    # Scaled by the largest component first, so a finite axis of extreme magnitude
    # neither overflows nor underflows its own length (F-348).
    peak = float(numpy.abs(array).max())
    scaled = array / peak if peak > 0.0 else array
    length = float(numpy.linalg.norm(scaled))
    if length == 0.0:
        raise InputArtifactError(
            "the rotor axis is the zero vector, which names no direction: with no "
            "axis there is no spanwise direction to section along and no tangential "
            "direction to tell the trailing edge from the leading one. Give the axis "
            "of rotation, pointing so that the rotor turns the right-handed way "
            "about it",
            kind=_KIND,
        )
    return scaled / length


def _loaded(mesh: Any) -> Any:
    """Return the mesh itself, reading it first when it is a file path."""
    if isinstance(mesh, (str, Path)):
        try:
            mesh = read_mesh(mesh)
        except (OSError, ValueError, NotImplementedError) as error:
            # The reader's own refusals are a bare ValueError for a path
            # that is not a file and a bare NotImplementedError for a
            # suffix it has no loader for, both raised out of a public
            # name of THIS package, which does not allow. Named
            # rather than re-worded, and chained, because the reader's
            # sentence is the specific one.
            raise InputArtifactError(
                f"the mesh file {mesh} could not be read: {error}. It is a surface "
                "mesh the reader understands from its suffix, ordinarily .obj or "
                ".stl; a path that does not exist and a suffix with no loader are "
                "the two that usually arrive here",
                kind=_KIND,
            ) from error
    return mesh


def _mesh_vertices(mesh: Any) -> numpy.ndarray:
    """Return the (n, 3) vertex array of a LOADED mesh, or refuse."""
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        raise InputArtifactError(
            f"the mesh is a {type(mesh).__name__} and carries no vertices, so there "
            "is nothing to section. Pass a path to a surface mesh file (.obj or "
            ".stl), or a loaded mesh. A file holding several separate bodies loads "
            "as a scene rather than as one mesh, and a blade is extracted one body "
            "at a time",
            kind=_KIND,
        )
    array = numpy.asarray(vertices, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise InputArtifactError(
            f"the mesh vertices have shape {tuple(array.shape)} and a surface mesh "
            "carries one row of three coordinates per vertex, so the array is "
            "(n, 3)",
            kind=_KIND,
        )
    if array.shape[0] < MINIMUM_SECTION_VERTICES:
        raise InputArtifactError(
            f"the mesh carries {array.shape[0]} vertices, which cannot describe a "
            "blade: a chordwise section needs at least "
            f"{MINIMUM_SECTION_VERTICES} of them before its own extent says "
            "anything about a chord direction",
            kind=_KIND,
        )
    if not bool(numpy.isfinite(array).all()):
        raise InputArtifactError(
            "the mesh carries a vertex coordinate that is not a finite number, so "
            "the spanwise coordinate of that vertex is undefined and it would fall "
            "into no section",
            kind=_KIND,
        )
    return array


def _mesh_faces(mesh: Any, vertex_count: int) -> numpy.ndarray:
    """Return the (m, 3) triangle array of a LOADED mesh, or refuse.

    The trailing-edge mid-points are of mesh EDGES, so a mesh with no
    faces names none; and an index outside the vertices would pair
    coordinates that are not an edge.
    """
    faces = getattr(mesh, "faces", None)
    array = numpy.asarray(faces if faces is not None else (), dtype=int)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 3:
        raise InputArtifactError(
            f"the mesh carries no triangular faces (their array has shape "
            f"{tuple(array.shape)}), and a trailing edge is found as a chain of mesh "
            "EDGES whose mid-points the import matches. A vertex cloud names no edge; "
            "pass the blade's surface mesh",
            kind=_KIND,
        )
    if int(array.min()) < 0 or int(array.max()) >= vertex_count:
        raise InputArtifactError(
            f"a face of the mesh names a vertex outside the {vertex_count} it carries, "
            "so its edges join coordinates that do not exist",
            kind=_KIND,
        )
    return array


def _chord_direction(points: numpy.ndarray, radial: numpy.ndarray) -> numpy.ndarray | None:
    """Return the unsigned chordwise direction of one section, or None.

    The chord is the longest dimension of an aerofoil section, so the
    direction is the first principal axis of the section's own vertices
    after the spanwise component is projected out. None when the
    section's vertices are effectively coincident, which leaves no
    direction to read.

    Parameters
    ----------
    points : numpy.ndarray
        Section vertices, shape (m, 3), in mesh length units.
    radial : numpy.ndarray
        Unit radial direction of this section, shape (3,). The component
        along it is removed so the extent is measured IN the section
        rather than across it.

    Returns
    -------
    numpy.ndarray or None
        Unit vector of shape (3,), with an arbitrary sign, or None.
    """
    centred = points - points.mean(axis=0)
    centred = centred - numpy.outer(centred @ radial, radial)
    scale = float(numpy.abs(centred).max())
    if scale == 0.0:
        return None
    centred = centred / scale
    eigenvalues, eigenvectors = numpy.linalg.eigh(centred.T @ centred)
    if float(eigenvalues[-1]) <= 0.0:
        return None
    direction = numpy.asarray(eigenvectors[:, -1], dtype=float)
    length = float(numpy.linalg.norm(direction))
    if length == 0.0:
        return None
    return direction / length


def extract_trailing_edge(
    mesh: Any,
    *,
    axis: Any = None,
    hub: Any = None,
    sections: int = DEFAULT_SECTIONS,
) -> TrailingEdge:
    """Extract the trailing edge of a blade mesh in the rotor frame.

    The criterion is the aftmost vertex of each chordwise section, with
    the chordwise direction taken from the section's own extent. Read
    the module docstring for why it is not a crease criterion and how
    the aft sense is fixed.

    Parameters
    ----------
    mesh : str, pathlib.Path or trimesh.Trimesh
        The blade surface, as a mesh file to read (``.obj`` or ``.stl``)
        or as a loaded mesh. Only its vertices are used. It need not be
        watertight and no proximity query is made, so this runs on a
        base install.
    axis : array_like
        REQUIRED. There is no default and the ``= None`` in the signature
        is a sentinel that exists only so the refusal can name both this
        and ``hub`` together; nothing here guesses an axis, because a
        guessed one silently returns the LEADING edge of a blade whose
        rotor turns the other way.

        Rotor axis of rotation, three components in the mesh's reference
        frame. Normalized internally, so only the direction matters, but
        the SENSE matters: the rotor is taken to turn the right-handed
        way about it, and that is what decides which end of the chord is
        aft. It is a second vocabulary for which way a rotor turns,
        beside the sign a matrix row states in ``RPM_SIGN`` or ``RPM``,
        and nothing reconciles the two; if the extraction returns a
        leading edge, this sign is the first thing to check.
    hub : array_like
        REQUIRED, on the same terms as ``axis`` above. A point on the
        axis of rotation, three components in the mesh's reference frame
        and length units. Together with ``axis`` it fixes the spanwise
        coordinate of every vertex.
    sections : int
        Number of chordwise sections to cut the span into, which is the
        number of nodes returned. At least two: one section yields one
        point, and a single point names no edge.

    Returns
    -------
    TrailingEdge
        The node list and what it was extracted with.

    Raises
    ------
    InputArtifactError
        If ``axis`` or ``hub`` is missing (the refusal names both,
        because neither alone defines a spanwise direction); if either
        is not three finite numbers, or the axis is the zero vector; if
        the mesh file cannot be read, or the loaded object carries no
        usable vertices; if ``sections`` is below
        two; if the vertices have no spanwise extent about the given
        axis; if a section holds too few vertices for its own extent to
        define a chord direction; or if no section's chord has a readable
        component along the tangential direction, which is the shape of
        a mesh, an axis and a hub that do not together describe a rotor
        blade.

        NOT among them, and named because the list would otherwise read
        as exhaustive of what can go wrong: two sections choosing one
        vertex. The sections partition the vertices, so that cannot
        happen; it is an internal invariant rather than a refusal (see
        the comment at the end of this function).

    Examples
    --------
    >>> from cfdbender.geometry.trailing_edges import extract_trailing_edge
    >>> edge = extract_trailing_edge(
    ...     "blade.obj", axis=(0.0, 0.0, 1.0), hub=(0.0, 0.0, 0.0)
    ... )  # doctest: +SKIP
    >>> edge.nodes.shape  # doctest: +SKIP
    (20, 3)
    """
    unit_axis, hub_point = _rotor_frame(axis, hub)
    vertices = _mesh_vertices(_loaded(mesh))
    cut = _sectioned(vertices, unit_axis, hub_point, sections)
    order = numpy.asarray(cut.chosen, dtype=int)
    return TrailingEdge(
        nodes=vertices[order],
        span=cut.span[order],
        axis=unit_axis,
        hub=hub_point,
        sections=sections,
    )


def _rotor_frame(axis: Any, hub: Any) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Read the rotor axis (normalized) and the hub, or refuse naming both."""
    if axis is None or hub is None:
        missing = ", ".join(name for name, value in (("axis", axis), ("hub", hub)) if value is None)
        raise InputArtifactError(
            f"the trailing-edge extraction was given no {missing}, and it needs BOTH: "
            "the rotor axis and the hub position are what define the spanwise "
            "direction to section along, and the tangential direction that tells the "
            "trailing edge from the leading one. Nothing here guesses a global axis, "
            "because a blade with twist has a trailing edge that a guessed axis "
            "silently gets wrong, which is the failure this extraction exists to "
            "remove. Pass axis=(x, y, z) and hub=(x, y, z) in the mesh's own frame",
            kind=_KIND,
        )
    return _unit_axis(axis), _three_vector(hub, "hub position")


@dataclass(frozen=True)
class _Sections:
    """The chordwise sectioning of one blade in its rotor frame.

    Attributes
    ----------
    span : numpy.ndarray
        Spanwise coordinate of every vertex, shape (n,).
    bounds : numpy.ndarray
        The sections' span boundaries, shape (sections + 1,).
    chords : numpy.ndarray
        Each section's unit chord direction, pointing AFT, shape
        (sections, 3).
    chosen : tuple of int
        Each section's aftmost vertex, by index, in span order.
    """

    span: numpy.ndarray
    bounds: numpy.ndarray
    chords: numpy.ndarray
    chosen: tuple[int, ...]

    def section_of(self, span: numpy.ndarray) -> numpy.ndarray:
        """Return the 0-based section each spanwise coordinate falls in."""
        last = len(self.bounds) - 2
        return numpy.clip(numpy.searchsorted(self.bounds, span, side="right") - 1, 0, last)


def _sectioned(
    vertices: numpy.ndarray, unit_axis: numpy.ndarray, hub_point: numpy.ndarray, sections: Any
) -> _Sections:
    """Cut the span into sections, orient each chord aft, pick each aftmost vertex."""
    # Any integer type counts, numpy's included; a boolean never does.
    if (
        not isinstance(sections, numbers.Integral)
        or isinstance(sections, bool | numpy.bool_)
        or sections < 2
    ):
        raise InputArtifactError(
            f"the extraction was asked for {sections!r} chordwise sections, and a "
            "trailing edge is a line along the span: one section yields one point, "
            "which names no edge, and a non-integer count names no sectioning at "
            f"all. Ask for two or more (the default is {DEFAULT_SECTIONS})",
            kind=_KIND,
        )
    sections = int(sections)
    if sections > len(vertices):
        # Refused before any array of that length is allocated.
        raise InputArtifactError(
            f"the extraction was asked for {sections} chordwise sections, and the mesh "
            f"has only {len(vertices)} vertices, fewer than one for each section. "
            "The span is being cut finer than the mesh resolves it: ask for fewer "
            "sections, or extract from a mesh with more spanwise stations",
            kind=_KIND,
        )

    offset = vertices - hub_point
    radial_vectors = offset - numpy.outer(offset @ unit_axis, unit_axis)
    span = numpy.linalg.norm(radial_vectors, axis=1)
    span_min = float(span.min())
    span_max = float(span.max())
    if span_max <= span_min:
        raise InputArtifactError(
            f"every vertex of the mesh lies {span_min!r} from the given rotor axis, "
            "so the blade has no spanwise extent to section along. The axis and the "
            "hub are probably not the ones this mesh was built about: the span is "
            "measured perpendicular to the axis, so a blade lying ALONG the axis "
            "collapses to a single radius",
            kind=_KIND,
        )

    edges = numpy.linspace(span_min, span_max, sections + 1)
    membership = numpy.clip(numpy.searchsorted(edges, span, side="right") - 1, 0, sections - 1)

    section_members: list[numpy.ndarray] = []
    section_chords: list[numpy.ndarray] = []
    section_tangentials: list[numpy.ndarray] = []
    for index in range(sections):
        selected = numpy.flatnonzero(membership == index)
        if selected.size < MINIMUM_SECTION_VERTICES:
            raise InputArtifactError(
                f"chordwise section {index + 1} of {sections}, spanning "
                f"{float(edges[index])!r} to {float(edges[index + 1])!r} from the "
                f"rotor axis, holds {int(selected.size)} vertices and a section "
                f"needs at least {MINIMUM_SECTION_VERTICES} for its own extent to "
                "define a chord direction. The span is being cut finer than the "
                "mesh resolves it: ask for fewer sections, or extract from a mesh "
                "with more spanwise stations",
                kind=_KIND,
            )
        points = vertices[selected]
        mean_radial = radial_vectors[selected].mean(axis=0)
        length = float(numpy.linalg.norm(mean_radial))
        if length == 0.0:
            raise InputArtifactError(
                f"chordwise section {index + 1} of {sections} has vertices spread "
                "evenly around the rotor axis, so the section has no radial "
                "direction of its own. That is the shape of a full disc or of a "
                "complete rotor rather than of one blade, and a trailing edge is "
                "extracted one blade at a time",
                kind=_KIND,
            )
        radial = mean_radial / length
        chord = _chord_direction(points, radial)
        if chord is None:
            raise InputArtifactError(
                f"chordwise section {index + 1} of {sections} has all its vertices "
                "at one point once the spanwise component is removed, so the "
                "section has no extent and no chord direction. A degenerate or "
                "duplicated band of vertices in the mesh is the usual cause",
                kind=_KIND,
            )
        tangential = numpy.cross(unit_axis, radial)
        tangential_length = float(numpy.linalg.norm(tangential))
        section_members.append(selected)
        section_chords.append(chord)
        section_tangentials.append(
            tangential / tangential_length if tangential_length else tangential
        )

    chords = _orient(section_chords, section_tangentials, sections)

    chosen: list[int] = []
    for members, chord in zip(section_members, chords, strict=True):
        chosen.append(int(members[int(numpy.argmax(vertices[members] @ chord))]))

    # Two sections cannot choose one vertex, because `membership` gives
    # every vertex exactly one section and each choice is made inside its
    # own section's members. This was written as a user-facing refusal
    # ("the sections are finer than the mesh resolves") and the
    # adversarial pass could not reach it: a refusal that cannot fire is
    # a promise nothing keeps, and this repository's rule is that a guard
    # is proven by restoring the defect it denies. Kept as an invariant
    # instead, so a future change that lets the sections overlap fails
    # here rather than writing a node list naming one edge twice.
    if len(set(chosen)) != len(chosen):
        raise AssertionError(
            "the chordwise sections no longer partition the vertices: two of them "
            f"chose the same mesh vertex out of {sections}"
        )
    return _Sections(
        span=span, bounds=edges, chords=numpy.asarray(chords, dtype=float), chosen=tuple(chosen)
    )


def _interior_edges(faces: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return every edge two faces share, and the far vertex of each face.

    An edge seen by one face is a boundary of the surface and an edge
    seen by three or more is not a surface edge anyone can call aft of
    both its faces, so only edges shared by exactly two faces are kept.

    Returns
    -------
    numpy.ndarray
        The edges, shape (k, 2), as sorted vertex-index pairs.
    numpy.ndarray
        For each edge, the vertex of each of its two faces that is not on
        it, shape (k, 2).
    """
    pairs = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    far = faces[:, [2, 0, 1]].reshape(-1)
    keys = numpy.sort(pairs, axis=1)
    order = numpy.lexsort((keys[:, 1], keys[:, 0]))
    keys = keys[order]
    far = far[order]
    fresh = numpy.ones(len(keys), dtype=bool)
    fresh[1:] = numpy.any(keys[1:] != keys[:-1], axis=1)
    starts = numpy.flatnonzero(fresh)
    counts = numpy.diff(numpy.append(starts, len(keys)))
    shared = starts[counts == 2]
    return keys[shared], numpy.stack([far[shared], far[shared + 1]], axis=1)


def _aft_ridge_edges(
    vertices: numpy.ndarray,
    faces: numpy.ndarray,
    unit_axis: numpy.ndarray,
    hub_point: numpy.ndarray,
    cut: _Sections,
) -> dict[int, list[tuple[int, float]]]:
    """Return the trailing-edge candidate edges as an adjacency, with their lengths.

    An interior edge is a candidate when the far vertex of EACH of its two
    faces lies forward of it: forward along the aft chord of the section
    its mid-point falls in, with the component along the edge itself
    removed, so an edge running across the chord at an angle is judged by
    the side its faces lie on and not by its own slant. Nothing here
    compares an angle with a threshold (DD-27).
    """
    edges, far = _interior_edges(faces)
    first = vertices[edges[:, 0]]
    second = vertices[edges[:, 1]]
    middle = (first + second) / 2.0
    direction = second - first
    length = numpy.linalg.norm(direction, axis=1)
    usable = length > 0.0
    direction[usable] /= length[usable][:, None]
    offset = middle - hub_point
    span = numpy.linalg.norm(offset - numpy.outer(offset @ unit_axis, unit_axis), axis=1)
    chord = cut.chords[cut.section_of(span)]
    across = chord - numpy.einsum("ij,ij->i", chord, direction)[:, None] * direction
    size = numpy.linalg.norm(across, axis=1)
    usable &= size > 1.0e-9
    across[usable] /= size[usable][:, None]
    ahead_first = numpy.einsum("ij,ij->i", vertices[far[:, 0]] - middle, across) < 0.0
    ahead_second = numpy.einsum("ij,ij->i", vertices[far[:, 1]] - middle, across) < 0.0
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for index in numpy.flatnonzero(usable & ahead_first & ahead_second):
        a, b = int(edges[index, 0]), int(edges[index, 1])
        adjacency.setdefault(a, []).append((b, float(length[index])))
        adjacency.setdefault(b, []).append((a, float(length[index])))
    return adjacency


def _shortest_path(
    adjacency: dict[int, list[tuple[int, float]]], start: int, goal: int
) -> tuple[list[int], int] | None:
    """Return the shortest path between two vertices, and how many share its length.

    Dijkstra over the candidate edges, counting the paths of equal length
    as it goes, so a caller can refuse a join it would otherwise pick
    between two equal answers arbitrarily. None when no path exists.
    """
    best: dict[int, float] = {start: 0.0}
    ways: dict[int, int] = {start: 1}
    previous: dict[int, int] = {}
    done: set[int] = set()
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if vertex in done:
            continue
        done.add(vertex)
        if vertex == goal:
            break
        for neighbour, length in adjacency.get(vertex, ()):
            if neighbour in done:
                continue
            reached = distance + length
            slack = 1.0e-12 * max(1.0, reached)
            known = best.get(neighbour)
            if known is None or reached < known - slack:
                best[neighbour] = reached
                ways[neighbour] = ways[vertex]
                previous[neighbour] = vertex
                heapq.heappush(queue, (reached, neighbour))
            elif abs(reached - known) <= slack:
                ways[neighbour] += ways[vertex]
    if goal not in done:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1], ways[goal]


def _walk_outward(
    chain: list[int], adjacency: dict[int, list[tuple[int, float]]], span: numpy.ndarray
) -> Iterator[tuple[bool, int]]:
    """Yield the vertices that extend the chain past its first and last section.

    From each end, the chain grows along the ONE candidate edge that moves
    outward in span, and stops where there is none or more than one: a
    capped tip or a split root offers a choice, and the extraction does not
    guess between them.
    """
    taken = set(chain)
    for at_root in (True, False):
        end = chain[0] if at_root else chain[-1]
        while True:
            outward = [
                neighbour
                for neighbour, _ in adjacency.get(end, ())
                if neighbour not in taken
                and (span[neighbour] < span[end] if at_root else span[neighbour] > span[end])
            ]
            if len(outward) != 1:
                break
            end = outward[0]
            taken.add(end)
            yield at_root, end


def trailing_edge_midpoints(
    mesh: Any,
    *,
    axis: Any = None,
    hub: Any = None,
    sections: int = DEFAULT_SECTIONS,
) -> numpy.ndarray:
    """Return the mid-point of every mesh edge on a blade's trailing edge.

    The form a points file is matched in: an edge is on the trailing edge when
    a point in the file lies within the tolerance of its mid-point; the
    edges' end vertices match nothing. The criterion is in the module
    docstring: the aftmost vertex of each chordwise section, joined section
    to section along the edges that are the aft ridge of both their faces.

    Parameters
    ----------
    mesh : str, pathlib.Path or trimesh.Trimesh
        The blade surface, as a mesh file to read (``.obj`` or ``.stl``)
        or as a loaded mesh. Its vertices AND faces are read.
    axis : array_like
        REQUIRED. Rotor axis of rotation; see :func:`extract_trailing_edge`,
        whose sense rule is this one's.
    hub : array_like
        REQUIRED. A point on the axis of rotation; see
        :func:`extract_trailing_edge`.
    sections : int
        Chordwise sections the span is cut into. Unlike the vertex
        extraction it does NOT set how many points come back: every mesh
        edge on the trailing edge gives one, whatever the count, as long as
        each section holds a vertex of it.

    Returns
    -------
    numpy.ndarray
        The mid-points, shape (m, 3), in the mesh's reference frame and
        length unit, in order along the edge from the root to the tip.

    Raises
    ------
    InputArtifactError
        Every refusal of :func:`extract_trailing_edge`; a mesh with no
        triangular faces; two consecutive sections whose aftmost vertices
        no path of candidate edges joins, or two paths of equal length
        join (the shape of a blunt trailing edge, which is not read here);
        and a chain that doubles back on itself.

    Examples
    --------
    >>> from cfdbender.geometry.trailing_edges import trailing_edge_midpoints
    >>> points = trailing_edge_midpoints(
    ...     "blade.obj", axis=(0.0, 0.0, 1.0), hub=(0.0, 0.0, 0.0)
    ... )  # doctest: +SKIP
    """
    unit_axis, hub_point = _rotor_frame(axis, hub)
    loaded = _loaded(mesh)
    vertices = _mesh_vertices(loaded)
    faces = _mesh_faces(loaded, len(vertices))
    cut = _sectioned(vertices, unit_axis, hub_point, sections)
    adjacency = _aft_ridge_edges(vertices, faces, unit_axis, hub_point, cut)

    chain = [cut.chosen[0]]
    for position in range(len(cut.chosen) - 1):
        found = _shortest_path(adjacency, cut.chosen[position], cut.chosen[position + 1])
        if found is None:
            raise InputArtifactError(
                f"no chain of trailing-edge mesh edges joins the aftmost vertex of "
                f"chordwise section {position + 1} of {sections} to that of section "
                f"{position + 2}: an edge is taken as trailing when both its faces lie "
                "forward of it, and none of those connects the two. A gap in the "
                "mesh along the trailing edge, or sections finer than its edges, are "
                "the usual causes; ask for fewer sections",
                kind=_KIND,
            )
        path, ways = found
        if ways > 1:
            raise InputArtifactError(
                f"{ways} chains of trailing-edge mesh edges of equal length join "
                f"chordwise sections {position + 1} and {position + 2} of {sections}, "
                "so which edges are the trailing edge cannot be told. That is the shape "
                "of a blunt trailing edge, two aft corners per station, which this "
                "extraction does not read",
                kind=_KIND,
            )
        chain.extend(path[1:])
    for at_root, vertex in _walk_outward(chain, adjacency, cut.span):
        if at_root:
            chain.insert(0, vertex)
        else:
            chain.append(vertex)
    if len(set(chain)) != len(chain):
        raise InputArtifactError(
            "the chain of trailing-edge mesh edges passes one vertex twice, so it "
            "doubles back rather than running along the span; ask for fewer sections, "
            "or check the axis and the hub",
            kind=_KIND,
        )
    ends = numpy.asarray(chain, dtype=int)
    return (vertices[ends[:-1]] + vertices[ends[1:]]) / 2.0


def _orient(
    chords: list[numpy.ndarray],
    tangentials: list[numpy.ndarray],
    sections: int,
) -> list[numpy.ndarray]:
    """Point every section's chord direction aft, consistently.

    Two steps, and the order is the point. Each section's extent gives a
    LINE, so the signs arrive independent of one another; they are first
    made CONSISTENT by walking the span and flipping any section that
    disagrees with its inboard neighbor, which is a comparison between
    two nearby chords and survives any amount of twist so long as the
    twist between neighboring sections is under a right angle. Only
    then is the single remaining global sign decided, on the section
    whose chord is most nearly tangential, where the answer is least
    ambiguous. Deciding per section against the tangential direction
    instead would put the question to the strongly twisted root sections
    individually, which is where it is hardest and where a wrong answer
    flips one node of the edge and nothing says so.

    Parameters
    ----------
    chords : list of numpy.ndarray
        Unsigned unit chord directions, one per section, in span order.
    tangentials : list of numpy.ndarray
        Unit tangential directions ``axis x radial``, one per section.
    sections : int
        Number of sections, for the refusal text.

    Returns
    -------
    list of numpy.ndarray
        The same directions, every one pointing aft.

    Raises
    ------
    InputArtifactError
        If no section's chord has a readable component along its own
        tangential direction, so aft cannot be told from forward.
    """
    oriented = [chords[0]]
    for index in range(1, len(chords)):
        chord = chords[index]
        oriented.append(-chord if float(chord @ oriented[-1]) < 0.0 else chord)

    alignments = [
        float(chord @ tangential) for chord, tangential in zip(oriented, tangentials, strict=True)
    ]
    anchor = int(numpy.argmax(numpy.abs(alignments)))
    if abs(alignments[anchor]) < _MINIMUM_TANGENTIAL_ALIGNMENT:
        raise InputArtifactError(
            f"no chordwise section of the {sections} has a chord direction with a "
            "readable component along the rotor's tangential direction, so which "
            "end of the chord is aft cannot be told from which end is forward. The "
            "aft sense comes from the tangential direction axis x radial, and a "
            "chord perpendicular to it at every station is not a blade section: "
            "check that the axis and the hub are the ones this mesh was built "
            "about",
            kind=_KIND,
        )
    if alignments[anchor] > 0.0:
        return [-chord for chord in oriented]
    return oriented


def write_trailing_edge_node_file(
    mesh: Any,
    path: str | Path,
    *,
    axis: Any = None,
    hub: Any = None,
    unit: str,
    sections: int = DEFAULT_SECTIONS,
    overwrite: bool = False,
) -> Path:
    """Extract a blade's trailing edge and write it as its points file.

    The whole route ```` names, in one call: read the mesh,
    find the mid-point of every mesh edge on the trailing edge
    (:func:`trailing_edge_midpoints`), and write them under a unit line
    (:func:`cfdbender._points.write_trailing_edge_points`),
    the points file a geometry names. Nothing of the mesh library crosses
    into the file: what is written is coordinates.

    Parameters
    ----------
    mesh : str, pathlib.Path or trimesh.Trimesh
        The blade surface; see :func:`trailing_edge_midpoints`.
    path : str or pathlib.Path
        Destination points file.
    axis : array_like
        Rotor axis of rotation; see :func:`extract_trailing_edge`.
    hub : array_like
        A point on the axis of rotation; see
        :func:`extract_trailing_edge`.
    unit : str
        Length unit the MESH coordinates are in, a unit word known to
        :func:`cfdbender._lengths.scale`, written as the file's first line. Required rather than
        defaulted: nothing here can read a unit out of a mesh file.
    sections : int
        Number of chordwise sections; see :func:`trailing_edge_midpoints`.
    overwrite : bool
        Replace an existing destination.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    InputArtifactError
        Every refusal of :func:`trailing_edge_midpoints` and of
        :func:`~cfdbender._points.write_trailing_edge_points`.
        The extraction runs FIRST, so a refused extraction leaves no
        file behind.

    Examples
    --------
    >>> from cfdbender.geometry.trailing_edges import (
    ...     write_trailing_edge_node_file,
    ... )
    >>> write_trailing_edge_node_file(
    ...     "blade.obj",
    ...     "blade.te.txt",
    ...     axis=(0.0, 0.0, 1.0),
    ...     hub=(0.0, 0.0, 0.0),
    ...     unit="METER",
    ... )  # doctest: +SKIP
    """
    points = trailing_edge_midpoints(mesh, axis=axis, hub=hub, sections=sections)
    return write_trailing_edge_points(path, points, unit=unit, overwrite=overwrite)

"""The CAD kernel interface of the geometry stages, with no kernel import.

A stage that builds, intersects, trims, queries or writes surfaces takes a :class:`CadKernel` and
never sees the library behind it. :mod:`cfdbender.mesh._gmsh` is the one module that
imports a kernel library and holds the adapter; a permissive kernel replaces it by adding an adapter
that satisfies this protocol and changing :func:`cfdbender.mesh._gmsh.open_kernel`.

Everything here is a plain type: a face is an opaque handle, points are tuples of floats in meters
(a file read in another unit arrives in meters), and a result is a frozen dataclass. The interface
is surfaces only: no method builds or returns a solid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CadKernel",
    "Face",
    "FileTopology",
    "FragmentMap",
    "NamedFaces",
    "Point",
    "UV",
    "UVCurve",
]

#: A point, in meters.
Point = tuple[float, float, float]
#: A surface parameter pair.
UV = tuple[float, float]
#: One curve of a trim loop in parameter space: the points it passes through, in (u, v). Two
#: points are a straight line; more are a cubic spline through them.
UVCurve = Sequence[UV]


@dataclass(frozen=True, order=True)
class Face:
    """An opaque handle to one surface of an open kernel; only the kernel that made it reads it."""

    tag: int


@dataclass(frozen=True)
class FragmentMap:
    """The result of :meth:`CadKernel.fragment`.

    ``faces`` is every face left after the operation. ``from_a`` and ``from_b`` give, for each input
    face of the two groups in order, the faces it became (one face when it was not cut).
    """

    faces: tuple[Face, ...]
    from_a: tuple[tuple[Face, ...], ...]
    from_b: tuple[tuple[Face, ...], ...]


@dataclass(frozen=True)
class NamedFaces:
    """The faces of one imported IGES file, each with the name the file gives it.

    The faces are in meters; ``unit`` is the unit the file stated.
    """

    faces: tuple[Face, ...]
    names: tuple[str, ...]
    unit: str | None = None
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class FileTopology:
    """What one IGES file holds once read alone, and sewn when a tolerance was given.

    ``names``, ``areas`` and ``perimeters`` are per face, in meters; each item of ``edges`` is
    ``(samples, faces)``: points along the edge and the indices of the faces that share it, so a
    free edge has one face and a non-manifold edge more than two.
    """

    names: tuple[str, ...]
    areas: tuple[float, ...]
    perimeters: tuple[float, ...]
    edges: tuple[tuple[np.ndarray, tuple[int, ...]], ...]


@runtime_checkable
class CadKernel(Protocol):
    """Surface construction, topology, measurements and export behind a replaceable kernel."""

    # Life
    @property
    def name(self) -> str:
        """The kernel's short name, recorded in a build identity."""
        ...

    @property
    def version(self) -> str:
        """The kernel library's version string, recorded in a build identity."""
        ...

    def close(self) -> None:
        """Release the kernel; the object is unusable afterwards. Closing twice is harmless."""
        ...

    # Build
    def add_bspline_surface(
        self,
        net: Sequence[Sequence[Point]],
        *,
        knots_u: Sequence[float],
        knots_v: Sequence[float],
        mult_u: Sequence[int],
        mult_v: Sequence[int],
        deg_u: int,
        deg_v: int,
    ) -> Face:
        """Add one non-rational B-spline surface; ``net[i][j]`` is the point at u index i."""
        ...

    def add_trimmed_face(self, base: Face, loops_uv: Sequence[Sequence[UVCurve]]) -> Face:
        """Add the part of the surface ``base`` that lies inside the outer loop of ``loops_uv``.

        Each loop is a closed chain of :data:`UVCurve` in the parameter space of ``base``; the
        first loop is the outer one and the rest are holes. The trim is exact, because it is
        stated in parameters and not as space curves. ``base`` stays in the model, so one base
        carries any number of faces; remove it with :meth:`remove` when its last face is made.
        An open loop is refused.
        """
        ...

    def add_planar_face(self, loop: Sequence[Point]) -> Face:
        """Add a planar face bounded by the closed polyline ``loop``."""
        ...

    def add_ribbon(self, curve_points: Sequence[Point], normal_vectors: Sequence[Point]) -> Face:
        """Add the ruled face between a curve and the same curve moved along ``normal_vectors``."""
        ...

    # Boolean
    def fragment(self, a: Sequence[Face], b: Sequence[Face], *, fuzzy: float = 0.0) -> FragmentMap:
        """Imprint the faces of ``a`` and ``b`` on each other and return what each became.

        ``fuzzy`` is the distance below which the operation treats two shapes as touching; 0
        keeps the kernel's own.
        """
        ...

    def imprint(
        self, faces: Sequence[Face], edges: Sequence[int], *, fuzzy: float = 0.0
    ) -> FragmentMap:
        """Split ``faces`` along the ``edges`` lying on them; ``from_a`` gives what each became.

        An edge that bounds one of ``faces`` stays shared by that face and the faces it split,
        so a projected junction is one curve of both sides.
        """
        ...

    # Edit
    def remove(self, faces: Sequence[Face]) -> None:
        """Delete the faces and what only they bound."""
        ...

    def split_face_uv(self, face: Face, direction: str, value: float) -> tuple[Face, ...]:
        """Split ``face`` along the iso line ``direction`` (``"u"`` or ``"v"``) at ``value``.

        ``face`` is replaced by the pieces and is not a handle afterwards. Return the pieces in
        increasing order of ``direction``; a face the line does not cross comes back whole. The
        cut is exact: it is an iso line of the face's own base. A face the kernel did not build
        from a base (an imported one) is refused.
        """
        ...

    def copy(self, faces: Sequence[Face]) -> tuple[Face, ...]:
        """Return copies of the faces, in order."""
        ...

    def transform(self, faces: Sequence[Face], matrix: Sequence[Sequence[float]]) -> None:
        """Apply a 4 by 4 homogeneous matrix, row-major, to the faces in place."""
        ...

    # Query
    def split_imported_face_uv(self, face: Face, direction: str, value: float) -> tuple[Face, ...]:
        """Split a face on its own iso-u or iso-v line, retaining its existing trims.

        Imported faces are supported without fitting a replacement surface. The input is
        replaced by its pieces, lower parameter side first. A finite value outside the
        face's parameter bounds returns the unchanged face. Invalid directions and
        non-finite values are refused before mutation.
        """
        ...

    def minimum_distance(self, a: Sequence[Face], b: Sequence[Face]) -> float:
        """Return native minimum distance between trimmed face sets, in meters.

        Contact and intersection give zero. This is a kernel extremum query, with no
        sampling of faces or edges. Empty sets and a failed kernel query are refused.
        """
        ...

    def faces(self) -> tuple[Face, ...]:
        """Return every face of the model, in creation order."""
        ...

    def boundary(self, face: Face) -> tuple[int, ...]:
        """Return the signed edge tags that bound ``face``, in loop order."""
        ...

    def adjacent(self, edge: int) -> tuple[Face, ...]:
        """Return the faces that share ``edge``; more than two means a non-manifold edge."""
        ...

    def area(self, face: Face) -> float:
        """Return the exact area of ``face``, in square meters."""
        ...

    def centroid(self, face: Face) -> Point:
        """Return the center of mass of ``face``."""
        ...

    def bbox(self, face: Face) -> tuple[Point, Point]:
        """Return the axis-aligned bounding box of ``face`` as ``(minimum, maximum)``."""
        ...

    def uv_bounds(self, face: Face) -> tuple[UV, UV]:
        """Return the parametric bounds of ``face`` as ``(minimum, maximum)``."""
        ...

    def value(self, face: Face, uv: UV) -> Point:
        """Return the point of ``face`` at ``uv``."""
        ...

    def normal(self, face: Face, uv: UV) -> Point:
        """Return the unit normal of ``face`` at ``uv``."""
        ...

    def derivative(self, face: Face, uv: UV) -> tuple[Point, Point]:
        """Return the partial derivatives ``(d/du, d/dv)`` of ``face`` at ``uv``."""
        ...

    def closest(self, face: Face, points: Sequence[Point]) -> tuple[tuple[Point, UV], ...]:
        """For each point, the closest point of ``face`` and its ``uv``."""
        ...

    def reparametrize(self, edge: int, face: Face, t: float) -> UV:
        """Return the ``(u, v)`` on ``face`` of the point of ``edge`` at its parameter ``t``."""
        ...

    def shared_curve(self, face_a: Face, face_b: Face) -> int | None:
        """Return the one edge the two faces share, or ``None`` when they share none.

        Two faces that share more than one edge have no one curve and are refused.
        """
        ...

    def is_inside(self, face: Face, uv: UV) -> bool:
        """Whether ``uv`` lies inside the trimmed face, not only inside its bounds."""
        ...

    # Batched surface and curve queries (a distance measure samples thousands of points)
    def values(self, face: Face, uvs: Sequence[UV]) -> np.ndarray:
        """Return the point of ``face`` at each ``uv``."""
        ...

    def derivatives(self, face: Face, uvs: Sequence[UV]) -> np.ndarray:
        """Return ``(d/du, d/dv)`` of ``face`` at each ``uv``."""
        ...

    def inside(self, face: Face, uvs: Sequence[UV]) -> np.ndarray:
        """Whether each ``uv`` lies inside the trimmed face."""
        ...

    def project(self, face: Face, points: Sequence[Point]) -> tuple[np.ndarray, np.ndarray]:
        """Return the closest point of ``face`` to each point, and its ``uv`` (see ``closest``)."""
        ...

    def parametrize(self, face: Face, points: Sequence[Point]) -> np.ndarray:
        """Return the ``uv`` of each point lying on ``face``."""
        ...

    def edge_bounds(self, edge: int) -> tuple[float, float]:
        """Return the parameter range of ``edge``."""
        ...

    def edge_length(self, edge: int) -> float:
        """Return the native arc length of a signed boundary edge, in meters."""
        ...

    def edge_values(self, edge: int, parameters: Sequence[float]) -> np.ndarray:
        """Return the point of ``edge`` at each parameter."""
        ...

    def edge_project(self, edge: int, points: Sequence[Point]) -> np.ndarray:
        """Return the closest point of the curve ``edge`` to each point; refuse a curveless edge."""
        ...

    def edge_ends(self, edge: int) -> tuple[Point, ...]:
        """Return the end vertices of ``edge``."""
        ...

    def edge_bbox(self, edge: int) -> tuple[Point, Point]:
        """Return the axis-aligned bounding box of ``edge`` as ``(minimum, maximum)``."""
        ...

    # Hull
    def triangulate(
        self, faces: Sequence[Face], size: float
    ) -> tuple[tuple[Point, ...], tuple[tuple[int, int, int], ...]]:
        """Triangulate the faces at element size ``size``: nodes, and triangles as node indices."""
        ...

    # IO
    def import_iges(self, path: str | Path) -> NamedFaces:
        """Read one IGES file into the model and return its faces with the names the file gives."""
        ...

    def inspect_file(self, path: str | Path, tolerance: float | None) -> FileTopology:
        """Read one IGES file alone, apart from the model, sew it at ``tolerance`` if given.

        The file is read in one import, because faces of two imports are never sewn to each
        other; the model of the kernel is left as it was.
        """
        ...

    def export_iges(
        self,
        groups: Mapping[str, Sequence[Face]],
        path: str | Path,
        *,
        preference: int = 1,
        unit: str = "m",
    ) -> None:
        """Write one IGES file, each group's faces carrying the group name as their NAME.

        Every 142 carries both parameter-space and model-space curves, with PREF 1.
        The parameter-space curves have entity use flag 05. Requests for preference
        2 (model space) or 3 (either) warn and write 1, since the observed reader
        requires the parameter-space boundary. Its coordinates are the exact ones
        supplied to :meth:`add_trimmed_face`.
        ``unit`` selects ``m`` or ``mm`` in the file; model coordinates remain meters.
        """
        ...

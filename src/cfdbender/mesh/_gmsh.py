"""The one guarded import of gmsh, and the named import of IGES files both IGES commands share.

gmsh is a base dependency but is imported only here, and only when a command that needs a CAD
kernel runs. :func:`gmsh_module` raises :class:`KernelUnavailableError` when it is absent or its
shared library fails to load; :class:`GmshKernel` is the adapter of the kernel interface.
:func:`import_named` reads each IGES file into one
OpenCASCADE model and keeps the component of every face from the file's own names
(:mod:`cfdbender.geometry._iges_names`), with every file scaled to the first file's unit.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import numpy as np

from cfdbender._digest import file_sha256
from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.geometry._geom_area_brep import brep_definition
from cfdbender.geometry._iges_names import (
    DEFAULT_NAME_ENCODING,
    IgesNames,
    read_iges_names,
)
from cfdbender.geometry._iges_write import _write_visible, read_iges, write_iges
from cfdbender.mesh._cad import (
    UV,
    Face,
    FileTopology,
    FragmentMap,
    NamedFaces,
    Point,
    UVCurve,
)
from cfdbender.mesh._extras import KernelUnavailableError

__all__ = [
    "GmshKernel",
    "NamedModel",
    "closest_on_face",
    "gmsh_module",
    "import_iges",
    "import_named",
    "keep_components",
    "open_kernel",
    "session",
]

#: The unit an OpenCASCADE IGES import returns its coordinates in, whatever the file states.
IMPORT_UNIT = "MILLIMETER"
#: The same unit as gmsh's target unit option names it.
_OCC_IMPORT_UNIT = "MM"
#: What gmsh raises when the process has not yet made the target unit it would set.
_NO_UNIT_YET = "Could not set OpenCASCADE target unit"
#: A parameter pair is trusted when it reproduces the projected point to this fraction of its size.
_PARAMETER_TOLERANCE = 1e-9
#: A trim loop is closed when each curve starts where the one before it ends, within this gap in
#: parameter units; the design's loop check (S7, ``trim.closed``) states 1e-12 and a kernel
#: refuses at a thousandth of a unit more so a loop one rounding off still builds.
_LOOP_GAP = 1e-9


def gmsh_module(purpose: str) -> ModuleType:
    """Return the gmsh module, or raise :class:`KernelUnavailableError` naming ``purpose``.

    ``ImportError`` (not installed) and ``OSError`` (a shared library it links against failed to
    load, which the import reports as ``OSError``) both give that one error.
    """
    try:
        import gmsh
    except (ImportError, OSError) as error:
        raise KernelUnavailableError(purpose=purpose, reason=str(error)) from error
    return gmsh


@contextlib.contextmanager
def _quiet_stdout() -> Iterator[None]:
    """Send the process's standard output to the null device for the block.

    The OpenCASCADE IGES reader prints a line of its own to the C standard output on every
    import, which gmsh's terminal option does not reach; a command that prints its record as
    JSON would otherwise print that line before it.
    """
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(null)


def import_iges(gmsh: Any, path: str | Path) -> list[int]:
    """Import one IGES file into the open model and return its faces, in import order.

    OpenCASCADE keeps the target unit of its last import in a process-wide static that an empty
    gmsh option leaves alone and ``gmsh.finalize`` does not reset, so a session that once asked
    for meters would change every later read in the process. The unit is named on every read and
    the option the caller's session held is put back.

    The static does not exist before the first import or export of the process, and gmsh refuses
    to set it then. That refusal proves no earlier read left a unit, so the read runs on the
    default, which is millimeters.
    """

    def read(unit: str) -> list[tuple[int, int]]:
        gmsh.option.setString("Geometry.OCCTargetUnit", unit)
        with _quiet_stdout():
            shapes: list[tuple[int, int]] = gmsh.model.occ.importShapes(
                str(path), highestDimOnly=True
            )
        return shapes

    held = gmsh.option.getString("Geometry.OCCTargetUnit")
    try:
        try:
            out = read(_OCC_IMPORT_UNIT)
        except Exception as error:
            if _NO_UNIT_YET not in str(error):
                raise
            out = read("")
    finally:
        gmsh.option.setString("Geometry.OCCTargetUnit", held)
    return [tag for dim, tag in out if dim == 2]


def closest_on_face(model: Any, face: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project ``points`` (n by 3) on a surface; return points on it and their parameters.

    gmsh 4.13 returns zeros for the parameters of a surface projection, and later versions return
    the true ones. Parameters that do not reproduce the projected point are recovered from the
    point. Where the projection lies outside the surface's bounds, the recovered parameters are a
    point of the surface inside its bounds but not the nearest one: it can lie far from the
    projection (measured on 4.13, a plate). The returned point is always the one at the returned
    parameters, never the projection beyond the bounds, so a distance to it is never shorter than
    the distance to the surface; read it as an upper bound there.
    """
    xyz, uv = model.getClosestPoint(2, face, points.ravel())
    xyz = np.array(xyz, dtype=float).reshape(-1, 3)
    uv = np.array(uv, dtype=float).reshape(-1, 2)
    at_uv = np.asarray(model.getValue(2, face, uv.ravel())).reshape(-1, 3)
    wrong = np.linalg.norm(at_uv - xyz, axis=1) > _PARAMETER_TOLERANCE * (
        1.0 + np.linalg.norm(xyz, axis=1)
    )
    if wrong.any():
        recovered = np.asarray(model.getParametrization(2, face, xyz[wrong].ravel())).reshape(-1, 2)
        uv[wrong] = recovered
        xyz[wrong] = np.asarray(model.getValue(2, face, recovered.ravel())).reshape(-1, 3)
    return xyz, uv


@contextlib.contextmanager
def session(purpose: str) -> Iterator[Any]:
    """Yield an initialized gmsh with its terminal output off; finalize it on the way out."""
    gmsh = gmsh_module(purpose)
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        yield gmsh
    finally:
        gmsh.finalize()


def _oriented(loop: Sequence[UVCurve], *, outer: bool) -> list[list[UV]]:
    """Return ``loop`` turning counterclockwise in (u, v) when ``outer`` and clockwise otherwise.

    The kernel measures a hole as an area that is taken away only when its wire turns against
    the outer one, and a caller states its loops in whatever sense it built them.
    """
    curves = [[(float(u), float(v)) for u, v in c] for c in loop]
    ring = [p for c in curves for p in c[:-1]]
    twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, [*ring[1:], ring[0]], strict=True))
    if (twice > 0) == outer:
        return curves
    return [c[::-1] for c in curves[::-1]]


def _check_loop_shape(loop: Sequence[UVCurve], low: UV, high: UV) -> None:
    """Refuse a trim loop that leaves the base's domain or encloses no area.

    Outside its domain a B-spline base is extrapolated, so such a loop builds a face that is not on
    the surface; a loop of zero area, or a curve of zero length, builds no face a reader can use.
    """
    points = [(float(u), float(v)) for c in loop for u, v in c]
    for u, v in points:
        if not (
            low[0] - _LOOP_GAP <= u <= high[0] + _LOOP_GAP
            and low[1] - _LOOP_GAP <= v <= high[1] + _LOOP_GAP
        ):
            raise InputValueError(
                f"a trim loop leaves the base's domain {low} to {high} at {(u, v)}"
            )
    if any(len(set(map(tuple, c))) < 2 for c in loop):
        raise InputValueError("a trim loop holds a curve of zero length")
    ring = [(float(u), float(v)) for c in loop for u, v in c[:-1]]
    twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, [*ring[1:], ring[0]], strict=True))
    size = (high[0] - low[0]) * (high[1] - low[1])
    if abs(twice) <= _LOOP_GAP * size:
        raise InputValueError("a trim loop encloses no area")


@dataclass
class NamedModel:
    """The faces of the imported files, each with its component, in import order."""

    unit: str
    faces: list[int]
    names: list[str]
    sources: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def components(self) -> dict[str, list[int]]:
        """Return the face tags of each component, in the order its first face was imported."""
        out: dict[str, list[int]] = {}
        for tag, name in zip(self.faces, self.names, strict=True):
            out.setdefault(name, []).append(tag)
        return out


def _unit_of(names: IgesNames) -> str:
    if names.unit is None:
        raise InputArtifactError(
            f"{names.path}: its global section names a length unit this package has no scale "
            f"for; give the file one of {', '.join(METRES_PER_UNIT)} and run again. Nothing "
            "was written.",
            kind="geometry",
        )
    return names.unit


def _stated_extent(path: str | Path) -> float | None:
    """Return the largest coordinate the file's global section states, or ``None`` if unreadable."""
    try:
        return float(read_iges(path)["global"][19].replace("D", "E"))
    except (IndexError, ValueError):
        return None


def import_named(
    gmsh: Any, paths: Sequence[str | Path], *, name_encoding: str = DEFAULT_NAME_ENCODING
) -> NamedModel:
    """Import every IGES file into the open gmsh model and name each face from its file.

    The i-th face the import returns is the i-th face :func:`read_iges_names` lists; the counts
    are compared and a file where they differ is refused with both. Coordinates are scaled from
    the import's unit to the first file's unit, so every file of the model is in that unit.
    ``name_encoding`` is the text encoding of the NAME bytes (:func:`read_iges_names`).
    """
    if not paths:
        raise InputArtifactError(
            "name at least one IGES file. Nothing was written.", kind="geometry"
        )
    listed: list[IgesNames] = [read_iges_names(p, name_encoding=name_encoding) for p in paths]
    unit = _unit_of(listed[0])
    model = NamedModel(unit=unit, faces=[], names=[])
    for names in listed:
        _unit_of(names)
        faces = import_iges(gmsh, names.path)
        if len(faces) != len(names.faces):
            raise InputArtifactError(
                f"{names.path}: the import returned {len(faces)} faces and the file's directory "
                f"lists {len(names.faces)} surfaces, so no face can be named from it. Write the "
                "file again from its CAD system and run again. Nothing was written.",
                kind="geometry",
            )
        model.faces += faces
        model.names += list(names.faces)
        model.sources.append(
            {
                "file": names.path.name,
                "sha256": file_sha256(names.path),
                "unit": names.unit,
                "faces": len(faces),
                "components": names.components,
            }
        )
        if names.unit != unit:
            model.notes.append(f"{names.path.name} is in {names.unit} and was scaled to {unit}")
    gmsh.model.occ.synchronize()
    factor = float(METRES_PER_UNIT[IMPORT_UNIT] / METRES_PER_UNIT[unit])
    if factor != 1.0:
        gmsh.model.occ.dilate([(2, t) for t in model.faces], 0.0, 0.0, 0.0, factor, factor, factor)
        gmsh.model.occ.synchronize()
    return model


def keep_components(gmsh: Any, model: NamedModel, names: Sequence[str]) -> NamedModel:
    """Remove the faces of every component not in ``names`` from the gmsh model; return the rest."""
    keep = set(names)
    dropped = [t for t, n in zip(model.faces, model.names, strict=True) if n not in keep]
    if dropped:
        gmsh.model.occ.remove([(2, t) for t in dropped], recursive=True)
        gmsh.model.occ.synchronize()
    pairs = [(t, n) for t, n in zip(model.faces, model.names, strict=True) if n in keep]
    return NamedModel(
        unit=model.unit,
        faces=[t for t, _ in pairs],
        names=[n for _, n in pairs],
        sources=model.sources,
        notes=[*model.notes, f"selected components: {', '.join(names)}"],
    )


class GmshKernel:
    """The :class:`~cfdbender.mesh._cad.CadKernel` adapter over an initialized gmsh.

    Build and edit calls mark the model as changed; every query synchronizes first, so a stage
    never calls ``synchronize``. The kernel finalizes gmsh in :meth:`close`.
    """

    name = "gmsh"

    def __init__(self, gmsh: Any) -> None:
        self._g = gmsh
        self._dirty = False
        self._open = True
        #: What each B-spline base was built from, so a split can build the same base again.
        self._specs: dict[int, dict[str, Any]] = {}
        #: The base of each face this kernel trimmed or split, by face tag.
        self._parent: dict[int, int] = {}

    @property
    def version(self) -> str:
        """Return the gmsh library version string."""
        return str(self._g.__version__)

    def close(self) -> None:
        """Finalize gmsh; closing twice is harmless."""
        if self._open:
            self._open = False
            self._g.finalize()

    def _model(self) -> Any:
        if self._dirty:
            self._g.model.occ.synchronize()
            self._dirty = False
        return self._g.model

    def _tags(self, faces: Sequence[Face]) -> list[tuple[int, int]]:
        return [(2, f.tag) for f in faces]

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
        occ = self._g.model.occ
        points = [occ.addPoint(*net[i][j]) for j in range(len(net[0])) for i in range(len(net))]
        tag = occ.addBSplineSurface(
            points,
            len(net),
            degreeU=deg_u,
            degreeV=deg_v,
            knotsU=list(knots_u),
            knotsV=list(knots_v),
            multiplicitiesU=list(mult_u),
            multiplicitiesV=list(mult_v),
        )
        self._dirty = True
        self._specs[tag] = {
            "net": [list(row) for row in net],
            "knots_u": list(knots_u),
            "knots_v": list(knots_v),
            "mult_u": list(mult_u),
            "mult_v": list(mult_v),
            "deg_u": deg_u,
            "deg_v": deg_v,
        }
        return Face(tag)

    def add_trimmed_face(self, base: Face, loops_uv: Sequence[Sequence[UVCurve]]) -> Face:
        """Add the part of ``base`` inside the outer loop of ``loops_uv`` (see the protocol)."""
        if not loops_uv:
            raise InputValueError("a trimmed face needs at least its outer loop of (u, v) curves")
        low, high = self.uv_bounds(base)
        for loop in loops_uv:
            _check_loop_shape(loop, low, high)
        wires = [self._uv_wire(_oriented(loop, outer=k == 0)) for k, loop in enumerate(loops_uv)]
        tag = self._g.model.occ.addTrimmedSurface(base.tag, wires, False)
        self._dirty = True
        if base.tag in self._specs:
            self._parent[tag] = base.tag
        return Face(tag)

    def _uv_wire(self, loop: Sequence[UVCurve]) -> int:
        """Build one closed wire of parameter-space curves (x is u, y is v, z is zero)."""
        occ = self._g.model.occ
        curves = [[(float(u), float(v)) for u, v in c] for c in loop]
        if not curves or any(len(c) < 2 for c in curves):
            raise InputValueError(
                "a trim loop is a chain of curves of at least two (u, v) points each"
            )
        for k, curve in enumerate(curves):
            nxt = curves[(k + 1) % len(curves)][0]
            if max(abs(curve[-1][0] - nxt[0]), abs(curve[-1][1] - nxt[1])) > _LOOP_GAP:
                raise InputValueError(
                    f"a trim loop is open: curve {k} ends at {curve[-1]} and the next "
                    f"starts at {nxt}"
                )
        starts = [occ.addPoint(c[0][0], c[0][1], 0.0) for c in curves]
        edges = []
        for k, curve in enumerate(curves):
            end = starts[(k + 1) % len(curves)]
            if len(curve) == 2:
                edges.append(occ.addLine(starts[k], end))
            else:
                inner = [occ.addPoint(u, v, 0.0) for u, v in curve[1:-1]]
                edges.append(occ.addSpline([starts[k], *inner, end]))
        return int(occ.addWire(edges))

    def add_planar_face(self, loop: Sequence[Point]) -> Face:
        """Add a planar face bounded by the closed polyline ``loop``."""
        occ = self._g.model.occ
        points = [occ.addPoint(*p) for p in loop]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        tag = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        self._dirty = True
        return Face(tag)

    def add_ribbon(self, curve_points: Sequence[Point], normal_vectors: Sequence[Point]) -> Face:
        """Add the ruled face between a curve and the same curve moved along ``normal_vectors``."""
        occ = self._g.model.occ
        wires = []
        for shift in (0.0, 1.0):
            points = [
                occ.addPoint(*(p[k] + shift * n[k] for k in range(3)))
                for p, n in zip(curve_points, normal_vectors, strict=True)
            ]
            wires.append(occ.addWire([occ.addSpline(points)]))
        out = occ.addThruSections(wires, makeSolid=False, makeRuled=True)
        faces = [t for d, t in out if d == 2]
        self._dirty = True
        return Face(faces[0])

    # Boolean
    def fragment(self, a: Sequence[Face], b: Sequence[Face], *, fuzzy: float = 0.0) -> FragmentMap:
        """Imprint the faces of ``a`` and ``b`` on each other and return what each became."""
        with self._fuzzy(fuzzy):
            out, mapping = self._g.model.occ.fragment(self._tags(a), self._tags(b))
        self._dirty = True
        rows = [tuple(Face(t) for d, t in row if d == 2) for row in mapping]
        return FragmentMap(
            faces=tuple(Face(t) for d, t in out if d == 2),
            from_a=tuple(rows[: len(a)]),
            from_b=tuple(rows[len(a) :]),
        )

    def imprint(
        self, faces: Sequence[Face], edges: Sequence[int], *, fuzzy: float = 0.0
    ) -> FragmentMap:
        """Split ``faces`` along the ``edges`` lying on them (see the protocol)."""
        self._model()
        tools = [(1, abs(e)) for e in dict.fromkeys(edges)]
        with self._fuzzy(fuzzy):
            out, mapping = self._g.model.occ.fragment(self._tags(faces), tools)
        self._dirty = True
        rows = [tuple(Face(t) for d, t in row if d == 2) for row in mapping[: len(faces)]]
        return FragmentMap(
            faces=tuple(Face(t) for d, t in out if d == 2), from_a=tuple(rows), from_b=()
        )

    @contextlib.contextmanager
    def _fuzzy(self, value: float) -> Iterator[None]:
        """Set the boolean operations' fuzzy value for one call, and restore the kernel's."""
        option = "Geometry.ToleranceBoolean"
        before = self._g.option.getNumber(option)
        self._g.option.setNumber(option, float(value))
        try:
            yield
        finally:
            self._g.option.setNumber(option, before)

    # Edit
    def remove(self, faces: Sequence[Face]) -> None:
        """Delete the faces and what only they bound."""
        self._g.model.occ.remove(self._tags(faces), recursive=True)
        self._dirty = True

    def split_face_uv(self, face: Face, direction: str, value: float) -> tuple[Face, ...]:
        """Split ``face`` along the iso line ``direction`` at ``value`` (see the protocol)."""
        if direction not in ("u", "v"):
            raise InputValueError(f"a split direction is 'u' or 'v', not {direction!r}")
        spec = self._specs.get(self._parent.get(face.tag, -1))
        if spec is None:
            raise InputValueError(
                f"face {face.tag} was not built from a B-spline base by this kernel, so it has no "
                "parameter space to split in; split a face made by add_trimmed_face"
            )
        axis = 0 if direction == "u" else 1
        knots = spec["knots_u"] if axis == 0 else spec["knots_v"]
        low, high = float(knots[0]), float(knots[-1])
        if not low < value < high:
            raise InputValueError(
                f"the {direction} value {value} is not inside the base domain ({low}, {high})"
            )
        other = spec["knots_v"] if axis == 0 else spec["knots_u"]
        o0, o1 = float(other[0]), float(other[-1])
        corners = [(value, o0), (high, o0), (high, o1), (value, o1)]
        if axis == 1:
            corners = [(o0, value), (o1, value), (o1, high), (o0, high)]
        scratch = self.add_bspline_surface(**spec)
        upper = self.add_trimmed_face(
            scratch, [[[corners[k], corners[(k + 1) % 4]] for k in range(4)]]
        )
        self._parent[upper.tag] = self._parent[face.tag]
        occ = self._g.model.occ
        _, mapping = occ.fragment([(2, face.tag)], [(2, upper.tag)])
        mine = [t for d, t in mapping[0] if d == 2]
        cutter = {t for d, t in mapping[1] if d == 2}
        occ.remove([(2, t) for t in cutter if t not in mine], recursive=False)
        occ.remove([(2, scratch.tag)], recursive=False)
        self._dirty = True
        base = self._parent[face.tag]
        for tag in mine:
            self._parent[tag] = base
        lower = [Face(t) for t in mine if t not in cutter]
        return tuple(lower + [Face(t) for t in mine if t in cutter])

    def copy(self, faces: Sequence[Face]) -> tuple[Face, ...]:
        """Return copies of the faces, in order."""
        out = self._g.model.occ.copy(self._tags(faces))
        self._dirty = True
        return tuple(Face(t) for d, t in out if d == 2)

    def transform(self, faces: Sequence[Face], matrix: Sequence[Sequence[float]]) -> None:
        """Apply a 4 by 4 homogeneous matrix, row-major, to the faces in place."""
        flat = [float(matrix[r][c]) for r in range(3) for c in range(4)]
        self._g.model.occ.affineTransform(self._tags(faces), flat)
        self._dirty = True

    # Query
    def split_imported_face_uv(self, face: Face, direction: str, value: float) -> tuple[Face, ...]:
        """Split on an iso line of the face's own base, preserving holes and outer trims.

        A parameter-space rectangle on the same base supplies the upper half cutter.
        Fragment provenance retains only descendants of the input, so the rectangle
        cannot fill a hole or extend the original outer boundary.
        """
        if direction not in ("u", "v"):
            raise InputValueError(f"split direction {direction!r}: give 'u' or 'v'")
        if not np.isfinite(value):
            raise InputValueError(f"split value {value!r}: give a finite parameter")
        low, high = self.uv_bounds(face)
        axis = 0 if direction == "u" else 1
        if not low[axis] < value < high[axis]:
            return (face,)
        u0, v0 = (value, low[1]) if axis == 0 else (low[0], value)
        corners = [(u0, v0), (high[0], v0), high, (u0, high[1])]
        cutter = self.add_trimmed_face(
            face, [[[corners[i], corners[(i + 1) % 4]] for i in range(4)]]
        )
        result = self.fragment([face], [cutter])
        mine, upper = set(result.from_a[0]), set(result.from_b[0])
        excess = upper - mine
        if excess:
            self._g.model.occ.remove(self._tags(sorted(excess)), recursive=False)
            self._dirty = True
        return tuple(sorted(mine - upper) + sorted(mine & upper))

    def minimum_distance(self, a: Sequence[Face], b: Sequence[Face]) -> float:
        """Return the native minimum distance of trimmed face sets in meters, zero on contact."""
        if not a or not b:
            raise InputValueError("minimum distance: empty face set; give two nonempty sets")
        self._model()
        minimum = float("inf")
        for first in a:
            for second in b:
                distance = float(self._g.model.occ.getDistance(2, first.tag, 2, second.tag)[0])
                if not np.isfinite(distance) or distance < 0:
                    raise InputValueError(
                        f"minimum distance: kernel returned {distance!r} for faces "
                        f"{first.tag}, {second.tag}; check the face handles and geometry"
                    )
                minimum = min(minimum, distance)
        return minimum

    def faces(self) -> tuple[Face, ...]:
        """Return every face of the model, in creation order."""
        return tuple(Face(t) for _, t in self._model().getEntities(2))

    def boundary(self, face: Face) -> tuple[int, ...]:
        """Return the signed edge tags that bound ``face``, in loop order."""
        out = self._model().getBoundary([(2, face.tag)], combined=False, oriented=True)
        return tuple(int(t) for d, t in out if d == 1)

    def adjacent(self, edge: int) -> tuple[Face, ...]:
        """Return the faces that share ``edge``; more than two means a non-manifold edge."""
        up, _ = self._model().getAdjacencies(1, abs(edge))
        return tuple(Face(int(t)) for t in up)

    def area(self, face: Face) -> float:
        """Return the exact area of ``face``, in square meters."""
        self._model()
        return float(self._g.model.occ.getMass(2, face.tag))

    def _export_area_definition(self, face: Face) -> dict[str, Any]:
        """Read this face's full-precision native topology for the export area proof.

        BREP preserves the kernel coefficients and edge parameter ranges. An
        IGES re-export would round them and can conceal small trim deviations.
        Visibility and the live model are restored by the native writer.
        """
        self._model()
        with TemporaryDirectory(prefix="cfdb-area-") as temporary:
            path = Path(temporary) / "face.brep"
            _write_visible(self._g, {"area": [face.tag]}, path)
            return brep_definition(path.read_text(encoding="ascii"))

    def centroid(self, face: Face) -> Point:
        """Return the center of mass of ``face``."""
        self._model()
        x, y, z = self._g.model.occ.getCenterOfMass(2, face.tag)
        return (float(x), float(y), float(z))

    def bbox(self, face: Face) -> tuple[Point, Point]:
        """Return the axis-aligned bounding box of ``face`` as ``(minimum, maximum)``."""
        b = self._model().getBoundingBox(2, face.tag)
        return (b[0], b[1], b[2]), (b[3], b[4], b[5])

    def uv_bounds(self, face: Face) -> tuple[UV, UV]:
        """Return the parametric bounds of ``face`` as ``(minimum, maximum)``."""
        low, high = self._model().getParametrizationBounds(2, face.tag)
        return (low[0], low[1]), (high[0], high[1])

    def value(self, face: Face, uv: UV) -> Point:
        """Return the point of ``face`` at ``uv``."""
        x, y, z = self._model().getValue(2, face.tag, list(uv))
        return (float(x), float(y), float(z))

    def normal(self, face: Face, uv: UV) -> Point:
        """Return the unit normal of ``face`` at ``uv``."""
        x, y, z = self._model().getNormal(face.tag, list(uv))
        return (float(x), float(y), float(z))

    def derivative(self, face: Face, uv: UV) -> tuple[Point, Point]:
        """Return the partial derivatives ``(d/du, d/dv)`` of ``face`` at ``uv``."""
        d = self._model().getDerivative(2, face.tag, list(uv))
        return (d[0], d[1], d[2]), (d[3], d[4], d[5])

    def closest(self, face: Face, points: Sequence[Point]) -> tuple[tuple[Point, UV], ...]:
        """Return, for each point, the closest point of ``face`` and its ``uv``."""
        model = self._model()
        out = []
        for p in points:
            xyz, uv = model.getClosestPoint(2, face.tag, list(p))
            out.append(((xyz[0], xyz[1], xyz[2]), (uv[0], uv[1])))
        return tuple(out)

    def reparametrize(self, edge: int, face: Face, t: float) -> UV:
        """Return the ``(u, v)`` on ``face`` of the point of ``edge`` at its parameter ``t``."""
        out = self._model().reparametrizeOnSurface(1, abs(edge), [float(t)], face.tag)
        return (float(out[0]), float(out[1]))

    def shared_curve(self, face_a: Face, face_b: Face) -> int | None:
        """Return the one edge the two faces share, or ``None`` when they share none."""
        common = {abs(e) for e in self.boundary(face_a)} & {abs(e) for e in self.boundary(face_b)}
        if len(common) > 1:
            raise InputValueError(
                f"faces {face_a.tag} and {face_b.tag} share {len(common)} edges, so there is no "
                f"one shared curve: {sorted(common)}"
            )
        return common.pop() if common else None

    def is_inside(self, face: Face, uv: UV) -> bool:
        """Return whether ``uv`` lies inside the trimmed face, not only inside its bounds."""
        return bool(self._model().isInside(2, face.tag, list(uv), parametric=True))

    # Batched surface and curve queries
    def values(self, face: Face, uvs: Sequence[UV]) -> np.ndarray:
        """Return the point of ``face`` at each ``uv``, as an n by 3 array."""
        flat = np.asarray(uvs, float).ravel()
        return np.asarray(self._model().getValue(2, face.tag, flat)).reshape(-1, 3)

    def derivatives(self, face: Face, uvs: Sequence[UV]) -> np.ndarray:
        """Return ``(d/du, d/dv)`` at each ``uv``, as an n by 2 by 3 array."""
        flat = np.asarray(uvs, float).ravel()
        return np.asarray(self._model().getDerivative(2, face.tag, flat)).reshape(-1, 2, 3)

    def inside(self, face: Face, uvs: Sequence[UV]) -> np.ndarray:
        """Return whether each ``uv`` lies inside the trimmed face."""
        model = self._model()
        return np.array(
            [bool(model.isInside(2, face.tag, p, parametric=True)) for p in np.asarray(uvs, float)]
        )

    def project(self, face: Face, points: Sequence[Point]) -> tuple[np.ndarray, np.ndarray]:
        """Return the closest point of ``face`` to each point and its ``uv`` (arrays)."""
        return closest_on_face(self._model(), face.tag, np.asarray(points, float))

    def parametrize(self, face: Face, points: Sequence[Point]) -> np.ndarray:
        """Return the ``uv`` of each point lying on ``face``, as an n by 2 array."""
        flat = np.asarray(points, float).ravel()
        return np.asarray(self._model().getParametrization(2, face.tag, flat)).reshape(-1, 2)

    def edge_bounds(self, edge: int) -> tuple[float, float]:
        """Return the parameter range of ``edge``."""
        low, high = self._model().getParametrizationBounds(1, abs(edge))
        return float(low[0]), float(high[0])

    def edge_length(self, edge: int) -> float:
        """Return the native arc length of a signed boundary edge, in meters."""
        from scipy.integrate import quad

        model = self._model()
        low, high = self.edge_bounds(edge)

        def speed(t):
            return float(np.linalg.norm(model.getDerivative(1, abs(edge), [t])))

        # Numerical integration accuracy, above QUADPACK's 50-epsilon minimum;
        # this is not a physical junction tolerance.
        value, _ = quad(speed, low, high, epsabs=0.0, epsrel=64 * np.finfo(float).eps, limit=200)
        return float(value)

    def edge_values(self, edge: int, parameters: Sequence[float]) -> np.ndarray:
        """Return the point of ``edge`` at each parameter, as an n by 3 array."""
        values = np.asarray(parameters, float)
        return np.asarray(self._model().getValue(1, abs(edge), values)).reshape(-1, 3)

    def edge_project(self, edge: int, points: Sequence[Point]) -> np.ndarray:
        """Return the closest point of the curve to each point.

        The batched parameters of the projection are unreliable, so they are recovered from the
        projected points.
        """
        model = self._model()
        low, high = self.edge_bounds(edge)
        xyz, _ = model.getClosestPoint(1, abs(edge), np.asarray(points, float).ravel())
        parameter = model.getParametrization(1, abs(edge), xyz)
        return self.edge_values(edge, np.clip(parameter, low, high))

    def edge_ends(self, edge: int) -> tuple[Point, ...]:
        """Return the end vertices of ``edge``."""
        model = self._model()
        vertices = model.getBoundary([(1, abs(edge))], oriented=False)
        return tuple(
            (float(p[0]), float(p[1]), float(p[2]))
            for p in (model.getValue(0, abs(v), []) for _, v in vertices)
        )

    def edge_bbox(self, edge: int) -> tuple[Point, Point]:
        """Return the axis-aligned bounding box of ``edge`` as ``(minimum, maximum)``."""
        b = self._model().getBoundingBox(1, abs(edge))
        return (b[0], b[1], b[2]), (b[3], b[4], b[5])

    # Hull
    def triangulate(
        self, faces: Sequence[Face], size: float
    ) -> tuple[tuple[Point, ...], tuple[tuple[int, int, int], ...]]:
        """Triangulate the faces at element size ``size``: nodes, and triangles as node indices."""
        model = self._model()
        self._g.option.setNumber("Mesh.MeshSizeMax", float(size))
        self._g.option.setNumber("Mesh.MeshSizeMin", float(size) * 0.5)
        model.mesh.generate(2)
        tags, coords, _ = model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(tags)}
        nodes = tuple(
            (coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]) for i in range(len(tags))
        )
        triangles: list[tuple[int, int, int]] = []
        for face in faces:
            _, _, node_tags = model.mesh.getElements(2, face.tag)
            for chunk in node_tags:
                for k in range(0, len(chunk), 3):
                    triangles.append(
                        (index[int(chunk[k])], index[int(chunk[k + 1])], index[int(chunk[k + 2])])
                    )
        return nodes, tuple(triangles)

    # IO
    def import_iges(self, path: str | Path) -> NamedFaces:
        """Read one IGES file into the model, in meters, with the names the file gives.

        Every file arrives in meters, like every face the kernel builds, so faces of two files
        agree whatever unit each one states; the file's unit is returned as ``unit``. A file whose
        unit has no scale, or whose size does not match what it states, is refused.
        """
        listed = read_iges_names(path)
        _unit_of(listed)
        tags = import_iges(self._g, path)
        self._g.model.occ.synchronize()
        if len(tags) != len(listed.faces):
            raise InputArtifactError(
                f"{path}: the import returned {len(tags)} faces and the file's directory lists "
                f"{len(listed.faces)} surfaces, so no face can be named from it. Nothing was "
                "read.",
                kind="geometry",
            )
        factor = float(METRES_PER_UNIT[IMPORT_UNIT])
        occ = self._g.model.occ
        occ.dilate([(2, t) for t in tags], 0.0, 0.0, 0.0, factor, factor, factor)
        occ.synchronize()
        faces = tuple(Face(t) for t in tags)
        self._check_extent(path, faces, listed.unit)
        return NamedFaces(faces=faces, names=tuple(listed.faces), unit=listed.unit)

    def inspect_file(self, path: str | Path, tolerance: float | None) -> FileTopology:
        """Read one IGES file alone in a model of its own, sewn when ``tolerance`` is given.

        The model of the kernel is not touched: the file goes to a scratch model that is removed
        afterwards. A face keeps the name of the face of the file nearest to it.
        """
        listed = read_iges_names(path)
        _unit_of(listed)
        self._model()
        current = self._g.model.getCurrent()
        self._g.model.add("inspect")
        try:
            occ = self._g.model.occ
            tags = import_iges(self._g, path)
            occ.synchronize()
            factor = float(METRES_PER_UNIT[IMPORT_UNIT])
            occ.dilate([(2, t) for t in tags], 0.0, 0.0, 0.0, factor, factor, factor)
            occ.synchronize()
            named = list(listed.faces) if len(listed.faces) == len(tags) else [""] * len(tags)
            centres = np.array([occ.getCenterOfMass(2, t) for t in tags]).reshape(-1, 3)
            if tolerance is not None:
                occ.healShapes([], float(tolerance), True, False, False, True, False)
                occ.synchronize()
            return _topology(self._g, named, centres)
        finally:
            self._g.model.remove()
            self._g.model.setCurrent(current)

    def _check_extent(self, path: str | Path, faces: Sequence[Face], unit: str | None) -> None:
        """Refuse a read-back whose size is far from the largest coordinate the file states.

        The scale above rests on what unit the import returns. A kernel that returned another one
        would give a model off by a factor of a thousand with no sign, so the file's own number
        (global parameter 20) is compared with what arrived.
        """
        stated = _stated_extent(path)
        if stated is None or not faces or unit is None:
            return
        expected = stated * float(METRES_PER_UNIT[unit])
        boxes = [self.bbox(face) for face in faces]
        measured = max(abs(c) for box in boxes for corner in box for c in corner)
        if expected > 0 and measured > 0 and not 0.01 < measured / expected < 100:
            raise InputArtifactError(
                f"{path}: the file states a largest coordinate of {expected} m and the import "
                f"returned {measured} m, so its unit was not read as stated. Nothing was read.",
                kind="geometry",
            )

    def export_iges(
        self,
        groups: Mapping[str, Sequence[Face]],
        path: str | Path,
        *,
        preference: int = 1,
        unit: str = "m",
    ) -> None:
        """Write both boundary curves with PREF 1; preferences 2 and 3 warn and normalize."""
        self._model()
        write_iges(
            self._g,
            {n: [f.tag for f in fs] for n, fs in groups.items()},
            path,
            preference=preference,
            unit=unit,
        )


def _topology(gmsh: Any, named: Sequence[str], centres: np.ndarray) -> FileTopology:
    """Return the faces and edges of the current model, each face named from ``named``."""
    model, occ = gmsh.model, gmsh.model.occ
    faces = [int(t) for _, t in model.getEntities(2)]
    index = {t: i for i, t in enumerate(faces)}
    names = []
    for t in faces:
        if not named:
            names.append("")
            continue
        centre = np.asarray(occ.getCenterOfMass(2, t))
        names.append(named[int(np.argmin(np.linalg.norm(centres - centre, axis=1)))])
    perimeters = []
    for t in faces:
        bound = model.getBoundary([(2, t)], oriented=False)
        perimeters.append(float(sum(occ.getMass(1, abs(e)) for _, e in bound)))
    edges = []
    for _, e in model.getEntities(1):
        up, _ = model.getAdjacencies(1, e)
        low, high = model.getParametrizationBounds(1, e)
        samples = np.asarray(model.getValue(1, e, np.linspace(low[0], high[0], 17))).reshape(-1, 3)
        edges.append((samples, tuple(index[int(t)] for t in up if int(t) in index)))
    return FileTopology(
        names=tuple(names),
        areas=tuple(float(occ.getMass(2, t)) for t in faces),
        perimeters=tuple(perimeters),
        edges=tuple(edges),
    )


def open_kernel(purpose: str) -> GmshKernel:
    """Return the configured CAD kernel, initialized, with its terminal output off.

    The caller closes it. The initialization is not interruptible, so it also works off the main
    thread; the library is still one process-wide state, so two kernels in one process share it
    and one :meth:`GmshKernel.close` ends both. Every kernel stage therefore runs in its own
    process (:mod:`cfdbender.mesh._gmsh_worker`).
    """
    gmsh = gmsh_module(purpose)
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add("kernel")
    return GmshKernel(gmsh)

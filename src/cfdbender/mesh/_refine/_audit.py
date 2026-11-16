"""The audit of a panel mesh, alone or against its source.

THREE KINDS OF RESULT. A GATE is judged on its own terms: the topology,
the orientation, the open boundary, the trailing edge, the shared nodes and
the faces a grid family at factor 1 keeps. A RELATIVE CHECK compares a
95th percentile of the level with the source's, because the source is the
accepted mesh. A FIGURE is reported and never judged. Without a
source only G1, G2 and G4 are judged and the rest is reported.

HOW IT MEASURES. Every measure is computed once on all the faces of a mesh
and read per family through the family of each face; an edge belongs to a
family when both its faces do. The audit needs numpy alone and imports
nothing of the refinement modules, so it judges a level
without them.

WHAT IT WRITES. ``<stem>.audit.json`` beside the mesh audited (a level's
stem is ``<stem>_<tag>``), with ``schema_version``, through the package's one
text route; a gate or check that fails is a warning naming it, the family and
its values. A refinement record names the level and source points files explicitly,
independent of marking declarations. Without that record, the points file is the one its
``<stem>.boundaries.toml`` names under ``[trailing_edges] file``, or else
``<stem>.te.txt`` beside it; its points are converted from the unit on its
first line to the mesh's (the ``[import] units`` of the boundaries file)
before they are matched to the mesh's edges. G4 compares the level's
trailing-edge chains with the source's; a source family whose trailing edge
the refinement detected has no points file to count, so the
record beside the level (``refine.json``, read when it names this level and
the source, as for G6) states it ``"trailing_edge": "detected"`` with the
mid-points of the detected line's edges (``"trailing_edge_points"``, in the
unit of the level's points file), and the source's chains are the connected
components of its listed edges and those edges together, so a detected line
that a listed edge continues is one chain. A family the record states
detected without its points (a record written by hand) counts one chain
when no listed point lies on its edges.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh._refine._geometry import (
    ASPECT_LIMIT,
    CHECK_TOLERANCE,
    DUPLICATE_FRACTION,
    GROWTH_DIHEDRAL_DEGREES,
    GROWTH_FLOOR,
    OPENING_FRACTION,
    PANEL_PRACTICE_COUNTS,
    PERCENTILE,
    QUALITY_GOOD,
    SCHEMA_VERSION,
    SKEWNESS_MARGIN,
    TE_POINT_FRACTION,
    WARP_FLOOR_DEGREES,
    ZERO_AREA_FRACTION,
    NearestIndex,
    below_family_area,
    boundary_edges,
    boundary_loops,
    close_pairs,
    degenerate,
    dihedral_degrees,
    edge_faces,
    normals_and_centroids,
    segment_distances,
    triangles,
)
from cfdbender.mesh._refine._obj import (
    KIND,
    TE_SUFFIX,
    ObjMesh,
    TrailingPoints,
    edge_midpoints,
    points_from_file,
    read_obj,
    te_points,
    te_route,
)
from cfdbender.mesh._sidecars import inventory_sidecar

Faces = list[list[int]]
Value = float | int | str | bool | None
Array = NDArray[Any]

#: The family name a result on all the faces of the mesh carries.
WHOLE_MESH = "(whole mesh)"
#: The verdicts of a gate or a check.
PASS, FAIL, NOT_JUDGED = "pass", "fail", "not judged"
#: The suffix of the audit written beside the mesh.
AUDIT_SUFFIX = ".audit.json"
#: The suffix of a level's refinement record, beside it, which the audit reads.
RECORD_SUFFIX = ".refine.json"
#: The relative checks of the name and the figure each compares.
CHECKS = (("skewness", "skewness_p95"), ("warp", "warp_p95"), ("growth", "growth_p95"))
_TINY = 1e-300


@dataclass(frozen=True)
class AuditItem:
    """One gate or relative check: its name, the family, its values and its verdict."""

    kind: str
    name: str
    family: str
    values: Mapping[str, Value]
    verdict: str

    @property
    def passed(self) -> bool:
        """Return whether the item did not fail (a result not judged did not fail)."""
        return self.verdict != FAIL

    def warning(self) -> str:
        """Return the warning of a failed item: its name, the family and its values."""
        said = ", ".join(f"{key} {_said(value)}" for key, value in self.values.items())
        return f"audit {self.kind} {self.name} failed on {self.family}: {said}"

    def line(self) -> str:
        """Return the console line of the item."""
        said = ", ".join(f"{key} {_said(value)}" for key, value in self.values.items())
        return f"{self.name} {self.verdict} on {self.family}: {said}"

    def as_json(self) -> dict[str, object]:
        """Return the item as written in ``audit.json``."""
        return {
            "name": self.name,
            "family": self.family,
            "verdict": self.verdict,
            "values": dict(self.values),
        }


@dataclass(frozen=True)
class MeshAudit:
    """The audit of one mesh, alone or against its source.

    Attributes
    ----------
    mesh : Path
        The OBJ audited.
    source : Path or None
        The source it was audited against, or None.
    gates : tuple of AuditItem
        G1 to G6, each with its family, its values and its verdict.
    checks : tuple of AuditItem
        The relative checks of, per family and on the whole mesh.
    figures : Mapping
        Per family (and ``"(whole mesh)"``), each reported figure and the
        95th percentiles the checks read.
    path : Path
        The ``<stem>.audit.json`` written beside the mesh.
    degenerate_faces : tuple of Mapping
        Each face G1 counts degenerate: its ``family``, its
        ``face`` index in that family (from 0, in file order), its
        ``centroid``, its ``area_ratio`` to its family's median face area and
        its ``min_angle_degrees``.
    """

    mesh: Path
    source: Path | None
    gates: tuple[AuditItem, ...]
    checks: tuple[AuditItem, ...]
    figures: Mapping[str, Mapping[str, Value]]
    path: Path
    degenerate_faces: tuple[Mapping[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether every gate and every check passed or was not judged."""
        return not self.failures

    @property
    def failures(self) -> tuple[AuditItem, ...]:
        """Return the gates and checks that failed, gates first."""
        return tuple(item for item in self.gates + self.checks if not item.passed)

    def as_json(self) -> dict[str, object]:
        """Return the audit as written in ``audit.json`` (``schema_version`` 1)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "mesh": self.mesh.name,
            "source": None if self.source is None else self.source.name,
            "passed": self.passed,
            "gates": [item.as_json() for item in self.gates],
            "checks": [item.as_json() for item in self.checks],
            "figures": {family: dict(values) for family, values in self.figures.items()},
            "degenerate_faces": [dict(face) for face in self.degenerate_faces],
        }

    def summary(self) -> list[str]:
        """Return the console summary: each gate, the checks counted, each failure."""
        against = "alone" if self.source is None else f"against {self.source.name}"
        lines = [f"audit of {self.mesh.name} {against}"]
        lines += [item.line() for item in self.gates]
        counts = Counter(item.verdict for item in self.checks)
        lines.append(
            f"relative checks: {counts[PASS]} pass, {counts[FAIL]} fail, "
            f"{counts[NOT_JUDGED]} not judged"
        )
        lines += [item.line() for item in self.checks if not item.passed]
        unjudged = sum(1 for item in (*self.gates, *self.checks) if item.verdict == NOT_JUDGED)
        if not self.passed:
            lines.append("the audit FAILED")
        elif unjudged:
            lines.append(f"every gate and check judged passed; {unjudged} not judged yet")
        else:
            lines.append("every gate and check passed")
        return lines

    @staticmethod
    def files_read(
        mesh: str | os.PathLike[str], against: str | os.PathLike[str] | None = None
    ) -> list[tuple[str, Path]]:
        """Return the files an audit of ``mesh`` reads, each with what it is.

        The one list every caller uses. The OBJ, its boundaries
        file, its refinement record (``<stem>.refine.json``), ``<stem>.te.txt`` and the
        trailing-edge points file the boundaries file names, read through the one
        reader of the boundaries file the refinement and the audit use, so a command
        that must not write over an input names the same files. With ``against``,
        the source's OBJ, boundaries file and points files follow. A boundaries
        file that cannot be read names no points file here; the audit refuses
        it on its own.

        Parameters
        ----------
        mesh : path-like
            The OBJ an audit would read.
        against : path-like, optional
            The source the audit would compare it with.

        Returns
        -------
        list of (str, pathlib.Path)
            What each file is, and its path; a file may not exist.

        Examples
        --------
        >>> from cfdbender.mesh._refine import MeshAudit
        >>> [what for what, _ in MeshAudit.files_read("W1/W1.obj")][:2]
        ['the OBJ', 'its boundaries file']
        """
        return _inputs(Path(mesh), None if against is None else Path(against))

    def write_csv(self, path: str | Path) -> Path:
        """Write one row per family and figure (``--out``, alias ``--csv``) and return the path.

        Raises
        ------
        InputArtifactError
            The path is a file this audit reads (:meth:`files_read` of the
            mesh and of its source) or the audit JSON it wrote; nothing was
            written.
        """
        target = Path(path)
        for what, read in [*self.files_read(self.mesh, self.source), ("the audit JSON", self.path)]:
            if _same_file(target, read):
                raise _refusal(
                    target,
                    f"the CSV would be written over {what}, {read}, which the audit reads or wrote",
                    "Name another file and run again.",
                )
        rows = (
            [family, figure, "" if value is None else str(value)]
            for family, values in self.figures.items()
            for figure, value in values.items()
        )
        _textio.write_csv(target, rows, header=("family", "figure", "value"))
        return target


@dataclass(frozen=True)
class _Trailing:
    """The trailing edge of a mesh: its points, those on an edge, its chains, its faces."""

    present: bool
    points: int
    on_edges: int
    chains: int
    faces: Array
    # G4 completeness: where the chains end, and their total length
    line: _Line | None = None
    # G4: points beyond the first that match an edge another point matches
    sharing: int = 0


@dataclass(frozen=True)
class _Line:
    """The trailing-edge line of a mesh: the positions its chains end at, and its length."""

    ends: Array
    length: float


@dataclass(frozen=True)
class _Mesh:
    """Every measure of one mesh, computed once on all its faces."""

    obj: ObjMesh
    faces: Faces
    family: Array
    size: float
    shape: Mapping[str, Array]
    pair: Array
    dihedral: Array
    flipped: int
    nonmanifold: int
    trailing: _Trailing
    # G4: the families the record states detected, and the chains of the listed and
    # detected edges together; None without a record that names the level and its source
    detected: int | None = None
    chains: int | None = None
    # and the line those edges form together, None in the same case
    line: _Line | None = None
    # G4: the fraction of the size within which a trailing-edge point names an edge
    fraction: float = TE_POINT_FRACTION


def audit_mesh(
    mesh: str | Path,
    *,
    against: str | Path | None = None,
    unchanged_grids: Sequence[str] = (),
) -> MeshAudit:
    """Audit an OBJ, alone or against its source, and write its audit beside it.

    Parameters
    ----------
    mesh : str or Path
        The OBJ to audit; its trailing-edge points file is the one its
        ``<stem>.boundaries.toml`` names, or else ``<stem>.te.txt``.
    against : str or Path, optional
        The source the mesh was made from. Without it only G1, G2 and G4 are
        judged and the rest is reported.
    unchanged_grids : sequence of str, optional
        The grid families a refinement wrote at factor 1, whose faces G6
        compares with the source's. Without them they are read from the
        level's ``<stem>.refine.json`` when it names this level and the
        source ``against`` (its ``unchanged_grids``), so a saved level is
        re-audited as the refinement audited it; with neither, G6 is not
        judged.

    Returns
    -------
    MeshAudit
        Each gate and check with its values and verdict, the figures, and
        ``passed``. ``<stem>.audit.json`` is written beside the mesh whatever
        the verdict, and each failed gate or check is a warning.

    Raises
    ------
    InputArtifactError
        A mesh or source that is not a readable OBJ, or a trailing-edge points
        file named and missing or malformed; nothing is written.
    """
    record = _record_of(Path(mesh))
    point_files = _refinement_points(record, Path(mesh), against)
    fraction = _recorded_fraction(record, Path(mesh))
    level = _read(Path(mesh), point_files=point_files, fraction=fraction)
    # like the grids and the detected edges, the components are read only from a
    # record that names this level and this source.
    components = _components_of(record) if _names_this_level(record, Path(mesh), against) else {}
    detected = _recorded_detected(record, Path(mesh), against)
    source = (
        None
        if against is None
        else _read(Path(against), components, detected, point_files, fraction)
    )
    target = Path(mesh).with_name(Path(mesh).stem + AUDIT_SUFFIX)
    _refuse_an_input_as_target(target, Path(mesh), None if against is None else Path(against))
    grids = tuple(unchanged_grids) or _recorded_grids(record, Path(mesh), against)
    gates = _gates(level, source, grids)
    figures = dict(_all_figures(level))
    checks = tuple(_checks(figures, None if source is None else dict(_all_figures(source))))
    audit = MeshAudit(
        mesh=Path(mesh),
        source=None if against is None else Path(against),
        gates=gates,
        checks=checks,
        figures=MappingProxyType(
            {name: MappingProxyType(values) for name, values in figures.items()}
        ),
        path=target,
        degenerate_faces=_degenerate_faces(level),
    )
    _textio.write_json(target, audit.as_json())
    for item in audit.failures:
        warnings.warn(item.warning(), GeometryWarning, stacklevel=2)
    return audit


def _refusal(subject: Path, reason: str, remedy: str) -> InputArtifactError:
    """Return a refusal naming the file, the reason and the remedy; nothing was written."""
    return InputArtifactError(f"{subject}: {reason}. {remedy} Nothing was written.", kind=KIND)


def _inputs(mesh: Path, against: Path | None) -> list[tuple[str, Path]]:
    """Return the files the audit of ``mesh`` reads, each with what it is.

    The OBJ, its boundaries file, its trailing-edge points file and its
    ``refine.json``; with ``against``, the source, its boundaries file and
    its points file. It is the one list: :meth:`MeshAudit.files_read` returns it
    and the audit's own refusal reads it. A file may not
    exist, and a boundaries file that cannot be read names no points file
    here; the audit refuses it on its own.
    """
    files = [
        ("the OBJ", mesh),
        ("its boundaries file", inventory_sidecar(mesh)),
        ("its refinement record", mesh.with_name(mesh.stem + RECORD_SUFFIX)),
    ]
    files += [("its trailing-edge points file", p) for p in _points_files(mesh)]
    if against is not None:
        files += [
            ("the source", against),
            ("the source's boundaries file", inventory_sidecar(against)),
        ]
        files += [("the source's trailing-edge points file", p) for p in _points_files(against)]
    chosen = _refinement_points(_record_of(mesh), mesh, against)
    if chosen is not None:
        files += [("the refinement's points input", path) for path in chosen.values() if path]
    return files


def _points_files(obj: Path) -> list[Path]:
    """Return the trailing-edge points files a mesh could be read with: none for ``none = true``.

    The file its boundaries file names and ``<stem>.te.txt`` beside the mesh.
    """
    try:
        route = te_route(obj)
    except (InputArtifactError, OSError):
        return [obj.with_name(obj.stem + TE_SUFFIX)]
    if route.none:
        return []
    files = [obj.with_name(obj.stem + TE_SUFFIX)]
    return files if route.file is None else [*files, route.file]


def _same_file(a: Path, b: Path) -> bool:
    """Return whether two paths name one file: one spelling, or one file on disk."""
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
        return os.path.normcase(a.resolve()) == os.path.normcase(b.resolve())
    except OSError:
        return False


def _refuse_an_input_as_target(target: Path, mesh: Path, against: Path | None) -> None:
    """Refuse an audit JSON that is a file the audit reads, before anything is written."""
    for what, path in _inputs(mesh, against):
        if _same_file(target, path):
            raise _refusal(
                target,
                f"the audit JSON would be written over {what}, {path}, which the audit reads",
                "Rename or move the mesh, or the file it would overwrite, and run again.",
            )


def _read(
    path: Path,
    components: Mapping[str, Sequence[str]] | None = None,
    detected: Mapping[str, list[list[float]] | None] | None = None,
    point_files: Mapping[Path, Path | None] | None = None,
    fraction: float = TE_POINT_FRACTION,
) -> _Mesh:
    """Read an OBJ and its trailing-edge points and measure them.

    With ``components``, the families each component lists are measured as
    that one family, so a level whose families were grouped into components is compared with the
    union of its members in the source.
    ``detected`` maps the families whose trailing edge the refinement
    detected to their recorded mid-points (:func:`_recorded_detected`); the
    chains of the listed and the detected edges together are counted once
    (:func:`_g4`). None, without a record, contributes no detected chains
    to G4. A trailing-edge point lies on the edge whose mid-point is within
    ``fraction`` of the mesh size.
    """
    if not path.is_file():
        raise _refusal(path, "no such OBJ file", "Name an existing OBJ and run again.")
    obj = read_obj(path)
    te = (
        te_points(path)
        if point_files is None or path not in point_files
        else points_from_file(path, point_files.get(path))
    )
    points = None if te is None else te.points
    measured = _measure(_grouped(obj, components) if components else obj, points, fraction)
    if detected is None:
        return measured
    count = sum(n in obj.families for n in detected)
    chains, line = _union_chains(obj, detected, te, fraction)
    return replace(measured, detected=count, chains=chains, line=line)


def _union_chains(
    obj: ObjMesh,
    detected: Mapping[str, list[list[float]] | None],
    te: TrailingPoints | None,
    fraction: float = TE_POINT_FRACTION,
) -> tuple[int, _Line | None]:
    """Return the chains of the listed and the recorded detected edges together, and their line.

    The recorded mid-points are in the unit of the level's points file, the
    source's points file's (``to_file``), else the mesh's. A family the
    record states detected without its points counts one chain when no
    listed point lies on its edges, and leaves the line unknown (None).
    """
    points = numpy.zeros((0, 3)) if te is None else te.points
    named = {n: p for n, p in detected.items() if n in obj.families}
    unlisted = sum(p is None and not _listed_on(obj, n, points, fraction) for n, p in named.items())
    scale = 1.0 if te is None else te.to_file
    union = [points] + [
        numpy.array(p, dtype=float).reshape(-1, 3) / scale for p in named.values() if p
    ]
    faces = [f for fs in obj.families.values() for f in fs]
    on = _on_edges(obj.verts, faces, numpy.vstack(union), fraction * obj.size)
    return _components(on) + unlisted, None if unlisted else _line(obj.verts, on)


def _line(verts: Array, edges: Array) -> _Line:
    """Return where the chains of trailing-edge edges end, and the edges' total length (G4).

    An end is a node of exactly one of the edges; a closed chain has none.
    """
    if not len(edges):
        return _Line(numpy.zeros((0, 3)), 0.0)
    # F-303: an edge listed twice (a duplicated row, or one both listed and detected)
    # is one edge; counted twice it gave its ends degree 2 and erased them.
    edges = numpy.unique(numpy.sort(edges, axis=1), axis=0)
    nodes, degree = numpy.unique(edges.reshape(-1), return_counts=True)
    length = float(numpy.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1).sum())
    return _Line(verts[nodes[degree == 1]], length)


def _listed_on(
    obj: ObjMesh, family: str, points: Array | None, fraction: float = TE_POINT_FRACTION
) -> bool:
    """Return whether a listed trailing-edge point lies on an edge of the family (G4)."""
    if points is None or not len(points) or not obj.families[family]:
        return False
    _, mids = edge_midpoints(obj.verts, obj.families[family])
    distance, _ = NearestIndex(mids).query(points)
    return bool((distance <= fraction * obj.size).any())


def _record_of(level: Path) -> dict[str, Any]:
    """Return the level's ``refine.json`` (an empty record when there is none or it is not JSON)."""
    path = level.with_name(level.stem + RECORD_SUFFIX)
    if not path.is_file():
        return {}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        record = None
    if not isinstance(record, dict):
        warnings.warn(
            f"{path}: not readable as a refinement record, so G4 and G6 are judged "
            "without what it records.",
            GeometryWarning,
            stacklevel=2,
        )
        return {}
    return record


def _refinement_points(
    record: Mapping[str, Any], level: Path, against: str | Path | None
) -> dict[Path, Path | None] | None:
    """Resolve the points selected by a refine, independently of marking declarations."""
    files = record.get("trailing_edge_files")
    if record.get("level") != level.stem or not isinstance(files, dict):
        return None
    paths = {"level": level}
    if against is not None and record.get("source") == Path(against).name:
        paths["source"] = Path(against)
    if any(files.get(key) is not None and not isinstance(files[key], str) for key in paths):
        return None
    return {
        path: path.parent / files[key] if files.get(key) else None for key, path in paths.items()
    }


def _components_of(record: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return the components a refinement recorded in the level's ``refine.json``, if any."""
    found = record.get("components")
    if not isinstance(found, dict):
        return {}
    return {
        str(c): [str(m) for m in members]
        for c, members in found.items()
        if isinstance(members, list)
    }


def _recorded_grids(
    record: Mapping[str, Any], level: Path, against: str | Path | None
) -> tuple[str, ...]:
    """Return the grid families the refinement wrote at factor 1, which G6 compares.

    They are read only from a record that names this level's stem and the
    source it is audited against, so a record of another refinement is not
    taken for this one's.
    """
    found = record.get("unchanged_grids")
    if not _names_this_level(record, level, against) or not isinstance(found, list):
        return ()
    return tuple(n for n in found if isinstance(n, str))


def _names_this_level(record: Mapping[str, Any], level: Path, against: str | Path | None) -> bool:
    """Return whether the record is the refinement of this level from the source ``against``."""
    if against is None or record.get("level") != level.stem:
        return False
    return record.get("source") == Path(against).name


def _recorded_fraction(record: Mapping[str, Any], level: Path) -> float:
    """Return the trailing-edge point fraction the level's refinement used (G4).

    A record that names this level and states a number between 0 and 1 gives
    it; any other record, or none, gives the default ``TE_POINT_FRACTION``.
    """
    found = record.get("trailing_edge_point_fraction")
    if (
        record.get("level") != level.stem
        or isinstance(found, bool)
        or not isinstance(found, int | float)
        or not math.isfinite(found)
        or not 0 < found < 1
    ):
        return TE_POINT_FRACTION
    return float(found)


def _recorded_detected(
    record: Mapping[str, Any], level: Path, against: str | Path | None
) -> dict[str, list[list[float]] | None] | None:
    """Return the source families whose trailing edge the refinement detected (G4).

    They are the families the record states with ``"trailing_edge":
    "detected"``, read only from a record that names this level and the
    source it is audited against, as for G6, each with its
    ``"trailing_edge_points"`` (None when the record holds no list of
    ``[x, y, z]`` numbers); None without such a record.
    """
    families = record.get("families")
    if not _names_this_level(record, level, against):
        return None
    if not isinstance(families, dict):
        return {}
    return {
        str(name): _recorded_points(entry.get("trailing_edge_points"))
        for name, entry in families.items()
        if isinstance(entry, dict) and entry.get("trailing_edge") in ("detected", "perimeter")
    }


def _recorded_points(found: object) -> list[list[float]] | None:
    """Return recorded mid-points as rows of three finite numbers, else None (G4)."""
    if not isinstance(found, list):
        return None
    rows = []
    for row in found:
        if not isinstance(row, list) or len(row) != 3 or not all(_finite(x) for x in row):
            return None
        rows.append([float(x) for x in row])
    return rows


def _finite(x: object) -> bool:
    """Return whether a JSON value is a finite number (a bool is not)."""
    return isinstance(x, int | float) and not isinstance(x, bool) and bool(numpy.isfinite(x))


def _grouped(obj: ObjMesh, components: Mapping[str, Sequence[str]]) -> ObjMesh:
    """Return the mesh with each component's member families merged, in the source's order."""
    holder = {m: c for c, members in components.items() for m in members}
    families: dict[str, list[list[int]]] = {}
    for name, faces in obj.families.items():
        families.setdefault(holder.get(name, name), []).extend(faces)
    return ObjMesh(obj.verts, families, obj.header, obj.family_tag)


def _measure(obj: ObjMesh, points: Array | None, fraction: float = TE_POINT_FRACTION) -> _Mesh:
    """Measure every face and interior edge of a mesh once."""
    faces = [f for fs in obj.families.values() for f in fs]
    family = numpy.repeat(
        numpy.arange(len(obj.families)), [len(fs) for fs in obj.families.values()]
    )
    ef = edge_faces(faces)
    keys = [e for e, fs in ef.items() if len(fs) == 2]
    directed = Counter((a, b) for f in faces for a, b in zip(f, f[1:] + f[:1], strict=True))
    dihedral = dihedral_degrees(obj.verts, faces)
    size = obj.size
    return _Mesh(
        obj=obj,
        faces=faces,
        family=family,
        size=size,
        shape=_face_shapes(obj.verts, faces),
        pair=numpy.array([ef[e] for e in keys], dtype=numpy.int64).reshape(-1, 2),
        dihedral=numpy.array([dihedral[e] for e in keys], dtype=float),
        flipped=sum(directed[e] != 1 for e in keys),
        nonmanifold=sum(len(fs) > 2 for fs in ef.values()),
        trailing=_trailing(obj.verts, faces, ef, points, fraction * size),
        fraction=fraction,
    )


def _face_shapes(verts: Array, faces: Faces) -> dict[str, Array]:
    """Return every per-face measure, polygons of one vertex count at a time."""
    n = len(faces)
    shape = {
        key: numpy.full(n, numpy.nan)
        for key in ("aspect", "skewness", "warp", "quad_min_angle", "tri_min_angle", "min_angle")
    }
    shape.update(perimeter=numpy.zeros(n), max_abs_y=numpy.zeros(n))
    sides = numpy.array([len(f) for f in faces])
    for count in numpy.unique(sides):
        rows = numpy.nonzero(sides == count)[0]
        _polygon_shapes(verts[numpy.array([faces[r] for r in rows])], rows, shape)
    normals, _ = normals_and_centroids(verts, faces)
    shape["area"] = 0.5 * numpy.linalg.norm(normals, axis=1)
    shape["sides"] = sides
    return shape


def _polygon_shapes(p: Array, rows: Array, shape: dict[str, Array]) -> None:
    """Fill the measures of polygons of one vertex count (``p`` is faces by vertices by 3)."""
    edge = numpy.roll(p, -1, axis=1) - p
    length = numpy.linalg.norm(edge, axis=2)
    shape["aspect"][rows] = length.max(axis=1) / numpy.maximum(length.min(axis=1), _TINY)
    shape["perimeter"][rows] = length.sum(axis=1)
    shape["max_abs_y"][rows] = numpy.abs(p[:, :, 1]).max(axis=1)
    before = numpy.roll(edge, 1, axis=1)
    cosine = -(edge * before).sum(axis=2) / numpy.maximum(
        length * numpy.roll(length, 1, axis=1), _TINY
    )
    angle = numpy.degrees(numpy.arccos(numpy.clip(cosine, -1.0, 1.0)))
    n = p.shape[1]
    ideal = 180.0 * (n - 2) / n
    shape["skewness"][rows] = numpy.maximum(
        (angle.max(axis=1) - ideal) / (180.0 - ideal), (ideal - angle.min(axis=1)) / ideal
    )
    shape["min_angle"][rows] = angle.min(axis=1)
    if n == 3:
        shape["tri_min_angle"][rows] = angle.min(axis=1)
    if n == 4:
        shape["quad_min_angle"][rows] = angle.min(axis=1)
        shape["warp"][rows] = _warp(p)


def _warp(p: Array) -> Array:
    """Return each quadrilateral's warp: the larger fold between the triangles of a diagonal."""
    folds = []
    for d in (0, 1):
        a, b, c, e = (p[:, (d + k) % 4] for k in range(4))
        n1 = numpy.cross(b - a, c - a)
        n2 = numpy.cross(c - a, e - a)
        cosine = (n1 * n2).sum(axis=1) / numpy.maximum(
            numpy.linalg.norm(n1, axis=1) * numpy.linalg.norm(n2, axis=1), _TINY
        )
        folds.append(numpy.degrees(numpy.arccos(numpy.clip(cosine, -1.0, 1.0))))
    return numpy.maximum(folds[0], folds[1])


def _trailing(
    verts: Array,
    faces: Faces,
    ef: Mapping[tuple[int, int], list[int]],
    points: Array | None,
    tolerance: float,
) -> _Trailing:
    """Match each trailing-edge point to the mesh edge whose mid-point it is (G4)."""
    touched = numpy.zeros(len(faces), dtype=bool)
    if points is None or not len(points):
        return _Trailing(points is not None, 0, 0, 0, touched, _line(verts, _NO_EDGES))
    on = _on_edges(verts, faces, points, tolerance)
    for a, b in on:
        touched[ef[(int(a), int(b))]] = True
    sharing = len(on) - len({(int(a), int(b)) for a, b in on})
    return _Trailing(
        True, len(points), len(on), _components(on), touched, _line(verts, on), sharing
    )


_NO_EDGES = numpy.zeros((0, 2), dtype=numpy.int64)


def _on_edges(verts: Array, faces: Faces, points: Array, tolerance: float) -> Array:
    """Return, per trailing-edge point within ``tolerance`` of an edge mid-point, that edge (G4)."""
    if not len(points) or not faces:
        return numpy.zeros((0, 2), dtype=numpy.int64)
    edges, mids = edge_midpoints(verts, faces)
    distance, nearest = NearestIndex(mids).query(points)
    return edges[nearest[distance <= tolerance]]


def _components(edges: Array) -> int:
    """Return the number of connected chains the edges form."""
    parent: dict[int, int] = {}

    def root(v: int) -> int:
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        parent[root(int(a))] = root(int(b))
    return len({root(v) for v in list(parent)})


def _p95(values: Array) -> float | None:
    """Return the 95th percentile (linear interpolation between ranks), or None when empty."""
    values = values[~numpy.isnan(values)]
    return float(numpy.percentile(values, PERCENTILE, method="linear")) if len(values) else None


def _selections(m: _Mesh) -> Iterator[tuple[str, Array, Array]]:
    """Yield each family, then the whole mesh, with its face mask and its edge mask."""
    for k, name in enumerate(m.obj.families):
        inside = m.family == k
        yield name, inside, inside[m.pair[:, 0]] & inside[m.pair[:, 1]]
    yield WHOLE_MESH, numpy.ones(len(m.faces), dtype=bool), numpy.ones(len(m.pair), dtype=bool)


def _all_figures(m: _Mesh) -> Iterator[tuple[str, dict[str, Value]]]:
    """Yield each family's figures, then the whole mesh's."""
    growth, quality = _neighbour_ratios(m)
    for name, faces, edges in _selections(m):
        yield name, _figures(m, faces, edges, growth, quality)


def _neighbour_ratios(m: _Mesh) -> tuple[Array, Array]:
    """Return the area ratio of every smooth interior edge and each face's quality ratio.

    The size growth is the larger area over the smaller across an edge whose
    dihedral is below the growth limit, NaN elsewhere; the face quality ratio
    is the largest ratio of inscribed radii (twice the area over the
    perimeter) between a face and its neighbours, 1 for a face with none.
    """
    area = m.shape["area"]
    a, b = m.pair[:, 0], m.pair[:, 1]
    growth = numpy.maximum(area[a], area[b]) / numpy.maximum(numpy.minimum(area[a], area[b]), _TINY)
    growth[m.dihedral >= GROWTH_DIHEDRAL_DEGREES] = numpy.nan
    radius = 2.0 * area / numpy.maximum(m.shape["perimeter"], _TINY)
    ratio = numpy.maximum(radius[a], radius[b]) / numpy.maximum(
        numpy.minimum(radius[a], radius[b]), _TINY
    )
    quality = numpy.ones(len(m.faces))
    numpy.maximum.at(quality, a, ratio)
    numpy.maximum.at(quality, b, ratio)
    return growth, quality


def _figures(
    m: _Mesh, faces: Array, edges: Array, growth: Array, quality: Array
) -> dict[str, Value]:
    """Return the figures of one selection of faces and edges ( and the checks' figures)."""
    shape = m.shape
    aspect = shape["aspect"][faces]
    out: dict[str, Value] = {
        "faces": int(faces.sum()),
        "triangles": int((shape["sides"][faces] == 3).sum()),
        "quadrilaterals": int((shape["sides"][faces] == 4).sum()),
        "skewness_p95": _p95(shape["skewness"][faces]),
        "warp_p95": _p95(shape["warp"][faces]),
        "growth_p95": _p95(growth[edges]),
        "aspect_p95": _p95(aspect),
        "aspect_max": float(aspect.max()) if len(aspect) else None,
    }
    for figure, measure, above, limit in PANEL_PRACTICE_COUNTS:
        x = shape[measure][faces]
        out[figure] = int((x > limit).sum() if above else (x < limit).sum())
    out["quality_ratio_p95"] = _p95(quality[faces])
    out["quality_ratio_over_2"] = int((quality[faces] > QUALITY_GOOD).sum())
    out["aspect_over_50"] = int((aspect > ASPECT_LIMIT).sum())
    sliver = m.trailing.faces & (shape["sides"] == 3) & (shape["aspect"] > ASPECT_LIMIT)
    out["trailing_edge_triangles_aspect_over_50"] = int(sliver[faces].sum())
    plane = shape["max_abs_y"] <= DUPLICATE_FRACTION * m.size
    out["faces_on_y0_plane"] = int(plane[faces].sum())
    return out


def _item(kind: str, name: str, family: str, values: dict[str, Value], failed: bool) -> AuditItem:
    """Return a judged item."""
    return AuditItem(kind, name, family, MappingProxyType(values), FAIL if failed else PASS)


def _reported(kind: str, name: str, family: str, values: dict[str, Value]) -> AuditItem:
    """Return an item reported and not judged."""
    return AuditItem(kind, name, family, MappingProxyType(values), NOT_JUDGED)


def _gates(level: _Mesh, source: _Mesh | None, grids: tuple[str, ...]) -> tuple[AuditItem, ...]:
    """Return G1 to G6 in order, the source-relative ones reported without a source."""
    return (
        _g1(level),
        *_g2(level),
        _g3(level, source),
        _g4(level, source),
        *_g5(level, source),
        *_g6(level, source, grids),
    )


def _zero_area(m: _Mesh) -> Array:
    """Return per face whether G1 counts it of zero area.

    A face is of zero area below ``ZERO_AREA_FRACTION`` of the size squared,
    or below ``DEGENERATE_AREA_FRACTION`` of its own family's median face
    area, so a sliver of a fine family is not hidden by the whole mesh's size.
    """
    area = m.shape["area"]
    return (area < ZERO_AREA_FRACTION * m.size**2) | below_family_area(area, m.family)


def _degenerate(m: _Mesh) -> Array:
    """Return per face whether it is degenerate: of zero area, or of an angle below 1 degree.

    The rule the remesh repairs by (:func:`._geometry.degenerate`),
    with the zero area of the size squared besides.
    """
    area = m.shape["area"]
    whole = area < ZERO_AREA_FRACTION * m.size**2
    return whole | degenerate(area, m.shape["min_angle"], m.family)


def _degenerate_faces(m: _Mesh) -> tuple[Mapping[str, object], ...]:
    """Return each degenerate face: family, face index, centroid, area ratio, least angle."""
    rows = numpy.nonzero(_degenerate(m))[0]
    if not len(rows):
        return ()
    names = list(m.obj.families)
    start = numpy.concatenate([[0], numpy.cumsum([len(fs) for fs in m.obj.families.values()])])
    out: list[Mapping[str, object]] = []
    for r in rows.tolist():
        family = int(m.family[r])
        mine = m.shape["area"][start[family] : start[family + 1]]
        median = float(numpy.median(mine))
        ratio = None if median <= 0.0 else float(f"{m.shape['area'][r] / median:.6g}")
        out.append(
            {
                "family": names[family],
                "face": r - int(start[family]),
                "centroid": [round(float(x), 9) for x in m.obj.verts[m.faces[r]].mean(axis=0)],
                "area_ratio": ratio,
                "min_angle_degrees": float(f"{m.shape['min_angle'][r]:.6g}"),
            }
        )
    return tuple(out)


def _g1(m: _Mesh) -> AuditItem:
    """G1: no edge of more than two faces, no two nodes at one position, no face of zero area.

    A face with an angle below 1 degree and an area above the zero-area rule
    is counted under ``faces_under_1_degree`` and listed with the degenerate
    faces, and judges nothing: a source's own trailing-edge or pole triangles
    may be that sharp, and a new failure of G1 would refuse a geometry build.
    """
    used = numpy.unique(numpy.fromiter((v for f in m.faces for v in f), dtype=numpy.int64))
    judged: dict[str, Value] = {
        "edges_of_more_than_two_faces": m.nonmanifold,
        "node_pairs_at_one_position": len(
            close_pairs(m.obj.verts[used], DUPLICATE_FRACTION * m.size)
        ),
        "faces_of_zero_area": int(_zero_area(m).sum()),
    }
    values = {**judged, "faces_under_1_degree": int((_degenerate(m) & ~_zero_area(m)).sum())}
    return _item("gate", "G1", WHOLE_MESH, values, any(v != 0 for v in judged.values()))


def _volume(verts: Array, faces: Faces) -> float | None:
    """Return the volume a closed set of faces encloses (right-hand rule), None when open."""
    if boundary_edges(faces):
        return None
    # Taken about the family's own centroid: the sum is the same at any offset, and the
    # triple product of absolute coordinates loses it to rounding far from the origin (F-321).
    used = numpy.unique(numpy.fromiter((v for f in faces for v in f), dtype=numpy.int64))
    p = verts[triangles(faces)] - verts[used].mean(axis=0)
    return float(numpy.einsum("ij,ij->i", p[:, 0], numpy.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def _inward_shells(obj: ObjMesh) -> int:
    """Return how many closed shells of the whole mesh enclose no positive volume (G2).

    A shell is the faces joined through shared edges whatever family holds them, so a
    closed body split into open families is judged as the one body it is (F-304).
    """
    faces = [face for family in obj.families.values() for face in family]
    holder = [name for name, family in obj.families.items() for _ in family]
    parent = list(range(len(faces)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for owners in edge_faces(faces).values():
        for other in owners[1:]:
            parent[root(other)] = root(owners[0])
    shells: dict[int, list[int]] = {}
    for index in range(len(faces)):
        shells.setdefault(root(index), []).append(index)
    inward = 0
    for members in shells.values():
        held = {holder[i] for i in members}
        # A closed family that is a shell by itself is judged on its own row.
        if len(held) == 1 and len(members) == len(obj.families[next(iter(held))]):
            continue
        volume = _volume(obj.verts, [faces[i] for i in members])
        inward += volume is not None and volume <= 0.0
    return inward


def _g2(m: _Mesh) -> Iterator[AuditItem]:
    """G2: no two neighbours of opposite orientation; a closed family encloses a positive volume."""
    inward = _inward_shells(m.obj)
    yield _item(
        "gate",
        "G2",
        WHOLE_MESH,
        {"neighbours_of_opposite_orientation": m.flipped, "inward_closed_shells": inward},
        m.flipped > 0 or inward > 0,
    )
    for name, faces in m.obj.families.items():
        volume = _volume(m.obj.verts, faces)
        if volume is not None:
            yield _item("gate", "G2", name, {"enclosed_volume": volume}, volume <= 0.0)


def _unmatched_loops(level: _Mesh, source: _Mesh) -> int:
    """Return how many open loops of either mesh close no opening of the other.

    Each level loop is offered the source loops its nodes are nearest to,
    most votes first, and takes the first not yet taken that closes the same
    opening (:func:`_same_opening`), so the match is one to one and spatial;
    the count is the loops of both meshes left without a partner.
    """
    old, new = boundary_loops(source.faces), boundary_loops(level.faces)
    if not old or not new:
        return len(old) + len(new)
    label = numpy.repeat(numpy.arange(len(old)), [len(loop) for loop in old])
    index = NearestIndex(source.obj.verts[numpy.concatenate(old)])
    taken: set[int] = set()
    for loop in new:
        points = level.obj.verts[loop]
        votes = numpy.bincount(label[index.query(points)[1]], minlength=len(old))
        for k in numpy.argsort(-votes, kind="stable"):
            if votes[k] == 0:
                break
            if int(k) not in taken and _same_opening(points, source.obj.verts[old[k]]):
                taken.add(int(k))
                break
    return (len(new) - len(taken)) + (len(old) - len(taken))


def _same_opening(mine: Array, theirs: Array) -> bool:
    """Return whether two closed loops bound the same opening.

    Every node of each lies within :data:`OPENING_FRACTION` of the source
    loop's extent from the other's polyline, so the loops coincide up to the
    resampling of their segments.
    """
    extent = float(numpy.linalg.norm(numpy.ptp(theirs, axis=0)))
    limit = OPENING_FRACTION * max(extent, _TINY)
    for points, loop in ((mine, theirs), (theirs, mine)):
        distance, _ = segment_distances(points, loop, numpy.roll(loop, -1, axis=0))
        if float(distance.max()) > limit:
            return False
    return True


def _g3(level: _Mesh, source: _Mesh | None) -> AuditItem:
    """G3: the open boundary loops of the level are the source's."""
    loops = len(boundary_loops(level.faces))
    if source is None:
        return _reported("gate", "G3", WHOLE_MESH, {"open_loops": loops})
    values: dict[str, Value] = {
        "open_loops": loops,
        "source_open_loops": len(boundary_loops(source.faces)),
        "unmatched_loops": _unmatched_loops(level, source),
    }
    failed = values["open_loops"] != values["source_open_loops"] or values["unmatched_loops"] != 0
    return _item("gate", "G3", WHOLE_MESH, values, failed)


def _g4(level: _Mesh, source: _Mesh | None) -> AuditItem:
    """G4: every trailing-edge point on its own mesh edge, and the source's number of chains.

    Two points on one mesh edge (a duplicated row) are counted in
    ``points_sharing_an_edge`` and fail the gate, as the run refuses such a
    file.

    The source's chains are the connected components of its listed edges and
    the detected edges the record states, together; ``source_chains_detected``
    is the number of families the record states detected (module docstring).

    COMPLETENESS. Equal chain counts do not prove the line is
    whole: a chain shortened by its last edge is still one chain. So every
    end of the source's chains must also end a chain of the level, matched
    within the gate's own point tolerance. Refinement and coarsening split or
    merge edges between the ends and keep the end stations, so they keep the
    ends; a dropped edge moves one. ``length`` and ``source_length`` (the
    edges' total length) are reported beside it and not judged: a curved
    line resampled by a spline changes its polyline length legitimately.
    """
    te = level.trailing
    values: dict[str, Value] = {
        "points": te.points,
        "points_on_edges": te.on_edges,
        "points_sharing_an_edge": te.sharing,
        "chains": te.chains,
    }
    if not te.present:
        # 0 < 0 is false, so the gate passes; the line says why rather than leave it silent.
        values["points_file"] = "none found"
    failed = te.on_edges < te.points or te.sharing > 0
    if source is not None:
        chains = source.trailing.chains if source.chains is None else source.chains
        values["source_chains"] = chains
        if source.detected is not None:
            values["source_chains_detected"] = source.detected
        failed = failed or te.chains != chains
    if te.line is not None:
        values["length"] = te.line.length
    if source is not None:
        line = source.trailing.line if source.detected is None else source.line
        if line is not None and te.line is not None:
            missing = _missing_ends(line.ends, te.line.ends, level.fraction * level.size)
            values["source_length"] = line.length
            values["source_ends"] = len(line.ends)
            values["source_ends_missing"] = missing
            failed = failed or missing > 0
    return _item("gate", "G4", WHOLE_MESH, values, failed)


def _missing_ends(wanted: Array, found: Array, tolerance: float) -> int:
    """Return how many of the source's chain ends no chain of the level ends at (G4)."""
    if not len(wanted):
        return 0
    if not len(found):
        return len(wanted)
    distance, _ = NearestIndex(found).query(wanted)
    return int((distance > tolerance).sum())


def _shared(obj: ObjMesh, a: str, b: str) -> int:
    """Return the number of nodes two families of a mesh share."""
    return len(numpy.intersect1d(obj.family_vertices(a), obj.family_vertices(b)))


def _sharing_pairs(obj: ObjMesh) -> Iterator[tuple[str, str, int]]:
    """Yield each pair of families sharing nodes, in family order, with the count."""
    names = list(obj.families)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if count := _shared(obj, a, b):
                yield a, b, count


def _g5(level: _Mesh, source: _Mesh | None) -> Iterator[AuditItem]:
    """G5: families that shared nodes in the source still share them."""
    if source is None:
        pairs = sum(1 for _ in _sharing_pairs(level.obj))
        yield _reported("gate", "G5", WHOLE_MESH, {"family_pairs_sharing_nodes": pairs})
        return
    for a, b, count in _sharing_pairs(source.obj):
        family = f"{a} and {b}"
        if a not in level.obj.families or b not in level.obj.families:
            # F-1203: a family that shared nodes in the source and is gone from the level
            # no longer shares them, so the gate fails rather than reporting nothing.
            values: dict[str, Value] = {"shared_nodes": None, "source_shared_nodes": count}
            yield _item("gate", "G5", family, values, True)
            continue
        now = _shared(level.obj, a, b)
        values = {"shared_nodes": now, "source_shared_nodes": count}
        yield _item("gate", "G5", family, values, now == 0)


def _differing_faces(level: _Mesh, source: _Mesh, name: str) -> int:
    """Return how many faces of a family differ from the source's, in order and coordinates."""
    new, old = level.obj.families[name], source.obj.families[name]
    tolerance = DUPLICATE_FRACTION * source.size
    differ = abs(len(new) - len(old))
    for f, g in zip(new, old, strict=False):
        if len(f) != len(g):
            differ += 1
        elif numpy.linalg.norm(level.obj.verts[f] - source.obj.verts[g], axis=1).max() > tolerance:
            differ += 1
    return differ


def _g6(level: _Mesh, source: _Mesh | None, grids: tuple[str, ...]) -> Iterator[AuditItem]:
    """G6: a grid family at factor 1 keeps the source's faces in coordinates and order."""
    if source is None or not grids:
        yield _reported("gate", "G6", WHOLE_MESH, {"grid_families_at_factor_1": len(grids)})
        return
    for name in grids:
        known = name in level.obj.families and name in source.obj.families
        values: dict[str, Value] = {
            "faces": len(level.obj.families.get(name, [])),
            "source_faces": len(source.obj.families.get(name, [])),
            "differing_faces": _differing_faces(level, source, name) if known else None,
        }
        yield _item("gate", "G6", name, values, not known or values["differing_faces"] != 0)


def _limit(check: str, source_value: float | None) -> float | None:
    """Return the limit of a relative check from the source's 95th percentile."""
    if check == "skewness":
        return None if source_value is None else source_value + SKEWNESS_MARGIN
    floor = WARP_FLOOR_DEGREES if check == "warp" else GROWTH_FLOOR
    return floor if source_value is None else max(source_value, floor)


def _checks(
    mine: Mapping[str, Mapping[str, Value]], theirs: Mapping[str, Mapping[str, Value]] | None
) -> Iterator[AuditItem]:
    """Yield the relative checks of each family and of the whole mesh.

    A family the source does not hold, and every family without a source, is
    reported and not judged.
    """
    theirs = theirs or {}
    for family, figures in mine.items():
        for check, figure in CHECKS:
            value = figures[figure]
            if family not in theirs:
                yield _reported("check", check, family, {"p95": value, "source_p95": None})
                continue
            base = theirs[family][figure]
            yield _check(check, family, value, base)


def _check(check: str, family: str, value: Value, base: Value) -> AuditItem:
    """Return one relative check judged against its limit."""
    limit = _limit(check, None if base is None else float(base))
    values: dict[str, Value] = {"p95": value, "source_p95": base, "limit": limit}
    if check != "skewness":
        floor = WARP_FLOOR_DEGREES if check == "warp" else GROWTH_FLOOR
        values["source_beyond_practice"] = base is not None and float(base) > floor * (
            1.0 + CHECK_TOLERANCE
        )
    if value is None or limit is None:
        return _reported("check", check, family, values)
    # a source beyond the floor is not an exemption. The limit is the larger of
    # the source's figure and the floor, so a level is judged not to be worse than a source
    # that already exceeds the practice; ``source_beyond_practice`` stays as information.
    failed = float(value) > limit + CHECK_TOLERANCE * abs(limit)
    return _item("check", check, family, values, failed)


def _said(value: Value) -> str:
    """Return a value as said in a warning, a console line or a CSV cell."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

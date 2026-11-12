"""Expand matrix rows into a read-only plan through an injected solver hook.

Adapters implement validate_row(row, inputs) -> RowRequirements. They inspect
inputs.setup, inputs.reference and inputs.post, and read any additional files
through inputs.read_bytes/read_toml so every byte read is included in provenance.
The hook validates and describes actions; it must not run a solver or write files.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from cfdbender.core.matrix import MatrixRow, parse_matrix
from cfdbender.core.naming import PointNameError, datapoint_dir_name, point_name
from cfdbender.core.settings import WorkspaceError, _validate


def _empty() -> Mapping[str, object]:
    return MappingProxyType({})


class PlanError(WorkspaceError):
    """A workspace input, solver validation or output claim prevents planning."""


def workspace_path(root: Path, name: str) -> Path:
    """Resolve a workspace-relative path, refusing escapes and ambiguous separators.

    Parameters
    ----------
    root : Path
        Resolved workspace root.
    name : str
        Relative POSIX path.

    Returns
    -------
    Path
        Contained absolute path, including symlink resolution.
    """
    path = Path(name)
    if not name or chr(92) in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise PlanError(f"{name!r}: path outside workspace or not a relative POSIX path")
    for part in path.parts:
        stem = part.split(".", 1)[0].upper()
        if (
            part.endswith((".", " "))
            or re.search(r"~[0-9]+(?:[.]|$)", part, re.IGNORECASE)
            or stem in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)[1-9]", stem)
            or any(ord(c) < 32 or c in '<>"|?*' for c in part)
        ):
            raise PlanError(f"{name!r}: ambiguous or invalid Windows path component")
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target == root:
        raise PlanError(f"{name!r}: path outside workspace")
    return target


class InputFiles:
    """Read byte snapshots once, tracking workspace-relative SHA-256 keys."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._bytes: dict[str, bytes] = {}

    def read_bytes(self, name: str) -> bytes:
        """Read and memoize an input, refusing a path outside this workspace.

        Parameters
        ----------
        name : str
            Workspace-relative input path, using forward slashes.

        Returns
        -------
        bytes
            Exact first-read bytes, also used for the recorded digest.
        """
        path = workspace_path(self.root, name)
        key = path.relative_to(self.root).as_posix()
        if key not in self._bytes:
            try:
                self._bytes[key] = path.read_bytes()
            except OSError as error:
                raise PlanError(f"{name}: cannot read: {error}") from error
        return self._bytes[key]

    def read_toml(self, name: str) -> Mapping[str, object]:
        """Read TOML from the same byte snapshot used for its digest.

        Parameters
        ----------
        name : str
            Workspace-relative TOML input path.

        Returns
        -------
        Mapping
            Parsed TOML fields. Solver-specific schema validation belongs to the hook.
        """
        data = self.read_bytes(name)
        try:
            return tomllib.loads(data.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as error:
            raise PlanError(f"{name}: invalid TOML: {error}") from error

    def digests(self) -> Mapping[str, str]:
        """Return a sorted, immutable digest snapshot of every file read."""
        return MappingProxyType(
            {k: hashlib.sha256(v).hexdigest() for k, v in sorted(self._bytes.items())}
        )


GEOMETRY_FOLDERS = ("inputs/geometries", "inputs/geometry")


def apply_variables(
    setup: Mapping[str, object], variables: Mapping[str, object]
) -> Mapping[str, object]:
    """Return the setup with a row's VAR_NAMES_VALUES overrides applied.

    Parameters
    ----------
    setup : Mapping
        Parsed setup file.
    variables : Mapping
        Row overrides; a dotted key such as ``solver.threads`` sets a key of a table.

    Returns
    -------
    Mapping
        A new mapping; the parsed setup is not modified.
    """
    result: dict[str, object] = dict(setup)
    for key, value in variables.items():
        *tables, name = key.split(".")
        target = result
        for part in tables:
            inner = target.get(part, {})
            if not isinstance(inner, Mapping):
                raise ValueError(f"VAR_NAMES_VALUES {key}: setup key {part} is not a table")
            copied = dict(inner)
            target[part] = copied
            target = copied
        target[name] = value
    return result


def resolve_geometry(root: Path, cell: str | None) -> str | None:
    """Resolve a GEOMETRY cell to a workspace-relative file, without reading it.

    Parameters
    ----------
    root : Path
        Resolved workspace root.
    cell : str or None
        A file name in inputs/geometries/ (or inputs/geometry/), a stem naming
        exactly one file there, or a workspace-relative path containing ``/``.

    Returns
    -------
    str or None
        The workspace-relative POSIX path, or None when the row states none.
    """
    if cell is None:
        return None
    if "/" in cell:
        candidates = [cell]
    else:
        candidates = [f"{folder}/{cell}" for folder in GEOMETRY_FOLDERS]
    found = [name for name in candidates if workspace_path(root, name).is_file()]
    if not found and "/" not in cell:
        found = [
            path.relative_to(root).as_posix()
            for folder in GEOMETRY_FOLDERS
            if (root / folder).is_dir()
            for path in sorted((root / folder).glob(f"{cell}.*"))
            if path.is_file()
        ]
    if len(found) != 1:
        raise ValueError(
            f"GEOMETRY {cell!r} names "
            + (f"several files: {', '.join(found)}" if found else "no file")
            + "; name one file of inputs/geometries/ (with its extension when stems repeat)"
        )
    return found[0]


class RowInputs(InputFiles):
    """One row's s/r/p tables plus tracked access to further workspace inputs.

    ``setup`` carries the row's VAR_NAMES_VALUES overrides. ``geometry`` is the
    resolved GEOMETRY file, or None; an adapter reads it through read_bytes.
    """

    def __init__(self, root: Path, row: MatrixRow, common: InputFiles) -> None:
        super().__init__(root)
        self._bytes.update(common._bytes)
        self.setup = apply_variables(
            self.read_toml(f"inputs/setups/{row.setup_id}.toml"), row.variables
        )
        self.reference = self.read_toml(f"inputs/references/{row.reference_id}.toml")
        self.post = self.read_toml(f"inputs/pproc/{row.post_id}.toml")
        self.geometry = resolve_geometry(root, row.geometry)


@dataclass(frozen=True)
class RowRequirements:
    """Solver hook response: actions and workspace-relative output templates.

    Templates may use {pol} and {point}. The point token is a stable DP name.
    Example: sims/sim_{pol}/datapoints/{point}/history.csv. No file is created.
    """

    actions: tuple[str, ...]
    outputs: tuple[str, ...]


class RowValidator(Protocol):
    """The solver-independent callable interface supplied by a composing layer."""

    def __call__(self, row: MatrixRow, inputs: RowInputs) -> RowRequirements:
        """Validate one row, read tracked inputs and describe the proposed work.

        Parameters
        ----------
        row : MatrixRow
            Matrix data and its input ids.
        inputs : RowInputs
            Parsed s/r/p and tracked readers for further inputs.

        Returns
        -------
        RowRequirements
            Actions and output paths. Raise ValueError to refuse the row.
        """
        ...


@dataclass(frozen=True)
class Point:
    """One stable point with all inputs, proposed actions and claimed output paths."""

    point_id: str
    pol: str
    setup_id: str
    reference_id: str
    post_id: str
    condition: Mapping[str, float]
    walltime: str | None
    directory: str
    actions: tuple[str, ...]
    outputs: tuple[str, ...]
    inputs_sha256: Mapping[str, str]
    geometry: str | None = None
    variables: Mapping[str, object] = field(default_factory=_empty)
    ncpus: int | None = None
    solver_build: str | None = None
    description: str = ""
    aircraft: str = ""
    configuration: str = ""
    hidden: bool = False


#: Point fields beyond the original record, with the value an older record implies.
POINT_EXTRAS: Mapping[str, object] = MappingProxyType(
    {
        "geometry": None,
        "variables": {},
        "ncpus": None,
        "solver_build": None,
        "description": "",
        "aircraft": "",
        "configuration": "",
        "hidden": False,
    }
)


@dataclass(frozen=True)
class RunPlan:
    """An in-memory, solver-named plan; creating it writes no workspace files."""

    root: Path
    matrix: str
    solver: str
    points: tuple[Point, ...]


def _points(root: Path, row: MatrixRow, inputs: RowInputs, needs: RowRequirements) -> list[Point]:
    if not needs.actions or any(not isinstance(a, str) or not a.strip() for a in needs.actions):
        raise PlanError("solver validation must describe at least one non-empty action")
    if not needs.outputs:
        raise PlanError("solver validation must declare output paths")
    points: list[Point] = []
    names: dict[str, float | None] = {}
    for value in row.sweep_values if row.sweep_key else (None,):
        condition = dict(row.condition)
        if row.sweep_key is not None and value is not None:
            condition[row.sweep_key] = value
        try:
            name = point_name(row.condition_order, condition)
        except PointNameError as error:
            raise PlanError(f"POL {row.pol}: {error}") from error
        if name in names:
            raise PlanError(
                f"POL {row.pol}: sweep values {names[name]!r} and {value!r} of "
                f"{row.sweep_key} round to the same point name {name}; a datapoint folder "
                "is named from its flight condition, so give the values distinct names "
                "(for example further apart)"
            )
        names[name] = value
        tag = datapoint_dir_name(name)
        directory = f"sims/sim_{row.pol}/datapoints/{tag}"
        outputs: list[str] = []
        for template in needs.outputs:
            try:
                output = template.format(pol=row.pol, point=tag)
            except (KeyError, IndexError, ValueError) as error:
                raise PlanError(f"invalid output template {template!r}: {error}") from error
            path = workspace_path(root, output)
            normalized = path.relative_to(root).as_posix()
            if not normalized.startswith(("sims/", "post/")):
                raise PlanError(f"{output}: output must be under sims/ or post/")
            if path.is_dir():
                raise PlanError(f"{output}: output is a directory")
            outputs.append(normalized)
        points.append(
            Point(
                point_id=f"{row.pol}/{tag}",
                pol=row.pol,
                setup_id=row.setup_id,
                reference_id=row.reference_id,
                post_id=row.post_id,
                condition=MappingProxyType(condition),
                walltime=row.walltime,
                directory=directory,
                actions=tuple(needs.actions),
                outputs=tuple(outputs),
                inputs_sha256=inputs.digests(),
                geometry=inputs.geometry,
                variables=row.variables,
                ncpus=row.ncpus,
                solver_build=row.solver_build,
                description=row.description,
                aircraft=row.aircraft,
                configuration=row.configuration,
                hidden=row.hidden,
            )
        )
    return points


def check_claims(points: tuple[Point, ...]) -> None:
    """Refuse duplicate point ids, overlapping directories and output path claims.

    Parameters
    ----------
    points : tuple of Point
        All points sharing a workspace, including any existing run records.
    """
    ids: set[str] = set()
    claims: list[tuple[str, str, bool]] = []
    for point in points:
        if point.point_id.casefold() in ids:
            raise PlanError(f"point {point.point_id!r} is recorded or planned twice")
        ids.add(point.point_id.casefold())
        for name, is_directory in [(point.directory, True), *((p, False) for p in point.outputs)]:
            folded = name.casefold().rstrip("/")
            for previous, owner, previous_dir in claims:
                equal = folded == previous
                overlaps = folded.startswith(previous + "/") or previous.startswith(folded + "/")
                own_file_in_directory = owner == point.point_id and (
                    (previous_dir and folded.startswith(previous + "/"))
                    or (is_directory and previous.startswith(folded + "/"))
                )
                if (equal or overlaps) and not own_file_in_directory:
                    raise PlanError(
                        f"{name}: point {point.point_id!r} and point {owner!r} "
                        "would write the same path; choose distinct output paths"
                    )
            claims.append((folded, point.point_id, is_directory))


def plan_matrix(root: str | Path, name: str, *, validators: Mapping[str, RowValidator]) -> RunPlan:
    """Plan a named inputs/matrices/<name>.fs without writing or executing anything.

    Parameters
    ----------
    root : str or Path
        Workspace root containing init.toml.
    name : str
        Matrix filename or stem, without directory components.
    validators : Mapping of str to RowValidator
        Solver name to hook, supplied by the application. Core imports no adapters.

    Returns
    -------
    RunPlan
        Stable points, actions, path claims and SHA-256 of exact input bytes.

    Raises
    ------
    PlanError
        Missing input, malformed TOML, hook refusal or colliding output claims.
    """
    target = Path(root).resolve()
    common = InputFiles(target)
    try:
        config = _validate(common.read_toml("init.toml"), target / "init.toml", validators.keys())
    except WorkspaceError as error:
        raise PlanError(str(error)) from error
    stem = name.removesuffix(".fs")
    if re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]*", stem) is None:
        raise PlanError(f"{name!r}: name a .fs matrix in inputs/matrices/")
    matrix = f"inputs/matrices/{stem}.fs"
    rows = parse_matrix(common.read_bytes(matrix), Path(matrix))
    points: list[Point] = []
    for row in rows:
        try:
            inputs = RowInputs(target, row, common)
            needs = validators[config.solver](row, inputs)
            if inputs.geometry is not None and inputs.geometry not in inputs.digests():
                raise PlanError(
                    f"solver {config.solver!r} does not read the GEOMETRY column; "
                    "name the geometry where its setup documents it and write - in the cell"
                )
            points.extend(_points(target, row, inputs, needs))
        except (ValueError, OSError) as error:
            raise PlanError(f"data row {row.row_number}, POL {row.pol}: {error}") from error
    result = tuple(points)
    check_claims(result)
    return RunPlan(target, matrix, config.solver, result)

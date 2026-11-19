"""Run steady CFD++ cases from a native grid folder; cfdbender writes the solver input.

Initialize with ``cfd-workspace init --solver cfdpp``. The setup file states the case
and nothing else: ``iterations``, ``convergence_orders`` (orders of residual drop that
end the run), ``cfl``, ``cfl_max``, ``cfl_ramp_steps``, ``turbulence`` (a model name,
``laminar`` or ``inviscid``) with the freestream ``viscosity_ratio`` or
``turbulence_intensity`` the model needs, and the ``[physics]``, ``[fluid]`` and
``[numerics]`` tables; one ``[boundaries.<family>]`` table for every family of the grid
(``type`` wall, farfield or symmetry; a viscous wall states ``treatment`` and
``thermal``); optional ``[groups]``; and ``[keywords]``, the one door for raw solver
keywords. No key has a default: ``cfd-workspace setups --template subsonic`` writes a
complete setup. The keys a version takes, and the keyword each sets, come from the
interface database of that version. The setup file holds no ``[solver]`` table.

The row's ``SOLVER_BUILD`` is required: it selects the version of the database and the
build entry of ``inputs/executables.toml`` (``serial``, ``mpi``, ``mpi_launcher``,
``partitioner``, ``tools_dir``). NCPUS 1 runs the serial executable, above 1 the MPI
launcher and executable. ``MESH`` names a native grid folder of ``inputs/mesh/``
(the three binary grid files and the family-name file); a mesh in another format is
converted once by the user with the solver's own converter. init.toml ``[solver]``
holds the machine options ``timeout_s``, ``ram_estimate_gib`` and ``stop_grace_s``.

The reference file states ``area``, ``length`` (also the Reynolds length) and
``moment_origin``; the post-processing file ``history = true``, ``output_frequency``
and ``restart_frequency``. cfdbender normalizes the solver's dimensional forces itself
along the axes of the grid (x aft, z up; y up in two dimensions): ``CFx`` .. ``CMz``,
their inviscid and viscous parts and the raw sums ``Fx`` .. ``Mz``. ``converged`` means
the residual fell by ``convergence_orders`` below the run's largest residual (R), or CL
and CD settled with the residual three orders down (C), with no failure in the log.

The adapter is incomplete and not yet verified against the solver. The supported
version is 22.5.U3; a row of any other version is refused at planning, naming the
supported ones.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.executables import has_registry, resolve_fields
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import (
    NO_READ_ENV,
    InputFiles,
    Point,
    RowInputs,
    RowRequirements,
    workspace_path,
)
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.results import Classification, Coefficient, ExecutionResult, NeutralResult
from cfdbender.core.settings import _validate, read_init
from cfdbender.solvers._host._config import steady_only
from cfdbender.solvers._host._exe import check_host_free
from cfdbender.solvers.cfdpp import _case, _config, _execute, _forces, _history, _mesh
from cfdbender.solvers.cfdpp._config import REFERENCE_KEYS, Case
from cfdbender.solvers.cfdpp._execute import EXECUTABLE_FIELDS as EXECUTABLE_FIELDS
from cfdbender.solvers.cfdpp._restart import restart_files as restart_files
from cfdbender.solvers.cfdpp._schema import Schema

SOLVER_NAME = "cfdpp"
SOLVER_TYPE = "CFD++"
CAPABILITIES = frozenset({"cluster_staging", "restart", "boundary_groups"})
STAGED_CONFIG = _execute.INPUT_NAME
#: The files a cluster job writes and collect reads.
JOB_OUTPUTS = (_history.INFO0_NAME, _history.INFO1_NAME, _history.LOG_NAME)
NATIVE_PASSTHROUGH = "the [keywords] table"
CONVERSION = (
    "MESH {name}: CFD++ reads a native grid folder; convert the mesh once with the "
    "solver's own converter and name the folder"
)
NO_GRID = (
    "CFD++ needs a grid: name a native grid folder of inputs/mesh/ in the MESH cell "
    "(or mesh in the setup file)"
)


@dataclass(frozen=True)
class _Result(NeutralResult):
    """The neutral result plus the dimension and the silent evidence fields of CFD++."""

    dimensions: int = 3
    schema_status: str = "harvested"
    raw_keywords: tuple[Mapping[str, object], ...] = field(default=())
    criterion: str = "none"
    drop_orders: float = 0.0
    redo_events: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize the common fields and the CFD++ evidence fields."""
        data = super().to_dict()
        data.update(
            dimensions=self.dimensions,
            schema_status=self.schema_status,
            raw_keywords=[dict(item) for item in self.raw_keywords],
            criterion=self.criterion,
            drop_orders=self.drop_orders if math.isfinite(self.drop_orders) else None,
            redo_events=self.redo_events,
        )
        return data


def _mesh_of(setup: Mapping[str, object], geometry: str | None) -> str:
    stated = setup.get("mesh")
    if geometry is not None and stated is not None and str(stated).rstrip("/") != geometry:
        raise ValueError(
            f"conflict: mesh is {stated!r} in the setup file and MESH names {geometry!r}; "
            "state it in one place"
        )
    mesh = geometry if geometry is not None else stated
    if not isinstance(mesh, str) or not mesh:
        raise ValueError(NO_GRID)
    return mesh.rstrip("/")


UNREAD_GRID = (
    "the facts of {names} (dimension, counts, selectors) are missing: the large-input check is "
    "off, so the file was not read and its facts are not remembered, and the input cannot be "
    "written. One plan with the check on is enough: plan once without --no-read-large-inputs "
    "(and without {env}), and later plans with it off reuse what that plan remembered"
)


def _grid(root: Path, mesh: str, files: InputFiles) -> _mesh.Grid:
    """Read the grid through the tracked reader: scanned once, tracked, never held whole."""
    folder = workspace_path(root, mesh)
    if not folder.is_dir() and folder.suffix.lower() != ".su2":
        raise ValueError(CONVERSION.format(name=mesh.rsplit("/", 1)[-1]))
    return _mesh.read_grid(folder, files=files, mesh=mesh)


def _case_for(
    schema: Schema,
    setup: Mapping[str, object],
    reference: Mapping[str, object],
    post: Mapping[str, object],
    condition: Mapping[str, float],
    mesh: str,
    grid: _mesh.Grid,
    restart: bool,
) -> Case:
    if grid.facts is None:
        names = ", ".join(f"{mesh}/{name}" for name in grid.unread)
        raise ValueError(UNREAD_GRID.format(names=names, env=NO_READ_ENV))
    return _config.config(
        schema,
        setup,
        reference,
        post,
        condition,
        mesh=mesh,
        names=grid.names,
        dimension=grid.facts.dimension,
        restart=restart,
    )


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Check a steady row against the database of its version, without writes.

    Parameters
    ----------
    row : MatrixRow
        Matrix row and sweep values.
    workspace : RowInputs
        Tracked setup, reference and post inputs; the grid files are read through it.
    """
    if row.workflow != "steady":
        raise steady_only(SOLVER_TYPE)
    schema = _config.database(row.solver_build)
    schema.require_grammar()
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", (SOLVER_NAME,)
    )
    _config.options(declaration.solver_options, schema)
    workspace.warn_unread(SOLVER_TYPE, reference=REFERENCE_KEYS, post=_config.post_keys(schema))
    mesh = _mesh_of(workspace.setup, workspace.geometry)
    grid = _grid(workspace.root, mesh, workspace)  # tracks every grid file it uses
    count = row.ncpus or 1
    staged_partition = count > 1 and count in grid.partitions
    if staged_partition:
        workspace.track(f"{mesh}/{grid.partitions[count].name}")
    if count > 1 and has_registry(workspace.root) and row.solver_build:
        table = resolve_fields(workspace.root, SOLVER_NAME, row.solver_build, EXECUTABLE_FIELDS)
        _execute.argv(table, count, partitioned=staged_partition, build=row.solver_build)
    conditions = [dict(row.condition)]
    if row.sweep_key:
        conditions = [dict(row.condition, **{row.sweep_key: v}) for v in row.sweep_values]
    lines: tuple[str, ...] = ()
    warned: set[str] = set()
    for condition in conditions if grid.facts is not None else ():
        case = _case_for(
            schema,
            workspace.setup,
            workspace.reference,
            workspace.post,
            condition,
            mesh,
            grid,
            row.restart_from is not None,
        )
        built = _case.build(case)
        _case.render(case)
        lines = _case.describe(built)
        for message in (*case.warnings, *built.advisories):
            if message not in warned:
                warned.add(message)
                warn_or_refuse(message, UserWarning, stacklevel=2)
    partition = count > 1 and not staged_partition
    names = [path.name for path in grid.files]
    if count > 1:
        names.append(f"{_mesh.PARTITION_PREFIX}{count}")
    outputs = (
        _execute.INPUT_NAME,
        *names,
        _execute.STDOUT,
        _execute.STDERR,
        _execute.EXECUTION,
        _history.INFO0_NAME,
        _history.INFO1_NAME,
        *((_execute.PARTITION_LOG,) if partition else ()),
    )
    actions = (
        *lines,
        "stage grid",
        *(("partition",) if partition else ()),
        f"run {SOLVER_TYPE}" + (f" on {count} processes" if count > 1 else ""),
        f"read {_history.INFO0_NAME} and {_history.INFO1_NAME}",
    )
    return RowRequirements(
        actions, tuple(f"sims/sim_{{pol}}/datapoints/{{point}}/{name}" for name in outputs)
    )


validate_row.default_setup_overrides = {"GEOMETRY": ("mesh",)}  # type: ignore[attr-defined]
validate_row.geometry_folders = True  # type: ignore[attr-defined]


def _point_case(point: AdapterPoint) -> tuple[Case, str]:
    schema = _config.database(point.point.solver_build)
    machine = read_init(point.root, available_solvers=(SOLVER_NAME,)).solver_options
    _config.options(machine, schema)
    if "line_endings" in machine and schema.number_format is not None:
        ending = "\r\n" if machine["line_endings"] == "crlf" else "\n"
        schema = replace(schema, number_format=replace(schema.number_format, line_ending=ending))
    mesh = _mesh_of(point.setup, point.geometry)
    grid = _grid(point.root, mesh, point.inputs())
    # A run binds the build's executables into a [solver] table; the case never reads it.
    setup = {key: value for key, value in point.setup.items() if key != "solver"}
    case = _case_for(
        schema,
        setup,
        point.reference,
        point.post,
        point.point.condition,
        mesh,
        grid,
        point.point.restart_from is not None,
    )
    return case, mesh


def translate(point: AdapterPoint) -> str:
    """Return the solver input of a verified point: deterministic ASCII.

    Parameters
    ----------
    point : AdapterPoint
        Prepared point with verified input hashes.
    """
    point.verify()
    case, _ = _point_case(point)
    return _case.render(case)


def _options(point: AdapterPoint) -> dict[str, float]:
    return _config.options(read_init(point.root, available_solvers=(SOLVER_NAME,)).solver_options)


def job_output_files(point: Point) -> tuple[str, ...]:
    """Name the numerical files an application-driven job writes."""
    prefix = point.directory + "/"
    return tuple(
        name.removeprefix(prefix)
        for name in point.outputs
        if name.removeprefix(prefix) in JOB_OUTPUTS
    )


def job_completion_files(root: Path, point: Point) -> tuple[str, ...] | None:
    """Use a script's completed execution record when present, including on early failure."""
    record = point.directory + "/" + _execute.EXECUTION
    return (record,) if workspace_path(root, record).is_file() else None


def stage(point: AdapterPoint) -> None:
    """Stage the input, the grid links and the job script for a cluster job, without running.

    Parameters
    ----------
    point : AdapterPoint
        Planned point whose folder does not exist yet.
    """
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError(f"existing {SOLVER_TYPE} point directory; refusing overwrite")
    case, mesh = _point_case(point)
    _execute.stage_files(point, _case.render(case), folder, mesh)


def _series(case: Case) -> Callable[[list[_history.StepForces]], dict[str, list[float]]]:
    def series(steps: list[_history.StepForces]) -> dict[str, list[float]]:
        return _history.lift_drag_series(
            steps,
            case.force_selectors(),
            case.freestream.dynamic_pressure,
            case.references,
            case.alpha,
            case.beta,
            case.dimension,
        )

    return series


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Stage and run one point, then read its evidence from the files it wrote.

    Parameters
    ----------
    point : AdapterPoint
        Planned point with no existing outputs, its build bound by the run.
    check_ram : callable
        Coordinator's memory guard, used besides the measured floor.
    """
    check_ram()
    case, mesh = _point_case(point)
    built = _case.build(case)
    text = _case.render(case)
    extra = {"schema_status": built.schema_status, "raw_keywords": list(built.raw_keywords)}
    code = _execute.run(point, text, mesh, case.schema, _options(point), check_ram, extra)
    return _history.read_evidence(
        workspace_path(point.root, point.point.directory),
        exit_code=code,
        iteration_limit=case.iterations,
        convergence_orders=case.convergence_orders,
        series=_series(case),
    )


def check_executable(point: AdapterPoint) -> None:
    """Refuse a point whose executables are not on this machine; nothing is launched."""
    table = _execute.entry(point)
    count = _execute.ncpus(point)
    mesh = _mesh_of(point.setup, point.geometry)
    staged = _execute.partitioned(workspace_path(point.root, mesh), count)
    build = point.point.solver_build
    for step in _execute.argv(table, count, partitioned=staged, build=build):
        tokens = step.argv[1:] if step.argv[0] == sys.executable else step.argv
        names = (
            ("partitioner",)
            if step.name == "partition"
            else ("serial",)
            if count == 1
            else ("mpi_launcher", "mpi")
        )
        checked = (tokens[0],) if len(names) == 1 else (tokens[0], tokens[-1])
        for name, command in zip(names, checked, strict=True):
            if not Path(command).is_file() and shutil.which(command) is None:
                raise ValueError(
                    f"{SOLVER_TYPE} executable not found: {command!r}; correct the {name} field "
                    f"of build {build!r} in inputs/executables.local.toml (this machine's "
                    "paths) or inputs/executables.toml"
                )


def preflight(point: AdapterPoint) -> None:
    """Refuse before any point starts: executables, then free RAM for the solver."""
    check_executable(point)
    check_host_free(SOLVER_TYPE, _options(point)["ram_estimate_gib"])


def classify(result: ExecutionResult) -> Classification:
    """Classify from the evidence alone: never from a zero exit code.

    Parameters
    ----------
    result : ExecutionResult
        Exit code, residual history and the evidence read from the files.
    """
    return _history.classify(result)


def _exit_code(folder: Path) -> int:
    path = folder / _execute.EXECUTION
    if not path.is_file():
        return 0  # collect works without execution.json (C-27)
    data = json.loads(path.read_text(encoding="ascii"))
    return int(data.get("exit_code", 1))


def collect(point: AdapterPoint) -> NeutralResult:
    """Collect body-axis coefficients from the last step of the force file, read-only.

    Parameters
    ----------
    point : AdapterPoint
        Recorded point with verified source references.
    """
    point.verify()
    case, _ = _point_case(point)
    built = _case.build(case)
    folder = workspace_path(point.root, point.point.directory)
    evidence = _history.read_evidence(
        folder,
        exit_code=_exit_code(folder),
        iteration_limit=case.iterations,
        convergence_orders=case.convergence_orders,
        series=_series(case),
    )
    classification = _history.classify(evidence)
    coefficients: dict[str, Coefficient] = {}
    groups: dict[str, Mapping[str, Coefficient]] = {}
    try:
        steps = _history.read_info1(folder)
    except (OSError, _history.HistoryError):
        steps = []
    if steps:
        coefficients = _coefficients(case, steps[-1], case.force_selectors())
        if case.groups:
            selector = {b.family: b.selector for b in case.boundaries}
            groups["TOTAL"] = dict(coefficients)
            for name, families in case.groups.items():
                groups[name] = _coefficients(
                    case, steps[-1], tuple(selector[family] for family in families)
                )
    elif classification != "failed":
        raise ValueError(f"{_history.INFO1_NAME} is missing; the point has no forces to collect")
    return _Result(
        coefficients,
        case.references,
        evidence.history,
        classification,
        evidence.solver_version,
        force_groups=groups,
        dimensions=case.dimension,
        schema_status=built.schema_status,
        raw_keywords=built.raw_keywords,
        criterion=evidence.criterion,
        drop_orders=evidence.drop_orders,
        redo_events=evidence.redo_events,
    )


def _coefficients(
    case: Case, step: _history.StepForces, selectors: tuple[int, ...]
) -> dict[str, Coefficient]:
    return _forces.coefficients(
        _history.step_forces(step, selectors, "total"),
        case.freestream.dynamic_pressure,
        case.references,
        inviscid=_history.step_forces(step, selectors, "inviscid"),
        viscous=_history.step_forces(step, selectors, "viscous"),
    )

"""SU2 adapter with deterministic s/r/p configuration and choosable routes.

The workspace declares solver = "su2" in init.toml. In inputs/setups/s<id>.toml,
neutral scalars (mach, alpha, sideslip, freestream_temperature, reynolds,
iterations, convergence_field) map to SU2 keys; uppercase SU2 keys in s/r/p pass through.
Reference area, length and moment_origin belong in r<id>.toml. The p file
requires history=true and accepts output_frequency. Mesh may be in s or r and
names a workspace-relative .su2 or .cgns file, tracked by inputs_sha256 and staged
per point as mesh.su2 or mesh.cgns. CGNS selects MESH_FORMAT=CGNS. The staged
mesh is always a relative symbolic link, or a hard link where the host cannot
make symbolic links, never a copy; staging refuses when neither works. mesh_link
= true is accepted and changes nothing; false is refused. Keep the source mesh
immutable for the lifetime of the run.
Duplicate assignments to a SU2 key refuse, including matrix conditions.

The matrix MESH column may name the mesh instead of the setup. A row whose
FLIGHT_CONDITION states the air with more than MACH (for example
MACH:0.85, REmi:5, TK:310.928) is resolved in the core and written as
MACH_NUMBER, FREESTREAM_TEMPERATURE, REYNOLDS_NUMBER and REYNOLDS_LENGTH (the r
file's length), with FREESTREAM_PRESSURE and FREESTREAM_DENSITY only where SU2 reads
them; a MUPas pin writes a constant viscosity. A value the s/r/p files also state
must agree with it.

The s file's optional [solver] table accepts route="pysu2" or "executable",
executable="<path>" (else CFDBENDER_SU2_EXE), threads=1, timeout_s=600 and
ram_estimate_gib=0.25. Omitted route prefers pysu2, with an announced fallback
when its module is absent. mpi_launcher=[] defaults to serial; on the executable
route tokens such as ["mpirun", "-np", "192"] or ["srun"] precede the executable.
Explicit pysu2 refuses if unavailable. Both routes
run in a child process; the serial wrapper uses CSinglezoneDriver(cfg, 1, 0),
StartSolver(), then Finalize(). A native hard exit becomes failed evidence.
pysu2_path="<directory>" (else CFDBENDER_PYSU2_PATH) adds a wrapper install
to the worker's PYTHONPATH; relative paths resolve from the workspace root.
Build the wrapper on Linux or WSL with a matching Python interpreter. A Linux
wrapper cannot be imported by the Windows interpreter.
For workspace defaults, replace the scalar solver declaration with [solver],
name="su2", and route or executable. The s file overrides workspace options.

Example setup (with a mesh and r/p files): mesh="inputs/mesh/airfoil.su2",
iterations=40, SOLVER="EULER", MACH_NUMBER=0.3, AOA=2.0,
MARKER_EULER=["airfoil"], MARKER_FAR=["farfield"]. For native execution add
[solver] with route="executable" and set CFDBENDER_SU2_EXE in the environment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from cfdbender.core import plan as plan_module
from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.flight import STATE_KEYS
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import Classification, Coefficient, ExecutionResult, NeutralResult
from cfdbender.core.rotor import periodic_state, rotor_block
from cfdbender.core.settings import _validate
from cfdbender.solvers._host._config import steady_only
from cfdbender.solvers._host._groups import groups, point_markers, validate_tracked_groups
from cfdbender.solvers.su2._config import (
    NEUTRAL_KEYS,
    config,
    mesh_name,
    options,
    reads_key,
    render,
    rotor_point,
    staged_mesh,
)
from cfdbender.solvers.su2._execute import job_command as job_command
from cfdbender.solvers.su2._execute import run
from cfdbender.solvers.su2._execute import stage as _stage
from cfdbender.solvers.su2._history import dimensions, force_groups, read, residual_field
from cfdbender.solvers.su2._restart import restart_files as restart_files
from cfdbender.solvers.su2._restart import restart_values

SOLVER_NAME = "su2"
CAPABILITIES = frozenset({"cluster_staging", "restart", "boundary_groups", "numerics_file"})
SOLVER_TYPE = "SU2"
STAGED_CONFIG = "su2.cfg"
#: The namespace of the s file that SU2 forwards unread (contract C-09): its own option names.
NATIVE_PASSTHROUGH = "upper-case native option names"
EXECUTABLE_KEYS = {"path": "executable", "launcher": "mpi_launcher"}
BUILD_OPTIONS = {"route": "executable"}
JOB_OUTPUTS = ("history.csv", "stdout.log")


@dataclass(frozen=True)
class _SU2Result(NeutralResult):
    """Add the route, the problem dimension and the residual's name to SU2's evidence."""

    route: str
    dimensions: int | None = None
    residual_field: str | None = None
    rotor: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the common fields, the route and, when known, dimension and residual."""
        data: dict[str, object] = {**super().to_dict(), "route": self.route}
        if self.dimensions is not None:
            data["dimensions"] = self.dimensions
        if self.residual_field is not None:
            data["residual_field"] = self.residual_field
        if self.rotor is not None:
            data["rotor"] = dict(self.rotor)
        return data


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Refuse invalid steady s/r/p data and track the mesh without writes.

    Parameters
    ----------
    row : MatrixRow
        Matrix conditions and workflow.
    workspace : RowInputs
        Tracked source reader.

    Returns
    -------
    RowRequirements
        Actions and claimed point outputs.
    """
    if row.workflow != "steady":
        raise steady_only("SU2")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("su2",)
    )
    options(workspace.setup, declaration.solver_options)
    workspace.warn_unread(
        "SU2",
        reference=tuple(key for key in workspace.reference if reads_key(key)),
        post=tuple(key for key in workspace.post if key == "history" or reads_key(key)),
    )
    conditions = [dict(row.condition)]
    if row.sweep_key:
        conditions = [dict(row.condition, **{row.sweep_key: value}) for value in row.sweep_values]
    for condition in conditions:
        values = config(
            workspace.setup,
            workspace.reference,
            workspace.post,
            condition,
            geometry=workspace.geometry,
        )
        validate_tracked_groups(workspace.setup, workspace, mesh_name(values))
    return RowRequirements(
        ("write SU2 cfg", "stage mesh", "run SU2", "read SU2 history"),
        tuple(
            f"sims/sim_{{pol}}/datapoints/{{point}}/{name}"
            for name in (
                "su2.cfg",
                staged_mesh(values),
                "stdout.log",
                "stderr.log",
                "execution.json",
                "history.csv",
            )
        ),
    )


_FLOW_SETUP_KEYS = {
    "MACH_NUMBER",
    "FREESTREAM_TEMPERATURE",
    "FREESTREAM_PRESSURE",
    "FREESTREAM_DENSITY",
    "REYNOLDS_NUMBER",
    "REYNOLDS_LENGTH",
}
validate_row.reads_symmetry = True  # type: ignore[attr-defined]
validate_row.default_setup_overrides = {  # type: ignore[attr-defined]
    **{
        key: tuple(
            _FLOW_SETUP_KEYS
            | {name for name, native in NEUTRAL_KEYS.items() if native in _FLOW_SETUP_KEYS}
        )
        for key in STATE_KEYS
        if key != "MACH"
    },
    "MACH": ("mach", "MACH_NUMBER"),
    "ALPHA": ("alpha", "angle_of_attack", "AOA"),
    "BETA": ("beta", "sideslip", "SIDESLIP_ANGLE"),
    "GEOMETRY": ("mesh", "MESH_FILENAME"),
}


def translate(point: AdapterPoint) -> str:
    """Return deterministic ASCII after verifying every planned input digest.

    Parameters
    ----------
    point : AdapterPoint
        Prepared and digest-checked inputs.
    """
    point.verify()
    if groups(point.setup):
        values = config(
            point.setup, point.reference, point.post, point.point.condition, geometry=point.geometry
        )
        point_markers(point, mesh_name(values))
    return render(
        restart_values(
            point,
            config(
                point.setup,
                point.reference,
                point.post,
                point.point.condition,
                geometry=point.geometry,
            ),
        )
    )


def stage(point: AdapterPoint) -> None:
    """Stage su2.cfg and the mesh, or its link, for a cluster job, without running SU2.

    The job runs SU2_CFD su2.cfg in the point folder and writes history.csv; its
    standard output must reach stdout.log there (the profile's {stdout_log}),
    because convergence is read from the solver log.

    Parameters
    ----------
    point : AdapterPoint
        Planned point whose folder does not exist yet.
    """
    _stage(point)


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Execute one cfg in its point folder and parse the saved SU2 evidence.

    Parameters
    ----------
    point : AdapterPoint
        Planned inputs and output claims.
    check_ram : callable
        Coordinator guard, called before and throughout execution in addition to
        the native measured 0.5 GiB floor and prelaunch memory estimate.
    """
    run(point, check_ram)
    return read(workspace_path(point.root, point.point.directory))[0]


def check_executable(point: AdapterPoint) -> None:
    """Refuse a point whose executable is not on this machine (executable route); no host check."""
    from cfdbender.solvers.su2._execute import check_executable as check

    check(point)


def preflight(point: AdapterPoint) -> None:
    """Refuse before any point starts: executable, host slot and free RAM (executable route)."""
    from cfdbender.solvers.su2._execute import preflight as check

    check(point)


def classify(result: ExecutionResult) -> Classification:
    """Require solver convergence evidence independently of process success.

    Parameters
    ----------
    result : ExecutionResult
        Parsed log/history outcome, including failed numerical evidence.
    """
    if result.exit_code != 0 or not result.history:
        return "failed"
    if result.history[-1].converged:
        return "converged"
    if result.history[-1].iteration >= result.iteration_limit:
        return "iteration_limit"
    return "failed"


def _rotor(
    point: AdapterPoint, coefficients: Mapping[str, Coefficient]
) -> dict[str, object] | None:
    """Return the rotor quantities of a point whose REF file states a rotor block, else None."""
    if rotor_block(point.reference) is None:
        return None
    values = config(
        point.setup, point.reference, point.post, point.point.condition, geometry=point.geometry
    )
    return rotor_point(
        values,
        point.reference,
        {k: c.value for k, c in coefficients.items()},
        periodic_state(point.setup),
    )


def collect(point: AdapterPoint) -> NeutralResult:
    """Collect SU2 coefficients with named axes and references from the used cfg.

    Parameters
    ----------
    point : AdapterPoint
        Recorded point. Collection reads saved cfg, logs and history without writes.
    """
    point.verify()
    folder = workspace_path(point.root, point.point.directory)
    result, coefficients, refs = read(folder)
    execution = folder / "execution.json"
    route = json.loads(execution.read_bytes())["route"] if execution.is_file() else "cluster"
    staged = folder / "mesh.su2"
    files = point.inputs()
    if not staged.is_file():
        found = dimensions(folder, mesh=False)  # a CGNS mesh: only the logs state it
    elif not files.reads_large() and staged.stat().st_size > plan_module.LARGE_INPUT_BYTES:
        # The large-input check is off: the mesh is not opened for its NDIME line.
        found = dimensions(folder, mesh=False)
        if found is None:
            source = mesh_name(
                config(
                    point.setup,
                    point.reference,
                    point.post,
                    point.point.condition,
                    geometry=point.geometry,
                )
            )
            files.warn(
                f"{source}|dimensions",
                f"{source}: problem dimension not read from mesh.su2; the "
                "large-input check is off, so the mesh is not read (use --rehash-inputs "
                "without --no-read-large-inputs to check it)",
            )
    else:
        found = dimensions(folder)
    return _SU2Result(
        coefficients,
        refs,
        result.history,
        classify(result),
        result.solver_version,
        route,
        found,
        residual_field(folder),
        _rotor(point, coefficients),
        force_groups=force_groups(folder, groups(point.setup), coefficients),
    )

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

The matrix GEOMETRY column may name the mesh instead of the setup. A row whose
FLIGHT_CONDITION states the air with more than MACH (for example
MACH:0.85, REmi:5, TK:310.928) is resolved in the core and written as
MACH_NUMBER, FREESTREAM_TEMPERATURE, FREESTREAM_PRESSURE, FREESTREAM_DENSITY,
REYNOLDS_NUMBER and REYNOLDS_LENGTH (the r file's length); a MUPas pin writes a
constant viscosity. A value the s/r/p files also state must agree with it.

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

Example setup (with a mesh and r/p files): mesh="inputs/meshes/airfoil.su2",
iterations=40, SOLVER="EULER", MACH_NUMBER=0.3, AOA=2.0,
MARKER_EULER=["airfoil"], MARKER_FAR=["farfield"]. For native execution add
[solver] with route="executable" and set CFDBENDER_SU2_EXE in the environment.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import _validate
from cfdbender.solvers.su2._config import config, mesh_name, options, render, staged_mesh
from cfdbender.solvers.su2._execute import run
from cfdbender.solvers.su2._execute import stage as _stage
from cfdbender.solvers.su2._history import read

SOLVER_NAME = "su2"
SOLVER_TYPE = "SU2"
STAGED_CONFIG = "su2.cfg"
EXECUTABLE_KEYS = {"path": "executable", "launcher": "mpi_launcher"}
BUILD_OPTIONS = {"route": "executable"}
JOB_OUTPUTS = ("history.csv", "stdout.log")


@dataclass(frozen=True)
class _SU2Result(NeutralResult):
    """Add the executed route to SU2's neutral result evidence."""

    route: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the common fields and the route from execution.json."""
        return {**super().to_dict(), "route": self.route}


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
        raise ValueError("SU2 WORKFLOW must be steady in this adapter")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("su2",)
    )
    options(workspace.setup, declaration.solver_options)
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
        workspace.read_bytes(mesh_name(values))
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


def translate(point: AdapterPoint) -> str:
    """Return deterministic ASCII after verifying every planned input digest.

    Parameters
    ----------
    point : AdapterPoint
        Prepared and digest-checked inputs.
    """
    point.verify()
    return render(
        config(
            point.setup,
            point.reference,
            point.post,
            point.point.condition,
            geometry=point.geometry,
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
    return _SU2Result(
        coefficients, refs, result.history, classify(result), result.solver_version, route
    )

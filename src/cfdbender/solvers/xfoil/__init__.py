"""Plan one XFOIL alpha per point from a NACA designation or tracked coordinates.

Initialize with ``cfd-workspace init --solver xfoil``. Set s001.toml to
``airfoil="NACA 0012"`` and ``reynolds=1000000``; r001.toml and p001.toml
may be empty because XFOIL defines unit chord and quarter-chord moment references.
Alternatively use ``airfoil_file="inputs/airfoils/profile.dat"``. The coordinate
bytes are tracked by inputs_sha256 and staged as airfoil.dat for portable scripts.
Mach defaults to 0, ncrit to 9, iterations to 200. Reynolds absent or 0 is inviscid.
Optional panels sets PPAR N before PANE; nondefault ncrit sets VPAR N in OPER.

A matrix with columns POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES and row
1001|1|001|001|001|alpha:sweep|0,2,4 plans three independent points through
``cfd-matrix plan study --local``. Scalar alpha uses alpha:4 and empty SWEEP_VALUES.
The core parser canonicalizes condition keys to uppercase; this adapter accepts
either case. Planning validates tracked inputs and returns deterministic ASCII
through translate without creating point folders.

Run with ``cfd-matrix run study --local`` and collect with
``cfd-matrix collect study``. Collection writes the neutral summary to
post/study/campaign_summary.json. In the s file, [solver] executable="<path>"
overrides CFDBENDER_XFOIL_EXE and then xfoil on PATH. A local .py executable
runs under sys.executable for offline stand-ins. Optional timeout_s=600 and
ram_estimate_gib=0.25 use the shared host guards, including the 0.5 GiB RAM floor
and shared process lease. Lease and RAM refusals name XFOIL.

For Windows to WSL, put launcher=["wsl", "-d", "Ubuntu", "--"] and the Linux
executable path in the s file's [solver] table. The launcher is prepended as
individual arguments and the executable passes verbatim; cwd is the Windows
point folder, which wsl.exe maps. With a launcher, host Python wrapping and host
executable resolution do not apply. Coordinate files and xfoil.in stay relative
to that point folder. No WSL process is needed for offline tests.

The polar row for the requested alpha is the convergence evidence. The neutral
history has one acceptance marker (iteration=1, residual=0), because the polar
does not report internal viscous iterations or residuals. Those values are
protocol sentinels. Coefficients CL, CD, CDp and CM retain the printed values,
unit chord/unit span references and XFOIL's moment origin at x/c=0.25.
"""

from __future__ import annotations

from collections.abc import Callable

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import _validate
from cfdbender.solvers.xfoil._config import config, options, render
from cfdbender.solvers.xfoil._execute import run
from cfdbender.solvers.xfoil._execute import run_script as run_script
from cfdbender.solvers.xfoil._execute import stage as _stage
from cfdbender.solvers.xfoil._polar import read, references

SOLVER_NAME = "xfoil"
STAGED_CONFIG = "xfoil.in"
EXECUTABLE_KEYS = {"path": "executable", "launcher": "launcher"}
JOB_OUTPUTS = ("polar.txt", "stdout.log")


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Validate an alpha row and track any coordinate input without writes.

    Parameters
    ----------
    row : MatrixRow
        Steady matrix row with scalar alpha or a sweep on alpha.
    workspace : RowInputs
        Tracked setup, reference, post and additional input reader.

    Returns
    -------
    RowRequirements
        Per-point actions and output claims.
    """
    if row.workflow != "steady":
        raise ValueError("XFOIL WORKFLOW must be steady")
    if row.sweep_key and row.sweep_key.lower() != "alpha":
        raise ValueError("XFOIL only supports an alpha sweep")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("xfoil",)
    )
    options(workspace.setup, declaration.solver_options)
    conditions = [dict(row.condition)]
    if row.sweep_key:
        conditions = [dict(row.condition, **{row.sweep_key: value}) for value in row.sweep_values]
    filename = None
    for condition in conditions:
        values = config(workspace.setup, condition, workspace.reference, workspace.geometry)
        filename = values.airfoil_file
        if filename:
            workspace.read_bytes(filename)
    names = ["xfoil.in", "stdout.log", "stderr.log", "execution.json", "polar.txt"]
    if filename:
        names.append("airfoil.dat")
    return RowRequirements(
        ("write XFOIL keystrokes", "run XFOIL", "read XFOIL polar"),
        tuple(f"sims/sim_{{pol}}/datapoints/{{point}}/{name}" for name in names),
    )


def translate(point: AdapterPoint) -> str:
    """Return deterministic ASCII after verifying the planned inputs.

    Parameters
    ----------
    point : AdapterPoint
        Digest-checked point with setup and scalar alpha.
    """
    point.verify()
    return render(config(point.setup, point.point.condition, point.reference, point.geometry))


def stage(point: AdapterPoint) -> None:
    """Stage xfoil.in and any airfoil.dat for a cluster job without running XFOIL.

    The job feeds xfoil.in to XFOIL's standard input in the point folder, writes
    polar.txt and sends its standard output to stdout.log (the profile's {stdout_log}).
    """
    _stage(point)


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Run one script with captured channels and read its saved polar evidence.

    Parameters
    ----------
    point : AdapterPoint
        Planned inputs with digest and solver binding.
    check_ram : callable
        Coordinator guard, checked before writes and throughout process execution.
    """
    run(point, check_ram)
    values = config(point.setup, point.point.condition, point.reference, point.geometry)
    return read(workspace_path(point.root, point.point.directory), values.alpha, values.iterations)[
        0
    ]


def classify(result: ExecutionResult) -> Classification:
    """Require a requested-alpha polar row; XFOIL does not use iteration_limit.

    Exit zero alone, or VISCAL: Convergence failed without a polar row, is failed.
    The history is a polar acceptance marker, not an internal iteration count.

    Parameters
    ----------
    result : ExecutionResult
        Process status and parsed polar acceptance evidence.
    """
    if result.exit_code == 0 and result.history and result.history[-1].converged:
        return "converged"
    return "failed"


def collect(point: AdapterPoint) -> NeutralResult:
    """Read printed coefficients with unit-chord and quarter-chord references.

    Parameters
    ----------
    point : AdapterPoint
        Recorded point; collection reads outputs without execution or writes.
    """
    point.verify()
    values = config(point.setup, point.point.condition, point.reference, point.geometry)
    result, coefficients = read(
        workspace_path(point.root, point.point.directory), values.alpha, values.iterations
    )
    return NeutralResult(
        coefficients, references(), result.history, classify(result), result.solver_version
    )

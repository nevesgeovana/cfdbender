"""PyFR 3.1 adapter for two-dimensional fluid-force workspaces."""

from collections.abc import Callable
from dataclasses import dataclass

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import _validate, read_init
from cfdbender.solvers.pyfr._config import config, options, render
from cfdbender.solvers.pyfr._execute import run, stage_case
from cfdbender.solvers.pyfr._history import Evidence, read

SOLVER_NAME = "pyfr"
STAGED_CONFIG = "case.ini"
EXECUTABLE_KEYS = {"path": "executable", "launcher": "launcher"}
JOB_OUTPUTS = ("forces.csv",)


@dataclass(frozen=True)
class PyFRResult(NeutralResult):
    """Neutral result with the averaging window and reference dynamic pressure."""

    window: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        """Return the neutral mapping plus window, q and moment availability."""
        return {
            **super().to_dict(),
            "averaging_window": self.window,
            "dynamic_pressure": 0.5,
            "moment": "not available",
        }


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Validate a steady row and its gmsh mesh without writing."""
    if row.workflow != "steady":
        raise ValueError("PyFR workflow must be steady")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("pyfr",)
    )
    options(workspace.setup, declaration.solver_options)
    if (
        workspace.reference
        or set(workspace.post) - {"history"}
        or workspace.post.get("history") is not True
    ):
        raise ValueError("PyFR uses unit references and requires history=true")
    conditions = [dict(row.condition)]
    if row.sweep_key:
        conditions = [dict(row.condition, **{row.sweep_key: value}) for value in row.sweep_values]
    for condition in conditions:
        case = config(workspace.setup, condition, workspace.reference, workspace.geometry)
        mesh = workspace.read_bytes(case.mesh)
        if not mesh.startswith((b"$MeshFormat\n", b"$MeshFormat\r\n")):
            raise ValueError("mesh must be gmsh format")
    return RowRequirements(
        ("stage mesh", "import mesh", "run PyFR", "read fluid force"),
        tuple(
            f"sims/sim_{{pol}}/datapoints/{{point}}/{name}"
            for name in (
                "case.ini",
                "mesh.msh",
                "mesh.pyfrm",
                "stdout.log",
                "stderr.log",
                "execution.json",
                "forces.csv",
            )
        ),
    )


def translate(point: AdapterPoint) -> str:
    """Return the deterministic case.ini text for one point."""
    point.verify()
    options(point.setup, read_init(point.root, available_solvers=("pyfr",)).solver_options)
    return render(config(point.setup, point.point.condition, point.reference, point.geometry))


def stage(point: AdapterPoint) -> None:
    """Stage case.ini and mesh.msh for a cluster job without running PyFR.

    The job runs pyfr import mesh.msh mesh.pyfrm, then pyfr run with
    mesh.pyfrm and case.ini, in the point folder; it writes forces.csv.
    """
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError("existing PyFR point directory; refusing overwrite")
    stage_case(point, translate(point), folder)


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Stage the mesh, import it and run PyFR under the host guards."""
    run(point, translate(point), check_ram)
    return read(workspace_path(point.root, point.point.directory), point)[0]


def classify(result: ExecutionResult) -> Classification:
    """Classify from the exit code and the window evidence."""
    if result.exit_code != 0 or not isinstance(result, Evidence) or result.failed:
        return "failed"
    if result.insufficient or not result.history or not result.history[-1].converged:
        return "iteration_limit"
    return "converged"


def collect(point: AdapterPoint) -> NeutralResult:
    """Reduce forces.csv over the averaging window to neutral results."""
    point.verify()
    result, coefficients, refs, window = read(
        workspace_path(point.root, point.point.directory), point
    )
    return PyFRResult(
        coefficients, refs, result.history, classify(result), result.solver_version, window
    )

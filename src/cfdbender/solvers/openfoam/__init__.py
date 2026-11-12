"""Run ESI simpleFoam cases from a tracked native or user-converted mesh.

Initialize with ``cfd-workspace init --solver openfoam``. A setup file contains
mesh="inputs/meshes/airfoil.msh", application="simpleFoam", speed=10, nu=1.5e-5,
turbulence="SpalartAllmaras" (or "laminar"), iterations=500 and optionally
residual_target=1e-5. Reference files supply area, length and moment_origin;
mesh may also name .cgns or .ugrid when mesh_converter supplies a shell command
template using {mesh} and {case}. The user installs and validates the converter;
the adapter supplies no CGNS or UGRID reader. Conversion precedes patch fixes.
post files require history=true. Matrix alpha and alpha sweeps rotate the
inflow and lift/drag axes in the XY plane. ESI's derived pitch axis is body -z
(liftDir cross dragDir); CmPitch is retained exactly as the solver writes it.

The setup [solver] table accepts foam_bashrc (a Linux etc/bashrc path, default
CFDBENDER_OPENFOAM_BASHRC), launcher=["wsl", "-d", "Ubuntu", "--"], timeout_s=600,
and threads=1. Omit launcher for a native Linux bash. Run from the Windows
point folder so WSL maps its working directory. processes=1 defaults to serial;
larger counts emit scotch decomposition and run decomposePar, the MPI solver,
then reconstructPar -latestTime. mpi_launcher=["mpirun"] supplies tokenized MPI
launcher arguments, followed by -np and the process count.
executable="stand_in.py" or
CFDBENDER_OPENFOAM_EXE replaces the launch command for offline testing.
Scalar options may also be workspace defaults in init.toml [solver] with
name="openfoam". Launcher lists belong in the setup file.

Use cfd-matrix plan study, run study, then collect study. Planning only reads
and claims outputs; execute materializes the translated case files. Collect
writes the neutral campaign summary under post/study. forceCoeffs supplies Cd,
Cl and CmPitch directly, with no coefficient recomputation. A real ESI v2412
run is required to establish physical convergence of a generated mesh.
"""

from __future__ import annotations

from collections.abc import Callable

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import _validate, read_init
from cfdbender.solvers.openfoam._config import config, files, options, staged_mesh
from cfdbender.solvers.openfoam._execute import run, stage_case
from cfdbender.solvers.openfoam._history import Evidence, read

SOLVER_NAME = "openfoam"
STAGED_CONFIG = "Allrun"
EXECUTABLE_KEYS = {"path": "foam_bashrc", "launcher": "launcher"}
JOB_OUTPUTS = ("log.simpleFoam", "postProcessing/forceCoeffs/0/coefficient.dat")


def validate_row(row: MatrixRow, workspace: RowInputs) -> RowRequirements:
    """Validate a steady alpha sweep and track its mesh without writes.

    Parameters
    ----------
    row : MatrixRow
        Matrix row and sweep values.
    workspace : RowInputs
        Tracked setup, reference and post inputs.
    """
    if row.workflow != "steady":
        raise ValueError("OpenFOAM WORKFLOW must be steady")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("openfoam",)
    )
    opts = options(workspace.setup, declaration.solver_options)
    conditions = [dict(row.condition)]
    if row.sweep_key:
        conditions = [dict(row.condition, **{row.sweep_key: value}) for value in row.sweep_values]
    names: tuple[str, ...] = ()
    for condition in conditions:
        case = config(
            workspace.setup,
            workspace.reference,
            workspace.post,
            condition,
            geometry=workspace.geometry,
        )
        mesh = workspace.read_bytes(case.mesh)
        if (
            case.mesh_converter is None
            and not mesh.startswith(b"$MeshFormat\n2.2 0 8\n")
            and not mesh.startswith(b"$MeshFormat\r\n2.2 0 8\r\n")
        ):
            raise ValueError("mesh must be gmsh 2.2 ASCII")
        names = tuple(files(case, opts))
    return RowRequirements(
        (
            "write OpenFOAM case",
            "stage mesh",
            "mesh_converter" if case.mesh_converter is not None else "gmshToFoam",
            "fix patches",
            "simpleFoam",
            "read forceCoeffs",
        ),
        tuple(
            f"sims/sim_{{pol}}/datapoints/{{point}}/{name}"
            for name in (
                *names,
                staged_mesh(case),
                *(("log.meshConverter",) if case.mesh_converter is not None else ()),
                "stdout.log",
                "stderr.log",
                "execution.json",
                "log.simpleFoam",
                *(("log.decomposePar", "log.reconstructPar") if opts["processes"] != 1 else ()),
                "postProcessing/forceCoeffs/0/coefficient.dat",
            )
        ),
    )


def translate(point: AdapterPoint) -> str:
    """Return case files and Allrun as one fixed-order ASCII text.

    Parameters
    ----------
    point : AdapterPoint
        Prepared point with verified input hashes.
    """
    point.verify()
    opts = options(
        point.setup, read_init(point.root, available_solvers=("openfoam",)).solver_options
    )
    case = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
    )
    return "".join(f"// FILE: {name}\n{body}" for name, body in files(case, opts).items())


def stage(point: AdapterPoint) -> None:
    """Stage the case tree, Allrun and the mesh for a cluster job without running it.

    The job runs bash Allrun in the point folder, which writes the log.* files
    and postProcessing/forceCoeffs that collect reads.

    Parameters
    ----------
    point : AdapterPoint
        Planned point whose folder does not exist yet.
    """
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError("existing OpenFOAM point directory; refusing overwrite")
    stage_case(point, translate(point), folder)


def execute(point: AdapterPoint, *, check_ram: Callable[[], None]) -> ExecutionResult:
    """Stage and execute one point with periodic memory and timeout checks.

    Parameters
    ----------
    point : AdapterPoint
        Planned point with no existing outputs.
    check_ram : callable
        Coordinator's memory guard, used in addition to the measured floor.
    """
    check_ram()
    run(point, translate(point), check_ram)
    return read(workspace_path(point.root, point.point.directory), point)[0]


def classify(result: ExecutionResult) -> Classification:
    """Require SIMPLE convergence or initial residual evidence and force output.

    Parameters
    ----------
    result : ExecutionResult
        Actual process exit and parsed numerical evidence.
    """
    if (
        result.exit_code != 0
        or not result.history
        or (isinstance(result, Evidence) and result.evidence_failed)
    ):
        return "failed"
    if result.history[-1].converged:
        return "converged"
    if result.history[-1].iteration >= result.iteration_limit:
        return "iteration_limit"
    return "failed"


def collect(point: AdapterPoint) -> NeutralResult:
    """Collect the last named coefficient row and the complete force-file history.

    Parameters
    ----------
    point : AdapterPoint
        Recorded point with verified source references.
    """
    point.verify()
    result, coefficients, refs = read(workspace_path(point.root, point.point.directory), point)
    return NeutralResult(
        coefficients, refs, result.history, classify(result), result.solver_version
    )

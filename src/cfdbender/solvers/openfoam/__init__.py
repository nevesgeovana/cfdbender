"""Run ESI simpleFoam cases from a tracked native or user-converted mesh.

Initialize with ``cfd-workspace init --solver openfoam``. A setup file contains
mesh="inputs/mesh/airfoil.msh", application="simpleFoam", speed=10, nu=1.5e-5,
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

import sys
from collections.abc import Callable
from dataclasses import replace

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.flight import STATE_KEYS
from cfdbender.core.matrix import MatrixRow
from cfdbender.core.plan import Point, RowInputs, RowRequirements, workspace_path
from cfdbender.core.results import Classification, ExecutionResult, NeutralResult
from cfdbender.core.settings import _validate, read_init
from cfdbender.solvers._host._config import steady_only
from cfdbender.solvers._host._gmsh import gmsh_header
from cfdbender.solvers._host._groups import point_markers, validate_tracked_groups
from cfdbender.solvers.openfoam._config import (
    POST_KEYS,
    REFERENCE_KEYS,
    config,
    files,
    options,
    solver_log,
    staged_mesh,
)
from cfdbender.solvers.openfoam._execute import run, stage_case
from cfdbender.solvers.openfoam._history import Evidence, force_groups, read
from cfdbender.solvers.openfoam._restart import restart_files as restart_files

SOLVER_NAME = "openfoam"
CAPABILITIES = frozenset({"cluster_staging", "restart", "boundary_groups"})
STAGED_CONFIG = "Allrun"
EXECUTABLE_KEYS = {"path": "foam_bashrc", "launcher": "launcher"}
#: The outputs a cluster job writes; only the solver log the point declares is awaited.
JOB_OUTPUTS = (
    "log.simpleFoam",
    "log.pimpleFoam",
    "postProcessing/forceCoeffs/0/coefficient.dat",
)


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
        raise steady_only("OpenFOAM")
    declaration = _validate(
        workspace.read_toml("init.toml"), workspace.root / "init.toml", ("openfoam",)
    )
    opts = options(workspace.setup, declaration.solver_options)
    workspace.warn_unread("OpenFOAM", reference=REFERENCE_KEYS, post=POST_KEYS)
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
        # The mesh is digested in one streamed pass that also scans it; it is never held whole.
        if case.groups:
            known = validate_tracked_groups(
                workspace.setup, workspace, case.mesh, skip_unread=False
            )
            case = replace(
                case, wall_markers=tuple(m for m in known if m not in {"farfield", "front", "back"})
            )
        if case.mesh_converter is None:
            if workspace.unread(case.mesh, "gmsh-info"):
                workspace.skip_content(case.mesh, "Gmsh 2.2 format line")
            elif gmsh_header(workspace, case.mesh) != "2.2":
                raise ValueError("mesh must be gmsh 2.2 ASCII")
        elif not case.groups:
            workspace.track(case.mesh)
        names = tuple(files(case, opts))
    return RowRequirements(
        (
            "write OpenFOAM case",
            "stage mesh",
            "mesh_converter" if case.mesh_converter is not None else "gmshToFoam",
            "fix patches",
            solver_log(case).removeprefix("log."),
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
                solver_log(case),
                *(("log.decomposePar", "log.reconstructPar") if opts["processes"] != 1 else ()),
                "postProcessing/forceCoeffs/0/coefficient.dat",
                *(f"postProcessing/forceCoeffs_{name}/0/coefficient.dat" for name in case.groups),
            )
        ),
    )


validate_row.default_setup_overrides = {  # type: ignore[attr-defined]
    **{key: ("speed", "nu") for key in STATE_KEYS},
    "GEOMETRY": ("mesh",),
}


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
    if case.groups:
        known = point_markers(point, case.mesh, skip_unread=False)
        case = replace(
            case, wall_markers=tuple(m for m in known if m not in {"farfield", "front", "back"})
        )
    translated = files(case, opts)
    if point.point.restart_from is not None:
        solution = restart_files(point)
        latest = float(next(iter(solution)).split("/")[0])
        translated["system/controlDict"] = (
            translated["system/controlDict"]
            .replace("startFrom startTime;", "startFrom latestTime;")
            .replace(f"endTime {case.iterations};", f"endTime {latest + case.iterations:g};")
        )
        # Function objects name their time directory from the restarted start time.
        # Preserve the adapter's declared, stable history path as a byte copy.
        for name in ("forceCoeffs", *(f"forceCoeffs_{name}" for name in case.groups)):
            translated["Allrun"] += (
                f"mkdir -p postProcessing/{name}/0\n"
                f"if [ -f postProcessing/{name}/{latest:g}/coefficient.dat ]; then\n"
                f"  cp postProcessing/{name}/{latest:g}/coefficient.dat "
                f"postProcessing/{name}/0/coefficient.dat\nfi\n"
            )
    return "".join(f"// FILE: {name}\n{body}" for name, body in translated.items())


def job_output_files(point: Point) -> tuple[str, ...]:
    """Name the numerical job outputs, including every requested boundary group."""
    prefix = point.directory + "/"
    return tuple(
        name.removeprefix(prefix)
        for name in point.outputs
        if name.removeprefix(prefix) in JOB_OUTPUTS or name.endswith("/coefficient.dat")
    )


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


def check_executable(point: AdapterPoint) -> None:
    """Refuse a point whose launcher or stand-in is not on this machine; no host check."""
    import shutil

    from cfdbender.solvers._host._exe import missing_executable
    from cfdbender.solvers.openfoam._execute import argv

    opts = options(
        point.setup, read_init(point.root, available_solvers=("openfoam",)).solver_options
    )
    command = argv(opts, point.root)
    if command[0] != sys.executable and not shutil.which(command[0]):
        raise missing_executable("OpenFOAM", command[0], "launcher")


def preflight(point: AdapterPoint) -> None:
    """Refuse before any point starts: launcher or stand-in, host slot and free RAM."""
    from cfdbender.solvers._host._exe import check_host_free

    check_executable(point)
    check_host_free("OpenFOAM", 0.25)


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
        coefficients,
        refs,
        result.history,
        classify(result),
        result.solver_version,
        force_groups=force_groups(
            workspace_path(point.root, point.point.directory), point, coefficients
        ),
    )

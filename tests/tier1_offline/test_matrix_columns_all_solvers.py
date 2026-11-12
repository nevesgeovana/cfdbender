"""Every registered library solver plans a matrix with the full pyflightstream header."""

from __future__ import annotations

from pathlib import Path

import pytest
from qprop_support import blade_obj

from cfdbender.core.adapter import load_point
from cfdbender.core.workspace import init_workspace
from cfdbender.solvers import available_solvers
from cfdbender.workspace.planning import plan_workspace

HEADER = (
    "POL | HIDDEN | RUN | AIRCRAFT | CONFIGURATION | DESCRIPTION | FLIGHT_CONDITION "
    "| SWEEP_VALUES | GEOMETRY | REF | SET | PPROC | SYMMETRY | SYMMETRY_LOADS | NCPUS "
    "| WALLTIME | FS_BUILD | WORKFLOW | VAR_NAMES_VALUES"
)
MSH2 = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
MSH4 = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"


def row(condition: str, sweep: str, geometry: str, variables: str = "-") -> str:
    return (
        f" 7001 | 1 | 1 | Model | clean | Full header row | {condition} | {sweep} "
        f"| {geometry} | 001 | 001 | 001 | - | - | 4 | 2h | 1.0 | - | {variables} "
    )


CASES = {
    "mock": (
        {
            "inputs/setups/s001.toml": '[mock]\noutcome = "converged"\niterations = 10\n',
            "inputs/references/r001.toml": "area = 1.5\nlength = 1.0\nmoment_origin = [0, 0, 0]\n"
            'axis_system = "body"\naxis_directions = ["aft", "right", "up"]\n',
            "inputs/pproc/p001.toml": "history = true\n",
            "inputs/geometries/model.dat": "mesh",
        },
        row("MACH:0.2, ALPHA:sweep", "0,2", "model.dat"),
    ),
    "su2": (
        {
            "inputs/setups/s001.toml": 'iterations = 40\nSOLVER = "EULER"\n',
            "inputs/references/r001.toml": "area = 1.0\nlength = 1.0\nmoment_origin = [0, 0, 0]\n",
            "inputs/pproc/p001.toml": "history = true\n",
            "inputs/geometries/model.su2": "NDIME= 2\n",
        },
        row("MACH:0.5, ALTFT:10000, ALPHA:sweep, BETA:0", "0,2", "model.su2"),
    ),
    "openfoam": (
        {
            "inputs/setups/s001.toml": 'application = "simpleFoam"\n'
            'turbulence = "laminar"\niterations = 4\n',
            "inputs/references/r001.toml": "area = 0.1\nlength = 1.0\nmoment_origin = [0, 0, 0]\n",
            "inputs/pproc/p001.toml": "history = true\n",
            "inputs/geometries/model.msh": MSH2,
        },
        row("TASmps:10, REmi:0.6, ALPHA:sweep, BETA:0", "0,4", "model.msh"),
    ),
    "pyfr": (
        {
            "inputs/setups/s001.toml": "order = 3\ntend = 3\nwindow_start = 0\n",
            "inputs/references/r001.toml": "",
            "inputs/pproc/p001.toml": "history = true\n",
            "inputs/geometries/model.msh": MSH4,
        },
        row("MACH:0.2, REmi:0.01, ALPHA:sweep, BETA:0", "4", "model.msh"),
    ),
    "xfoil": (
        {
            "inputs/setups/s001.toml": 'airfoil = "NACA 0012"\n',
            "inputs/references/r001.toml": "",
            "inputs/pproc/p001.toml": "",
        },
        row("MACH:0.1, REmi:1, ALPHA:sweep", "0,2", "-", "ncrit: 7"),
    ),
    "qprop": (
        {
            "inputs/setups/s001.toml": 'n_blades = 3\nspacing = "cosine"\ncut = "plane"\n'
            "radius = 0.2\n",
            "inputs/references/r001.toml": "",
            "inputs/pproc/p001.toml": "",
            "inputs/geometries/blade.obj": blade_obj(rings=31),
        },
        row("TASmps:20, RPM:6000, BETA07:30, ALPHA:0, BETA:0", "-", "blade.obj", "n_sections: 5"),
    ),
}


def build(root: Path, solver: str) -> Path:
    init_workspace(root, solver=solver, available_solvers=available_solvers())
    files, line = CASES[solver]
    for name, text in {
        **files,
        "inputs/matrices/full.fs": f"{HEADER}\n{'-' * 60}\n{line}\n",
    }.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    return root


@pytest.mark.parametrize("solver", sorted(CASES))
def test_every_solver_plans_the_full_header(tmp_path, solver):
    assert set(CASES) <= set(available_solvers()) | {"mock"}
    plan = plan_workspace(build(tmp_path, solver), "full")
    assert plan.points, f"{solver} planned no point"
    point = plan.points[0]
    assert (point.description, point.aircraft, point.configuration) == (
        "Full header row",
        "Model",
        "clean",
    )
    assert (point.ncpus, point.solver_build, point.hidden) == (4, "1.0", True)
    if CASES[solver][1].split("|")[8].strip() != "-":
        assert point.geometry is not None and point.geometry in point.inputs_sha256


def test_openfoam_speed_and_viscosity_come_from_the_row(tmp_path):
    from cfdbender.solvers.openfoam._config import config

    root = build(tmp_path, "openfoam")
    context = load_point(root, plan_workspace(root, "full").points[0], solver="openfoam")
    case = config(
        context.setup,
        context.reference,
        context.post,
        context.point.condition,
        geometry=context.geometry,
    )
    assert case.speed == pytest.approx(10.0)
    assert case.nu == pytest.approx(10.0 * 1.0 / 0.6e6)
    assert case.mesh == "inputs/geometries/model.msh"


def test_two_dimensional_solvers_take_mach_and_reynolds_from_the_row(tmp_path):
    from cfdbender.solvers.pyfr._config import config as pyfr_config
    from cfdbender.solvers.xfoil._config import config as xfoil_config

    root = build(tmp_path / "pyfr", "pyfr")
    context = load_point(root, plan_workspace(root, "full").points[0], solver="pyfr")
    case = pyfr_config(context.setup, context.point.condition, context.reference, context.geometry)
    assert (case.mach, case.reynolds, case.alpha) == (pytest.approx(0.2), pytest.approx(1e4), 4)
    root = build(tmp_path / "xfoil", "xfoil")
    context = load_point(root, plan_workspace(root, "full").points[0], solver="xfoil")
    values = xfoil_config(context.setup, context.point.condition, context.reference)
    assert (values.mach, values.reynolds, values.ncrit) == (
        pytest.approx(0.1),
        pytest.approx(1e6),
        7,
    )
    with pytest.raises(ValueError, match="reynolds"):
        xfoil_config(
            {"airfoil": "NACA 0012", "reynolds": 2e6},
            {"MACH": 0.1, "REmi": 1.0, "ALPHA": 0.0},
        )


def test_qprop_free_stream_comes_from_the_row(tmp_path):
    from cfdbender.solvers.qprop._config import operating

    op = operating({"TASmps": 20.0, "ADVANCE_RATIO": 0.5, "ALPHA": 0.0})
    assert (op.speed, op.j, op.rpm) == (20.0, 0.5, None)
    assert op.rho == pytest.approx(1.225, abs=1e-3)
    assert op.a == pytest.approx(340.294, abs=1e-3)
    with pytest.raises(ValueError, match="V"):
        operating({"TASmps": 20.0, "V": 25.0, "RPM": 6000.0})


def test_xfoil_takes_an_incompressible_reynolds_number_from_the_row():
    from cfdbender.solvers.xfoil._config import config as xfoil_config

    values = xfoil_config({"airfoil": "NACA 0012"}, {"MACH": 0.0, "REmi": 3.0, "ALPHA": 2.0})
    assert (values.mach, values.reynolds, values.alpha) == (0.0, pytest.approx(3e6), 2.0)
    with pytest.raises(ValueError, match="state one"):
        xfoil_config(
            {"airfoil": "NACA 0012"}, {"MACH": 0.0, "REmi": 3.0, "RHOkgm3": 1.2, "ALPHA": 0.0}
        )


def test_pyfr_refuses_a_row_reynolds_beside_the_resolved_one():
    from cfdbender.solvers.pyfr._config import config as pyfr_config

    setup = {
        "mesh": "inputs/meshes/a.msh",
        "order": 2,
        "tend": 1.0,
        "window_start": 0.5,
    }
    with pytest.raises(ValueError, match="REYNOLDS"):
        pyfr_config(setup, {"MACH": 0.2, "REmi": 3.0, "REYNOLDS": 1e6, "ALPHA": 0.0})

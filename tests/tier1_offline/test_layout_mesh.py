"""One mesh resolver: canonical inputs/mesh/, legacy folder names still read."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from cfdbender.core import mesh as mesh_module
from cfdbender.core.workspace import init_workspace
from cfdbender.solvers import available_solvers
from cfdbender.workspace.planning import plan_workspace

HEADER = "POL | RUN | FLIGHT_CONDITION | SWEEP_VALUES | MESH | REF | SETUP | PPROC | WALLTIME"
ROW = " 5001 | 1 | MACH:0.5, ALPHA:sweep | 1,2 | wb.su2 | 001 | 001 | 001 | 1h"


def workspace(root: Path, mesh_folder: str | None, *, extra: tuple[str, ...] = ()) -> Path:
    init_workspace(root, solver="su2", available_solvers=available_solvers())
    files = {
        "inputs/setups/s001.toml": 'iterations = 10\nSOLVER = "RANS"\n',
        "inputs/references/r001.toml": "area = 1.0\nlength = 0.5\nmoment_origin = [0, 0, 0]\n",
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/m.fs": "\n".join([HEADER, "-" * 60, ROW]) + "\n",
    }
    for folder in ([mesh_folder] if mesh_folder else []) + list(extra):
        files[f"{folder}/wb.su2"] = f"NDIME= 3 {folder}\n"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="ascii")
    return root


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(mesh_module, "_warned", set())


@pytest.mark.parametrize("solver", ["su2", "openfoam", "pyfr", "xfoil", "qprop"])
def test_layout_init_creates_mesh_not_geometries(tmp_path, solver):
    init_workspace(tmp_path, solver=solver, available_solvers=available_solvers())
    assert (tmp_path / "inputs/mesh").is_dir()
    assert not (tmp_path / "inputs/geometries").exists()


def test_layout_canonical_mesh_plans_without_warning(tmp_path):
    workspace(tmp_path, "inputs/mesh")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = plan_workspace(tmp_path, "m")
    assert plan.points[0].geometry == "inputs/mesh/wb.su2"


@pytest.mark.parametrize("legacy", ["inputs/geometry", "inputs/geometries", "inputs/meshes"])
def test_layout_each_legacy_name_alone_plans_ok_and_warns_once(tmp_path, legacy):
    workspace(tmp_path, legacy)
    with pytest.warns(UserWarning, match="legacy mesh folder") as caught:
        plan = plan_workspace(tmp_path, "m")
        plan_workspace(tmp_path, "m")
    assert plan.points[0].geometry == f"{legacy}/wb.su2"
    assert len([w for w in caught if "legacy mesh folder" in str(w.message)]) == 1


def test_layout_mesh_wins_over_every_legacy_name(tmp_path):
    workspace(
        tmp_path,
        "inputs/mesh",
        extra=("inputs/geometry", "inputs/geometries", "inputs/meshes"),
    )
    plan = plan_workspace(tmp_path, "m")
    assert plan.points[0].geometry == "inputs/mesh/wb.su2"


def test_layout_legacy_fallback_order_is_geometry_geometries_meshes(tmp_path):
    workspace(tmp_path, "inputs/meshes", extra=("inputs/geometries",))
    with pytest.warns(UserWarning):
        plan = plan_workspace(tmp_path, "m")
    assert plan.points[0].geometry == "inputs/geometries/wb.su2"


def test_layout_setup_mesh_path_resolves_through_the_same_resolver(tmp_path):
    root = tmp_path
    (root / "inputs/meshes").mkdir(parents=True)
    (root / "inputs/meshes/a.msh").write_text("x")
    assert mesh_module.resolve_mesh_path(root, "inputs/mesh/a.msh") == "inputs/meshes/a.msh"
    (root / "inputs/mesh").mkdir()
    (root / "inputs/mesh/a.msh").write_text("y")
    # An explicit legacy path that names an existing file is honoured; one warning says both exist.
    mesh_module._warned.clear()
    with pytest.warns(UserWarning, match="both exist") as caught:
        for _ in range(2):
            assert (
                mesh_module.resolve_mesh_path(root, "inputs/meshes/a.msh") == "inputs/meshes/a.msh"
            )
    assert len(caught) == 1
    assert mesh_module.resolve_mesh_path(root, "inputs/mesh/a.msh") == "inputs/mesh/a.msh"

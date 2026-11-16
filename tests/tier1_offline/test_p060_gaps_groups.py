"""P060: setup [groups] exposes boundary force sets and TOTAL through neutral records."""

import json

import pytest
import test_p0000_item3d_config as su2_case
import test_p0000_openfoam_adapter as foam_case
from test_p0000_item3f_plan import profile

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.matrix.collection import collect_workspace
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace


def workspace(root, solver, *, marker="wing"):
    if solver == "su2":
        su2_case.workspace(root, extra='MARKER_EULER = ["wing", "tail"]\n')
        mesh = root / "inputs/meshes/tiny.su2"
        mesh.write_text(
            "NDIME=2\nNMARK=3\nMARKER_TAG=wing\nMARKER_ELEMS=0\n"
            "MARKER_TAG=tail\nMARKER_ELEMS=0\nMARKER_TAG=farfield\nMARKER_ELEMS=0\n"
        )
    else:
        foam_case.case(root, sweep="4")
        mesh = root / "inputs/meshes/tiny.msh"
        mesh.write_text(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n"
            '2 1 "wing"\n2 2 "tail"\n2 3 "farfield"\n2 4 "front"\n'
            '2 5 "back"\n$EndPhysicalNames\n'
        )
    setup = root / "inputs/setups/s001.toml"
    text = setup.read_text()
    groups = f'\n[groups]\nmain = ["{marker}"]\ncombined = ["wing", "tail"]\n'
    before, sep, after = text.partition("[solver]")
    setup.write_text(before + groups + (sep + after if sep else ""))


def saved_results(root, point, solver):
    folder = root / point.directory
    for name in point.outputs:
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    if solver == "su2":
        (folder / "execution.json").write_text('{"exit_code":0,"route":"executable"}')
        (folder / "stdout.log").write_text("SU2 version 8.6.0\nConvergence criteria satisfied.\n")
        (folder / "history.csv").write_text(
            "Inner_Iter,rms[Rho],CL,CD,CMz,CL(wing),CD(wing),CMz(wing),"
            "CL(tail),CD(tail),CMz(tail)\n"
            "1,-10,0.9,0.09,0.03,0.4,0.04,0.01,0.5,0.05,0.02\n"
        )
    else:
        (folder / "execution.json").write_text('{"exit_code":0}')
        (folder / "log.simpleFoam").write_text(
            "OpenFOAM-v2412\nTime = 4\n"
            "Solving for Ux, Initial residual = 1e-7, Final residual = 1e-9\n"
            "Solving for Uy, Initial residual = 1e-7, Final residual = 1e-9\n"
            "Solving for p, Initial residual = 1e-7, Final residual = 1e-9\n"
            "Solving for nuTilda, Initial residual = 1e-7, Final residual = 1e-9\n"
            "SIMPLE solution converged in 4 iterations\n"
        )
        for name, values in [
            ("forceCoeffs", "0.09 0.9 0.03"),
            ("forceCoeffs_main", "0.04 0.4 0.01"),
            ("forceCoeffs_combined", "0.09 0.9 0.03"),
        ]:
            path = folder / f"postProcessing/{name}/0/coefficient.dat"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Time Cd Cl CmPitch\n4 " + values + "\n")


@pytest.mark.parametrize("solver", ["su2", "openfoam"])
def test_p060_force_groups_reach_collect_records_without_group_csvs(tmp_path, solver):
    workspace(tmp_path, solver)
    try:
        plan = plan_workspace(tmp_path, "study")
    except ValueError as error:
        pytest.fail(f"valid boundary groups must be planned: {error}")
    adapter = load_solver(solver)
    assert "boundary_groups" in adapter.CAPABILITIES
    context = load_point(tmp_path, plan.points[0], solver=solver)
    text = adapter.translate(context)
    if solver == "su2":
        assert "AERO_COEFF_SURF" in text and "MARKER_MONITORING= ( wing, tail )" in text
    else:
        assert "forceCoeffs_main" in text and "patches (wing);" in text
        assert "forceCoeffs_combined" in text and "patches (wing tail);" in text
    registry = RunRegistry(tmp_path)
    registry.record(plan)
    registry.transition(context.point.point_id, "RUNNING")
    adapter.stage(context)
    saved_results(tmp_path, context.point, solver)
    registry.transition(context.point.point_id, "COMPLETED", classification="converged")
    result = adapter.collect(context).to_dict()
    groups = result.get("force_groups", {})
    assert set(groups) == {"main", "combined", "TOTAL"}
    lift = "CL" if solver == "su2" else "Cl"
    assert groups["main"][lift]["value"] == pytest.approx(0.4)
    assert groups["combined"][lift]["value"] == pytest.approx(0.9)
    assert groups["TOTAL"][lift]["value"] == pytest.approx(0.9)
    assert groups["main"][lift]["axis"] == result["coefficients"][lift]["axis"]
    assert collect_workspace(tmp_path, "study").collected == 1
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert summary["points"][0]["result"]["force_groups"] == groups
    assert {p.name for p in (tmp_path / "post").rglob("*.csv")} == {
        "FAM-DIM-P1001.csv",
        "FAM-ACOEF-P1001.csv",
        "P1001-TOTAL-DIM.csv",
        "P1001-TOTAL-ACOEF.csv",
        "P1001-AIRFRAME-DIM.csv",
        "P1001-AIRFRAME-ACOEF.csv",
    }


@pytest.mark.parametrize("solver", ["su2", "openfoam"])
def test_p060_unknown_group_marker_names_the_mesh_inventory(tmp_path, solver):
    workspace(tmp_path, solver, marker="misspelt")
    with pytest.raises(ValueError) as caught:
        plan_workspace(tmp_path, "study")
    text = str(caught.value)
    assert "misspelt" in text and all(marker in text for marker in ["wing", "tail", "farfield"])


def test_p060_openfoam_cluster_waits_for_every_group_force_file(tmp_path):
    workspace(tmp_path, "openfoam")
    matrix = tmp_path / "inputs/matrices/study.fs"
    header, row = matrix.read_text().strip().splitlines()
    matrix.write_text(header + "|SOLVER_BUILD\n" + row + "|v2412\n")
    path = profile(tmp_path, fields={"name": "{job_name}"})
    path.write_text(path.read_text().replace('"mock-1" =', '"v2412" ='))
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    record = RunRegistry(tmp_path).read()[0]
    assert any("forceCoeffs_combined" in name for name in record.submission["job_outputs"])
    saved_results(tmp_path, record.point, "openfoam")
    path = (
        tmp_path / record.point.directory / "postProcessing/forceCoeffs_combined/0/coefficient.dat"
    )
    data = path.read_bytes()
    path.unlink()
    assert collect_workspace(tmp_path, "study").outstanding == 1
    path.write_bytes(data)
    assert collect_workspace(tmp_path, "study").collected == 1

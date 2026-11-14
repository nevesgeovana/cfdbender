"""FIX87I: the OpenFOAM solver log is named after the application that ran."""

from __future__ import annotations

import test_p0000_openfoam_adapter as foam

from cfdbender.matrix.cli import main as matrix


def _lts(root):
    foam.case(root, sweep="4")
    setup = root / "inputs/setups/s001.toml"
    text = setup.read_text(encoding="utf-8")
    setup.write_text(text.replace("[solver]", 'time_integration = "lts"\n[solver]', 1))
    return foam.plan_workspace(root, "study")


def test_lts_route_writes_and_declares_log_pimplefoam(tmp_path):
    plan = _lts(tmp_path)
    allrun = foam.split(foam.api().translate(foam.point(plan)))["Allrun"]
    assert "pimpleFoam > log.pimpleFoam 2>&1" in allrun
    assert "log.simpleFoam" not in allrun
    outputs = "\n".join(plan.points[0].outputs)
    assert "/log.pimpleFoam" in outputs and "/log.simpleFoam" not in outputs


def test_steady_route_keeps_log_simplefoam(tmp_path):
    plan = foam.case(tmp_path, sweep="4")
    allrun = foam.split(foam.api().translate(foam.point(plan)))["Allrun"]
    assert "simpleFoam > log.simpleFoam 2>&1" in allrun


def test_lts_run_is_collected_from_log_pimplefoam(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_OPENFOAM_MODE", "residual")
    plan = _lts(tmp_path)
    assert matrix(["run", "study", "--workspace", str(tmp_path)]) == 0
    folder = tmp_path / plan.points[0].directory
    assert (folder / "log.pimpleFoam").is_file() and not (folder / "log.simpleFoam").exists()
    assert foam.api().collect(foam.point(plan)).classification == "converged"


def test_job_outputs_name_both_logs():
    assert {"log.simpleFoam", "log.pimpleFoam"} <= set(foam.api().JOB_OUTPUTS)

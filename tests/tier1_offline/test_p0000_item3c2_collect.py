"""P0000 item 3C2: cfd-matrix collect [study] --workspace ROOT."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.solvers import mock
from cfdbender.workspace.planning import plan_workspace


def executed(root, outcome="converged"):
    workspace(root, outcome)
    plan = plan_workspace(root, "study")
    run_api(plan, check_ram=lambda: None)
    return plan


@pytest.mark.parametrize("outcome,code", [("converged", 0), ("iteration_limit", 0), ("failed", 1)])
def test_p0000_item3c2_collect_neutral_summary(tmp_path, monkeypatch, outcome, code):
    plan = executed(tmp_path, outcome)
    calls = []
    original = mock.collect

    def collect(point):
        calls.append(point.point.point_id)
        return original(point)

    monkeypatch.setattr(mock, "collect", collect)
    before = (tmp_path / "runs.json").read_bytes()
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == code
    assert calls == [plan.points[0].point_id], "must read results through adapter.collect"
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert summary["matrix"] == plan.matrix
    assert summary["solver"] == "mock"
    row = summary["points"][0]
    assert row["point_id"] == plan.points[0].point_id
    assert row["inputs_sha256"] == dict(plan.points[0].inputs_sha256)
    assert row["outputs_sha256"]
    result = row["result"]
    assert result["classification"] == outcome
    assert result["references"] == {
        "area": 2.0,
        "length": 3.0,
        "moment_origin": [1.0, 2.0, 3.0],
        "axis_system": "body",
        "axis_directions": ["aft", "right", "up"],
    }
    assert {column["axis"] for column in result["coefficients"].values()} == {"x", "y", "z"}
    assert result["solver_version"] == "mock-1"
    assert result["history"]
    assert (tmp_path / "runs.json").read_bytes() == before


@pytest.mark.parametrize("problem", ["input", "output", "missing", "classification"])
def test_p0000_item3c2_collect_reports_bad_evidence(tmp_path, monkeypatch, capsys, problem):
    plan = executed(tmp_path)
    if problem == "input":
        (tmp_path / "inputs/pproc/p001.toml").write_text("history = false")
    elif problem == "output":
        with (tmp_path / plan.points[0].outputs[1]).open("ab") as stream:
            stream.write(b"changed")
    elif problem == "missing":
        (tmp_path / plan.points[0].outputs[1]).unlink()
    else:
        original = mock.collect
        monkeypatch.setattr(
            mock, "collect", lambda p: replace(original(p), classification="failed")
        )
    assert cli(["collect", "--workspace", str(tmp_path)]) == 1
    assert "failed 1" in capsys.readouterr().out
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert summary["points"][0]["error"]
    assert summary["points"][0]["result"] is None


@pytest.mark.parametrize("state", ["PLANNED", "RUNNING"])
def test_p0000_item3c2_collect_outstanding_is_not_success(tmp_path, state, capsys):
    workspace(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    registry = RunRegistry(tmp_path)
    registry.record(plan)
    if state == "RUNNING":
        getattr(registry, "transition", lambda *a: None)(plan.points[0].point_id, "RUNNING")
    assert cli(["collect", "--workspace", str(tmp_path)]) == 3
    assert "outstanding 1" in capsys.readouterr().out


def test_p0000_item3c2_collect_rebuilds_without_reexecution(tmp_path, monkeypatch):
    executed(tmp_path)
    monkeypatch.setattr(mock, "execute", lambda *a, **kw: pytest.fail("collect must not execute"))
    for _ in range(2):
        assert cli(["collect", "--workspace", str(tmp_path)]) == 0
        path = tmp_path / "post/study/campaign_summary.json"
        data = path.read_bytes()
        path.unlink()
    assert data


def test_p0000_item3c2_collect_filters_by_recorded_matrix(tmp_path):
    executed(tmp_path)
    matrix = tmp_path / "inputs/matrices/other.fs"
    matrix.write_text("POL|RUN|SET|REF|PPROC" + chr(10) + "1002|1|001|001|001")
    run_api(plan_workspace(tmp_path, "other"), check_ram=lambda: None)
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert not (tmp_path / "post/other/campaign_summary.json").exists()
    assert cli(["collect", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "post/other/campaign_summary.json").exists()


def test_p0000_item3c2_collect_refuses_absent_records(tmp_path, capsys):
    workspace(tmp_path)
    assert cli(["collect", "--workspace", str(tmp_path)]) == 2
    assert "no recorded points" in capsys.readouterr().err
    assert not (tmp_path / "post").exists()


def test_collect_reads_records_whose_folders_carry_an_old_hash_name(tmp_path):
    plan = executed(tmp_path)
    old = {p.point_id.split("/")[1]: "DP-" + "0123456789abcdef0123"[:20] for p in plan.points}
    assert len(old) == len(plan.points) == 1
    ((name, hashed),) = old.items()
    sims = tmp_path / "sims"
    for folder in sims.rglob(name):
        folder.rename(folder.with_name(hashed))
    manifest = tmp_path / "runs.json"
    manifest.write_text(manifest.read_text().replace(name, hashed))
    assert cli(["collect", "--workspace", str(tmp_path)]) == 0

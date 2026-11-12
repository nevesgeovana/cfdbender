"""P0000: collect --hpc reads saved numerical evidence without launching work."""

from __future__ import annotations

import json

import pytest
from test_p0000_item3f_plan import cli, prepared
from test_p0000_item3f_registry import submitted

from cfdbender.core.registry import RunRegistry
from cfdbender.solvers import mock


def outputs(root, point, outcome="converged"):
    folder = root / point.directory
    (folder / "execution.json").write_text(
        json.dumps(
            {
                "exit_code": int(outcome == "failed"),
                "iteration_limit": 4,
                "solver_version": "mock-1",
            }
        ),
        encoding="ascii",
    )
    (folder / "history.csv").write_text(
        "iteration,residual,converged,CX,CY,CZ,Cl,Cm,Cn"
        + chr(10)
        + "1,0.1,0,0.1,0.2,0.3,0.4,0.5,0.6"
        + chr(10)
        + f"4,0.001,{int(outcome == 'converged')},0.1,0.2,0.3,0.4,0.5,0.6"
        + chr(10),
        encoding="ascii",
    )


@pytest.mark.parametrize("outcome,code", [("converged", 0), ("iteration_limit", 0), ("failed", 1)])
def test_p0000_item3f_collect_submitted_numerical_outputs(tmp_path, monkeypatch, outcome, code):
    registry, point, _ = submitted(tmp_path)
    assert registry.read()[0].status == "SUBMITTED"
    original = registry.read()[0].submission
    outputs(tmp_path, point, outcome)
    monkeypatch.setattr(mock, "execute", lambda *a, **kw: pytest.fail("collect must not execute"))
    assert cli(tmp_path, "collect", "--hpc", "h001") == code
    record = registry.read()[0]
    assert record.status == "COMPLETED" and record.classification == outcome
    assert record.submission == original and len(record.outputs_sha256) == 3
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    row = summary["points"][0]
    assert row["status"] == "COMPLETED" and row["classification"] == outcome
    assert row["submission"]["descriptor_sha256"] == original["descriptor_sha256"]
    assert row["result"]["references"]["area"] == 2.0
    assert row["result"]["solver_version"] == "mock-1"
    assert {v["axis"] for v in row["result"]["coefficients"].values()} == {"x", "y", "z"}
    before = registry.path.read_bytes()
    assert cli(tmp_path, "collect", "--local") == code
    assert before == registry.path.read_bytes(), "completed records remain sealed"


@pytest.mark.parametrize("problem", ["input", "descriptor", "config", "malformed", "changing"])
def test_p0000_item3f_collect_refuses_untrusted_submitted_evidence(tmp_path, monkeypatch, problem):
    registry, point, _ = submitted(tmp_path)
    assert registry.read()[0].status == "SUBMITTED"
    outputs(tmp_path, point)
    folder = tmp_path / point.directory
    if problem == "input":
        (tmp_path / "inputs/pproc/p001.toml").write_text("history = false")
    elif problem == "descriptor":
        (folder / "submit.json").write_text("changed")
    elif problem == "config":
        (folder / "mock.in").write_text("changed")
    elif problem == "malformed":
        (folder / "history.csv").write_text("invalid columns")
    else:
        original = mock.collect

        def collect(point):
            result = original(point)
            with (folder / "history.csv").open("a") as stream:
                stream.write(chr(10))
            return result

        monkeypatch.setattr(mock, "collect", collect)
    before = registry.path.read_bytes()
    assert cli(tmp_path, "collect", "--hpc", "h001") == 1
    assert registry.path.read_bytes() == before
    row = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())["points"][0]
    assert row["error"] and row["result"] is None


def test_p0000_item3f_fake_submit_is_not_numerical_completion(tmp_path):
    prepared(tmp_path)
    assert cli(tmp_path, "run", "--hpc", "h001") == 0
    registry = RunRegistry(tmp_path)
    before = registry.path.read_bytes()
    assert cli(tmp_path, "collect", "--hpc", "h001") == 3
    assert registry.read()[0].status == "SUBMITTED"
    assert before == registry.path.read_bytes()

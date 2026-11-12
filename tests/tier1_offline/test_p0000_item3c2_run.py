"""P0000 item 3C2: cfd-matrix run study --local --workspace ROOT."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.solvers import mock
from cfdbender.workspace.planning import plan_workspace


def cli(args):
    try:
        return main(args)
    except SystemExit as error:
        return error.code


def run_api(plan, adapter=mock, **kwargs):
    try:
        module = importlib.import_module("cfdbender.core.run")
    except ModuleNotFoundError:
        return ()
    return module.execute_plan(plan, adapter, **kwargs)


@pytest.mark.parametrize("local", [[], ["--local"]])
@pytest.mark.parametrize("outcome,code", [("converged", 0), ("iteration_limit", 0), ("failed", 2)])
def test_p0000_item3c2_cli_run_records_classification(tmp_path, local, outcome, code):
    workspace(tmp_path, outcome)
    assert cli(["run", "study", "--workspace", str(tmp_path), *local]) == code
    records = RunRegistry(tmp_path).read()
    assert len(records) == 1, "run must record every point"
    record = records[0]
    assert record.status == "COMPLETED"
    assert record.classification == outcome
    assert record.created_at <= record.started_at <= record.completed_at
    assert len(record.point.inputs_sha256) == 5
    assert len(record.outputs_sha256) == 3


def test_p0000_item3c2_run_claims_before_execute_and_refuses_second(tmp_path):
    workspace(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    registry = RunRegistry(tmp_path)
    registry.record(plan)
    calls = []

    def execute(point, **kwargs):
        assert registry.read()[0].status == "RUNNING"
        calls.append(point.point.point_id)
        return mock.execute(point, **kwargs)

    adapter = SimpleNamespace(execute=execute, classify=mock.classify)
    records = run_api(plan, adapter, check_ram=lambda: None)
    assert len(records) == 1
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="already recorded"):
        run_api(plan, adapter, check_ram=lambda: None)
    assert len(calls) == 1
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["solver", "io", "history"])
def test_p0000_item3c2_failure_does_not_skip_next_point(tmp_path, failure):
    workspace(tmp_path)
    matrix = tmp_path / "inputs/matrices/study.fs"
    matrix.write_text(
        "POL|RUN|SET|REF|PPROC" + chr(10) + "1001|1|001|001|001" + chr(10) + "1002|1|001|001|001"
    )
    plan = plan_workspace(tmp_path, "study")
    calls = []

    def execute(point, **kwargs):
        calls.append(point.point.point_id)
        result = mock.execute(point, **kwargs)
        if len(calls) == 1:
            if failure == "io":
                raise OSError("could not read history")
            return (
                replace(result, exit_code=1) if failure == "solver" else replace(result, history=())
            )
        return result

    records = run_api(
        plan, SimpleNamespace(execute=execute, classify=mock.classify), check_ram=lambda: None
    )
    assert len(records) == 2
    assert [r.classification for r in records] == ["failed", "converged"]
    assert records[0].error


def test_p0000_item3c2_invalid_row_refuses_before_writes(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text("history = false")
    assert cli(["run", "study", "--local", "--workspace", str(tmp_path)]) == 2
    assert not (tmp_path / "runs.json").exists()
    assert not (tmp_path / "sims").exists()


@pytest.mark.parametrize("problem", ["drift", "existing", "ram"])
def test_p0000_item3c2_run_preflight_refuses_without_outputs(tmp_path, problem):
    workspace(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    if problem == "drift":
        (tmp_path / "inputs/pproc/p001.toml").write_text("history = false")
    elif problem == "existing":
        (tmp_path / plan.points[0].directory).mkdir(parents=True)

    def check_ram():
        if problem == "ram":
            raise ValueError("free RAM below 0.5 GiB")

    with pytest.raises(ValueError):
        run_api(plan, check_ram=check_ram)
    assert not (tmp_path / "runs.json").exists()
    assert not list(tmp_path.rglob("history.csv"))


def test_p0000_item3c2_second_cli_run_preserves_every_byte(tmp_path, capsys):
    workspace(tmp_path)
    assert cli(["run", "study", "--local", "--workspace", str(tmp_path)]) == 0
    before = (tmp_path / "runs.json").read_bytes()
    assert cli(["run", "study", "--local", "--workspace", str(tmp_path)]) == 2
    assert "already recorded" in capsys.readouterr().err
    assert (tmp_path / "runs.json").read_bytes() == before
    assert json.loads(before)["records"][0]["status"] == "COMPLETED"

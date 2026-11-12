"""P0000: run --hpc invokes only a neutral fake, captures it and protects records."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys

import pytest
from test_p0000_item3f_plan import cli, prepared, profile, tree

from cfdbender.core.registry import RunRegistry
from cfdbender.solvers import mock
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("options", [[], ["--hpc", "h001"]])
def test_p0000_item3f_run_submits_real_fake_argv(tmp_path, monkeypatch, options):
    tmp_path = tmp_path / "workspace with spaces"
    prepared(tmp_path)
    monkeypatch.setattr(
        mock, "execute", lambda *a, **kw: pytest.fail("HPC must not execute locally")
    )
    assert cli(tmp_path, "run", *options) == 0
    record = RunRegistry(tmp_path).read()[0]
    assert record.status == "SUBMITTED" and record.classification is None
    data = record.submission
    descriptor = tmp_path / data["descriptor"]
    folder = descriptor.parent
    assert json.loads((folder / "fake-submit-argv.json").read_bytes()) == [descriptor.as_posix()]
    assert data["command"][-1] == descriptor.as_posix()
    assert data["descriptor_sha256"] == hashlib.sha256(descriptor.read_bytes()).hexdigest()
    assert data["config_sha256"] == hashlib.sha256((folder / "mock.in").read_bytes()).hexdigest()
    assert "job offline-001 accepted" in data["stdout"]
    assert "fake submit diagnostic" in data["stderr"]
    assert data["stdout"] == (tmp_path / data["stdout_path"]).read_bytes().decode()
    assert data["stderr"] == (tmp_path / data["stderr_path"]).read_bytes().decode()
    assert data["return_code"] == 0 and not data["timed_out"]
    assert 0 < data["timeout_s"] <= 60
    assert record.started_at <= data["started_at"] <= data["finished_at"]
    assert json.loads(descriptor.read_bytes())["config"] == (folder / "mock.in").as_posix()
    assert json.loads((folder / "mock.in").read_bytes())["setup"]["mock"]
    before = tree(tmp_path)
    assert cli(tmp_path, "run", *options) == 2
    assert before == tree(tmp_path)
    assert cli(tmp_path, "run", "--local") == 2
    assert before == tree(tmp_path)


def test_p0000_item3f_submit_uses_no_shell_and_hidden_windows(tmp_path, monkeypatch):
    prepared(tmp_path)
    try:
        module = importlib.import_module("cfdbender.hpc._submit")
    except ImportError:
        module = None
    assert module is not None, "HPC must provide a thin submit executor"
    original = module.subprocess.run
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] > 0
        assert kwargs["stdout"] is not None and kwargs["stderr"] is not None
        if sys.platform == "win32":
            assert kwargs["creationflags"] & module.subprocess.CREATE_NO_WINDOW
        assert RunRegistry(tmp_path).read()[0].status == "RUNNING"
        return original(argv, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert cli(tmp_path, "run", "--hpc", "h001") == 0
    assert len(calls) == 1


@pytest.mark.parametrize("problem", ["nonzero", "missing", "timeout"])
def test_p0000_item3f_submit_failure_is_recorded_and_never_retried(
    tmp_path, monkeypatch, capsys, problem
):
    path = prepared(tmp_path, extra=["--exit-code", "7"] if problem == "nonzero" else [])
    if problem == "missing":
        path.write_text(
            path.read_text().replace(
                json.dumps(sys.executable.replace(chr(92), "/")), '"absent-neutral-submit-client"'
            )
        )
    elif problem == "timeout":
        profile(tmp_path, extra=["--sleep", "10"])
        try:
            module = importlib.import_module("cfdbender.hpc._submit")
        except ImportError:
            module = None
        assert module is not None, "submission needs a bounded executor"
        monkeypatch.setattr(module, "SUBMIT_TIMEOUT_S", 0.5)
    assert cli(tmp_path, "run", "--hpc", "h001") == 2
    records = RunRegistry(tmp_path).read()
    assert len(records) == 1, "failed submission must leave durable evidence"
    record = records[0]
    assert (
        record.submission["return_code"] == {"nonzero": 7, "missing": 127, "timeout": 124}[problem]
    )
    assert record.status == ("SUBMITTED" if problem == "timeout" else "COMPLETED")
    assert record.classification == (None if problem == "timeout" else "failed")
    if problem == "timeout":
        assert record.submission["timed_out"]
        assert "unknown" in capsys.readouterr().err
    before = tree(tmp_path)
    assert cli(tmp_path, "run", "--hpc", "h001") == 2
    assert before == tree(tmp_path)


@pytest.mark.parametrize("problem", ["existing", "finished", "descriptor", "second-row"])
def test_p0000_item3f_run_preflights_all_before_writes(tmp_path, capsys, problem):
    path = prepared(tmp_path)
    point = plan_workspace(tmp_path, "study").points[0]
    if problem == "existing":
        (tmp_path / point.directory).mkdir(parents=True)
    elif problem == "finished":
        assert cli(tmp_path, "run", "--local") == 0
    elif problem == "descriptor":
        path.write_text(path.read_text().replace('name = "submit.json"', 'name = "history.csv"'))
    else:
        matrix = tmp_path / "inputs/matrices/study.fs"
        matrix.write_text(
            "POL|RUN|SET|REF|PPROC"
            + chr(10)
            + "1001|1|001|001|001"
            + chr(10)
            + "1002|1|001|001|001"
        )
        second = plan_workspace(tmp_path, "study").points[1]
        (tmp_path / second.directory).mkdir(parents=True)
    before = tree(tmp_path)
    assert cli(tmp_path, "run", "--hpc", "h001") == 2
    assert before == tree(tmp_path)
    assert "exist" in capsys.readouterr().err or problem in ("finished", "descriptor")


def test_p0000_item3f_local_with_profile_runs_adapter(tmp_path):
    prepared(tmp_path)
    assert cli(tmp_path, "run", "--local", "--hpc", "h001") == 0
    record = RunRegistry(tmp_path).read()[0]
    assert record.status == "COMPLETED" and record.classification == "converged"
    assert record.submission is None
    assert not list(tmp_path.rglob("submit.json"))
    assert not list(tmp_path.rglob("fake-submit-argv.json"))

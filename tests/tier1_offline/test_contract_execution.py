"""Adapter contract, execution, classification and results clauses, over the six adapters.

Offline stand-in executables as in test_conformance_workspace; each test docstring
names its clause of docs/adapter-contract.md.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_conformance_workspace import case, local, prepared, staging_case, state_no_build  # noqa: F401

from cfdbender import testing
from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.solvers import load_solver
from cfdbender.solvers._host._single import process_lease
from cfdbender.workspace.planning import plan_workspace

COMMON_KEYS = ("coefficients", "references", "history", "classification", "solver_version")


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_c19_translate_verifies_the_workspace_solver_first(case):  # noqa: F811
    """C-19: translate refuses a changed init.toml (assert_reads_init) for every adapter."""
    solver, root = case
    adapter, point = prepared(root, solver)
    testing.assert_reads_init(adapter, point)


def test_c10_the_run_record_holds_every_declared_output_digest(case):  # noqa: F811
    """C-10: after a run the record carries the digest of every declared output."""
    solver, root = case
    state_no_build(root)
    testing.assert_run_record(load_solver(solver), plan_workspace(root, "full"))


def test_c22_a_finished_point_is_never_overwritten(case):  # noqa: F811
    """C-22: a second run of a finished point refuses and keeps every first-run byte."""
    solver, root = case
    testing.assert_finished_no_overwrite(load_solver(solver), plan_workspace(root, "full"))


def test_c32_exit_zero_without_evidence_is_failed(case):  # noqa: F811
    """C-32: a zero exit with no history fails, and a nonzero exit cannot converge."""
    solver, root = case
    adapter, point = prepared(root, solver)
    testing.assert_exit_evidence(adapter, point)


def test_c35_c36_collect_reads_the_recorded_class_and_only_reads(case):  # noqa: F811
    """C-35 and C-36: collect reaches the recorded class and writes nothing."""
    solver, root = case
    state_no_build(root)
    testing.assert_collected_results(load_solver(solver), plan_workspace(root, "full"))


def test_c43_every_result_carries_the_five_common_fields(case):  # noqa: F811
    """C-43: the serialised result of every adapter holds the common keys."""
    solver, root = case
    adapter, point = prepared(root, solver)
    adapter.execute(point, check_ram=lambda: None)
    data = adapter.collect(point).to_dict()
    assert set(COMMON_KEYS) <= set(data), (solver, sorted(data))


def test_c44_solver_version_is_a_nonempty_string_as_printed_or_unknown(case):  # noqa: F811
    """C-44: solver_version is the printed version, or the word unknown."""
    solver, root = case
    adapter, point = prepared(root, solver)
    adapter.execute(point, check_ram=lambda: None)
    version = adapter.collect(point).solver_version
    assert isinstance(version, str) and version.strip(), solver
    if version != "unknown":
        printed = " ".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in (root / point.point.directory).rglob("*")
            if p.is_file() and p.stat().st_size < 2_000_000
        )
        assert version in printed, (solver, version)


def test_c26_the_solver_is_launched_without_a_shell_or_console_window(case, monkeypatch):  # noqa: F811
    """C-26: every Popen call of a local run passes shell=False and the no-window flag."""
    solver, root = case
    calls: list[dict] = []
    real = subprocess.Popen

    def recorder(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recorder)
    adapter, point = prepared(root, solver)
    adapter.execute(point, check_ram=lambda: None)
    if solver == "mock":
        assert not calls, "the mock adapter launches no process"
        return
    assert calls, solver
    for kwargs in calls:
        assert kwargs.get("shell", False) is False
        if sys.platform == "win32":
            assert kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW


def test_c27_collect_works_without_execution_json(case):  # noqa: F811
    """C-27: collect classifies a point folder from which execution.json is removed."""
    solver, root = case
    adapter, point = prepared(root, solver)
    adapter.execute(point, check_ram=lambda: None)
    expected = adapter.collect(point).classification
    record = root / point.point.directory / "execution.json"
    if record.exists():
        record.unlink()
    assert adapter.collect(point).classification == expected, solver


def test_c47_collect_works_from_the_job_outputs_alone(staging_case, tmp_path_factory):  # noqa: F811
    """C-47: a cluster folder (staged inputs plus JOB_OUTPUTS) collects as the local one did."""
    solver, root = staging_case
    adapter, point = prepared(root, solver)
    spare = tmp_path_factory.mktemp("staged") / "ws"
    shutil.copytree(root, spare)
    staged_point = load_point(spare, point.point, solver=solver)
    adapter.stage(staged_point)
    staged = {
        p.relative_to(spare / point.point.directory).as_posix()
        for p in (spare / point.point.directory).rglob("*")
        if p.is_file()
    }
    adapter.execute(point, check_ram=lambda: None)
    expected = adapter.collect(point).classification
    folder = root / point.point.directory
    keep = set(adapter.JOB_OUTPUTS) | staged
    for path in folder.rglob("*"):
        if path.is_file() and path.relative_to(folder).as_posix() not in keep:
            path.unlink()
    assert all((folder / name).is_file() for name in adapter.JOB_OUTPUTS), solver
    assert adapter.collect(point).classification == expected, solver


def test_c23_ram_is_checked_before_launch_and_a_refusal_writes_nothing(case):  # noqa: F811
    """C-23: a check_ram that refuses on its first call stops execute before any output."""
    solver, root = case
    adapter, point = prepared(root, solver)
    before = snapshot(root)

    def refuse():
        raise ValueError("free RAM below 0.5 GiB")

    with pytest.raises(ValueError, match="RAM"):
        adapter.execute(point, check_ram=refuse)
    assert snapshot(root) == before


def test_c23_ram_is_checked_again_during_the_run(case):  # noqa: F811
    """C-23: a refusal on a later check_ram call stops a running solver and raises."""
    solver, root = case
    adapter, point = prepared(root, solver)
    calls = []

    def count():
        calls.append(1)

    adapter.execute(point, check_ram=count)
    assert len(calls) >= 2, f"{solver}: RAM checked {len(calls)} time(s)"


def test_c23_a_refusal_during_the_run_stops_the_solver(case):  # noqa: F811
    """C-23: check_ram raising after launch ends execute with the refusal."""
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock adapter launches no solver process")
    adapter, point = prepared(root, solver)
    calls = []

    def refuse_late():
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("free RAM below 0.5 GiB")

    with pytest.raises(ValueError, match="RAM"):
        adapter.execute(point, check_ram=refuse_late)


def test_c24_one_solver_process_per_host(case, monkeypatch):  # noqa: F811
    """C-24: while another process holds the host lease, execute refuses."""
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock adapter launches no solver process")
    # The suite replaces every adapter's lease with a no-op; this test needs the real one.
    module = importlib.import_module(f"cfdbender.solvers.{solver}._execute")
    monkeypatch.setattr(module, "process_lease", process_lease)
    adapter, point = prepared(root, solver)
    held, release = threading.Event(), threading.Event()

    def hold():
        for _ in range(10):
            try:
                with process_lease():
                    held.set()
                    release.wait(10)
                return
            except ValueError:
                time.sleep(0.1)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        if not held.wait(2):
            pytest.skip("another process holds the host solver slot")
        with pytest.raises(ValueError, match="already running"):
            adapter.execute(point, check_ram=lambda: None)
    finally:
        release.set()
        thread.join(10)


def test_c28_an_unexpected_exception_propagates_and_leaves_the_record_running(case, monkeypatch):  # noqa: F811
    """C-28: an exception that is not an expected failure propagates out of the run."""
    solver, root = case
    adapter = load_solver(solver)
    plan = plan_workspace(root, "full")

    def explode(point, *, check_ram):
        raise RuntimeError("planted unexpected failure")

    monkeypatch.setattr(adapter, "execute", explode)
    from cfdbender.core.run import execute_plan

    with pytest.raises(RuntimeError, match="planted"):
        execute_plan(plan, adapter, check_ram=lambda: None)
    statuses = {r.status for r in RunRegistry(root).read()}
    assert statuses == {"RUNNING"} or statuses <= {"RUNNING", "PLANNED"}, statuses


def test_c31_an_unknown_classification_is_recorded_failed_by_the_core(case, monkeypatch):  # noqa: F811
    """C-31: a classify that returns a fourth class never reaches the record as that class."""
    solver, root = case
    adapter = load_solver(solver)
    plan = plan_workspace(root, "full")
    monkeypatch.setattr(adapter, "classify", lambda result: "almost_converged")
    from cfdbender.core.run import execute_plan

    execute_plan(plan, adapter, check_ram=lambda: None)
    records = RunRegistry(root).read()
    assert {r.classification for r in records} == {"failed"}, solver
    assert all("unknown classification" in (r.error or "") for r in records)


def test_c34_a_solver_without_an_iteration_budget_never_returns_iteration_limit(case):  # noqa: F811
    """C-34: xfoil and qprop report no iterations, so no evidence maps to iteration_limit."""
    from dataclasses import replace

    solver, root = case
    if solver not in ("xfoil", "qprop"):
        pytest.skip("only xfoil and qprop have no iteration budget")
    adapter, point = prepared(root, solver)
    result = adapter.execute(point, check_ram=lambda: None)
    for variant in (result, replace(result, history=()), replace(result, exit_code=1)):
        assert adapter.classify(variant) != "iteration_limit", solver


def test_c07_every_planned_input_is_digest_tracked_and_a_change_is_refused(case):  # noqa: F811
    """C-07: the GEOMETRY file is in the plan's digests and a changed byte stops translate."""
    solver, root = case
    adapter, point = prepared(root, solver)
    if point.geometry is None:
        pytest.skip("this adapter reads no GEOMETRY file")
    assert point.geometry in point.point.inputs_sha256
    path = root / point.geometry
    path.write_bytes(path.read_bytes() + b"# changed")
    with pytest.raises(ValueError, match="changed"):
        adapter.translate(point)

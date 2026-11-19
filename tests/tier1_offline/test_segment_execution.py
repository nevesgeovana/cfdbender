"""A point with an export window must execute both generated configurations."""

from __future__ import annotations

import json

import pytest
from test_runtypes_b1_workflows import workspace

from cfdbender.core.adapter import load_point
from cfdbender.solvers.su2 import _execute
from cfdbender.workspace.planning import plan_workspace


def point_with_exports(root, *, segmented=True):
    workspace(
        root,
        workflow="unsteady",
        condition="MACH:0.0529",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 20 / LAST_ITERS_AVG: 10 / "
        f"EXPORT_UNSTEADY_AFTER_ITER: {10 if segmented else 1}",
    )
    plan = plan_workspace(root, "rt")
    return load_point(root, plan.points[0], solver="su2")


@pytest.mark.parametrize("first_status", [0, 7])
@pytest.mark.parametrize("segmented", [False, True])
def test_local_executes_all_segments_until_failure(tmp_path, monkeypatch, first_status, segmented):
    point = point_with_exports(tmp_path, segmented=segmented)
    values = _execute.config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )
    calls = []

    class Child:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            self.status = first_status if len(calls) == 1 else 0

        def wait(self, **kwargs):
            return self.status

        def poll(self):
            return self.status

    monkeypatch.setattr(_execute.subprocess, "Popen", Child)
    monkeypatch.setattr(_execute._ram, "require_ram", lambda *a, **k: None)
    folder = tmp_path / point.point.directory
    _execute._run_owned(
        point,
        lambda: None,
        {"threads": 1, "ram_estimate_gib": 0.25},
        values,
        ["solver", "su2.cfg"],
        5.0,
        folder,
        "executable",
        {},
    )
    expected = [["solver", "su2.cfg"]]
    if segmented and first_status == 0:
        expected.append(["solver", "su2_segment2.cfg"])
    assert calls == expected
    assert json.loads((folder / "execution.json").read_bytes())["exit_code"] == first_status


def test_batch_command_runs_the_segment_script(tmp_path):
    point = point_with_exports(tmp_path)
    assert _execute.job_command(point) == ["sh", "run_segments.sh", "SU2_CFD"]


@pytest.mark.parametrize("elapsed", [0.0, 3.0, 6.0])
@pytest.mark.parametrize("first_timeout", [False, True])
def test_segments_share_timeout_and_logs(tmp_path, monkeypatch, elapsed, first_timeout):
    point = point_with_exports(tmp_path)
    values = _execute.config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )
    now = [0.0]
    calls = []

    class Child:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            self.status = None
            kwargs["stdout"].write((argv[-1] + "\n").encode())
            kwargs["stderr"].write((argv[-1] + "\n").encode())

        def wait(self, **kwargs):
            if first_timeout:
                now[0] = 6.0
                raise _execute.subprocess.TimeoutExpired("solver", kwargs["timeout"])
            now[0] += elapsed
            if len(calls) == 2 and elapsed:
                raise _execute.subprocess.TimeoutExpired("solver", kwargs["timeout"])
            self.status = 0
            return 0

        def poll(self):
            return self.status

    monkeypatch.setattr(_execute.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(_execute.subprocess, "Popen", Child)
    monkeypatch.setattr(_execute, "_kill_tree", lambda child: setattr(child, "status", 124))
    monkeypatch.setattr(_execute._ram, "require_ram", lambda *a, **k: None)
    folder = tmp_path / point.point.directory
    _execute._run_owned(
        point,
        lambda: None,
        {"threads": 1, "ram_estimate_gib": 0.25},
        values,
        ["solver", "su2.cfg"],
        5.0,
        folder,
        "executable",
        {},
    )
    expected = ["su2.cfg"] + (["su2_segment2.cfg"] if elapsed < 5 and not first_timeout else [])
    assert [argv[-1] for argv in calls] == expected
    for name in ("stdout.log", "stderr.log"):
        assert (folder / name).read_text().splitlines() == expected
    record = json.loads((folder / "execution.json").read_bytes())
    assert record["exit_code"] == (124 if elapsed or first_timeout else 0)
    assert record["timed_out"] is bool(elapsed or first_timeout)

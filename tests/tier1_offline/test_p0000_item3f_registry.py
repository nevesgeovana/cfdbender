"""P0000: submission evidence survives RUNNING -> SUBMITTED -> COMPLETED."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_p0000_item3f_plan import prepared

from cfdbender.core.registry import RunRegistry
from cfdbender.workspace.planning import plan_workspace


def evidence(root, point):
    folder = root / point.directory
    folder.mkdir(parents=True)
    for name in ("submit.json", "mock.in", "submit.stdout.txt", "submit.stderr.txt"):
        (folder / name).write_text("saved", encoding="utf-8")
    return {
        "descriptor": f"{point.directory}/submit.json",
        "descriptor_sha256": hashlib.sha256(b"saved").hexdigest(),
        "config_path": f"{point.directory}/mock.in",
        "config_sha256": hashlib.sha256(b"saved").hexdigest(),
        "profile": "inputs/hpc/h001.toml",
        "command": ["neutral-client", str(folder / "submit.json")],
        "stdout_path": f"{point.directory}/submit.stdout.txt",
        "stderr_path": f"{point.directory}/submit.stderr.txt",
        "stdout": "job offline-001 accepted",
        "stderr": "diagnostic",
        "started_at": "2026-11-10T12:00:01+00:00",
        "finished_at": "2026-11-10T12:00:02+00:00",
        "return_code": 0,
        "timeout_s": 60.0,
        "timed_out": False,
    }


def submitted(root):
    prepared(root)
    plan = plan_workspace(root, "study")
    registry = RunRegistry(root)
    registry.record(plan, created_at="2026-11-10T12:00:00+00:00")
    point = plan.points[0]
    registry.transition(point.point_id, "RUNNING", timestamp="2026-11-10T12:00:01+00:00")
    data = evidence(root, point)
    try:
        registry.transition(
            point.point_id, "SUBMITTED", timestamp="2026-11-10T12:00:02+00:00", submission=data
        )
    except (TypeError, ValueError):
        pass  # Base has no submission contract; assert the observable state below.
    return registry, point, data


def test_p0000_item3f_submission_roundtrip_and_completion(tmp_path):
    registry, point, data = submitted(tmp_path)
    record = registry.read()[0]
    assert record.status == "SUBMITTED"
    assert record.classification is None and record.completed_at is None
    assert dict(record.submission) == {**data, "command": tuple(data["command"])}
    for name in point.outputs:
        (tmp_path / name).write_text("numerical output", encoding="utf-8")
    finished = registry.transition(point.point_id, "COMPLETED", classification="converged")
    assert finished.status == "COMPLETED"
    assert finished.submission == record.submission
    assert len(finished.outputs_sha256) == 3
    assert registry.read()[0] == finished


@pytest.mark.parametrize("status", ["PLANNED", "RUNNING", "SUBMITTED"])
def test_p0000_item3f_submitted_refuses_reexecution(tmp_path, status):
    registry, point, _ = submitted(tmp_path)
    assert registry.read()[0].status == "SUBMITTED"
    before = registry.path.read_bytes()
    with pytest.raises(ValueError, match="refusing transition"):
        registry.transition(point.point_id, status)
    assert before == registry.path.read_bytes()


@pytest.mark.parametrize("fault", ["digest", "path", "clock", "command", "terminal"])
def test_p0000_item3f_submission_decode_refuses_bad_evidence(tmp_path, fault):
    registry, _, data = submitted(tmp_path)
    assert registry.read()[0].status == "SUBMITTED"
    document = json.loads(registry.path.read_bytes())
    row = document["records"][0]
    if fault == "digest":
        row["submission"]["descriptor_sha256"] = "wrong"
    elif fault == "path":
        row["submission"]["descriptor"] = "inputs/hpc/h001.toml"
    elif fault == "clock":
        row["submission"]["finished_at"] = "2026-11-10T11:00:00+00:00"
    elif fault == "command":
        row["submission"]["command"] = "a shell string"
    else:
        row["classification"] = "converged"
    registry.path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        registry.read()
    assert data["return_code"] == 0

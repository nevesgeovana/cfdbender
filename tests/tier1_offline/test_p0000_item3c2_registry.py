"""P0000 item 3C2: record -> transition(RUNNING) -> transition(COMPLETED)."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.workspace.planning import plan_workspace

CREATED = "2026-01-01T00:00:00+00:00"
STARTED = "2026-01-01T00:00:01+00:00"
FINISHED = "2026-01-01T00:00:02+00:00"


def registered(root):
    workspace(root)
    plan = plan_workspace(root, "study")
    registry = RunRegistry(root)
    record = registry.record(plan, created_at=CREATED)[0]
    return registry, record


def transition(registry, point_id, status, **kwargs):
    method = getattr(registry, "transition", None)
    if method is None:
        return registry.read()[0]
    return method(point_id, status, **kwargs)


@pytest.mark.parametrize("classification", ["converged", "iteration_limit", "failed"])
def test_p0000_item3c2_lifecycle_roundtrip(tmp_path, classification):
    registry, original = registered(tmp_path)
    running = transition(registry, original.point_id, "RUNNING", timestamp=STARTED)
    assert running.status == "RUNNING"
    assert running.started_at == STARTED
    assert running.completed_at is None
    for name in original.point.outputs:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"saved evidence")
    completed = transition(
        registry,
        original.point_id,
        "COMPLETED",
        timestamp=FINISHED,
        classification=classification,
        error="solver failed" if classification == "failed" else None,
    )
    assert completed.status == "COMPLETED"
    assert completed.classification == classification
    assert completed.created_at == CREATED
    assert completed.started_at == STARTED
    assert completed.completed_at == FINISHED
    assert completed.point == original.point
    assert dict(completed.outputs_sha256) == {
        name: hashlib.sha256(b"saved evidence").hexdigest() for name in original.point.outputs
    }
    assert registry.read() == (completed,)


@pytest.mark.parametrize("status", ["PLANNED", "RUNNING", "COMPLETED"])
def test_p0000_item3c2_finished_record_is_immutable(tmp_path, status):
    registry, record = registered(tmp_path)
    transition(registry, record.point_id, "RUNNING", timestamp=STARTED)
    transition(
        registry,
        record.point_id,
        "COMPLETED",
        timestamp=FINISHED,
        classification="failed",
        error="no outputs",
    )
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError, match="already recorded|transition"):
        transition(registry, record.point_id, status, timestamp=FINISHED, classification="failed")
    assert registry.path.read_bytes() == before


@pytest.mark.parametrize("problem", ["skip", "clock", "classification", "missing", "lock", "drift"])
def test_p0000_item3c2_transition_refuses_without_replacing(tmp_path, problem):
    registry, record = registered(tmp_path)
    if problem in ("clock", "classification", "missing"):
        transition(registry, record.point_id, "RUNNING", timestamp=STARTED)
    if problem == "lock":
        (tmp_path / "runs.json.lock").write_text("another writer")
    if problem == "drift":
        (tmp_path / "inputs/pproc/p001.toml").write_text("history = false")
    before = registry.path.read_bytes()
    with pytest.raises(RegistryError):
        if problem in ("lock", "drift"):
            transition(registry, record.point_id, "RUNNING", timestamp=STARTED)
        else:
            transition(
                registry,
                record.point_id,
                "COMPLETED",
                timestamp=CREATED if problem == "clock" else FINISHED,
                classification="nonsense" if problem == "classification" else "converged",
            )
    assert registry.path.read_bytes() == before


def test_p0000_item3c2_decode_rejects_inconsistent_lifecycle(tmp_path):
    registry, record = registered(tmp_path)
    transition(registry, record.point_id, "RUNNING", timestamp=STARTED)
    data = json.loads(registry.path.read_bytes())
    assert data["records"][0]["status"] == "RUNNING"
    data["records"][0]["classification"] = "converged"
    registry.path.write_text(json.dumps(data))
    with pytest.raises(RegistryError):
        registry.read()

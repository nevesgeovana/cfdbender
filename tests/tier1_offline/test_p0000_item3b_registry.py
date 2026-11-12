"""P0000 item 3B: RunRegistry(root).record(plan, created_at=timestamp)."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_p0000_item3b_plan import plan_api, snapshot, study, validator


def registry_api():
    if importlib.util.find_spec("cfdbender.core.registry") is None:
        return SimpleNamespace(
            RunRegistry=lambda *a: SimpleNamespace(record=lambda *a, **kw: (), read=lambda: ()),
            RegistryError=ValueError,
        )
    return importlib.import_module("cfdbender.core.registry")


def planned(root):
    study(root)
    return plan_api().plan_matrix(root, "study", validators={"mock": validator})


def test_p0000_item3b_registry_records_per_point_digests_status_and_time(tmp_path):
    plan = planned(tmp_path)
    registry = registry_api().RunRegistry(tmp_path)
    records = registry.record(plan, created_at="2026-01-02T03:04:05+00:00")
    assert len(records) == 3, "every planned point needs its own record"
    assert records == registry.read()
    data = json.loads((tmp_path / "runs.json").read_text())
    assert data["schema_version"] == 1
    assert len(data["records"]) == 3
    for raw, point in zip(data["records"], plan.points, strict=True):
        assert raw["point_id"] == point.point_id
        assert raw["status"] == "PLANNED"
        assert raw["created_at"] == "2026-01-02T03:04:05+00:00"
        assert raw["inputs_sha256"] == dict(point.inputs_sha256)
        assert raw["solver"] == "mock"
        assert raw["matrix"] == "inputs/matrices/study.fs"
        assert raw["outputs"] == list(point.outputs)


def test_p0000_item3b_registry_refuses_duplicate_point_without_replacing_evidence(tmp_path):
    plan = planned(tmp_path)
    registry = registry_api().RunRegistry(tmp_path)
    registry.record(plan)
    before = snapshot(tmp_path)
    with pytest.raises(registry_api().RegistryError, match="point"):
        registry.record(plan)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "changed",
    [
        "init.toml",
        "inputs/matrices/study.fs",
        "inputs/setups/s002.toml",
        "inputs/references/r003.toml",
        "inputs/pproc/p001.toml",
        "inputs/geometries/mesh.dat",
    ],
)
def test_p0000_item3b_registry_refuses_changed_bytes_in_every_input(tmp_path, changed):
    plan = planned(tmp_path)
    path = tmp_path / changed
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(registry_api().RegistryError, match="changed.*" + path.name):
        registry_api().RunRegistry(tmp_path).record(plan)
    assert not (tmp_path / "runs.json").exists()
    assert not (tmp_path / "runs.json.lock").exists()


def test_p0000_item3b_registry_checks_existing_claims_across_matrices(tmp_path):
    plan = planned(tmp_path)
    registry = registry_api().RunRegistry(tmp_path)
    records = registry.record(plan)
    assert len(records) == 3
    point = replace(
        plan.points[0],
        point_id="other/DP-1",
        pol="other",
        directory="sims/sim_other/datapoints/DP-1",
    )
    other = replace(plan, matrix="inputs/matrices/another.fs", points=(point,))
    before = snapshot(tmp_path)
    with pytest.raises(registry_api().RegistryError, match="history.csv.*point"):
        registry.record(other)
    assert snapshot(tmp_path) == before


def test_p0000_item3b_registry_serializes_writers_and_never_steals_a_lock(tmp_path):
    plan = planned(tmp_path)
    lock = tmp_path / "runs.json.lock"
    lock.write_text("another writer")
    with pytest.raises(registry_api().RegistryError, match="locked"):
        registry_api().RunRegistry(tmp_path).record(plan)
    assert lock.read_text() == "another writer"
    assert not (tmp_path / "runs.json").exists()


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        '{"schema_version": 2, "records": []}',
        '{"schema_version": 1, "records": [{}]}',
    ],
)
def test_p0000_item3b_registry_refuses_corrupt_or_unknown_records(tmp_path, body):
    (tmp_path / "runs.json").write_text(body)
    with pytest.raises(registry_api().RegistryError, match="runs.json"):
        registry_api().RunRegistry(tmp_path).read()
    assert (tmp_path / "runs.json").read_text() == body


def test_p0000_item3b_registry_deterministic_with_explicit_clock(tmp_path):
    root1, root2 = tmp_path / "one", tmp_path / "two"
    for root in (root1, root2):
        plan = planned(root)
        registry_api().RunRegistry(root).record(plan, created_at="2026-01-02T03:04:05Z")
    assert (root1 / "runs.json").is_file(), "record must persist the plan"
    assert (root1 / "runs.json").read_bytes() == (root2 / "runs.json").read_bytes()


@pytest.mark.parametrize("clock", ["yesterday", "2026-01-02T03:04:05", ""])
def test_p0000_item3b_registry_requires_timezone_in_creation_time(tmp_path, clock):
    plan = planned(tmp_path)
    with pytest.raises(registry_api().RegistryError, match="created_at"):
        registry_api().RunRegistry(tmp_path).record(plan, created_at=clock)


def test_p0000_item3b_registry_refuses_foreign_workspace(tmp_path):
    plan = planned(tmp_path / "one")
    (tmp_path / "two").mkdir()
    with pytest.raises(registry_api().RegistryError, match="workspace"):
        registry_api().RunRegistry(tmp_path / "two").record(plan)

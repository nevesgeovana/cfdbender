"""P060: reading a 0.5.x run registry upgrades it only after archiving its bytes."""

import json

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.workspace.planning import plan_workspace


def test_p060_registry_migrates_schema_one_and_archives_original_once(tmp_path):
    workspace(tmp_path)
    registry = RunRegistry(tmp_path)
    registry.record(plan_workspace(tmp_path, "study"))
    document = json.loads(registry.path.read_bytes())
    document["schema_version"] = 1
    old = json.dumps(document, indent=1).encode()
    registry.path.write_bytes(old)
    records = registry.read()
    assert len(records) == 1
    assert json.loads(registry.path.read_bytes())["schema_version"] == 2
    archives = list((tmp_path / "archive").glob("*/runs.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == old
    registry.read()
    assert list((tmp_path / "archive").glob("*/runs.json")) == archives


def test_p060_invalid_legacy_registry_is_never_archived_or_rewritten(tmp_path):
    path = tmp_path / "runs.json"
    old = b'{"schema_version": 1, "records": [{}]}'
    path.write_bytes(old)
    with pytest.raises(RegistryError):
        RunRegistry(tmp_path).read()
    assert path.read_bytes() == old and not (tmp_path / "archive").exists()

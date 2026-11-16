"""P060: recover runs.json only when its recorded writer process is gone."""

import os
import subprocess
import sys

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.workspace.planning import plan_workspace


def test_p060_dead_registry_writer_is_taken_over_with_warning(tmp_path):
    workspace(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    lock = tmp_path / "runs.json.lock"
    departed = subprocess.Popen([sys.executable, "-c", "pass"])
    departed.wait(timeout=10)
    lock.write_text(str(departed.pid))
    try:
        with pytest.warns(UserWarning, match="writer.*gone|dead writer"):
            records = RunRegistry(tmp_path).record(plan)
    except RegistryError as error:
        pytest.fail(f"a dead writer must be recoverable: {error}")
    assert len(records) == 1 and not lock.exists()


def test_p060_registry_transaction_always_identifies_live_writer(tmp_path):
    registry = RunRegistry(tmp_path)
    with registry.transaction() as writer:
        assert writer.readable(), "the transaction must expose its recorded writer identity"
        writer.seek(0)
        assert writer.read().strip() == str(os.getpid())
        with pytest.raises(RegistryError, match="locked"):
            with RunRegistry(tmp_path).transaction():
                pytest.fail("a second writer acquired a live lock")
    assert not registry.lock_path.exists()

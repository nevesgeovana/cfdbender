"""Dead writer recovery must never truncate an aliased user file."""

import os

import pytest

from cfdbender.core import registry


def test_dead_lock_hardlink_preserves_user_bytes(tmp_path, monkeypatch):
    original = tmp_path / "user.txt"
    original.write_bytes(b"123456789")
    lock = tmp_path / "runs.json.lock"
    os.link(original, lock)
    monkeypatch.setattr(registry, "_writer_alive", lambda pid: False)
    with pytest.raises(registry.RegistryError, match="lock"):
        with registry.RunRegistry(tmp_path).transaction():
            pass
    assert original.read_bytes() == b"123456789"
    assert lock.read_bytes() == b"123456789"

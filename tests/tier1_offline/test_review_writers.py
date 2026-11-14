"""Management changes must respect the run writer for the whole operation."""

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.matrix.cli import main


@pytest.mark.parametrize("command", ["free-space", "sync"])
def test_held_writer_lock_prevents_file_changes(tmp_path, command):
    root, other = tmp_path / "main", tmp_path / "other"
    workspace(root)
    workspace(other)
    victim = root / "sims/sim_1001/scratch.tmp"
    victim.parent.mkdir(parents=True)
    victim.write_text("keep")
    recipes = root / "inputs/management"
    recipes.mkdir()
    (recipes / "clean.toml").write_text('[[delete_extensions]]\nextensions = [".tmp"]\n')
    (root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n'
    )
    (other / "post").mkdir()
    (other / "post/result.csv").write_text("new")
    lock = root / "runs.json.lock"
    lock.write_text("another writer")
    option = "clean" if command == "free-space" else "post"
    assert main([command, option, "--workspace", str(root), "--apply"]) == 2
    assert victim.read_text() == "keep"
    assert not (root / "post/result.csv").exists()
    assert lock.read_text() == "another writer"


def test_nested_registry_write_keeps_outer_transaction_locked(tmp_path):
    registry = RunRegistry(tmp_path)
    with registry.transaction():
        registry.replace_records({})
        assert registry.lock_path.exists()
        with pytest.raises(RegistryError, match="locked"):
            RunRegistry(tmp_path).replace_records({})
    assert not registry.lock_path.exists()


def test_delete_keeps_lock_until_folder_removal(tmp_path, monkeypatch):
    import shutil

    from cfdbender.matrix import storage  # noqa: F401

    workspace(tmp_path)
    (tmp_path / "sims/sim_1001").mkdir(parents=True)
    real = shutil.rmtree

    def remove(path):
        assert (tmp_path / "runs.json.lock").exists()
        with pytest.raises(RegistryError, match="locked"):
            RunRegistry(tmp_path).replace_records({})
        real(path)

    monkeypatch.setattr(shutil, "rmtree", remove)
    assert main(["delete-sims", "--workspace", str(tmp_path), "--sims", "1001", "--apply"]) == 0

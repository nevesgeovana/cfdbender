"""Workspace initialization excludes disposable digest state from version control."""

from cfdbender.core.workspace import init_workspace


def test_workspace_ignores_digest_cache(tmp_path):
    init_workspace(tmp_path, solver="mock", available_solvers=("mock",))
    ignore = tmp_path / ".gitignore"
    assert ignore.is_file()
    assert "/.input-digests.cache" in ignore.read_text().splitlines()


def test_existing_ignore_rules_survive(tmp_path):
    ignore = tmp_path / ".gitignore"
    ignore.write_text("custom-file\n")
    init_workspace(tmp_path, solver="mock", available_solvers=("mock",))
    assert "custom-file" in ignore.read_text().splitlines()
    assert "/.input-digests.cache" in ignore.read_text().splitlines()

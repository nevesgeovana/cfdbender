"""Sync must preserve previews and each overwritten generation."""

from test_p0000_item3c_adapter import workspace

from cfdbender.matrix.cli import main


def test_sync_preview_writes_nothing(tmp_path):
    root, other = tmp_path / "main", tmp_path / "other"
    workspace(root)
    workspace(other)
    (root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n'
    )
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["sync", "runs", "--workspace", str(root)]) == 0
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before

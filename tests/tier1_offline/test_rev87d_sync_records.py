"""R2-07: sync must not combine a point record with another execution's files."""

from test_cli_verbs import FIRST, ran
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def _sources(root, other):
    (root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n'
    )


def test_sync_does_not_fill_a_kept_records_missing_file_from_a_conflicting_run(tmp_path):
    root, other = tmp_path / "main", tmp_path / "other"
    ran(root)
    ran(other, "iteration_limit")
    _sources(root, other)
    history = root / "sims/sim_1001/datapoints/DP-NOCOND/history.csv"
    history.unlink()
    assert main(["sync", "all", "--overwrite", "--apply", "--workspace", str(root)]) == 0
    assert not history.exists(), "a kept record cannot borrow another run's output"


def test_sync_refuses_to_import_a_record_when_its_files_cannot_replace_conflicts(tmp_path):
    root, other = tmp_path / "main", tmp_path / "other"
    workspace(root)
    ran(other)
    _sources(root, other)
    history = root / "sims/sim_1001/datapoints/DP-NOCOND/history.csv"
    history.parent.mkdir(parents=True)
    history.write_text("orphaned earlier output")
    assert main(["sync", "all", "--apply", "--workspace", str(root)]) == 2
    assert FIRST not in {r.point_id for r in RunRegistry(root).read()}
    assert history.read_text() == "orphaned earlier output"

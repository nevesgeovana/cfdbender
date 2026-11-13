"""F8-005: sync never overwrites the files of a point whose record main keeps."""

from __future__ import annotations

from test_cli_verbs import FIRST, ran

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def test_overwrite_leaves_the_files_of_a_kept_record(tmp_path, capsys):
    main_root, other = tmp_path / "main", tmp_path / "other"
    main_root.mkdir()
    other.mkdir()
    ran(main_root)
    ran(other, "iteration_limit")
    (main_root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n'
    )
    history = main_root / "sims/sim_1001/datapoints/DP-NOCOND/history.csv"
    mine = history.read_bytes()
    assert mine != (other / "sims/sim_1001/datapoints/DP-NOCOND/history.csv").read_bytes()
    capsys.readouterr()
    sync = ["sync", "all", "--overwrite", "--apply", "--workspace", str(main_root)]
    assert main(sync) == 0
    text = capsys.readouterr().out
    assert f"conflict {FIRST}: main's record kept" in text
    assert "sims/sim_1001/datapoints/DP-NOCOND/history.csv: main's copy kept" in text
    assert history.read_bytes() == mine
    record = {r.point_id: r for r in RunRegistry(main_root).read()}[FIRST]
    assert record.classification == "converged"
    assert main(["collect", "study", "--workspace", str(main_root)]) == 0, "main still collects"

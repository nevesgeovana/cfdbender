"""F8-008: sync --overwrite replaces a linked file, it never writes through the link."""

from __future__ import annotations

import os

from test_cli_verbs import ran

from cfdbender.matrix.cli import main


def test_overwrite_replaces_a_hard_linked_file_and_leaves_its_source(tmp_path):
    main_root, other = tmp_path / "main", tmp_path / "other"
    main_root.mkdir()
    other.mkdir()
    ran(main_root)
    ran(other, "iteration_limit")
    (main_root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n'
    )
    relative = "sims/sim_1001/datapoints/DP-NOCOND/history.csv"
    # A staged mesh is a hard link to the workspace input where symbolic links are refused.
    shared = main_root / "inputs/shared-input.csv"
    linked = main_root / relative
    os.replace(linked, shared)
    os.link(shared, linked)
    before = shared.read_bytes()
    theirs = (other / relative).read_bytes()
    assert before != theirs
    sync = ["sync", "all", "--prefer-other", "--overwrite", "--apply"]
    assert main([*sync, "--workspace", str(main_root)]) == 0
    assert linked.read_bytes() == theirs
    assert shared.read_bytes() == before, "the copy wrote through the link into inputs/"
    saved = list((main_root / "archive").glob(f"*-sync/{relative}"))
    assert len(saved) == 1 and saved[0].read_bytes() == before

"""Regression probes for management path containment."""

import os
import subprocess
from argparse import Namespace

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix.cli import main
from cfdbender.matrix.storage import _copy_files


def link_folder(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("directory symlinks unavailable")
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "New-Item -ItemType Junction -Path $env:REVIEW_LINK -Target $env:REVIEW_TARGET",
            ],
            env={**os.environ, "REVIEW_LINK": str(link), "REVIEW_TARGET": str(target)},
            check=True,
            capture_output=True,
        )


def test_delete_refuses_a_linked_sims_parent(tmp_path):
    root = tmp_path / "workspace"
    workspace(root)
    outside = tmp_path / "outside"
    victim = outside / "sim_1001" / "evidence.txt"
    victim.parent.mkdir(parents=True)
    victim.write_text("keep")
    if (root / "sims").exists():
        (root / "sims").rmdir()
    link_folder(root / "sims", outside)
    assert main(["delete-sims", "--workspace", str(root), "--sims", "1001", "--apply"]) == 2
    assert victim.read_text() == "keep"


def test_sync_refuses_a_linked_destination_parent(tmp_path):
    root, other, outside = (tmp_path / name for name in ("main", "other", "outside"))
    root.mkdir()
    outside.mkdir()
    (other / "post").mkdir(parents=True)
    (other / "post/result.csv").write_text("new")
    link_folder(root / "post", outside)
    args = Namespace(include_archives=False, overwrite=True, apply=True)
    with pytest.raises(WorkspaceError):
        _copy_files(root, other, "post", args, [])
    assert not (outside / "result.csv").exists()

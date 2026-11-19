"""Linked product folders must not redirect collect outside the workspace."""

import os
import subprocess

import pytest
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from test_wiring_products import PPROC

from cfdbender.workspace.planning import plan_workspace


def test_linked_sections_folder_cannot_escape(tmp_path):
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace(root)
    (root / "inputs/pproc/p001.toml").write_text(PPROC)
    plan = plan_workspace(root, "study")
    run_api(plan, check_ram=lambda: None)
    post = root / "post/study"
    post.mkdir(parents=True, exist_ok=True)
    try:
        (post / "sections").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory links unavailable: {error}")
        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(post / "sections"), str(outside)],
            capture_output=True,
        )
        assert made.returncode == 0, made.stderr
    status = cli(["collect", "study", "--workspace", str(root)])
    assert list(outside.iterdir()) == [], "collect wrote through a linked product folder"
    assert status == 0

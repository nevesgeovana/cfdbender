"""Deep point trees beyond 260 characters work on Windows through the extended-length spelling."""

from __future__ import annotations

import os
import sys

import pytest
from cli_route import main
from test_cli_verbs import ran

from cfdbender.core import longpath

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the limit is a Windows one")


def _deep(folder):
    """Create a file whose path passes 260 characters; return its plain path."""
    deep = folder
    while len(str(deep)) < 300:
        deep = deep / ("processor0_with_a_long_name_" + "x" * 30)
    target = deep / "U"
    os.makedirs(longpath.extended_length(deep), exist_ok=True)
    longpath.long_path(target).write_bytes(b"field")
    assert len(str(target)) > 260
    return target


def test_the_spelling_is_unchanged_off_windows_and_prefixed_on_it():
    assert longpath.extended_length("a/b", windows=False) == "a/b"
    assert longpath.extended_length("D:/data/a", windows=True) == "\\\\?\\D:\\data\\a"
    assert longpath.extended_length("//host/share/a", windows=True) == "\\\\?\\UNC\\host\\share\\a"
    assert longpath.plain("\\\\?\\D:\\data\\a") == longpath.plain("D:\\data\\a")


@windows_only
def test_walk_size_and_remove_a_tree_deeper_than_260_characters(tmp_path):
    target = _deep(tmp_path / "t")
    files = list(longpath.iter_files(tmp_path / "t"))
    assert files == [target]
    assert longpath.file_size(target) == 5
    longpath.rmtree(tmp_path / "t")
    assert not (tmp_path / "t").exists()


@windows_only
def test_delete_sims_removes_a_point_tree_deeper_than_260_characters(tmp_path, capsys):
    ran(tmp_path)
    sim = tmp_path / "sims/sim_1001"
    target = _deep(sim / "datapoints" / "DP-NOCOND")
    assert len(str(target)) > 260
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["space-in-use", "--workspace", str(tmp_path)]) == 0
    base = ["delete-sims", "--sims", "1001", "--matrix-products", "points-only"]
    assert main([*base, "--apply", "--workspace", str(tmp_path)]) == 0
    assert not sim.exists()

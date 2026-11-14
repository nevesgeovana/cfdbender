"""R2-03: sync must create its own temporary file without following an existing link."""

import os
from argparse import Namespace

from cfdbender.matrix.storage import _copy_files


def test_sync_does_not_write_through_a_preexisting_temporary_hardlink(tmp_path):
    root, other = tmp_path / "main", tmp_path / "other"
    (root / "post").mkdir(parents=True)
    (other / "post").mkdir(parents=True)
    victim = root / "input.txt"
    victim.write_text("original input")
    planted = root / "post/.sync-result.csv.tmp"
    os.link(victim, planted)
    (other / "post/result.csv").write_text("new result")
    _copy_files(
        root, other, "post", Namespace(include_archives=False, overwrite=True, apply=True), []
    )
    assert victim.read_text() == "original input"
    assert planted.read_text() == "original input"
    assert (root / "post/result.csv").read_text() == "new result"

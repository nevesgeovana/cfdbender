"""Review round rev080-r1: regressions found reviewing c748af8..d5213ce (REV080A-NNN)."""

from __future__ import annotations

from test_080_continuation import TWO, collected, cont, finished, grow

from cfdbender.core.registry import RunRegistry
from cfdbender.solvers.su2._time_history import time_history


def test_rev080a_001_a_continuation_that_marched_nothing_does_not_block_the_retry(tmp_path):
    """A continuation whose run wrote nothing (its submission failed) is continued again."""
    record, folder = finished(tmp_path)
    grow(tmp_path, TWO)
    assert cont(tmp_path) == 0
    # Leave the folder as a continuation that ran nothing would: no history, no new restart.
    for path in folder.glob("restart_flow_*.dat"):
        if int(path.stem.rsplit("_", 1)[1]) > 35:
            path.unlink()
    for path in list(folder.glob("history*.csv")):
        path.unlink()
    assert cont(tmp_path) == 0
    assert list(time_history(folder).steps) == list(range(1, 73))
    current = RunRegistry(tmp_path).read()[-1]
    assert current.classification != "failed", current.error
    assert collected(tmp_path, current)["history"][-1]["iteration"] == 72
    segments = current.point.continuation["segments"]
    assert [(s["first_step"], s["last_step"], s["folder"]) for s in segments] == [
        (1, 36, "continued/1"),
        (37, 72, "."),
    ]

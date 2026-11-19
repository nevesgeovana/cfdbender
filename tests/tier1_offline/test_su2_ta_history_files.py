"""The history files SU2 8.6.0 writes for a restarted or two-segment time-accurate point.

Measured on a real run: a restarted time-accurate run writes ``history_<RESTART_ITER>.csv``
(five digits) and no ``history.csv``; a point with an export window has one history per
segment. The readers join them in step order and name every file they looked for.
"""

from __future__ import annotations

import pytest
from test_segment_execution import point_with_exports

from cfdbender.solvers import su2
from cfdbender.solvers.su2._history import read
from cfdbender.solvers.su2._time_history import time_history

REFS = "REF_AREA= 1\nREF_LENGTH= 1\nREF_ORIGIN_MOMENT_X= 0\nREF_ORIGIN_MOMENT_Y= 0\n"
REFS += "REF_ORIGIN_MOMENT_Z= 0\n"
LOADS = "Time_Iter,Inner_Iter,rms[Rho],CFx,CFy,CFz,CMx,CMy,CMz"


def _history(folder, name, steps, header=LOADS):
    rows = [header] + [f"{k},0,-3,0.1,0.2,0.3,0.4,0.5,0.6" for k in steps]
    (folder / name).write_text("\n".join(rows) + "\n", encoding="ascii")


def _segments(folder, first="TIME_DOMAIN= YES\nTIME_ITER= 3\n"):
    folder.mkdir()
    (folder / "su2.cfg").write_text(REFS + first, encoding="ascii")
    (folder / "su2_segment2.cfg").write_text(
        REFS + "TIME_DOMAIN= YES\nRESTART_SOL= YES\nRESTART_ITER= 3\nTIME_ITER= 6\n",
        encoding="ascii",
    )


def test_the_segments_join_in_step_order(tmp_path):
    folder = tmp_path / "p"
    _segments(folder)
    _history(folder, "history.csv", [0, 1, 2])
    _history(folder, "history_00003.csv", [3, 4, 5])
    assert list(time_history(folder).steps) == [1, 2, 3, 4, 5, 6]
    result = read(folder)[0]
    assert [i.iteration for i in result.history] == [1, 2, 3, 4, 5, 6]
    assert result.iteration_limit == 6


def test_a_steady_start_counts_the_rows_steps_from_one(tmp_path):
    folder = tmp_path / "p"
    _segments(folder, first="TIME_DOMAIN= YES\nRESTART_SOL= YES\nRESTART_ITER= 2\nTIME_ITER= 3\n")
    _history(folder, "history_00002.csv", [2])
    _history(folder, "history_00003.csv", [3, 4, 5])
    assert list(time_history(folder).steps) == [1, 2, 3, 4]
    result = read(folder)[0]
    assert [i.iteration for i in result.history] == [1, 2, 3, 4]
    assert result.iteration_limit == 4


@pytest.mark.parametrize(("steps", "word"), [([0, 1], "gap"), ([0, 1, 2, 3], "overlap")])
def test_a_gap_or_an_overlap_between_segments_is_refused_by_name(tmp_path, steps, word):
    folder = tmp_path / "p"
    _segments(folder)
    _history(folder, "history.csv", steps)
    _history(folder, "history_00003.csv", [3, 4, 5])
    with pytest.raises(ValueError, match=f"{word}.*history.csv.*history_00003.csv"):
        time_history(folder)
    assert word in str(read(folder)[0].evidence_error)


def test_a_missing_history_names_the_files_looked_for(tmp_path):
    folder = tmp_path / "p"
    _segments(folder)
    with pytest.raises(ValueError, match="history.csv, history_00003.csv"):
        time_history(folder)
    assert "history.csv, history_00003.csv" in str(read(folder)[0].evidence_error)


def test_a_two_segment_point_declares_and_waits_for_both_histories(tmp_path):
    point = point_with_exports(tmp_path).point
    names = [name.rsplit("/", 1)[-1] for name in point.outputs]
    assert "history.csv" in names and "history_00009.csv" in names
    assert set(su2.job_output_files(point)) == {"stdout.log", "history.csv", "history_00009.csv"}

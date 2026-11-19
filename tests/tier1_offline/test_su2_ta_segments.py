"""The export window of a time-accurate SU2 point: segment two reads what segment one wrote.

Measured on a real SU2 8.6.0 run: segment two at RESTART_ITER R reads
<SOLUTION_FILENAME stem>_<R-2> and _<R-1>, the restart files segment one wrote under its
RESTART_FILENAME. The generated configurations must run as they are, with no edit.
"""

from __future__ import annotations

from test_su2_time_restart import WINDOW, collected, keys, run_ta, workspace

from cfdbender.solvers.su2._time_history import time_history


def test_the_export_window_hands_over_to_segment_two_as_generated(tmp_path):
    record, folder = run_ta(workspace(tmp_path, WINDOW))
    assert record.classification != "failed", record.error
    first, second = keys(folder / "su2.cfg"), keys(folder / "su2_segment2.cfg")
    assert (first["RESTART_ITER"], first["TIME_ITER"]) == ("2", "29")
    assert (second["RESTART_ITER"], second["TIME_ITER"]) == ("29", "38")
    assert second["SOLUTION_FILENAME"] == "restart_flow.dat"
    for k in (27, 28):
        assert (folder / f"restart_flow_{k:05d}.dat").is_file()
    surfaces = sorted(path.name for path in folder.glob("surface_flow_*.vtu"))
    assert surfaces == [f"surface_flow_{k:05d}.vtu" for k in range(29, 38)]
    assert sorted(p.name for p in folder.glob("history*.csv")) == [
        "history_00002.csv",
        "history_00029.csv",
    ]
    assert list(time_history(folder).steps) == list(range(1, 37))
    average = collected(tmp_path, record)["time_average"]
    assert (average["FIRST_STEP"], average["LAST_STEP"]) == (28, 36)
    assert len(average["per_revolution"]) == 1


def test_a_cold_start_export_window_reads_what_segment_one_wrote(tmp_path):
    record, folder = run_ta(workspace(tmp_path, WINDOW, restart=False))
    assert record.classification != "failed", record.error
    second = keys(folder / "su2_segment2.cfg")
    assert (second["RESTART_ITER"], second["SOLUTION_FILENAME"]) == ("27", "restart_flow.dat")
    assert sorted(p.name for p in folder.glob("history*.csv")) == [
        "history.csv",
        "history_00027.csv",
    ]
    assert list(time_history(folder).steps) == list(range(1, 37))


def test_an_average_longer_than_the_export_window_runs_both_segments(tmp_path):
    # SU2 8.6.0 refuses a restart whose WINDOW_START_ITER is below RESTART_ITER (measured):
    # segment two starts SU2's own time average at its first iteration.
    var = WINDOW.replace("LAST_REVS_AVG: 0.25", "LAST_REVS_AVG: 0.5")
    record, folder = run_ta(workspace(tmp_path, var))
    assert record.classification != "failed", record.error
    first, second = keys(folder / "su2.cfg"), keys(folder / "su2_segment2.cfg")
    assert (first["WINDOW_START_ITER"], second["WINDOW_START_ITER"]) == ("20", "29")
    average = collected(tmp_path, record)["time_average"]
    assert (average["FIRST_STEP"], average["LAST_STEP"], average["STEPS"]) == (19, 36, 18)

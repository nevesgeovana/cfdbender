"""What a time-accurate SU2 point writes to disk, step by step.

pyflightstream exports an unsteady point's files at its last step when the row states no
``EXPORT_UNSTEADY_*`` key. A real SU2 8.6.0 run with no output frequency wrote a volume, a
surface and a restart file at every step (88, 1.1 and 44 MiB per step on a 476k-point mesh).
SU2 writes each file at every multiple of its ``OUTPUT_WRT_FREQ`` and always at its last time
iteration, which these tests rely on.
"""

from __future__ import annotations

import test_runtypes_b1_workflows as rt
from test_su2_time_restart import CLOCK, WINDOW, keys, run_ta, workspace


def _names(folder, pattern):
    return sorted(path.name for path in folder.glob(pattern))


def test_without_an_export_window_only_the_last_step_is_written(tmp_path):
    record, folder = run_ta(workspace(tmp_path, CLOCK))
    assert record.classification != "failed", record.error
    cfg = keys(folder / "su2.cfg")
    assert cfg["OUTPUT_FILES"] == "( RESTART, PARAVIEW, SURFACE_PARAVIEW )"
    assert cfg["OUTPUT_WRT_FREQ"] == "( 38, 38, 38 )"
    assert _names(folder, "*flow_0*.vtu") == ["flow_00037.vtu", "surface_flow_00037.vtu"]
    # Second-order dual time: SU2 writes the last two restart files, a continuation's pair.
    assert _names(folder, "restart_flow_*.dat") == [
        "restart_flow_00036.dat",
        "restart_flow_00037.dat",
    ]


def test_the_segments_keep_only_the_restart_files_the_hand_over_reads(tmp_path):
    record, folder = run_ta(workspace(tmp_path, WINDOW))
    assert record.classification != "failed", record.error
    first, second = keys(folder / "su2.cfg"), keys(folder / "su2_segment2.cfg")
    # Segment one's frequency is its TIME_ITER: SU2 writes its last two iterations only.
    assert (first["OUTPUT_FILES"], first["OUTPUT_WRT_FREQ"]) == ("( RESTART )", "29")
    assert second["OUTPUT_WRT_FREQ"] == "( 38, 1, 1 )"
    assert _names(folder, "restart_flow_*.dat") == [
        "restart_flow_00027.dat",
        "restart_flow_00028.dat",
        "restart_flow_00036.dat",
        "restart_flow_00037.dat",
    ]
    assert _names(folder, "flow_*.vtu") == [f"flow_{k:05d}.vtu" for k in range(29, 38)]


def test_a_stated_output_frequency_wins_on_a_row_without_an_export_window(tmp_path):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=CLOCK, setup="OUTPUT_WRT_FREQ = 4\n")
    _, lines, _ = rt.deck(tmp_path)
    assert lines["OUTPUT_WRT_FREQ"] == "4" and "OUTPUT_FILES" not in lines


def test_a_window_from_step_one_writes_every_step_but_the_restart_at_the_end(tmp_path):
    rt.workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 20 / LAST_ITERS_AVG: 10 / "
        "EXPORT_UNSTEADY_AFTER_ITER: 1",
    )
    _, lines, files = rt.deck(tmp_path)
    assert lines["OUTPUT_WRT_FREQ"] == "( 20, 1, 1 )" and files == {}

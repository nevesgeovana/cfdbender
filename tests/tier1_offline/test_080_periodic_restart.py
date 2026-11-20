"""RESTART_EVERY_REV (provisional name): a time-accurate point writes restart files periodically.

A point killed by its walltime keeps nothing to continue from when its restart files are written
only at its last step (``OUTPUT_WRT_FREQ`` = ``TIME_ITER``). The row key makes SU2 write the
restart file every ``r`` revolutions (with the second-order pair SU2 writes before each multiple)
without writing the volume and surface files more often. SU2 8.6.0 writes each file of
``OUTPUT_FILES`` at every time iteration that is a multiple of its own ``OUTPUT_WRT_FREQ`` entry.
"""

from __future__ import annotations

import warnings

import pytest
import test_runtypes_b1_workflows as rt
from test_su2_time_restart import CLOCK, keys, run_ta, workspace

from cfdbender.workspace.planning import plan_workspace

EVERY = CLOCK + " / RESTART_EVERY_REV: 0.25"


def flat(text):
    return text.replace(" ", "")


def test_the_restart_file_is_written_every_quarter_revolution(tmp_path):
    record, folder = run_ta(workspace(tmp_path, EVERY, restart=False))
    assert record.classification != "failed", record.error
    cfg = keys(folder / "su2.cfg")
    # 36 steps per revolution: the restart every 9, the volume and surface at the end only.
    assert flat(cfg["OUTPUT_FILES"]) == "(RESTART,PARAVIEW,SURFACE_PARAVIEW)"
    assert flat(cfg["OUTPUT_WRT_FREQ"]) == "(9,36,36)"
    restarts = sorted(int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("restart_flow_*.dat"))
    assert {8, 9, 17, 18, 26, 27, 34, 35} <= set(restarts)
    volumes = sorted(p.name for p in folder.glob("flow_*.vtu"))
    assert volumes == ["flow_00000.vtu", "flow_00035.vtu"]


def test_both_segments_of_an_export_window_write_the_periodic_restart(tmp_path):
    var = EVERY + " / EXPORT_UNSTEADY_LAST_ITER: 9"
    record, folder = run_ta(workspace(tmp_path, var, restart=False))
    assert record.classification != "failed", record.error
    first, second = keys(folder / "su2.cfg"), keys(folder / "su2_segment2.cfg")
    assert (flat(first["OUTPUT_FILES"]), first["OUTPUT_WRT_FREQ"]) == ("(RESTART)", "9")
    assert flat(second["OUTPUT_WRT_FREQ"]) == "(9,1,1)"


def test_the_key_is_recorded_with_its_steps(tmp_path):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=EVERY)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (point,) = plan_workspace(tmp_path, "rt").points
    assert point.variables["RESTART_EVERY_REV"] in (0.25, "0.25")


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_a_value_that_is_not_a_positive_number_is_refused(tmp_path, value):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=CLOCK + f" / RESTART_EVERY_REV: {value}")
    with pytest.raises(Exception, match="RESTART_EVERY_REV"):
        plan_workspace(tmp_path, "rt")


def test_a_stated_output_frequency_and_the_key_are_refused_together(tmp_path):
    rt.workspace(
        tmp_path,
        workflow="unsteady_rotor",
        var=EVERY,
        setup="OUTPUT_WRT_FREQ = 5\n",
    )
    with pytest.raises(Exception, match="RESTART_EVERY_REV.*OUTPUT_WRT_FREQ"):
        plan_workspace(tmp_path, "rt")


def test_a_row_with_no_rotor_clock_warns_and_skips_the_key(tmp_path):
    rt.workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0, BETA:0",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 20 / LAST_ITERS_AVG: 5 / RESTART_EVERY_REV: 1",
    )
    with pytest.warns(UserWarning, match="RESTART_EVERY_REV"):
        (point,) = plan_workspace(tmp_path, "rt").points
    assert point.point_id

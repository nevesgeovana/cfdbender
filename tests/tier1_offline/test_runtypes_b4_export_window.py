"""The export window: pyflightstream's four keys, resolved agnostically; SU2 runs two segments."""

from __future__ import annotations

import warnings

import pytest
from test_runtypes_b1_workflows import deck, workspace

from cfdbender.core.export_window import export_window
from cfdbender.core.plan import PlanError


def window(variables, n=108, spr=36.0):
    return export_window(
        variables,
        pol="7001",
        time_iterations=n,
        steps_per_revolution=spr,
    )


@pytest.mark.parametrize(
    ("variables", "first"),
    [
        ({"EXPORT_UNSTEADY_AFTER_ITER": 10}, 10),
        ({"EXPORT_UNSTEADY_AFTER_REV": 2}, 72),
        ({"EXPORT_UNSTEADY_LAST_ITER": 8}, 101),
        ({"EXPORT_UNSTEADY_LAST_REV": 1}, 73),
        ({"EXPORT_UNSTEADY_LAST_REV": 0.1}, 105),
    ],
)
def test_each_key_resolves_to_its_first_step_and_ends_at_the_last(variables, first):
    found = window(variables)
    assert found is not None and (found.first_step, found.last_step) == (first, 108)
    assert found.record()["form"] == next(iter(variables))


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        (
            {"EXPORT_UNSTEADY_AFTER_ITER": 1, "EXPORT_UNSTEADY_LAST_ITER": 2},
            "states EXPORT_UNSTEADY_LAST_ITER and EXPORT_UNSTEADY_AFTER_ITER both",
        ),
        ({"EXPORT_UNSTEADY_LAST_ITER": 0}, "so it is a positive number"),
        ({"EXPORT_UNSTEADY_AFTER_ITER": 2.5}, "not a whole number of steps"),
        ({"EXPORT_UNSTEADY_LAST_ITER": 200}, "more than the 108 steps the run is long"),
        ({"EXPORT_UNSTEADY_AFTER_REV": 4}, "which is step 144, and the run is 108 steps long"),
    ],
)
def test_the_refusals(variables, message):
    with pytest.raises(ValueError, match=message):
        window(variables)


def test_the_export_and_averaging_windows_are_independent(tmp_path):
    # pyflightstream refuses an export window beginning after the average only for a surface
    # time average ([time_averaging] of the pproc), which cfdbender does not read.
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 5 / REVOLUTIONS: 1 / LAST_REVS_AVG: 0.5 / EXPORT_UNSTEADY_LAST_ITER: 12",
    )
    _, lines, files = deck(tmp_path)
    assert lines["WINDOW_START_ITER"] == "36" and lines["TIME_ITER"] == "60"
    second = dict(line.split("= ", 1) for line in files["su2_segment2.cfg"].splitlines())
    assert second["RESTART_ITER"] == "60"


def test_a_revolutions_form_without_a_rotor_clock_is_warned_and_skipped():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert window({"EXPORT_UNSTEADY_LAST_REV": 1}, spr=None) is None
    assert any("has no rotor clock" in str(w.message) for w in caught)


def test_su2_runs_the_point_in_two_segments_of_one_job(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var=("DELTA_THETA: 3 / REVOLUTIONS: 5 / LAST_REVS_AVG: 2.5 / EXPORT_UNSTEADY_LAST_REV: 3"),
    )
    point, lines, files = deck(tmp_path)
    assert (lines["TIME_ITER"], lines["OUTPUT_FILES"]) == ("240", "( RESTART )")
    second = dict(line.split("= ", 1) for line in files["su2_segment2.cfg"].splitlines())
    assert (second["TIME_ITER"], second["RESTART_SOL"], second["RESTART_ITER"]) == (
        "600",
        "YES",
        "240",
    )
    # Segment one writes only the two restart files segment two reads (238 and 239): its
    # frequency is its TIME_ITER, and SU2 writes the last two of a second-order run.
    assert lines["OUTPUT_WRT_FREQ"] == "240"
    assert second["OUTPUT_WRT_FREQ"] == "( 600, 1, 1 )"
    assert second["OUTPUT_FILES"] == "( RESTART, PARAVIEW, SURFACE_PARAVIEW )"
    assert {k for k in second if k not in lines} == {
        "RESTART_SOL",
        "RESTART_ITER",
        "SOLUTION_FILENAME",
        "READ_BINARY_RESTART",
    }
    # Segment two reads the restart files segment one writes under SU2's default name.
    assert (second["SOLUTION_FILENAME"], second["READ_BINARY_RESTART"]) == ("restart", "YES")
    script = files["run_segments.sh"]
    assert script.index("su2.cfg") < script.index("su2_segment2.cfg")
    for name in ("su2_segment2.cfg", "run_segments.sh"):
        assert any(output.endswith("/" + name) for output in point.outputs)


def test_an_export_window_from_the_first_step_is_one_segment(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 20 / LAST_ITERS_AVG: 10 / "
        "EXPORT_UNSTEADY_AFTER_ITER: 1",
    )
    _, lines, files = deck(tmp_path)
    assert lines["OUTPUT_WRT_FREQ"] == "( 20, 1, 1 )" and files == {}


def test_an_export_key_on_a_steady_row_is_warned_and_skipped(tmp_path):
    workspace(
        tmp_path, condition="MACH:0.0529, ALPHA:0", rotor="", var="EXPORT_UNSTEADY_LAST_ITER: 5"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lines = deck(tmp_path)[1]
    assert "TIME_DOMAIN" not in lines
    assert any(
        "EXPORT_UNSTEADY_LAST_ITER is not registered for run type 'steady'" in str(w.message)
        for w in caught
    )


def test_a_planning_refusal_names_the_row(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 20 / LAST_ITERS_AVG: 10 / "
        "EXPORT_UNSTEADY_LAST_ITER: 30",
    )
    with pytest.raises(PlanError, match="POL 7001"):
        deck(tmp_path)

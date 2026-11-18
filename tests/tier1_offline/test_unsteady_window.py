"""The averaging window and the time average over it, as pure functions.

Every expected window and mean below is ARITHMETIC ON THE ROW AND THE CLOCK, worked out
in the test and never read off the function under test. A window is inclusive, counted in
solver steps from one, ends at the last step of the run and is exactly as long as the row
states: one revolution of 100 steps ending at step 400 is steps 301 to 400.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfdbender.core.settings import WorkspaceError
from cfdbender.core.unsteady_window import (
    LAST_ITERS_AVG,
    LAST_REVS_AVG,
    TimeHistory,
    averaging_span,
    averaging_steps,
    refuse_uncovered_window,
    stated_key,
    step_clock,
    window_average,
    window_is_covered,
)


def test_the_span_is_the_last_revolutions_of_the_clock_or_the_last_iterations() -> None:
    # 2 revolutions of 250 steps ending at step 1000 is 501 to 1000.
    assert averaging_span({"LAST_REVS_AVG": "2"}, last_step=1000, per_revolution=250.0) == (
        501,
        1000,
    )
    # Half a revolution is 125 steps: 876 to 1000. It takes a float.
    assert averaging_span({"LAST_REVS_AVG": "0.5"}, last_step=1000, per_revolution=250.0) == (
        876,
        1000,
    )
    assert averaging_span({"LAST_ITERS_AVG": "100"}, last_step=1000, per_revolution=None) == (
        901,
        1000,
    )
    # Longer than the run is the whole run, never before step one.
    assert averaging_span({"LAST_REVS_AVG": "99"}, last_step=1000, per_revolution=250.0) == (
        1,
        1000,
    )


def test_the_window_keeps_the_last_revolutions_and_is_as_long_as_the_row_states() -> None:
    # Four revolutions of 100 steps; the last ONE is 301 to 400, and 101 to 400 would be the
    # rule that discards the first revolution.
    assert averaging_span({"LAST_REVS_AVG": "1.0"}, last_step=400, per_revolution=100) == (301, 400)
    for stated, steps in (("100", 100), ("1", 1), ("37", 37)):
        span = averaging_span({"LAST_ITERS_AVG": stated}, last_step=400, per_revolution=None)
        assert span is not None
        first, last = span
        assert last == 400 and last - first + 1 == steps
    # More revolutions move the FIRST step earlier; the last step does not move.
    assert averaging_span({"LAST_REVS_AVG": "3.0"}, last_step=400, per_revolution=100) == (101, 400)


@pytest.mark.parametrize("variables", [{}, {"LAST_REVS_AVG": "-"}, {"LAST_ITERS_AVG": "NA"}])
def test_a_row_that_states_no_usable_key_resolves_to_nothing(variables: dict[str, str]) -> None:
    assert stated_key(variables) is None
    assert averaging_span(variables, last_step=1000, per_revolution=250.0) is None


def test_revolutions_without_a_clock_resolve_to_nothing_rather_than_a_guess() -> None:
    assert averaging_span({"LAST_REVS_AVG": "1"}, last_step=1000, per_revolution=None) is None
    assert averaging_span({"LAST_REVS_AVG": "1"}, last_step=1000, per_revolution=math.nan) is None
    assert averaging_span({"LAST_REVS_AVG": "1"}, last_step=1000, per_revolution=0.0) is None
    # A count of iterations needs no clock and a last step that is not positive has no window.
    assert averaging_span({"LAST_ITERS_AVG": "5"}, last_step=0, per_revolution=None) is None


@pytest.mark.parametrize(
    "variables",
    [
        {"LAST_REVS_AVG": "0"},
        {"LAST_REVS_AVG": "-2"},
        {"LAST_ITERS_AVG": "nope"},
        {"LAST_ITERS_AVG": "inf"},
        {"LAST_ITERS_AVG": True},
        {"LAST_REVS_AVG": "1", "LAST_ITERS_AVG": "10"},
    ],
)
def test_an_invalid_stated_window_is_refused(variables: dict[str, object]) -> None:
    with pytest.raises(WorkspaceError):
        averaging_span(variables, last_step=1000, per_revolution=250.0)


def test_both_keys_are_refused_naming_both() -> None:
    with pytest.raises(WorkspaceError) as refused:
        stated_key({"LAST_REVS_AVG": "1", "LAST_ITERS_AVG": "10"})
    text = str(refused.value)
    assert "LAST_REVS_AVG = 1" in text and "LAST_ITERS_AVG = 10" in text
    assert "state only one averaging key" in text


def test_a_window_under_one_step_is_refused() -> None:
    # 0.001 revolution of 250 steps is a quarter of a step.
    with pytest.raises(WorkspaceError, match="averages nothing"):
        averaging_steps({"LAST_REVS_AVG": "0.001"}, per_revolution=250.0)


@pytest.mark.parametrize(("steps", "expected"), [(2.5, 2), (3.5, 4), (2.4, 2), (2.6, 3), (1.5, 2)])
def test_a_fractional_window_rounds_to_the_nearest_step_halves_to_even(
    steps: float, expected: int
) -> None:
    assert averaging_steps({"LAST_ITERS_AVG": steps}, per_revolution=None) == expected


def test_the_keys_are_spelled_once_in_upper_case() -> None:
    assert (LAST_REVS_AVG, LAST_ITERS_AVG) == ("LAST_REVS_AVG", "LAST_ITERS_AVG")
    assert stated_key({"LAST_REVS_AVG": "2.5"}) == ("LAST_REVS_AVG", 2.5)
    # a lower case key is another key and states nothing
    assert stated_key({"last_revs_avg": "2.5"}) is None


# ---------------------------------------------------------------- the step clock


def test_the_step_clock_is_the_stated_one_where_it_is_strictly_increasing_whole_numbers() -> None:
    assert step_clock(["3", "4", "5"]).tolist() == [3, 4, 5]
    assert step_clock([10, 20, 30]).tolist() == [10, 20, 30]
    assert step_clock(np.array([1.0, 2.0, 4.0])).tolist() == [1, 2, 4]


@pytest.mark.parametrize(
    "cells",
    [["3", "3", "5"], ["5", "4", "6"], ["1.5", "2.5"], ["a", "2"], [1.0, math.nan], [None, 1]],
)
def test_the_step_clock_falls_back_to_the_ordinal_for_the_whole_table(cells: list[object]) -> None:
    assert step_clock(cells).tolist() == list(range(1, len(cells) + 1))


def test_an_empty_clock_is_empty() -> None:
    assert step_clock([]).tolist() == []


# ---------------------------------------------------------------- the average


def _history(values: list[float], first: int = 1) -> TimeHistory:
    steps = np.arange(first, first + len(values))
    return TimeHistory(
        steps=steps,
        columns={"cp": np.array(values), "twice": 2.0 * np.array(values)},
    )


def test_the_average_is_inclusive_at_both_ends_and_ignores_what_is_outside() -> None:
    # Steps 2 to 4 hold 2, 3, 4: the mean is 3 and the frames outside never reach it.
    average = window_average(_history([1.0, 2.0, 3.0, 4.0, 100.0]), window=(2, 4))
    assert average.n_frames == 3
    assert average.window == (2, 4)
    assert average.source_window == (2, 4)
    assert average.means["cp"] == 3.0
    assert average.means["twice"] == 6.0


def test_the_average_uses_the_steps_the_history_states_not_the_row_index() -> None:
    # A clock that does not begin at one: steps 101 to 104 hold 1, 2, 3, 4.
    average = window_average(_history([1.0, 2.0, 3.0, 4.0], first=101), window=(102, 103))
    assert average.n_frames == 2 and average.means["cp"] == 2.5
    assert average.source_window == (102, 103)


def test_the_average_reports_the_steps_it_read() -> None:
    read: set[int] = set()
    window_average(_history([1.0, 2.0, 3.0, 4.0]), window=(3, 9), read_steps=read)
    assert read == {3, 4}


def test_the_average_of_a_sparse_clock_names_the_steps_actually_averaged() -> None:
    history = TimeHistory(
        steps=np.array([2, 4, 6, 8]), columns={"x": np.array([1.0, 3.0, 5.0, 7.0])}
    )
    average = window_average(history, window=(3, 7))
    assert average.n_frames == 2 and average.means["x"] == 4.0
    assert average.window == (3, 7) and average.source_window == (4, 6)


def test_the_time_is_carried_and_never_averaged() -> None:
    history = TimeHistory(
        steps=np.array([1, 2]), columns={"x": np.array([1.0, 3.0])}, time_s=np.array([0.1, 0.2])
    )
    assert window_average(history, window=(1, 2)).means == {"x": 2.0}


def test_the_average_refuses_a_window_holding_no_frame() -> None:
    with pytest.raises(WorkspaceError, match="no frame"):
        window_average(_history([1.0, 2.0]), window=(40, 50))


def test_the_average_refuses_a_backwards_window_and_an_empty_history() -> None:
    with pytest.raises(WorkspaceError, match="ends before it starts"):
        window_average(_history([1.0, 2.0]), window=(2, 1))
    empty = TimeHistory(steps=np.array([], dtype=int), columns={"x": np.array([])})
    with pytest.raises(WorkspaceError, match="holds no frame"):
        window_average(empty, window=(1, 2))


def test_a_nan_in_the_window_makes_its_mean_nan_and_leaves_the_others() -> None:
    history = TimeHistory(
        steps=np.array([1, 2, 3]),
        columns={"a": np.array([1.0, math.nan, 3.0]), "b": np.array([1.0, 2.0, 3.0])},
    )
    average = window_average(history, window=(1, 3))
    assert math.isnan(average.means["a"]) and average.means["b"] == 2.0


def test_a_column_that_does_not_match_the_steps_is_refused() -> None:
    with pytest.raises(WorkspaceError, match="one value per step"):
        TimeHistory(steps=np.array([1, 2, 3]), columns={"x": np.array([1.0, 2.0])})


# ---------------------------------------------------------------- the coverage


def test_a_history_that_holds_every_step_of_the_window_is_covered() -> None:
    assert window_is_covered(_history([1.0] * 10), (4, 10)) is None
    assert window_is_covered(_history([1.0] * 10, first=101), (103, 105)) is None


def test_a_history_that_stops_early_is_left_out_with_the_reason() -> None:
    reason = window_is_covered(_history([1.0] * 8), (5, 12))
    assert reason == (
        "the row states steps 5 to 12 and the history holds steps 1 to 8, "
        "with a missing sample or gap in that window"
    )


def test_a_gap_inside_the_window_is_not_a_cover() -> None:
    history = TimeHistory(steps=np.array([1, 2, 4, 5]), columns={"x": np.zeros(4)})
    assert window_is_covered(history, (1, 5)) is not None
    assert window_is_covered(history, (4, 5)) is None


def test_an_empty_history_covers_nothing() -> None:
    empty = TimeHistory(steps=np.array([], dtype=int), columns={})
    reason = window_is_covered(empty, (1, 3))
    assert reason is not None and "holds no step" in reason


def test_a_window_reaching_outside_the_history_is_refused_by_name() -> None:
    history = _history([1.0] * 4, first=11)
    with pytest.raises(WorkspaceError) as refused:
        refuse_uncovered_window(history, (12, 20), reduction="time_average", index=2)
    text = str(refused.value)
    assert "from step 11 to step 14" in text
    assert "time_average window 2 spans steps 12 to 20" in text
    assert "an average of a run that did not finish writing" in text
    with pytest.raises(WorkspaceError):
        refuse_uncovered_window(history, (10, 12), reduction="time_average")
    # inside the history, nothing is said
    refuse_uncovered_window(history, (12, 14), reduction="time_average")

"""The time average of a history over a window of solver steps, and the step clock."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfdbender.core.settings import WorkspaceError
from cfdbender.post.time_average import (
    TimeHistory,
    refuse_uncovered_window,
    step_clock,
    window_average,
    window_is_covered,
)

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

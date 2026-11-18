"""The averaging window and the time average over it, as pure functions.

Every expected window and mean below is ARITHMETIC ON THE ROW AND THE CLOCK, worked out
in the test and never read off the function under test. A window is inclusive, counted in
solver steps from one, ends at the last step of the run and is exactly as long as the row
states: one revolution of 100 steps ending at step 400 is steps 301 to 400.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.core.settings import WorkspaceError
from cfdbender.core.unsteady_window import (
    LAST_ITERS_AVG,
    LAST_REVS_AVG,
    averaging_span,
    averaging_steps,
    stated_key,
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

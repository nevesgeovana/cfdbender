"""The clock of an unsteady row and the azimuth of the rotor at each step, as pure functions.

Every expected number is worked by hand from the row: one revolution lasts 60/rpm seconds,
so one degree lasts 1/(6 rpm) seconds; a revolution is 360 degrees, so a ten degree step is
thirty-six steps exactly.
"""

from __future__ import annotations

import math

import pytest

from cfdbender.core.settings import WorkspaceError
from cfdbender.core.unsteady_clock import (
    BladeDatum,
    TimeStepping,
    blade_azimuth_deg,
    blade_datum,
    blade_one_azimuth,
    revolution_steps,
    rotor_time_stepping,
    unsteady_time_stepping,
)

RPM = 473.1778


def _angular(**over: str) -> dict[str, object]:
    row: dict[str, object] = {"DELTA_THETA": "10.0", "REVOLUTIONS": "1.5"}
    row.update(over)
    return row


def test_the_azimuthal_step_and_the_revolutions_set_the_whole_clock() -> None:
    stepping = rotor_time_stepping("7001", _angular(), rpm=RPM)
    assert stepping.stated_form == "angular"
    # 1.5 revolutions x 360 / 10 degrees
    assert stepping.time_iterations == 54
    assert stepping.steps_per_revolution == 36.0
    assert stepping.delta_time_s == 10.0 / (6.0 * RPM)
    assert stepping.delta_theta_deg == 10.0 and stepping.revolutions == 1.5
    assert stepping.rpm == RPM


def test_the_direction_of_rotation_does_not_change_the_clock() -> None:
    forward = rotor_time_stepping("7001", _angular(), rpm=RPM)
    reverse = rotor_time_stepping("7001", _angular(), rpm=-RPM)
    assert reverse.delta_time_s == forward.delta_time_s
    assert reverse.time_iterations == forward.time_iterations
    assert reverse.rpm == -RPM


def test_the_explicit_clock_is_carried_through_unconverted() -> None:
    stepping = rotor_time_stepping(
        "7001", {"DELTA_TIME": "1e-4", "TIME_ITERATIONS": "720"}, rpm=-1200.0
    )
    assert stepping.stated_form == "explicit"
    assert stepping.delta_time_s == 1e-4 and stepping.time_iterations == 720
    assert stepping.delta_theta_deg is None and stepping.revolutions is None
    assert stepping.rpm == -1200.0
    # 60 / (1200 x 1e-4) = 500 steps a revolution, from the seconds because no angle was stated
    assert stepping.steps_per_revolution == pytest.approx(500.0)


def test_an_explicit_clock_with_no_speed_has_no_revolution() -> None:
    stepping = rotor_time_stepping("7001", {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "10"})
    assert stepping.rpm is None and stepping.steps_per_revolution is None


def test_steps_per_revolution_follow_the_stated_angle_not_the_rounded_seconds() -> None:
    # 36.0238 would be read back from seconds rounded to five decimals; the angle says 36.
    stepping = TimeStepping("angular", 0.00352, 54, 10.0, 1.5, 473.1723)
    assert stepping.steps_per_revolution == 36.0


@pytest.mark.parametrize("rpm", [0.0, None])
def test_a_rotor_at_rest_has_no_revolution(rpm: float | None) -> None:
    assert TimeStepping("explicit", 0.01, 10, None, None, rpm).steps_per_revolution is None
    assert TimeStepping("explicit", 5e-324, 10, None, None, 1e-3).steps_per_revolution is None


def test_the_record_carries_the_divisor_beside_the_quotient() -> None:
    record = rotor_time_stepping("7001", _angular(), rpm=RPM).record()
    assert record == {
        "form": "angular",
        "delta_time_s": 10.0 / (6.0 * RPM),
        "time_iterations": 54,
        "delta_theta_deg": 10.0,
        "revolutions": 1.5,
        "rpm": RPM,
        "steps_per_revolution": 36.0,
    }


def test_revolutions_that_are_not_a_whole_number_of_steps_are_refused() -> None:
    with pytest.raises(WorkspaceError) as raised:
        rotor_time_stepping("7001", _angular(DELTA_THETA="7.0"), rpm=RPM)
    message = str(raised.value)
    # 1.5 x 360 / 7 = 77.142857 steps; 77 whole steps are 1.49722 revolutions
    assert "7.0" in message and "1.5" in message and "77.1429" in message
    assert "1.49722 revolutions is 77 whole steps" in message


@pytest.mark.parametrize(
    ("stated", "missing"),
    [({"DELTA_THETA": "10.0"}, "REVOLUTIONS"), ({"REVOLUTIONS": "1.5"}, "DELTA_THETA")],
)
def test_half_an_angular_clock_is_refused_naming_the_missing_key(
    stated: dict[str, object], missing: str
) -> None:
    with pytest.raises(WorkspaceError) as raised:
        rotor_time_stepping("7001", stated, rpm=RPM)
    message = str(raised.value)
    assert f"and not {missing}" in message
    assert message.index(next(iter(stated))) < message.index(f"and not {missing}")


@pytest.mark.parametrize(
    ("stated", "missing"),
    [({"DELTA_TIME": "0.001"}, "TIME_ITERATIONS"), ({"TIME_ITERATIONS": "720"}, "DELTA_TIME")],
)
def test_half_an_explicit_clock_is_refused_naming_the_missing_key(
    stated: dict[str, object], missing: str
) -> None:
    with pytest.raises(WorkspaceError, match=f"and not {missing}"):
        rotor_time_stepping("7001", stated, rpm=RPM)
    with pytest.raises(WorkspaceError, match=f"and not {missing}"):
        unsteady_time_stepping("7001", stated)


def test_stating_the_clock_in_both_forms_is_refused_naming_both() -> None:
    with pytest.raises(WorkspaceError) as raised:
        rotor_time_stepping("7001", _angular(DELTA_TIME="0.01", TIME_ITERATIONS="5"), rpm=RPM)
    message = str(raised.value)
    assert "two forms at once" in message
    assert "DELTA_THETA" in message and "DELTA_TIME" in message
    assert "'7001'" in message


def test_stating_no_clock_is_refused_offering_both_forms() -> None:
    with pytest.raises(WorkspaceError) as raised:
        rotor_time_stepping("7001", {"LAST_REVS_AVG": "1"}, rpm=RPM)
    message = str(raised.value)
    assert "states no physical clock" in message
    assert "DELTA_THETA: <deg>" in message and "DELTA_TIME: <s>" in message


@pytest.mark.parametrize(
    ("key", "value"),
    [("DELTA_THETA", "nan"), ("REVOLUTIONS", "inf"), ("DELTA_THETA", "ten")],
)
def test_a_cell_that_is_no_finite_number_is_refused_by_case_and_key(key: str, value: str) -> None:
    with pytest.raises(WorkspaceError) as raised:
        rotor_time_stepping("7001", _angular(**{key: value}), rpm=RPM)
    message = str(raised.value)
    assert key in message and "7001" in message and ("finite" in message or "number" in message)


@pytest.mark.parametrize("theta", ["0", "-10", "361"])
def test_an_azimuthal_step_outside_one_revolution_is_refused(theta: str) -> None:
    with pytest.raises(WorkspaceError, match="resolves nothing inside one turn"):
        rotor_time_stepping("7001", _angular(DELTA_THETA=theta), rpm=RPM)


def test_a_run_of_no_revolutions_is_refused() -> None:
    with pytest.raises(WorkspaceError, match="positive number of revolutions"):
        rotor_time_stepping("7001", _angular(REVOLUTIONS="0"), rpm=RPM)


def test_an_angular_clock_that_makes_no_whole_step_is_refused() -> None:
    # 0.01 revolution at 20 degrees is 0.18 of a step
    with pytest.raises(WorkspaceError, match="at least one time step"):
        rotor_time_stepping("7001", {"DELTA_THETA": "20", "REVOLUTIONS": "0.01"}, rpm=1200.0)
    assert (
        rotor_time_stepping(
            "7001", {"DELTA_THETA": "20", "REVOLUTIONS": "2"}, rpm=1200.0
        ).time_iterations
        == 36
    )


def test_an_angular_clock_on_a_rotor_that_does_not_turn_is_refused() -> None:
    with pytest.raises(WorkspaceError, match="rotor speed of zero"):
        rotor_time_stepping("7001", _angular(), rpm=0.0)
    with pytest.raises(WorkspaceError, match="resolves no rotor speed"):
        rotor_time_stepping("7001", _angular())


@pytest.mark.parametrize(
    ("key", "value", "phrase"),
    [
        ("DELTA_TIME", "0", "positive number of seconds"),
        ("DELTA_TIME", "-1", "positive number of seconds"),
        ("TIME_ITERATIONS", "0", "at least one time step"),
        ("TIME_ITERATIONS", "-3", "at least one time step"),
        ("TIME_ITERATIONS", "2.5", "no fractional part"),
    ],
)
def test_a_bad_explicit_clock_is_refused(key: str, value: str, phrase: str) -> None:
    row: dict[str, object] = {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "5", key: value}
    with pytest.raises(WorkspaceError, match=phrase):
        rotor_time_stepping("7001", row, rpm=RPM)


def test_a_step_count_keeps_all_its_digits() -> None:
    row = {"DELTA_TIME": "0.01", "TIME_ITERATIONS": "9007199254740993"}
    assert rotor_time_stepping("7001", row).time_iterations == 9007199254740993


def test_a_speed_so_large_the_step_underflows_is_refused_by_case() -> None:
    with pytest.raises(WorkspaceError, match="positive, finite number of seconds"):
        rotor_time_stepping("7001", _angular(DELTA_THETA="1e-300"), rpm=1e300)


# ---------------------------------------------------------------- the rotorless row


def test_a_rotorless_row_states_its_clock_directly() -> None:
    stepping = unsteady_time_stepping("7003", {"DELTA_TIME": "0.002", "TIME_ITERATIONS": "40"})
    assert stepping.stated_form == "explicit" and stepping.rpm is None
    assert (stepping.delta_time_s, stepping.time_iterations) == (0.002, 40)
    assert stepping.steps_per_revolution is None


def test_a_rotorless_row_with_an_angular_pair_needs_a_speed_beside_it() -> None:
    with pytest.raises(WorkspaceError) as raised:
        unsteady_time_stepping("7003", {"DELTA_THETA": "20", "REVOLUTIONS": "2"})
    assert "ADVANCE_RATIO" in str(raised.value)
    assert "'RPM: <rev/min>'" not in str(raised.value)
    with pytest.raises(WorkspaceError, match="no rotor speed"):
        unsteady_time_stepping("7003", {"DELTA_THETA": "20", "REVOLUTIONS": "2"}, rpm=0.0)
    # with the speed it resolves as the rotor row does
    resolved = unsteady_time_stepping("7003", {"DELTA_THETA": "20", "REVOLUTIONS": "2"}, rpm=900.0)
    assert resolved == rotor_time_stepping(
        "7003", {"DELTA_THETA": "20", "REVOLUTIONS": "2"}, rpm=900.0
    )
    assert resolved.time_iterations == 36


def test_a_rotorless_row_with_no_clock_is_refused() -> None:
    with pytest.raises(WorkspaceError, match="states no physical clock"):
        unsteady_time_stepping("7003", {})


# ---------------------------------------------------------------- revolutions to steps


def test_revolutions_make_their_steps_rounded_up_with_a_margin_for_rounding_error() -> None:
    assert revolution_steps(1.5, 24.0) == 36
    assert revolution_steps(0.1, 36.0) == 4  # 3.6 rounds up
    assert revolution_steps(0.1 * 3, 10.0) == 3  # 3.0000000000000004 does not gain a step
    assert revolution_steps(1e300, 1e300) == 10**15
    assert revolution_steps(math.inf, 2.0) == 10**15


# ---------------------------------------------------------------- the azimuth of each step


@pytest.mark.parametrize(("rpm", "expected"), [(2000.0, 100.0), (-2000.0, 280.0)])
def test_the_azimuth_turns_in_the_sense_of_the_rotor(rpm: float, expected: float) -> None:
    # step 18 of 72 is a quarter turn: 10 + 90 = 100, or 10 - 90 = -80 = 280
    assert blade_azimuth_deg(10.0, step=18, steps_per_revolution=72, rpm=rpm) == expected


def test_the_azimuth_at_step_zero_is_the_datum_and_wraps_to_one_turn() -> None:
    assert blade_azimuth_deg(45.0, step=0, steps_per_revolution=36, rpm=1.0) == 45.0
    assert blade_azimuth_deg(350.0, step=3, steps_per_revolution=36, rpm=1.0) == 20.0
    assert blade_azimuth_deg(10.0, step=36, steps_per_revolution=36, rpm=1.0) == 10.0
    # a negative result wraps up, and a negative zero is never printed
    assert blade_azimuth_deg(0.0, step=36, steps_per_revolution=36, rpm=-1.0) == 0.0
    assert str(blade_azimuth_deg(0.0, step=36, steps_per_revolution=36, rpm=-1.0)) == "0.0"


def test_only_the_sign_of_the_speed_is_read() -> None:
    fast = blade_azimuth_deg(10.0, step=5, steps_per_revolution=72, rpm=5000.0)
    slow = blade_azimuth_deg(10.0, step=5, steps_per_revolution=72, rpm=1.0)
    assert fast == slow == 35.0


@pytest.mark.parametrize(
    "bad",
    [
        {"datum_deg": None},
        {"datum_deg": math.nan},
        {"step": "3"},
        {"step": True},
        {"step": math.inf},
        {"steps_per_revolution": 0},
        {"steps_per_revolution": -72},
        {"rpm": 0.0},
        {"rpm": None},
        {"rpm": math.nan},
    ],
)
def test_a_clock_that_is_not_stated_has_no_azimuth(bad: dict[str, object]) -> None:
    stated: dict[str, object] = {
        "datum_deg": 10.0,
        "step": 18,
        "steps_per_revolution": 72,
        "rpm": 2000.0,
    }
    stated.update(bad)
    assert (
        blade_azimuth_deg(
            stated["datum_deg"],
            step=stated["step"],
            steps_per_revolution=stated["steps_per_revolution"],
            rpm=stated["rpm"],
        )
        is None
    )


def test_blade_one_follows_the_sense_and_refuses_an_unstated_clock() -> None:
    # 3 of 8 steps is 135 degrees: 10 - 135 = -125 = 235
    assert blade_one_azimuth(3, datum_deg=10.0, sense=-1.0, steps_per_revolution=8.0) == 235.0
    assert blade_one_azimuth(3, datum_deg=10.0, sense=1.0, steps_per_revolution=8.0) == 145.0
    # a zero sense reads as forward
    assert blade_one_azimuth(3, datum_deg=10.0, sense=0.0, steps_per_revolution=8.0) == 145.0
    with pytest.raises(WorkspaceError, match="do not state a clock"):
        blade_one_azimuth(3, datum_deg=10.0, sense=1.0, steps_per_revolution=0.0)


def test_the_azimuth_over_a_run_matches_the_clock_of_the_row() -> None:
    stepping = rotor_time_stepping("7001", {"DELTA_THETA": "15", "REVOLUTIONS": "1"}, rpm=-900.0)
    per_revolution = stepping.steps_per_revolution
    assert per_revolution == 24.0
    azimuths = [
        blade_azimuth_deg(30.0, step=step, steps_per_revolution=per_revolution, rpm=stepping.rpm)
        for step in range(stepping.time_iterations + 1)
    ]
    # a left-handed rotor loses 15 degrees a step, and returns to its datum after one turn
    assert azimuths[:4] == [30.0, 15.0, 0.0, 345.0]
    assert azimuths[-1] == 30.0


# ---------------------------------------------------------------- the datum of blade one


def test_the_datum_is_read_from_the_blade1_table() -> None:
    assert blade_datum({"blade1": {"azimuth_deg": 45.0, "zero": "-y"}}) == BladeDatum(45.0, "-Y")
    assert blade_datum({"blade1": {"azimuth_deg": 30, "zero": "Z"}}) == BladeDatum(30.0, "Z")


def test_without_a_blade1_table_the_datum_is_zero_from_x() -> None:
    assert blade_datum({}) == BladeDatum(0.0, "X")
    assert blade_datum({"blade1": {}}) == BladeDatum(0.0, "X")
    assert blade_datum({"blade1": {"azimuth_deg": 12.5}}) == BladeDatum(12.5, "X")


@pytest.mark.parametrize(
    ("blade1", "phrase"),
    [
        ("0", "must be a table"),
        ({"azimuth": 1.0}, "unknown keys: azimuth"),
        ({"azimuth_deg": "10"}, "not a finite number"),
        ({"azimuth_deg": math.nan}, "not a finite number"),
        ({"azimuth_deg": True}, "not a finite number"),
        ({"zero": "W"}, "not an axis"),
        ({"zero": "+-X"}, "not an axis"),
        ({"zero": 3}, "not an axis"),
    ],
)
def test_a_bad_blade1_table_is_refused(blade1: object, phrase: str) -> None:
    with pytest.raises(WorkspaceError, match=phrase):
        blade_datum({"blade1": blade1})

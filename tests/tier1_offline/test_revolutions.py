"""Per-revolution means of a history, their drift and the warning that judges the last one.

Every expected number is worked by hand from the values written in the test: four steps per
revolution, three complete revolutions and two steps of a partial fourth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfdbender.core.revolutions import (
    DEFAULT_DRIFT_LIMIT_PCT,
    FORCE_PLOT_PARAMETERS,
    is_force_or_moment_column,
    last_complete_revolution,
    load_kind,
    per_revolution_table,
    revolution_drift_pct,
    revolution_drift_warnings,
    whole_revolution_steps,
)
from cfdbender.core.settings import WorkspaceError
from cfdbender.core.unsteady_window import TimeHistory

#: FX: revolution means 10, 10.5, 10.5; the partial revolution is 50 and 60 and never averaged.
FX = [9.0, 11.0, 9.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.4, 10.6, 10.4, 10.6, 50.0, 60.0]
#: FY: means 0, 1, 1; the first mean is ZERO, so the second drift is NA.
FY = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 9.0, 9.0]
#: MACH_P1 is no load: means 0.125, 0.125, 0.25, a drift of 100 per cent.
MACH = [0.125] * 8 + [0.25] * 4 + [0.9, 0.9]


def _history(first: int = 1) -> TimeHistory:
    steps = np.arange(first, first + len(FX))
    return TimeHistory(
        steps=steps,
        columns={"FX_P": np.array(FX), "FY_P": np.array(FY), "MACH_P1": np.array(MACH)},
    )


def test_each_revolution_is_one_row_with_exact_means() -> None:
    windows, means, left = per_revolution_table(
        _history(), ["FX_P", "FY_P", "MACH_P1"], revolution_steps=4
    )
    assert windows == [(1, 4), (5, 8), (9, 12)]
    assert left == 2
    assert [m["FX_P"] for m in means] == [10.0, 10.5, 10.5]
    assert [m["FY_P"] for m in means] == [0.0, 1.0, 1.0]
    assert [m["MACH_P1"] for m in means] == [0.125, 0.125, 0.25]


def test_the_revolutions_are_counted_from_the_first_row_of_the_history() -> None:
    windows, means, left = per_revolution_table(_history(first=101), ["FX_P"], revolution_steps=4)
    assert windows == [(101, 104), (105, 108), (109, 112)]
    assert [m["FX_P"] for m in means] == [10.0, 10.5, 10.5] and left == 2


def test_only_the_columns_asked_are_averaged() -> None:
    _, means, _ = per_revolution_table(_history(), ["FY_P"], revolution_steps=4)
    assert all(set(m) == {"FY_P"} for m in means)


def test_less_than_one_revolution_has_no_row_and_names_what_is_left() -> None:
    windows, means, left = per_revolution_table(_history(), ["FX_P"], revolution_steps=20)
    assert (windows, means, left) == ([], [], 14)
    exact = per_revolution_table(_history(), ["FX_P"], revolution_steps=14)
    assert exact[0] == [(1, 14)] and exact[2] == 0


def test_a_revolution_of_under_one_step_cannot_be_cut() -> None:
    with pytest.raises(WorkspaceError, match="under one solver step"):
        per_revolution_table(_history(), ["FX_P"], revolution_steps=0)


def test_the_means_are_those_of_the_one_average() -> None:
    read: set[int] = set()
    per_revolution_table(_history(), ["FX_P"], revolution_steps=4, read_steps=read)
    assert read == set(range(1, 13))


# ---------------------------------------------------------------- the drift


def test_the_drift_is_relative_to_the_previous_mean_whatever_its_sign() -> None:
    assert revolution_drift_pct(10.5, 10.0) == pytest.approx(5.0)
    assert revolution_drift_pct(-9.5, -10.0) == pytest.approx(5.0)
    assert revolution_drift_pct(101.0, 100.0) == pytest.approx(1.0)
    # the sign says whether the mean rose or fell: -11 against -10 is a fall of 10 per cent
    assert revolution_drift_pct(-11.0, -10.0) == pytest.approx(-10.0)
    assert revolution_drift_pct(5.0, 10.0) == pytest.approx(-50.0)


@pytest.mark.parametrize(
    ("current", "previous"),
    [(3.0, 0.0), (math.nan, 1.0), (1.0, math.nan), (math.inf, 1.0), (1.0, math.inf)],
)
def test_the_drift_is_undefined_where_the_previous_mean_is_zero_or_a_number_is_not(
    current: float, previous: float
) -> None:
    assert revolution_drift_pct(current, previous) is None


def test_a_difference_that_overflows_keeps_its_sign() -> None:
    # 1.7e308 - (-1.7e308) overflows; the ratio is taken as current/|previous| - sign(previous)
    drift = revolution_drift_pct(1.7e308, -1.7e308)
    assert drift == pytest.approx(200.0)


def test_the_drift_of_a_history_is_the_ratio_of_its_revolution_means() -> None:
    _, means, _ = per_revolution_table(_history(), ["FX_P", "FY_P"], revolution_steps=4)
    assert revolution_drift_pct(means[1]["FX_P"], means[0]["FX_P"]) == pytest.approx(5.0)
    assert revolution_drift_pct(means[2]["FX_P"], means[1]["FX_P"]) == 0.0
    assert revolution_drift_pct(means[1]["FY_P"], means[0]["FY_P"]) is None


def test_the_default_limit_is_one_per_cent() -> None:
    assert DEFAULT_DRIFT_LIMIT_PCT == 1.0


# ---------------------------------------------------------------- what is a load


def test_a_load_is_named_by_its_parameter_not_by_its_first_letter() -> None:
    assert is_force_or_moment_column("MZ_ROTOR_PROP") and is_force_or_moment_column("CL_MRP_TOTAL")
    assert not is_force_or_moment_column("MACH_P1") and not is_force_or_moment_column("FXX_TOTAL")
    assert is_force_or_moment_column("FX")  # the bare parameter is a load too
    assert set(FORCE_PLOT_PARAMETERS) == {
        "CL", "CDI", "CDO", "CDP", "CDV", "CD", "FX", "FY", "FZ", "MX", "MY", "MZ"
    }  # fmt: skip


def test_the_kind_and_group_of_a_load_column() -> None:
    assert load_kind("FY_PROP_X") == ("force", "PROP_X")
    assert load_kind("MZ_PROP_X") == ("moment", "PROP_X")
    assert load_kind("CDI_WING") == ("coefficient", "WING")
    assert load_kind("CL_MRP_TOTAL") == ("coefficient", "MRP_TOTAL")
    assert load_kind("MACH_P1") is None and load_kind("FXX_TOTAL") is None


# ---------------------------------------------------------------- the warning


#: Thrust and torque settled, in-plane loads tiny and moving -70 per cent: the measured shape of
#: a rotor in uniform inflow. Each list is the three revolution means.
ROTOR = {
    "FX_PROP_X": [-350.0, -350.1, -350.2],
    "FY_PROP_X": [0.08, 0.025, 0.0075],
    "FZ_PROP_X": [-0.05, 0.02, 0.006],
    "MX_PROP_X": [30.0, 30.01, 30.02],
    "MY_PROP_X": [0.01, 0.003, 0.0009],
    "MZ_PROP_X": [-0.004, 0.002, 0.0006],
    "FX_PROP_Blade1": [-10.0, -10.0, -10.0],
    "FY_PROP_Blade1": [0.5, 0.5, 0.5],
    "MACH_P1": [0.1, 0.1, 0.2],
}


def _warn(means: dict[str, list[float]], limit: float = 1.0) -> list[str]:
    previous = {name: values[-2] for name, values in means.items()}
    current = {name: values[-1] for name, values in means.items()}
    return revolution_drift_warnings(previous, current, {n: n for n in means}, limit_pct=limit)


def test_near_zero_in_plane_loads_with_steady_thrust_do_not_warn() -> None:
    assert _warn(ROTOR) == []
    # THE TABLE STILL STATES THE RELATIVE DRIFT: minus 70 per cent
    assert revolution_drift_pct(0.0075, 0.025) == pytest.approx(-70.0)
    assert revolution_drift_pct(0.0009, 0.003) == pytest.approx(-70.0)


def test_an_in_plane_load_moving_against_its_thrust_or_torque_warns() -> None:
    moving = {
        **ROTOR,
        # 0.025 -> 10: a change of 9.975 against the thrust 350.1, 2.8492 per cent.
        "FY_PROP_X": [0.08, 0.025, 10.0],
        # 0.002 -> 1: 0.998 against the torque 30.01, 3.3256 per cent; against the thrust it
        # would be 0.285 per cent, so a scale that mixed kinds stays quiet.
        "MZ_PROP_X": [-0.004, 0.002, 1.0],
        # 0.5 -> 0.7 on the blade: 2 per cent of ITS thrust 10, 0.057 of the rotor's.
        "FY_PROP_Blade1": [0.5, 0.5, 0.7],
    }
    fy, mz, blade = _warn(moving)
    assert fy.startswith("column FY_PROP_X drifts +2.8492 per cent of its scale 350.1 ")
    assert "the magnitude of FX_PROP_X, the largest force mean of its group" in fy
    assert fy.endswith("a change of +9.975)")
    assert mz.startswith("column MZ_PROP_X drifts +3.3256 per cent of its scale 30.01 ")
    assert "the magnitude of MX_PROP_X, the largest moment mean of its group" in mz
    assert blade.startswith("column FY_PROP_Blade1 drifts +2.0000 per cent of its scale 10 ")
    assert "the magnitude of FX_PROP_Blade1" in blade
    # THE LIMIT IS A PARAMETER: 3.5 per cent quiets all three
    assert _warn(moving, limit=3.5) == []


def test_the_largest_column_of_its_group_warns_exactly_as_its_relative_drift() -> None:
    # THE THRUST ITSELF MOVES: -350.1 -> -355, -1.3996 per cent, over 1 and under 1.5.
    thrust = {**ROTOR, "FX_PROP_X": [-350.0, -350.1, -355.0]}
    (line,) = _warn(thrust)
    written = revolution_drift_pct(-355.0, -350.1)
    assert written is not None and written == pytest.approx((-355.0 + 350.1) / 350.1 * 100.0)
    assert f"column FX_PROP_X drifts {written:+.4f} per cent of its scale 350.1 " in line
    assert _warn(thrust, limit=1.5) == []
    # AND OVER A GRID, the largest column warns iff its relative drift exceeds the limit
    names = {"MX_A": "MX_A", "MY_A": "MY_A"}
    for previous in (-40.0, -2.0, 0.5, 30.0):
        for step in (-0.021, -0.0099, 0.0, 0.0101, 0.3):
            current = previous * (1.0 + step)
            for limit in (0.5, 1.0, 2.0):
                drift = revolution_drift_pct(current, previous)
                assert drift is not None
                clauses = revolution_drift_warnings(
                    {"MX_A": previous, "MY_A": 0.001},
                    {"MX_A": current, "MY_A": 0.001},
                    names,
                    limit_pct=limit,
                )
                if abs(drift) > limit:
                    assert clauses and clauses[0].startswith(
                        f"column MX_A drifts {drift:+.4f} per cent of its scale"
                    ), (previous, step, limit, clauses)
                else:
                    assert clauses == [], (previous, step, limit, clauses)


def test_a_column_that_is_no_load_is_never_warned_about() -> None:
    assert _warn({"MACH_P1": [0.1, 0.1, 0.9], "VX_probe1": [1.0, 1.0, 5.0]}) == []


def test_a_group_with_no_nonzero_finite_mean_is_quiet() -> None:
    zero = {"FX_G": [0.0, 0.0, 5.0]}
    assert _warn(zero) == []
    not_a_number = {"FX_G": [1.0, math.nan, 5.0], "FY_G": [1.0, math.nan, 5.0]}
    assert _warn(not_a_number) == []


def test_a_mean_that_is_no_number_does_not_set_the_scale_of_its_group() -> None:
    # FX_G, first in column order, has no mean in the earlier revolution; FY_G still warns
    # against its own 100 and not against a scale that is not a number.
    (clause,) = revolution_drift_warnings(
        {"FX_G": math.nan, "FY_G": 100.0},
        {"FX_G": 1.0, "FY_G": 105.0},
        {"FX_G": "FX_G", "FY_G": "FY_G"},
        limit_pct=1.0,
    )
    assert clause.startswith("column FY_G drifts +5.0000 per cent of its scale 100 ")


def test_a_mean_that_is_no_number_in_the_last_revolution_is_quiet() -> None:
    assert _warn({"FX_G": [1.0, 100.0, math.nan]}) == []


def test_the_clauses_come_in_column_order_and_use_the_printed_names() -> None:
    previous = {"FX_G": 100.0, "FZ_G": 100.0}
    current = {"FX_G": 105.0, "FZ_G": 95.0}
    shown = {"FX_G": "FX_G [N]", "FZ_G": "FZ_G [N]"}
    first, second = revolution_drift_warnings(previous, current, shown, limit_pct=1.0)
    assert first.startswith("column FX_G [N] drifts +5.0000 per cent of its scale 100 ")
    assert second.startswith("column FZ_G [N] drifts -5.0000 per cent of its scale 100 ")
    assert "the magnitude of FX_G [N]" in first


def test_the_drift_at_the_limit_does_not_warn() -> None:
    assert (
        revolution_drift_warnings({"FX_G": 100.0}, {"FX_G": 101.0}, {"FX_G": "FX_G"}, limit_pct=1.0)
        == []
    )


def test_the_classifier_of_load_columns_is_a_parameter() -> None:
    def su2_kind(name: str) -> tuple[str, str] | None:
        # a solver's own history names: CFx[1] is a force of group 1, CMz[1] a moment of it
        if name.startswith("CF"):
            return ("force", name.split("[")[-1])
        if name.startswith("CM"):
            return ("moment", name.split("[")[-1])
        return None

    previous = {"CFx[1]": 10.0, "CFy[1]": 0.01, "CMz[1]": 2.0, "Time(sec)": 1.0}
    current = {"CFx[1]": 10.0, "CFy[1]": 0.5, "CMz[1]": 2.0, "Time(sec)": 5.0}
    shown = {name: name for name in previous}
    (clause,) = revolution_drift_warnings(previous, current, shown, limit_pct=1.0, kind_of=su2_kind)
    assert clause.startswith("column CFy[1] drifts +4.9000 per cent of its scale 10 ")
    # with the default classifier none of these names is a load
    assert revolution_drift_warnings(previous, current, shown, limit_pct=1.0) == []


# ---------------------------------------------------------------- the revolution of a clock


def test_a_clock_with_a_fractional_remainder_is_refused_not_rounded() -> None:
    assert whole_revolution_steps(36.0) == 36
    assert whole_revolution_steps(36.0 + 1e-14) == 36
    with pytest.raises(WorkspaceError, match="fractional remainder 0.5"):
        whole_revolution_steps(36.5)
    with pytest.raises(WorkspaceError, match="finite number"):
        whole_revolution_steps(math.nan)
    with pytest.raises(WorkspaceError, match="finite number"):
        whole_revolution_steps(math.inf)


def test_the_last_complete_revolution_is_counted_from_the_first_step_of_the_history() -> None:
    assert last_complete_revolution(3, 13, 4.0) == (7, 10)  # 11 steps: two revolutions and 3
    assert last_complete_revolution(1, 8, 4.0) == (5, 8)
    assert last_complete_revolution(1, 4, 4.0) == (1, 4)
    assert last_complete_revolution(3, 5, 4.0) is None
    assert last_complete_revolution(5, 1, 4.0) is None
    with pytest.raises(WorkspaceError, match="fractional remainder"):
        last_complete_revolution(1, 100, 36.4)

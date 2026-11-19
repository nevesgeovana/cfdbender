"""Implemented row variables are read, never warned; the others are skipped."""

from __future__ import annotations

import warnings

import pytest
from test_runtypes_b1_workflows import deck, workspace

from cfdbender.core.rotor import is_row_control
from cfdbender.core.run_types import CONTROLLED_KEYS

TIME_KEYS = "DELTA_THETA: 30 / REVOLUTIONS: 3 / LAST_REVS_AVG: 1 / LAST_ITERS_AVG: 5"


@pytest.mark.parametrize(
    "key",
    [
        "LAST_ITERS_AVG",
        "DELTA_TIME",
        "TIME_ITERATIONS",
        "TURBULENCE_MODEL",
        "PASSAGE_POSITIONS",
        "EXPORT_UNSTEADY_LAST_REV",
        "CLOCK_MOTION",
        "MOTIONS",
    ],
)
def test_every_row_variable_of_the_run_types_is_a_control_never_a_solver_key(key):
    assert is_row_control(key)
    assert key in CONTROLLED_KEYS or key in ("CLOCK_MOTION", "MOTIONS")


def caught(root):
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        lines = deck(root)[1]
    return lines, [str(w.message) for w in found]


def test_the_variables_of_the_row_s_own_run_type_are_read_without_a_warning(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 30 / REVOLUTIONS: 3 / LAST_REVS_AVG: 1 / EXPORT_UNSTEADY_LAST_ITER: 12",
    )
    lines, said = caught(tmp_path)
    assert not [text for text in said if "skipped" in text or "not registered" in text]
    assert lines["TIME_ITER"] == "24", "the export window starts the second segment at step 25"


def test_both_window_keys_are_refused_as_pyflightstream_refuses_them(tmp_path):
    from cfdbender.core.plan import PlanError

    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var=TIME_KEYS,
    )
    with pytest.raises(PlanError, match="LAST_REVS_AVG.*LAST_ITERS_AVG"):
        deck(tmp_path)


@pytest.mark.parametrize("key", ["CLOCK_MOTION: ROTOR", "MOTIONS: ROTATE"])
def test_a_variable_this_package_does_not_offer_yet_is_warned_and_skipped(tmp_path, key):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var=f"DELTA_THETA: 30 / REVOLUTIONS: 3 / LAST_REVS_AVG: 1 / {key}",
    )
    lines, said = caught(tmp_path)
    name = key.split(":")[0]
    assert any(text.startswith(f"POL 7001: {name} is skipped") for text in said)
    assert name not in lines

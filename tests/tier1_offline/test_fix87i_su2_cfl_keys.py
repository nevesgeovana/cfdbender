"""FIX87I: the neutral cfl_max and cfl_growth keys set the SU2 CFL ramp without restating it."""

from __future__ import annotations

import pytest
import test_p0000_item3d_config as su2_case
from test_fix87f_su2_cfl_ramp import _lines


def test_cfl_max_sets_the_ramp_maximum_only(tmp_path):
    lines = _lines(tmp_path, "cfl_max = 1000")
    assert "CFL_NUMBER= 5" in lines
    assert "CFL_ADAPT= YES" in lines
    assert "CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 1000 )" in lines


def test_cfl_growth_and_start_set_their_slots(tmp_path):
    lines = _lines(tmp_path, "cfl = 10\ncfl_growth = 1.5\ncfl_max = 500")
    assert "CFL_NUMBER= 10" in lines
    assert "CFL_ADAPT= YES" in lines
    assert "CFL_ADAPT_PARAM= ( 0.5, 1.5, 10, 500 )" in lines


@pytest.mark.parametrize(
    "extra,problem",
    [
        ("cfl_max = 100\nCFL_ADAPT_PARAM = [0.5, 1.2, 5, 25]", "CFL_ADAPT_PARAM"),
        ("cfl_growth = 1.1\nCFL_ADAPT = false", "CFL_ADAPT"),
        ("cfl_growth = 1.0", "cfl_growth"),
        ("cfl = 10\ncfl_max = 5", "cfl_max"),
    ],
)
def test_ramp_keys_refuse_contradictions(tmp_path, extra, problem):
    with pytest.raises(ValueError, match=problem):
        _lines(tmp_path, extra)


def test_ramp_keys_are_known_to_the_planner(tmp_path):
    su2_case.workspace(tmp_path, extra="cfl_max = 1000")
    su2_case.plan(tmp_path)


def test_neutral_and_native_cfl_number_conflict(tmp_path):
    with pytest.raises(ValueError, match="CFL_NUMBER"):
        _lines(tmp_path, "cfl = 10\nCFL_NUMBER = 10")

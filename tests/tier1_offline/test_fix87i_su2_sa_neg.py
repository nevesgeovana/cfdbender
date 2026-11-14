"""FIX87I: SU2 SA runs default to the standard SA-neg variant; the user's SA_OPTIONS win."""

from __future__ import annotations

from test_fix87f_su2_cfl_ramp import _lines


def test_sa_without_options_gets_standard_sa_neg(tmp_path):
    lines = _lines(tmp_path, 'KIND_TURB_MODEL = "SA"')
    assert "SA_OPTIONS= ( NEGATIVE, WITHFT2 )" in lines


def test_user_sa_options_win_entirely(tmp_path):
    lines = _lines(tmp_path, 'KIND_TURB_MODEL = "SA"\nSA_OPTIONS = ["NEGATIVE", "EXPERIMENTAL"]')
    assert "SA_OPTIONS= ( NEGATIVE, EXPERIMENTAL )" in lines
    assert sum(line.startswith("SA_OPTIONS=") for line in lines) == 1


def test_other_turbulence_models_get_no_sa_options(tmp_path):
    lines = _lines(tmp_path, 'KIND_TURB_MODEL = "SST"')
    assert not any(line.startswith("SA_OPTIONS=") for line in lines)
    assert not any(line.startswith("SA_OPTIONS=") for line in _lines(tmp_path / "e"))

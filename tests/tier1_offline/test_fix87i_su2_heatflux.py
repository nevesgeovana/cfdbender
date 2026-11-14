"""FIX87I: an SU2 heat-flux marker given by name alone is written with its adiabatic value."""

from __future__ import annotations

from test_fix87f_su2_cfl_ramp import _lines


def test_heatflux_names_get_an_adiabatic_value(tmp_path):
    lines = _lines(tmp_path, 'MARKER_HEATFLUX = ["slat", "main"]')
    assert "MARKER_HEATFLUX= ( slat, 0.0, main, 0.0 )" in lines


def test_heatflux_single_name_gets_an_adiabatic_value(tmp_path):
    lines = _lines(tmp_path, 'MARKER_HEATFLUX = "airfoil"')
    assert "MARKER_HEATFLUX= ( airfoil, 0.0 )" in lines


def test_heatflux_with_values_is_written_as_stated(tmp_path):
    lines = _lines(tmp_path, 'MARKER_HEATFLUX = ["airfoil", 1500.0]')
    assert "MARKER_HEATFLUX= ( airfoil, 1500 )" in lines
    lines = _lines(tmp_path / "t", 'MARKER_HEATFLUX = ["airfoil", "0.0"]')
    assert "MARKER_HEATFLUX= ( airfoil, 0.0 )" in lines

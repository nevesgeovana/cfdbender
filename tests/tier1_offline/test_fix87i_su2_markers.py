"""FIX87I: SU2 wall markers are plotted and monitored by default; the user's lists win."""

from __future__ import annotations

from test_fix87f_su2_cfl_ramp import _lines


def test_wall_markers_are_plotted_and_monitored_by_default(tmp_path):
    extra = 'MARKER_HEATFLUX = ["main"]\nMARKER_ISOTHERMAL = ["flap", 300.0]\nMARKER_FAR = ["far"]'
    lines = _lines(tmp_path, extra)
    assert "MARKER_PLOTTING= ( main, flap )" in lines
    assert "MARKER_MONITORING= ( main, flap )" in lines


def test_euler_wall_is_plotted_and_monitored_by_default(tmp_path):
    lines = _lines(tmp_path, 'MARKER_EULER = ["airfoil"]')
    assert "MARKER_PLOTTING= ( airfoil )" in lines
    assert "MARKER_MONITORING= ( airfoil )" in lines


def test_user_plotting_and_monitoring_win(tmp_path):
    extra = 'MARKER_HEATFLUX = ["main", "flap"]\nMARKER_MONITORING = ["main"]'
    lines = _lines(tmp_path, extra)
    assert "MARKER_MONITORING= ( main )" in lines
    assert "MARKER_PLOTTING= ( main, flap )" in lines


def test_no_wall_marker_writes_no_plotting(tmp_path):
    lines = _lines(tmp_path)
    assert not any(line.startswith(("MARKER_PLOTTING=", "MARKER_MONITORING=")) for line in lines)

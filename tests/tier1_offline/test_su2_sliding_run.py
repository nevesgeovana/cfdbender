"""A sliding-mesh point: the interpolator it states, and the history a multizone run writes."""

from __future__ import annotations

from test_runtypes_b1_workflows import deck, workspace

SLIDING = '[sliding]\nzones = 2\nrotating_zone = 0\ninterface = ["X_OUT", "X_IN"]\n'


def _sliding(tmp_path, setup=""):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 3 / REVOLUTIONS: 5 / LAST_REVS_AVG: 2.5",
        setup=setup + SLIDING,
    )
    return deck(tmp_path)


def test_a_sliding_interface_defaults_to_the_nearest_neighbour_interpolation(tmp_path):
    point, lines, _ = _sliding(tmp_path)
    assert lines["KIND_INTERPOLATION"] == "NEAREST_NEIGHBOR"
    assert point.origins["KIND_INTERPOLATION"] == "run type unsteady_rotor"


def test_a_setup_that_states_the_interpolation_keeps_it(tmp_path):
    _, lines, _ = _sliding(tmp_path, 'KIND_INTERPOLATION = "ISOPARAMETRIC"\n')
    assert lines["KIND_INTERPOLATION"] == "ISOPARAMETRIC"

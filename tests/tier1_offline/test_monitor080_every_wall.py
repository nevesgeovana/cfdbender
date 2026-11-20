"""Every wall marker is monitored, whatever the setup lists (0.8.0).

The rule of 0.8.0: every mesh family is monitored. A ``MARKER_MONITORING`` list that
misses a wall of ``MARKER_EULER``, ``MARKER_HEATFLUX`` or ``MARKER_ISOTHERMAL`` is completed
after its own markers, with a one-line note in the plan output, never a refusal.
"""

from __future__ import annotations

import warnings

import test_p0000_item3d_config as su2_case
from test_fix87i_su2_markers import _lines
from test_runtypes_b1_workflows import workspace

from cfdbender.core.adapter import load_point
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace


def test_a_monitoring_list_that_misses_a_wall_is_completed(tmp_path):
    extra = 'MARKER_HEATFLUX = ["blade", "nacelle"]\nMARKER_MONITORING = ["blade"]'
    lines = _lines(tmp_path, extra)
    assert "MARKER_MONITORING= ( blade, nacelle )" in lines


def test_every_wall_kind_is_monitored_with_the_listed_markers_first(tmp_path):
    extra = (
        'MARKER_EULER = ["spinner"]\nMARKER_HEATFLUX = ["blade"]\n'
        'MARKER_ISOTHERMAL = ["nacelle", 300.0]\nMARKER_MONITORING = ["nacelle"]'
    )
    lines = _lines(tmp_path, extra)
    assert "MARKER_MONITORING= ( nacelle, spinner, blade )" in lines


def test_the_plan_notes_the_walls_it_adds_in_one_line_and_does_not_refuse(tmp_path):
    su2_case.workspace(
        tmp_path, extra='MARKER_HEATFLUX = ["blade", "nacelle"]\nMARKER_MONITORING = ["blade"]'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        su2_case.plan(tmp_path)
    said = [str(w.message) for w in caught if "MARKER_MONITORING" in str(w.message)]
    assert said == [
        "setup s001: MARKER_MONITORING misses the wall marker(s) nacelle; every wall is "
        "monitored, so they are added"
    ]


def test_a_complete_monitoring_list_gets_no_note(tmp_path):
    su2_case.workspace(tmp_path, extra='MARKER_HEATFLUX = ["blade", "nacelle"]')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        su2_case.plan(tmp_path)
    assert not [w for w in caught if "MARKER_MONITORING" in str(w.message)]


def test_each_zone_of_a_sliding_run_monitors_every_wall(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 30 / REVOLUTIONS: 1 / LAST_REVS_AVG: 1",
        setup='MARKER_HEATFLUX = ["B1", "cowl", "nacelle"]\nMARKER_MONITORING = ["B1"]\n'
        '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_BOXD1", "X_BOX1"]\n',
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(tmp_path, "rt")
    point = load_point(tmp_path, plan.points[0], solver="su2")
    lines = load_solver("su2").translate(point).splitlines()
    # the zone files state their own keys only, so the master list is every zone's list
    assert "MARKER_MONITORING= ( B1, cowl, nacelle )" in lines

"""The rotor's loads are its own markers', never SU2's totals (0.8.0).

Every wall is monitored (0.8.0), and SU2's total coefficients sum the monitored markers, so a
monitored nacelle (one zone) or a monitored non-rotor wall of the turning zone (two zones) is
in the totals. The rotor row and the rotor plot group are summed over the rotor's markers (the
REF declaration's ``families_blades`` and ``families_general``) from the per-surface history
columns, and do not move with that wall's force.
"""

from __future__ import annotations

import warnings

import pytest
import test_su2_rotor_coefficients as steady_rotor
from test_runtypes_b1_workflows import workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import workspace_path
from cfdbender.solvers import load_solver
from cfdbender.solvers.su2._config import config, extra_files
from cfdbender.workspace.planning import plan_workspace

LOADS = ("CD", "CL", "CSF", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
BLADE = (0.01, 0.2, 0.001, -0.012, 0.003, 0.004, 0.0004, 0.0002, 0.0001)
BIG = 50.0  # the other wall's force, far above the blade's


def _steady_history(other):
    """A rotating-frame history: the totals hold the blade and a wall of force ``other``."""
    names = [*LOADS, *(f"{k}({m})" for m in ("Blade1", "nacelle") for k in LOADS)]
    head = "Inner_Iter,rms[Rho]," + ",".join(names)
    blade = dict(zip(LOADS, BLADE, strict=True))
    total = [blade[k] + other for k in LOADS]
    cells = [*total, *BLADE, *(other for _ in LOADS)]
    return head + "\n0,-5.0," + ",".join(format(v, ".17g") for v in cells) + "\n"


def _steady(root, other):
    setup = steady_rotor.SETUP_COLLECT + 'MARKER_HEATFLUX = ["Blade1", "nacelle"]\n'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, plan = steady_rotor.collected(root, 6, setup=setup)
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    (folder / "history.csv").write_text(_steady_history(other), encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_solver("su2").collect(point).to_dict()


def test_a_monitored_nacelle_does_not_move_the_rotor_row(tmp_path):
    bare = _steady(tmp_path / "bare", 0.0)
    loaded = _steady(tmp_path / "loaded", BIG)
    # the totals hold the nacelle ...
    assert loaded["coefficients"]["CFx"]["value"] == pytest.approx(BLADE[3] + BIG)
    # ... and the rotor row does not
    for key, value in bare["rotor"]["row"].items():
        assert loaded["rotor"]["row"][key] == pytest.approx(value, rel=1e-12, abs=1e-15), key
    assert loaded["rotor"]["loads"] == bare["rotor"]["loads"]


def test_the_rotor_row_of_the_blade_equals_the_row_of_its_own_loads(tmp_path):
    loaded = _steady(tmp_path, BIG)
    row, _ = steady_rotor.expected(k=6.0)
    for key, value in row.items():
        assert loaded["rotor"]["row"][key] == pytest.approx(value, rel=1e-9), key


def _zone_history(other):
    """Two zones: zone 0 holds the nacelle, the turning zone 1 the blade and a cowl."""
    walls = {0: ("nacelle",), 1: ("B1", "cowl")}
    head = ["Time_Iter", "Outer_Iter", "Inner_Iter", "rms[Rho][0]", "rms[Rho][1]"]
    head += [f"{k}[{z}]" for z in (0, 1) for k in LOADS]
    head += [f"{k}[{z}]({m})" for z, ms in walls.items() for m in ms for k in LOADS]
    blade = dict(zip(LOADS, BLADE, strict=True))
    lines = [",".join(head)]
    for step in range(12):
        cells = [step, 0, 0, -6.0, -6.0]
        cells += [other for _ in LOADS] + [blade[k] + other for k in LOADS]
        cells += [other for _ in LOADS] + list(BLADE) + [other for _ in LOADS]
        lines.append(",".join(format(v, ".17g") for v in cells))
    return "\n".join(lines) + "\n"


def _zones(root, other, *, quiet=True):
    workspace(
        root,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 30 / REVOLUTIONS: 1 / LAST_REVS_AVG: 1",
        setup='MARKER_HEATFLUX = ["B1", "cowl", "nacelle"]\n'
        '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_BOXD1", "X_BOX1"]\n',
    )
    plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    solver = load_solver("su2")
    (folder / "su2.cfg").write_text(solver.translate(point), encoding="ascii")
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )
    for name, text in extra_files(values).items():
        (folder / name).write_text(text, encoding="ascii")
    (folder / "su2.csv").write_text(_zone_history(other), encoding="ascii")
    if not quiet:
        return solver.collect(point).to_dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solver.collect(point).to_dict()


def test_a_monitored_wall_of_the_turning_zone_does_not_move_the_rotor_row(tmp_path):
    bare = _zones(tmp_path / "bare", 0.0)
    loaded = _zones(tmp_path / "loaded", BIG)
    assert loaded["coefficients"]["CFx"]["value"] == pytest.approx(BLADE[3] + BIG)
    for key, value in bare["rotor"]["row"].items():
        assert loaded["rotor"]["row"][key] == pytest.approx(value, rel=1e-12, abs=1e-15), key


def test_the_rotor_plot_group_of_two_zones_sums_the_rotor_markers(tmp_path):
    bare = _zones(tmp_path / "bare", 0.0)["time_average"]["means"]
    loaded = _zones(tmp_path / "loaded", BIG)["time_average"]["means"]
    rotor = [name for name in loaded if name.endswith("_ROTOR_ROTOR")]
    assert rotor
    for name in rotor:
        assert loaded[name] == pytest.approx(bare[name], rel=1e-12, abs=1e-12), name
    # the airframe total keeps every wall: it is the totals' sum, legitimately
    total = [name for name in loaded if name.endswith("_MRP_TOTAL") and name.startswith("FX")]
    assert total and loaded[total[0]] != pytest.approx(bare[total[0]])


def test_the_walls_left_out_of_the_rotor_row_are_named_in_one_line(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _zones(tmp_path, BIG, quiet=False)
    said = {str(w.message) for w in caught if str(w.message).startswith("rotor ROTOR:")}
    assert said == {
        "rotor ROTOR: its row sums B1; the monitored wall(s) cowl are outside its "
        "families_blades and families_general, so they are not in it"
    }


REF = {
    "ROTOR": {
        "kind": "rotor",
        "axis": "X",
        "diameter_m": 1.0,
        "families_blades": ["B1", "B2"],
        "families_general": ["spinner"],
        "blade1": {"azimuth_deg": 0.0, "zero": "Z"},
    }
}


@pytest.mark.parametrize(
    ("walls", "surfaces", "multizone", "markers", "reason"),
    [
        (["B1", "B2", "spinner"], [], False, None, None),  # the totals are the rotor's alone
        (
            ["B1", "B2", "spinner", "nacelle"],
            ["B1", "B2", "spinner", "nacelle"],
            False,
            ("B1", "B2", "spinner"),
            None,
        ),
        (["B1", "nacelle"], [], False, None, "no per-surface loads"),
        (["nacelle"], ["nacelle"], False, None, "no marker of the rotor"),
        ([], ["B1", "cowl"], True, ("B1",), None),
        ([], [], True, None, None),  # the turning zone's total
        ([], ["BLADE_1", "hub"], True, None, None),  # the turning zone names the rotor
    ],
)
def test_the_rotor_markers_are_told_from_the_other_walls(
    walls, surfaces, multizone, markers, reason
):
    from cfdbender.solvers.su2._rotor import rotor_surfaces

    found, why = rotor_surfaces(REF, walls, surfaces, multizone=multizone)
    assert found == markers
    assert (why is None) if reason is None else (reason in why)

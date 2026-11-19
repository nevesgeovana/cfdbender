"""A sliding-mesh point reports the stationary zone's loads (the airframe) beside the rotor's."""

from __future__ import annotations

import csv
import io
import warnings

import pytest
import test_runtypes_b1_workflows as rt

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import workspace_path
from cfdbender.matrix._unsteady_polar import airframe_absent, unsteady_polars
from cfdbender.matrix.exports import _cell
from cfdbender.solvers import load_solver
from cfdbender.solvers.su2._history import AIRFRAME_SUFFIX, airframe
from cfdbender.workspace.planning import plan_workspace

LOADS = ("CL", "CD", "CSF", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
STEPS = 36
CLOCK = "DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_REVS_AVG: 0.25"
SLIDING = '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_OUT", "X_IN"]\n'


def _value(zone, column, step):
    """A load that differs by zone and column and moves with the step."""
    return 0.01 * (1 + LOADS.index(column)) * (1 + 9 * zone) + 0.001 * step * (1 + zone)


def _history(folder, *, stationary=True, zero=False):
    columns = [f"{name}[{zone}]" for zone in (0, 1) for name in ("rms[Rho]", *LOADS)]
    if not stationary:
        columns = [c for c in columns if not (c.endswith("[0]") and not c.startswith("rms"))]
    header = ["Time_Iter", "Outer_Iter", "Inner_Iter", *columns]
    rows = [",".join(f'"{h}"' for h in header)]
    for step in range(STEPS):
        for inner in range(2):
            cells = [step, inner, 0]
            for column in columns:
                name, zone = column[:-3], int(column[-2])
                if name == "rms[Rho]":
                    cells.append(-8.0 - 0.01 * step)
                elif zone == 0 and zero:
                    cells.append(0.0)
                else:
                    # the last inner row of a step is the one kept
                    cells.append(_value(zone, name, step + 1) + (0.5 if inner == 0 else 0.0))
            rows.append(",".join(str(c) for c in cells))
    (folder / "su2.csv").write_text("\n".join(rows) + "\n", encoding="ascii")


def _point(root, **history):
    rt.workspace(
        root,
        workflow="unsteady_rotor",
        var=CLOCK,
        setup='MARKER_PLOTTING = ["B1"]\n' + SLIDING,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    (folder / "su2.cfg").write_text(
        "MULTIZONE= YES\nCONFIG_LIST= ( zone_0.cfg, zone_1.cfg )\nTIME_DOMAIN= YES\n"
        f"TIME_ITER= {STEPS}\nREF_AREA= 1\nREF_LENGTH= 0.5\n"
        "REF_ORIGIN_MOMENT_X= 0\nREF_ORIGIN_MOMENT_Y= 0\nREF_ORIGIN_MOMENT_Z= 0\n"
    )
    (folder / "zone_0.cfg").write_text("GRID_MOVEMENT= NONE\n")
    (folder / "zone_1.cfg").write_text("GRID_MOVEMENT= RIGID_MOTION\nROTATION_RATE= 1 0 0\n")
    _history(folder, **history)
    return point, folder


def _collect(point):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_solver("su2").collect(point).to_dict()


def test_the_stationary_zones_last_loads_are_read_under_the_single_zone_names(tmp_path):
    _, folder = _point(tmp_path)
    found = airframe(folder)
    assert found is not None and found.absent is None and found.zone == 0
    assert tuple(found.coefficients) == LOADS
    for name in LOADS:
        assert found.coefficients[name].value == pytest.approx(_value(0, name, STEPS))
    assert found.coefficients["CMy"].axis == "y" and found.coefficients["CMy"].kind == "moment"


def test_a_single_zone_point_has_no_airframe_block(tmp_path):
    folder = tmp_path / "single"
    folder.mkdir()
    (folder / "su2.cfg").write_text("TIME_DOMAIN= NO\n")
    assert airframe(folder) is None


def test_the_result_carries_the_airframe_beside_the_rotor_loads(tmp_path):
    point, _ = _point(tmp_path)
    result = _collect(point)
    window = range(28, STEPS + 1)
    # the rotor's loads keep their names and values
    rotor_cl = sum(_value(1, "CL", step) for step in window) / len(window)
    assert result["coefficients"]["CL"]["value"] == pytest.approx(rotor_cl)
    block = result["airframe"]
    assert block["zone"] == 0 and "absent" not in block
    assert set(block["coefficients"]) == set(LOADS)
    for name in LOADS:
        expected = sum(_value(0, name, step) for step in window) / len(window)
        assert block["means"][name] == pytest.approx(expected)
        assert result["time_average"]["means"][name + AIRFRAME_SUFFIX] == pytest.approx(expected)
        rotor = sum(_value(1, name, step) for step in window) / len(window)
        assert result["time_average"]["means"][name] == pytest.approx(rotor)
    # the per-revolution table carries the airframe columns too
    assert "CL" + AIRFRAME_SUFFIX in result["time_average"]["per_revolution"][0]


def test_the_unsteady_polar_has_the_airframe_columns_after_the_rotor_ones(tmp_path):
    point, _ = _point(tmp_path)
    row = {"pol": "7001", "point_id": "7001/DP", "condition": {}, "result": _collect(point)}
    written = {}

    def write(path, text):
        written[path.name] = text
        return path

    unsteady_polars(tmp_path / "post/rt", [row], write=write, cell=_cell)
    (text,) = written.values()
    header = next(csv.reader(io.StringIO(text)))
    names = [*LOADS, *(name + AIRFRAME_SUFFIX for name in LOADS)]
    assert all(name in header for name in names)
    assert header.index("CL") < header.index("CL" + AIRFRAME_SUFFIX)
    cells = next(csv.DictReader(io.StringIO(text)))
    assert float(cells["CD" + AIRFRAME_SUFFIX]) == pytest.approx(
        sum(_value(0, "CD", s) for s in range(28, STEPS + 1)) / 9
    )


@pytest.mark.parametrize("layout", ["zero", "unmonitored"])
def test_a_stationary_zone_that_monitors_no_wall_leaves_the_columns_absent(tmp_path, layout):
    point, folder = _point(tmp_path, zero=layout == "zero", stationary=layout == "zero")
    result = _collect(point)
    block = result["airframe"]
    assert set(block) == {"zone", "absent"} and block["zone"] == 0
    assert "monitors no wall" in block["absent"] or "no load column" in block["absent"]
    assert not any(name.endswith(AIRFRAME_SUFFIX) for name in result["time_average"]["means"])
    # the rotor's quantities are those of a point without the airframe
    assert set(result["time_average"]["means"]) == set(LOADS)
    row = {"pol": "7001", "point_id": "7001/DP", "condition": {}, "result": result}
    (found,) = airframe_absent([row], "rt")
    assert found[0].endswith("_uns_avg.csv#airframe,point=7001/DP") or "#airframe" in found[0]
    assert block["absent"] in found[1]
    written = {}
    unsteady_polars(
        tmp_path / "post/rt", [row], write=lambda p, t: written.setdefault(p.name, t), cell=_cell
    )
    (text,) = written.values()
    assert AIRFRAME_SUFFIX not in text.splitlines()[0]
    del folder


def test_a_point_with_an_airframe_names_no_absent_column(tmp_path):
    point, _ = _point(tmp_path)
    row = {"pol": "7001", "point_id": "7001/DP", "condition": {}, "result": _collect(point)}
    assert airframe_absent([row], "rt") == ()

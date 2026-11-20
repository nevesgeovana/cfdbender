"""A post file's ``[plots]`` groups become ``<parameter>_<group>`` columns of a SU2 point."""

from __future__ import annotations

import warnings

import pytest
import test_runtypes_b1_workflows as rt

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import workspace_path
from cfdbender.core.pproc import PprocError, read_plots
from cfdbender.matrix._unsteady_polar import plots_absent, unsteady_polars
from cfdbender.matrix.exports import _cell
from cfdbender.solvers import load_solver
from cfdbender.solvers.su2._config import config
from cfdbender.workspace.planning import plan_workspace

LOADS = ("CL", "CD", "CSF", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
STEPS = 36
WINDOW = range(28, STEPS + 1)
CLOCK = "DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_REVS_AVG: 0.25"
SLIDING = '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_OUT", "X_IN"]\n'
ZONES = (("wing", "wing_te", "fuselage"), ("BLADE_1", "spinner"))
MARKERS = [m for zone in ZONES for m in zone]
POST_A = """history = true
[plots]
parameters = ["CL", "CDI", "CD", "FX", "MY"]
[plots.families]
W = ["wing", "wing_te"]
[[plots.groups]]
name = "MRP_TOTAL"
families = "all"
[[plots.groups]]
name = "MRP_{family}"
families = "each"
"""
POST_B = """history = true
[plots]
parameters = ["CL", "CD", "FX", "FY", "FZ", "MX", "MY", "MZ"]
[[plots.groups]]
name = "MRP_TOTAL"
families = "all"
[[plots.groups]]
name = "MRP_AIRFRAME"
families = ["wing", "wing_te", "fuselage"]
[[plots.groups]]
name = "MRP_PROP"
families = ["BLADE_1", "spinner"]
"""


def _value(marker, column, step):
    """A per-marker load that differs by marker and column and moves with the step."""
    return 0.001 * (1 + MARKERS.index(marker)) * (1 + LOADS.index(column)) + 0.0001 * step


def _mean(markers, column):
    return sum(_value(m, column, s) for m in markers for s in WINDOW) / len(WINDOW)


def _history(folder, *, multizone, dead=()):
    """Per-marker columns and the totals SU2 sums from them; markers in ``dead`` carry zero."""
    zones = ZONES if multizone else (tuple(MARKERS),)
    columns = []
    for zone, markers in enumerate(zones):
        tag = f"[{zone}]" if multizone else ""
        columns.append(f"rms[Rho]{tag}")
        columns += [f"{c}{tag}" for c in LOADS]
        columns += [f"{c}{tag}({m})" for m in markers for c in LOADS]
    header = ["Time_Iter", "Outer_Iter", "Inner_Iter", *columns]
    rows = [",".join(f'"{h}"' for h in header)]
    for step in range(STEPS):
        for inner in range(2):
            cells = [step, inner, 0]
            for column in columns:
                if column.startswith("rms"):
                    cells.append(-8.0)
                    continue
                name = column.split("[")[0].split("(")[0]
                zone = int(column.split("[")[1][0]) if multizone else 0
                if "(" in column:
                    marker = column.split("(")[1][:-1]
                    value = 0.0 if marker in dead else _value(marker, name, step + 1)
                else:
                    value = sum(
                        0.0 if m in dead else _value(m, name, step + 1) for m in zones[zone]
                    )
                cells.append(value + (0.5 if inner == 0 else 0.0))
            rows.append(",".join(str(c) for c in cells))
    stem = "su2.csv" if multizone else "history.csv"
    (folder / stem).write_text("\n".join(rows) + "\n", encoding="ascii")


def _point(root, post=POST_A, *, multizone=True, dead=()):
    rt.workspace(
        root,
        workflow="unsteady_rotor",
        var=CLOCK,
        setup=SLIDING if multizone else "",
    )
    (root / "inputs/pproc/p001.toml").write_text(post, encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    head = "MULTIZONE= YES\nCONFIG_LIST= ( zone_0.cfg, zone_1.cfg )\n" if multizone else ""
    (folder / "su2.cfg").write_text(
        head + "TIME_DOMAIN= YES\n"
        f"TIME_ITER= {STEPS}\nREF_AREA= 1\nREF_LENGTH= 0.5\n"
        "REF_ORIGIN_MOMENT_X= 0\nREF_ORIGIN_MOMENT_Y= 0\nREF_ORIGIN_MOMENT_Z= 0\n"
    )
    if multizone:
        (folder / "zone_0.cfg").write_text("GRID_MOVEMENT= NONE\n")
        (folder / "zone_1.cfg").write_text("GRID_MOVEMENT= RIGID_MOTION\nROTATION_RATE= 1 0 0\n")
    _history(folder, multizone=multizone, dead=dead)
    return plan.points[0], point, folder


def _collect(point):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_solver("su2").collect(point).to_dict()


def _means(result):
    return result["time_average"]["means"]


def test_the_plots_table_has_pyflightstreams_defaults_and_refusals():
    plots = read_plots({"plots": {"groups": [{"name": "MRP_TOTAL", "families": "all"}]}})
    assert plots.parameters == ("CL", "CDI", "CDO", "CD", "FX", "FY", "FZ", "MX", "MY", "MZ")
    assert plots.groups[0].frame == "MRP"
    assert read_plots({"history": True}) is None
    bad = {
        "an unknown key": {"plots": {"group": []}},
        "an unknown parameter": {"plots": {"parameters": ["CX"]}},
        "each without the placeholder": {"plots": {"groups": [{"name": "M", "families": "each"}]}},
        "the placeholder without each": {
            "plots": {"groups": [{"name": "A_{family}", "families": "all"}]}
        },
        "a name outside the alphabet": {"plots": {"groups": [{"name": "A-B", "families": "all"}]}},
        "each_blade": {"plots": {"groups": [{"name": "A", "families": "each_blade"}]}},
        "a marker in two families": {"plots": {"families": {"A": ["x", "y"], "B": ["y"]}}},
    }
    for post in bad.values():
        with pytest.raises(PprocError):
            read_plots(post)


def test_a_multizone_point_gets_a_column_per_parameter_and_group_from_its_markers(tmp_path):
    _, point, _ = _point(tmp_path)
    result = _collect(point)
    means = _means(result)
    families = {
        "MRP_TOTAL": MARKERS,
        "MRP_W": ["wing", "wing_te"],
        "MRP_fuselage": ["fuselage"],
        "MRP_BLADE_1": ["BLADE_1"],
        "MRP_spinner": ["spinner"],
    }
    for group, markers in families.items():
        assert means[f"CL_{group}"] == pytest.approx(_mean(markers, "CL"))
        assert means[f"CD_{group}"] == pytest.approx(_mean(markers, "CD"))
        assert means[f"MY_{group}"] / _mean(markers, "CMy") == pytest.approx(
            means["MY_MRP_TOTAL"] / _mean(MARKERS, "CMy")
        )
        assert means[f"FX_{group}"] / _mean(markers, "CFx") == pytest.approx(
            means["FX_MRP_TOTAL"] / _mean(MARKERS, "CFx")
        )
    assert not any(name.startswith(("CDI_", "CDO_")) for name in means)
    assert "induced" in result["plots"]["absent"]["CDI"]
    # the plot columns lead the table, ahead of the history's own columns
    assert list(means)[0] == "CL_MRP_TOTAL"
    assert "CL_MRP_TOTAL" in result["time_average"]["per_revolution"][0]
    # the turning zone's own columns are what they were
    assert means["CL"] == pytest.approx(_mean(ZONES[1], "CL"))


def test_a_new_post_file_regroups_the_same_run_and_the_total_is_the_same(tmp_path):
    plan_point, point, folder = _point(tmp_path)
    first = _means(_collect(point))
    history = (folder / "su2.csv").read_bytes()
    (tmp_path / "inputs/pproc/p001.toml").write_text(POST_B, encoding="ascii")
    with pytest.raises(ValueError, match="input changed after planning"):
        load_point(tmp_path, plan_point, solver="su2")
    again = load_point(tmp_path, plan_point, solver="su2", post_may_change=True)
    second = _means(_collect(again))
    assert (folder / "su2.csv").read_bytes() == history
    for name in ("CL_MRP_TOTAL", "CD_MRP_TOTAL", "FX_MRP_TOTAL"):
        assert first[name] == second[name]
    assert second["CL_MRP_AIRFRAME"] == pytest.approx(_mean(ZONES[0], "CL"))
    assert second["CL_MRP_PROP"] == pytest.approx(_mean(ZONES[1], "CL"))
    assert "CL_MRP_AIRFRAME" not in first and "CL_MRP_W" not in second
    assert second["CL_MRP_AIRFRAME"] + second["CL_MRP_PROP"] == pytest.approx(
        second["CL_MRP_TOTAL"]
    )


def test_a_single_zone_point_sums_its_markers_too(tmp_path):
    _, point, _ = _point(tmp_path, POST_B, multizone=False)
    means = _means(_collect(point))
    assert means["CL_MRP_TOTAL"] == pytest.approx(_mean(MARKERS, "CL"))
    assert means["CL_MRP_PROP"] == pytest.approx(_mean(["BLADE_1", "spinner"], "CL"))


def test_a_group_whose_markers_carry_no_load_is_absent_with_its_reason(tmp_path):
    _, point, _ = _point(tmp_path, dead=ZONES[0])
    result = _collect(point)
    means = _means(result)
    assert "CL_MRP_fuselage" not in means and "CL_MRP_W" not in means
    assert "no load" in result["plots"]["absent"]["MRP_W"]
    assert means["CL_MRP_TOTAL"] == pytest.approx(_mean(ZONES[1], "CL"))
    row = {"pol": "7001", "point_id": "7001/DP", "condition": {}, "result": result}
    assert any("#plots:MRP_W,point=7001/DP" in product for product, _ in plots_absent([row], "rt"))
    written = {}
    unsteady_polars(
        tmp_path / "post/rt", [row], write=lambda p, t: written.setdefault(p.name, t), cell=_cell
    )
    (text,) = written.values()
    assert "CL_MRP_W" not in text.splitlines()[0] and "CL_MRP_TOTAL" in text.splitlines()[0]


def test_one_zero_column_does_not_make_a_group_absent(tmp_path):
    _, point, folder = _point(tmp_path)
    lines = (folder / "su2.csv").read_text(encoding="ascii").splitlines()
    header = [cell.strip('"') for cell in lines[0].split(",")]
    at = header.index("CMy[0](fuselage)")
    rows = [line.split(",") for line in lines[1:]]
    for cells in rows:
        cells[at] = "0.0"
    (folder / "su2.csv").write_text(
        "\n".join([lines[0], *(",".join(cells) for cells in rows)]) + "\n", encoding="ascii"
    )
    assert "MY_MRP_fuselage" in _means(_collect(point))


def test_without_plots_only_a_multizone_point_gets_the_two_groups_its_zones_can_fill(tmp_path):
    _, point, _ = _point(tmp_path, "history = true\n")
    means = _means(_collect(point))
    assert means["CL_MRP_TOTAL"] == pytest.approx(_mean(MARKERS, "CL"))
    assert means["CL_ROTOR_ROTOR"] == pytest.approx(_mean(ZONES[1], "CL"))
    assert not any(name.endswith("_wing") for name in means)
    single = tmp_path / "single"
    single.mkdir()
    _, point, _ = _point(single, "history = true\n", multizone=False)
    result = _collect(point)
    assert "plots" not in result
    assert not any("_MRP_" in name for name in _means(result))


def test_a_setup_and_zone_files_that_disagree_on_the_turning_zone_are_refused(tmp_path):
    _, point, folder = _point(tmp_path)
    (folder / "zone_0.cfg").write_text("GRID_MOVEMENT= RIGID_MOTION\nROTATION_RATE= 1 0 0\n")
    (folder / "zone_1.cfg").write_text("GRID_MOVEMENT= NONE\n")
    with pytest.raises(ValueError, match=r"rotating_zone.*zone_0\.cfg"):
        _collect(point)


def test_a_family_naming_a_marker_the_history_lacks_is_refused(tmp_path):
    post = POST_A.replace('["wing", "wing_te"]', '["wing", "nothing"]')
    _, point, _ = _point(tmp_path, post)
    with pytest.raises(ValueError, match="nothing"):
        _collect(point)


@pytest.mark.parametrize("multizone", [True, False])
def test_every_point_asks_for_the_per_surface_loads_of_all_its_markers(tmp_path, multizone):
    _, point, _ = _point(tmp_path, "history = true\n", multizone=multizone)
    keys = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )
    wanted = ["AERO_COEFF_SURF[0]", "AERO_COEFF_SURF[1]"] if multizone else ["AERO_COEFF_SURF"]
    assert [g for g in keys["HISTORY_OUTPUT"] if "AERO_COEFF_SURF" in str(g)] == wanted

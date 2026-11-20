"""A post file's ``[plots]`` groups become ``<parameter>_<group>`` columns of a SU2 point."""

from __future__ import annotations

import json
import sys
import warnings

import pytest
import test_runtypes_b1_workflows as rt

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import new_post_file, workspace_path
from cfdbender.core.pproc import PprocError, read_plots
from cfdbender.matrix._steady_plots import steady_plot_polars
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
    with new_post_file():
        again = load_point(tmp_path, plan_point, solver="su2")
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
    with pytest.raises(ValueError, match="nothing") as refused:
        _collect(point)
    assert "BLADE_1" in str(refused.value) and "fuselage" in str(refused.value)


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


# ---- a steady point: the same groups, the same names, from the last row of its history

STEADY_MARKERS = ["wing", "wing_te", "fuselage", "tail"]
ITERATIONS = 12


def _steady_value(marker, column, iteration):
    base = 0.01 * (1 + STEADY_MARKERS.index(marker)) * (1 + LOADS.index(column))
    return round(base + 0.001 * iteration, 6)


def _steady(root, post, *, dead=()):
    rt.workspace(root, setup="", condition="MACH:0.0529, ALPHA:0, BETA:0")
    (root / "inputs/pproc/p001.toml").write_text(post, encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    (folder / "su2.cfg").write_text(
        f"ITER= {ITERATIONS}\nREF_AREA= 1\nREF_LENGTH= 0.5\n"
        "REF_ORIGIN_MOMENT_X= 0\nREF_ORIGIN_MOMENT_Y= 0\nREF_ORIGIN_MOMENT_Z= 0\n"
    )
    columns = ["rms[Rho]", *LOADS, *[f"{c}({m})" for m in STEADY_MARKERS for c in LOADS]]
    rows = [",".join(f'"{h}"' for h in ["Inner_Iter", *columns])]
    for it in range(ITERATIONS):
        cells = [it]
        for column in columns:
            if column.startswith("rms"):
                cells.append(-8.0)
            elif "(" in column:
                marker = column.split("(")[1][:-1]
                name = column.split("(")[0]
                cells.append(0.0 if marker in dead else _steady_value(marker, name, it))
            else:
                live = [m for m in STEADY_MARKERS if m not in dead]
                cells.append(round(sum(_steady_value(m, column, it) for m in live), 6))
        rows.append(",".join(str(c) for c in cells))
    (folder / "history.csv").write_text("\n".join(rows) + "\n", encoding="ascii")
    return plan.points[0], point


def _steady_table(root, result):
    row = {"pol": "7001", "point_id": "7001/DP", "condition": {}, "result": result}
    written = {}
    steady_plot_polars(
        root / "post/rt", [row], write=lambda p, t: written.setdefault(p.name, t), cell=_cell
    )
    return written, row


STEADY_POST = """history = true
[plots]
parameters = ["CL", "CDI", "CD", "FX", "MY"]
[plots.families]
Wing = ["wing", "wing_te"]
[[plots.groups]]
name = "MRP_TOTAL"
families = "all"
[[plots.groups]]
name = "MRP_{family}"
families = "each"
"""


def test_a_steady_point_gets_the_same_columns_from_the_last_row(tmp_path):
    _, point = _steady(tmp_path, STEADY_POST)
    result = _collect(point)
    last = ITERATIONS - 1
    values = result["plots"]["values"]
    wing = sum(_steady_value(m, "CL", last) for m in ("wing", "wing_te"))
    assert values["CL_MRP_Wing"] == pytest.approx(wing)
    assert values["CD_MRP_tail"] == pytest.approx(_steady_value("tail", "CD", last))
    assert not any(name.startswith("CDI_") for name in values)
    assert "induced" in result["plots"]["absent"]["CDI"]
    written, row = _steady_table(tmp_path, result)
    (text,) = written.values()
    header = text.splitlines()[0].split(",")
    assert header[:2] == ["POL", "ITER"] and "CL_MRP_TOTAL" in header and "CL_MRP_Wing" in header
    assert any("#plots:CDI,point=7001/DP" in p for p, _ in plots_absent([row], "rt"))
    # the steady coefficients are those of a point without plots
    total = sum(_steady_value(m, "CL", last) for m in STEADY_MARKERS)
    assert result["coefficients"]["CL"]["value"] == pytest.approx(total)


def test_the_total_is_the_histories_total_within_the_rounding_of_its_digits(tmp_path):
    """SU2 prints the total and each marker's value apart, each to six significant digits.

    The sum of the markers then differs from the total by at most half a unit of the sixth
    digit per marker: four markers differ by less than 2e-6, and the test allows 5e-6.
    """
    _, point = _steady(tmp_path, STEADY_POST)
    values = _collect(point)["plots"]["values"]
    for parameter in ("CL", "CD"):
        parts = sum(values[f"{parameter}_MRP_{f}"] for f in ("Wing", "fuselage", "tail"))
        assert values[f"{parameter}_MRP_TOTAL"] == pytest.approx(parts, abs=5e-6)


def test_a_new_post_regroups_a_steady_point_and_its_total_is_unchanged(tmp_path):
    plan_point, point = _steady(tmp_path, STEADY_POST)
    first = _collect(point)["plots"]["values"]
    other = (
        'history = true\n[plots]\nparameters = ["CL"]\n'
        '[[plots.groups]]\nname = "MRP_TOTAL"\nfamilies = "all"\n'
        '[[plots.groups]]\nname = "MRP_BODY"\nfamilies = ["fuselage", "tail"]\n'
    )
    (tmp_path / "inputs/pproc/p001.toml").write_text(other, encoding="ascii")
    with new_post_file():
        again = load_point(tmp_path, plan_point, solver="su2")
    second = _collect(again)["plots"]["values"]
    assert first["CL_MRP_TOTAL"] == second["CL_MRP_TOTAL"]
    assert "CL_MRP_BODY" in second and "CL_MRP_BODY" not in first


def test_a_steady_group_whose_markers_carry_no_load_is_absent(tmp_path):
    _, point = _steady(tmp_path, STEADY_POST, dead=("tail",))
    result = _collect(point)
    assert "CL_MRP_tail" not in result["plots"]["values"]
    assert "no load" in result["plots"]["absent"]["MRP_tail"]


def test_a_steady_point_without_plots_is_unchanged(tmp_path):
    _, point = _steady(tmp_path, "history = true\n")
    result = _collect(point)
    assert "plots" not in result
    written, _ = _steady_table(tmp_path, result)
    assert written == {}


def test_collect_and_post_take_an_edited_post_file_only_with_the_option(tmp_path, capsys):
    from test_p0000_item3c2_collect import executed
    from test_p0000_item3c2_run import cli

    executed(tmp_path)
    assert cli(["collect", "--workspace", str(tmp_path)]) == 0
    (tmp_path / "inputs/pproc/p001.toml").write_text("history = false", encoding="ascii")
    assert cli(["collect", "--workspace", str(tmp_path)]) == 1
    capsys.readouterr()
    assert cli(["collect", "--workspace", str(tmp_path), "--new-pproc"]) == 0
    assert cli(["post", "--workspace", str(tmp_path), "--new-pproc"]) == 0
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert not summary["points"][0]["error"]


# ---- the absolute scale of FX..MZ, and the dynamic pressure of each zone

NL = chr(10)
# SU2's dynamic pressure at Mach 0.0529, 288.15 K, 101325 Pa, STANDARD_AIR
Q = 0.5 * 1.2249781262066513 * (0.0529 * 340.2970287557621) ** 2
AREA, LENGTH = 2.5, 0.8


@pytest.fixture
def reference(monkeypatch):
    original = rt.workspace

    def wrapped(root, **kw):
        original(root, **kw)
        f = root / "inputs/references/r001.toml"
        text = f.read_text().replace("area = 1.0", f"area = {AREA}")
        f.write_text(text.replace("length = 0.5", f"length = {LENGTH}"))

    monkeypatch.setattr(rt, "workspace", wrapped)


def _expect(means, name, coefficient, markers, factor):
    assert means[name] == pytest.approx(factor * _mean(markers, coefficient)), name


def test_time_accurate_loads_are_coefficient_times_q_a_and_q_a_l(tmp_path, reference):
    _, point, _ = _point(tmp_path, POST_B)
    means = _means(_collect(point))
    for parameter, coefficient in (("FX", "CFx"), ("FY", "CFy"), ("FZ", "CFz")):
        _expect(means, f"{parameter}_MRP_TOTAL", coefficient, MARKERS, Q * AREA)
    for parameter, coefficient in (("MX", "CMx"), ("MY", "CMy"), ("MZ", "CMz")):
        _expect(means, f"{parameter}_MRP_TOTAL", coefficient, MARKERS, Q * AREA * LENGTH)
        _expect(means, f"{parameter}_MRP_PROP", coefficient, ZONES[1], Q * AREA * LENGTH)


def test_steady_loads_are_coefficient_times_q_a_and_q_a_l(tmp_path, reference):
    six = '"FX", "FY", "FZ", "MX", "MY", "MZ"'
    _, point = _steady(tmp_path, STEADY_POST.replace('"FX", "MY"', six))
    values = _collect(point)["plots"]["values"]
    last = ITERATIONS - 1
    wing = ("wing", "wing_te")
    fx = Q * AREA * sum(_steady_value(m, "CFx", last) for m in wing)
    mz = Q * AREA * LENGTH * sum(_steady_value(m, "CMz", last) for m in wing)
    assert values["FX_MRP_Wing"] == pytest.approx(fx)
    assert values["MZ_MRP_Wing"] == pytest.approx(mz)
    mx = Q * AREA * LENGTH * _steady_value("tail", "CMx", last)
    assert values["MX_MRP_tail"] == pytest.approx(mx)


def _moving(monkeypatch, tmp_path, motion, *, condition=None):
    monkeypatch.setattr(sys.modules[__name__], "SLIDING", f"MACH_MOTION = {motion}" + NL + SLIDING)
    if condition is not None:
        original = rt.workspace

        def wrapped(root, **kw):
            kw["condition"] = condition
            original(root, **kw)

        monkeypatch.setattr(rt, "workspace", wrapped)
    _, point, folder = _point(tmp_path, POST_B)
    zone = "GRID_MOVEMENT= RIGID_MOTION" + NL + "ROTATION_RATE= 1 0 0" + NL
    (folder / "zone_1.cfg").write_text(zone + f"MACH_MOTION= {motion}" + NL)
    return point


def test_a_stated_mach_motion_scales_the_turning_zone_by_its_own_dynamic_pressure(
    tmp_path, monkeypatch
):
    point = _moving(monkeypatch, tmp_path, 0.1)
    means = _means(_collect(point))
    prop = Q * (0.1 / 0.0529) ** 2 * _mean(ZONES[1], "CFx")
    frame = Q * _mean(ZONES[0], "CFx")
    assert means["FX_MRP_PROP"] == pytest.approx(prop)
    assert means["FX_MRP_AIRFRAME"] == pytest.approx(frame)
    assert means["FX_MRP_TOTAL"] == pytest.approx(prop + frame)
    # a group over both zones: its lift is the dimensional sum over the free-stream q
    cl = (Q * (0.1 / 0.0529) ** 2 * _mean(ZONES[1], "CL") + Q * _mean(ZONES[0], "CL")) / Q
    assert means["CL_MRP_TOTAL"] == pytest.approx(cl)
    # a group in one zone is SU2's own coefficient
    assert means["CL_MRP_PROP"] == pytest.approx(_mean(ZONES[1], "CL"))


def test_a_hover_has_no_free_stream_pressure_so_mixed_coefficients_are_absent(
    tmp_path, monkeypatch
):
    point = _moving(monkeypatch, tmp_path, 0.1, condition="MACH:0, ALPHA:0, RPM:1000")
    result = _collect(point)
    means = _means(result)
    # the turning zone alone: SU2's own coefficient, and a load from its MACH_MOTION
    assert means["CL_MRP_PROP"] == pytest.approx(_mean(ZONES[1], "CL"))
    assert means["FX_MRP_PROP"] > 0.0
    # the fixed zone has no pressure to scale it: no load, never 0.0
    assert "FX_MRP_AIRFRAME" not in means and "FX_MRP_TOTAL" not in means
    assert "dynamic pressure" in result["plots"]["absent"]["FX_MRP_TOTAL"]
    # both zones: no free-stream reference for the coefficient
    assert "CL_MRP_TOTAL" not in means
    assert "free-stream" in result["plots"]["absent"]["CL_MRP_TOTAL"]


# ---- authoring errors and edge cases of the groups

HEAD = "history = true" + NL + "[plots]" + NL + 'parameters = ["CL"]' + NL


def _rewrite_header(folder, change):
    lines = (folder / "su2.csv").read_text(encoding="ascii").splitlines()
    cells = [change(c.strip('"')) for c in lines[0].split(",")]
    header = ",".join(f'"{c}"' for c in cells)
    (folder / "su2.csv").write_text(NL.join([header, *lines[1:]]) + NL, encoding="ascii")


def test_a_misspelt_word_makes_its_group_absent_with_the_reason_and_the_rest_goes_on(tmp_path):
    post = (
        HEAD
        + '[[plots.groups]]\nname = "MRP_TOTAL"\nfamilies = "all"\n'.replace("\n", NL)
        + '[[plots.groups]]\nname = "A"\nfamilies = ["wing", "fuselag"]\n'.replace("\n", NL)
    )
    _, point, _ = _point(tmp_path, post)
    result = _collect(point)
    assert "CL_A" not in _means(result) and "CL_MRP_TOTAL" in _means(result)
    (skipped,) = result["plots"]["skipped"]
    assert skipped["group"] == "A"
    assert "'fuselag'" in skipped["reason"] and "fuselage" in skipped["reason"]


def test_a_family_named_like_an_unclaimed_marker_is_refused(tmp_path):
    post = (
        HEAD
        + '[plots.families]\nwing = ["fuselage"]\n'.replace("\n", NL)
        + '[[plots.groups]]\nname = "M_{family}"\nfamilies = "each"\n'.replace("\n", NL)
    )
    _, point, _ = _point(tmp_path, post)
    with pytest.raises(ValueError, match="wall marker 'wing'"):
        _collect(point)


def test_each_on_a_history_without_per_surface_columns_says_why(tmp_path):
    post = HEAD + '[[plots.groups]]\nname = "M_{family}"\nfamilies = "each"\n'.replace("\n", NL)
    _, point, folder = _point(tmp_path, post)
    lines = (folder / "su2.csv").read_text(encoding="ascii").splitlines()
    keep = [i for i, c in enumerate(lines[0].split(",")) if "(" not in c]
    rows = [",".join(line.split(",")[i] for i in keep) for line in lines]
    (folder / "su2.csv").write_text(NL.join(rows) + NL, encoding="ascii")
    result = _collect(point)
    (skipped,) = result["plots"]["skipped"]
    assert "before per-surface output" in skipped["reason"]
    assert "CL_MRP_TOTAL" in _means(result)


def test_an_unclaimed_marker_that_cannot_name_a_plot_is_skipped_from_each(tmp_path):
    post = HEAD + '[[plots.groups]]\nname = "M_{family}"\nfamilies = "each"\n'.replace("\n", NL)
    _, point, folder = _point(tmp_path, post)
    _rewrite_header(folder, lambda c: c.replace("(fuselage)", "(fuse-lage)"))
    result = _collect(point)
    means = _means(result)
    assert "CL_M_wing" in means and not any("fuse" in name for name in means)
    (skipped,) = result["plots"]["skipped"]
    assert "'fuse-lage'" in skipped["reason"] and "[plots.families]" in skipped["reason"]


def test_a_family_word_of_several_markers_sums_them_all(tmp_path):
    post = (
        HEAD
        + '[plots.families]\nW = ["wing", "wing_te"]\n'.replace("\n", NL)
        + '[[plots.groups]]\nname = "WF"\nfamilies = ["W", "fuselage"]\n'.replace("\n", NL)
    )
    _, point, _ = _point(tmp_path, post)
    means = _means(_collect(point))
    assert means["CL_WF"] == pytest.approx(_mean(["wing", "wing_te", "fuselage"], "CL"))


def test_a_run_never_takes_the_option_and_the_others_inputs_stay_checked(tmp_path, capsys):
    from test_p0000_item3c2_collect import executed
    from test_p0000_item3c2_run import cli

    executed(tmp_path)
    assert cli(["run", "study", "--workspace", str(tmp_path), "--new-pproc"]) == 2
    for name in ("references/r001", "setups/s001"):
        path = tmp_path / f"inputs/{name}.toml"
        original = path.read_text(encoding="ascii")
        path.write_text(original + NL + "# edited" + NL, encoding="ascii")
        assert cli(["collect", "--workspace", str(tmp_path), "--new-pproc"]) == 1
        path.write_text(original, encoding="ascii")
    capsys.readouterr()


def test_tables_made_with_a_new_post_file_name_it_and_keep_a_copy(tmp_path):
    import hashlib

    from test_p0000_item3c2_collect import executed
    from test_p0000_item3c2_run import cli

    executed(tmp_path)
    assert cli(["collect", "--workspace", str(tmp_path)]) == 0
    name = "inputs/pproc/p001.toml"
    summary = tmp_path / "post/study/campaign_summary.json"
    planned = json.loads(summary.read_bytes())["points"][0]["inputs_sha256"][name]
    edited = b"history = true" + NL.encode() + b"# a new grouping" + NL.encode()
    (tmp_path / name).write_bytes(edited)
    assert cli(["collect", "--workspace", str(tmp_path), "--new-pproc"]) == 0
    digest = hashlib.sha256(edited).hexdigest()
    row = json.loads(summary.read_bytes())["points"][0]
    assert row["inputs_sha256"][name] == digest != planned
    assert (tmp_path / f"post/study/pproc/{digest}.toml").read_bytes() == edited

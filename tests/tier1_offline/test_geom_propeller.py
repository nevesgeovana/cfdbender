"""Geometry: classical propeller inputs and measured geometry contracts."""

import json
import tomllib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline

import cfdbender.geometry as geometry
from cfdbender.geometry import resolve_config
from cfdbender.geometry.stages import unpack

FIXTURES = Path(__file__).parent / "fixtures" / "propeller"


def definition(name="simple_naca"):
    return tomllib.loads((FIXTURES / f"{name}.toml").read_text(encoding="utf-8"))


def test_geom_propeller_component_retains_strict_definition():
    """the user supplied definition reaches the generator."""
    supplied = definition()
    result = resolve_config(
        {"components": {"rotor": {"type": "propeller", "definition": supplied}}}
    )
    plain = unpack(result)
    assert "definition" in plain["components"]["rotor"]
    assert plain["components"]["rotor"]["definition"]["propeller"]["blades"] == 3
    assert not any("deferred" in note or "unknown" in note for note in result["warnings"])


@pytest.mark.parametrize(
    "table", [(), ("propeller",), ("stations",), ("options",), ("output",), ("airfoils", "n4412")]
)
def test_geom_propeller_refuses_misspelt_keys(table):
    """every schema nesting refuses a misspelt key by name."""
    supplied = deepcopy(definition())
    node = supplied
    for key in table:
        node = node.setdefault(key, {})
    node["misspelt"] = 1
    with pytest.raises(ValueError, match="misspelt"):
        resolve_config({"type": "propeller", "definition": supplied})


def design(name="simple_naca"):
    assert hasattr(geometry, "Propeller"), "geometry facade exposes Propeller"
    return geometry.Propeller.from_toml(FIXTURES / f"{name}.toml")


def at(patch, u, v):
    bu = BSpline(
        np.repeat(patch.knots_u, patch.multiplicities_u),
        np.eye(len(patch.controls)),
        patch.degree_u,
    )
    bv = BSpline(
        np.repeat(patch.knots_v, patch.multiplicities_v),
        np.eye(patch.controls.shape[1]),
        patch.degree_v,
    )
    return np.einsum("i,j,ijc->c", bu(u), bv(v), patch.controls)


def test_geom_propeller_pitch_loft_and_rotor():
    """sections interpolate at r/R, and pitch is an immutable offset."""
    p = design()
    a, b = p.blade(), p.with_pitch(43).blade()
    assert len(a.patches) == 5
    assert len(p.rotor().patches) == 15
    assert p.config["propeller"]["pitch_setting"] == 28
    np.testing.assert_allclose(np.array(b.info["beta_deg"]) - a.info["beta_deg"], 15, atol=1e-12)
    for patch in a.patches[:2]:
        assert patch.degree_v == 3
        assert max(patch.multiplicities_v[1:-1]) == 1
        for r in p.config["stations"]["r_R"]:
            np.testing.assert_allclose(at(patch, 0.31, r)[2], 600 * r, atol=1e-10)
    report = p.report()
    assert report["pitch"]["beta_0p7R_deg"] == pytest.approx(28, abs=1e-12)
    assert report["ok"], report


@pytest.mark.parametrize("hand", ["positive_x", "negative_x"])
def test_geom_propeller_skew_rake_plane_and_units(hand):
    """skew preserves radius and section planarity in either hand."""
    raw = definition()
    raw["propeller"]["rotation"] = hand
    assert hasattr(geometry, "Propeller")
    p = geometry.Propeller(raw)
    q = p.with_skew(30).with_rake(0.05)
    r = np.array(raw["stations"]["r_R"])
    equivalent = p.with_skew(r * np.pi / 6, units="s/R").with_rake(0.05)
    for a, b in zip(q.blade().patches, equivalent.blade().patches, strict=True):
        np.testing.assert_allclose(a.controls, b.controls, atol=1e-10)
    sign = 1 if hand == "positive_x" else -1
    radial = np.array([0, sign * 0.5, np.sqrt(3) / 2])
    for patch in q.blade().patches[:2]:
        for station in r:
            for u in (0, 0.32, 1):
                assert at(patch, u, station) @ radial == pytest.approx(600 * station, abs=1e-10)
    assert q.report()["handedness"]["ok"]


def test_geom_propeller_analytic_integrals():
    """constant chord has independently integrable design measures."""
    raw = definition()
    raw["propeller"].update(blades=4, pitch_setting=32)
    raw["stations"]["c_R"] = [0.1] * 7
    raw["stations"]["beta_deg"] = [32] * 7
    assert hasattr(geometry, "Propeller")
    report = geometry.Propeller(raw).report()
    assert report["solidity"]["disk"] == pytest.approx(4 * 0.1 * 0.8 / np.pi, rel=1e-11)
    assert report["solidity"]["local_0p7R"] == pytest.approx(0.4 / (2 * np.pi * 0.7), rel=1e-11)
    assert report["activity_factor"]["per_blade"] == pytest.approx(
        1e5 / 16 * 0.05 * (1 - 0.2**4) / 4, rel=1e-11
    )


def test_geom_propeller_runner_records_and_log(tmp_path):
    """standalone TOML reaches measured propeller stages and the run log."""
    from cfdbender.geometry import run_geometry

    report = run_geometry(FIXTURES / "simple_naca.toml", out=tmp_path, until="S8")
    assert report["status"] == "accepted", report
    text = (tmp_path / "geom-run.log").read_text(encoding="utf-8")
    assert "propeller.section_fit" in text
    assert "propeller.handedness" in text
    assert "propeller.closure" in text
    assert "threshold" in text and "inputs of" in text


def test_geom_propeller_api_toml_identity(tmp_path):
    """the API and standalone TOML resolve the same parameter set."""
    from cfdbender.geometry import run_geometry

    a = run_geometry(FIXTURES / "simple_naca.toml", out=tmp_path / "toml", until="S8")
    b = run_geometry(design().component(), out=tmp_path / "api", until="S8")
    assert a["status"] == b["status"] == "accepted"
    assert a["parameter_id"] == b["parameter_id"]


def test_geom_propeller_toml_roundtrip_and_keywords(tmp_path):
    """explicit input and API preserve the same surfaces."""
    p = design("skewed_naca").with_pitch(35).with_skew(lambda r: 12 * np.sin(5 * r))
    file = tmp_path / "design.toml"
    p.to_toml(file)
    q = geometry.Propeller.from_toml(file)
    assert q.config == p.config
    for a, b in zip(p.rotor().patches, q.rotor().patches, strict=True):
        np.testing.assert_array_equal(a.controls, b.controls)
    raw = definition()
    rows = [
        dict(zip(raw["stations"], values, strict=True))
        for values in zip(*raw["stations"].values(), strict=True)
    ]
    q = geometry.Propeller(
        radius=600,
        blades=3,
        hub_radius=120,
        stations=rows,
        airfoils=raw["airfoils"],
        pitch_setting=28,
    )
    np.testing.assert_array_equal(
        q.blade().patches[0].controls, design().blade().patches[0].controls
    )


@pytest.mark.parametrize("target", ["blunt", "sharp"])
def test_geom_propeller_edges_extensions_and_caps(target):
    """cap options, copied root and requested edge gap reach the surfaces."""
    raw = definition()
    raw["options"].update(trailing_edge=target, te_thickness=0.006, root_extension_r_R=0.15)
    p = geometry.Propeller(raw)
    model = p.blade()
    assert len(model.patches) == (5 if target == "blunt" else 4)
    for r, chord in zip(raw["stations"]["r_R"], raw["stations"]["c_R"], strict=True):
        gap = np.linalg.norm(at(model.patches[0], 1, r) - at(model.patches[1], 1, r))
        assert gap == pytest.approx((0.006 if target == "blunt" else 0) * chord * 600, abs=1e-10)
    for patch in model.patches[:2]:
        delta = at(patch, 0.4, 0.15) - at(patch, 0.4, 0.2)
        np.testing.assert_allclose(delta, [0, 0, -30], atol=1e-10)
    assert p.report()["ok"]
    assert json.loads(json.dumps(p.report()))["closure"]["free_edges"] == 0
    raw["options"].update(root_cap="none", tip="none")
    opened = geometry.Propeller(raw)
    assert len(opened.blade().patches) == (3 if target == "blunt" else 2)
    assert opened.report()["closure"]["expected_open_edges"] == (6 if target == "blunt" else 4)


def test_geom_propeller_rotations_and_frame():
    """opposite rotation is a reflection, and rigid frames commute with build."""
    p = design().with_skew(13).with_rake(0.04)
    raw = p.config
    raw["propeller"]["rotation"] = "negative_x"
    negative = geometry.Propeller(raw)
    for a, b in zip(p.blade().patches, negative.blade().patches, strict=True):
        np.testing.assert_allclose(a.controls * [1, -1, 1], b.controls, atol=1e-10)
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    origin = np.array([31, -22, 7])
    raw = p.config
    raw["frame"] = {
        "pitch_axis_point_mm": origin.tolist(),
        "blade_dir": (rotation @ [0, 0, 1]).tolist(),
        "thrust_dir": (rotation @ [-1, 0, 0]).tolist(),
        "rotation_dir_at_blade": (rotation @ [0, -1, 0]).tolist(),
    }
    framed = geometry.Propeller(raw)
    for a, b in zip(p.rotor().patches, framed.rotor().patches, strict=True):
        np.testing.assert_allclose(a.controls @ rotation.T + origin, b.controls, atol=1e-10)


def test_geom_propeller_cst_and_coordinate_profiles(tmp_path):
    """Selig, Lednicer and CST routes produce measured closed surfaces."""
    x = np.linspace(0, 1, 161) ** 2
    z = 0.10 * np.sqrt(x) * (1 - x) + 0.001 * x
    upper, lower = np.column_stack((x, z)), np.column_stack((x, -z))
    selig = tmp_path / "profile.dat"
    lednicer = tmp_path / "counted.dat"
    np.savetxt(selig, np.vstack((upper[::-1], lower[1:])) * 4 + [7, -3], header="Synthetic profile")
    np.savetxt(
        lednicer, np.vstack(([len(x), len(x)], upper, lower)), header="Synthetic counted profile"
    )
    raw = definition()
    raw["airfoils"]["n4412"] = {"file": str(selig)}
    a = geometry.Propeller(raw)
    raw["airfoils"]["n4412"] = {"file": str(lednicer)}
    b = geometry.Propeller(raw)
    for one, two in zip(a.blade().patches, b.blade().patches, strict=True):
        np.testing.assert_allclose(one.controls, two.controls, atol=1e-8)
    raw["airfoils"]["n4412"] = {
        "cst": {
            "upper": [0.1, 0.1, 0.1],
            "lower": [-0.1, -0.1, -0.1],
            "trailing_edge_thickness": 0.002,
        }
    }
    c = geometry.Propeller(raw)
    assert a.report()["ok"] and b.report()["ok"] and c.report()["ok"]
    (tmp_path / "empty.dat").write_text("")
    raw["airfoils"]["n4412"] = {"file": str(tmp_path / "empty.dat")}
    with pytest.raises(ValueError, match="file"):
        geometry.Propeller(raw)
    raw["airfoils"]["n4412"] = {"cst": {"upper": [-0.1, -0.1], "lower": [0.1, 0.1]}}
    with pytest.raises(ValueError, match="swapped"):
        geometry.Propeller(raw).blade()


def test_geom_propeller_natural_sharp_profile():
    """a zero-gap source closes without a zero-area trailing face."""
    raw = definition()
    raw["airfoils"]["n4412"] = {"cst": {"upper": [0.1, 0.1], "lower": [-0.1, -0.1]}}
    p = geometry.Propeller(raw)
    assert len(p.blade().patches) == 4
    assert p.report()["ok"]


def test_geom_propeller_handedness_at_every_station():
    """a reversed root section cannot hide behind a valid reference section."""
    raw = definition()
    raw["stations"]["beta_deg"][0] = 210
    report = geometry.Propeller(raw).report()
    assert not report["handedness"]["ok"]


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("propeller", "radius"), "600", "radius"),
        (("propeller", "blades"), True, "blades"),
        (("propeller", "units"), "m", "units"),
        (("propeller", "rotation"), "clockwise", "rotation"),
        (("stations", "c_R"), [0.1], "c_R"),
        (("stations", "r_R"), [0.2, 0.5, 0.4, 1], "r_R"),
        (("stations", "units"), {"skew": "rad"}, "skew"),
        (("stations", "units"), {"skwe": "deg"}, "skwe"),
        (("stations", "rake"), [float("nan")] * 7, "rake"),
        (("options", "root_cap"), "rounded", "root_cap"),
        (("options", "loft"), "linear", "loft"),
        (("options", "rake_tip_deg"), 90, "rake"),
        (("options", "skew_power"), 0, "skew_power"),
        (("output", "blades"), "some", "blades"),
        (("frame", "blade_dir"), [0, 0, 0], "blade_dir"),
        (("frame", "unknown"), [], "unknown"),
        (
            ("airfoils", "n4412", "cst"),
            {"upper": [0.1, 0.1], "lower": [-0.1, -0.1], "unkn": 1},
            "unkn",
        ),
    ],
)
def test_geom_propeller_bad_input(path, value, match):
    """bad input names its failed parameter at the API boundary."""
    raw = definition()
    if path[-1] == "cst":
        raw["airfoils"]["n4412"] = {}
    node = raw
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        geometry.Propeller(raw)


def test_geom_propeller_definition_units_and_sources():
    """embedded mm and distribution units retain their meanings."""
    raw = definition()
    raw["stations"].update(skew=[0.1] * 7, rake=[5.0] * 7, units={"skew": "s/R", "rake": "deg"})
    result = resolve_config({"components": {"rotor": {"type": "propeller", "definition": raw}}})
    value = result["components"]["rotor"]["definition"]
    assert value["propeller"]["radius"]["unit"] == "mm"
    assert value["propeller"]["pitch_setting"]["unit"] == "deg"
    assert value["propeller"]["rotation"]["unit"] == "1"
    assert value["stations"]["skew"]["unit"] == "s/R"
    assert value["stations"]["rake"]["unit"] == "deg"
    assert value["options"]["fit_tol_chord"]["source"] == "default"
    assert value["propeller"]["radius"]["source"] == "given"
    assert unpack(resolve_config(result)) == unpack(result)


def test_geom_propeller_geometric_pitch_rake_and_tip():
    """physical points verify pitch, axial shift and the demo tip law."""
    from scipy.interpolate import CubicSpline

    p = design()
    a, b = p.blade(), p.with_rake(0.037).blade()
    for first, moved in zip(a.patches, b.patches, strict=True):
        np.testing.assert_array_equal(first.controls[..., 1:], moved.controls[..., 1:])
        np.testing.assert_allclose(
            moved.controls[..., 0] - first.controls[..., 0], 600 * 0.037, atol=1e-10
        )
    raw = definition()
    offset = 28 - float(CubicSpline(raw["stations"]["r_R"], raw["stations"]["beta_deg"])(0.7))
    for r, beta in zip(raw["stations"]["r_R"], raw["stations"]["beta_deg"], strict=True):
        le = (at(a.patches[0], 0, r) + at(a.patches[1], 0, r)) / 2
        te = (at(a.patches[0], 1, r) + at(a.patches[1], 1, r)) / 2
        assert np.rad2deg(np.arctan2((te - le)[0], (te - le)[1])) == pytest.approx(
            beta + offset, abs=1e-9
        )
    raw = definition("skewed_naca")
    raw["stations"].update(x_ref=[0.0] * 7, y_ref=[0.0] * 7)
    tip = at(geometry.Propeller(raw).blade().patches[0], 0, 1)
    np.testing.assert_allclose(tip, 500 * np.array([0.05, 0.5, np.sqrt(3) / 2]), atol=1e-9)


def test_geom_propeller_until_uses_stage_identity_in_filtered_plan(tmp_path):
    """a generic override plan stops at the requested registered row."""
    from cfdbender.geometry import run_geometry
    from cfdbender.geometry.pipeline import Stage, StageOutcome, default_stages
    from cfdbender.geometry.stages import check_entry

    plan = []
    for stage in default_stages():

        def passing(context, row=stage.row):
            return StageOutcome(checks=[check_entry(cid, "passed", 0, 0) for cid, _ in row.checks])

        plan.append(Stage(stage.row, passing))
    report = run_geometry(
        {"components": {"body": {"type": "body"}}},
        out=tmp_path,
        until="S11.R",
        stages=plan,
    )
    assert report["status"] == "accepted"
    assert report["stages"][-1]["stage"] == "S11.R"
    assert not (tmp_path / "S11.S.json").exists()


@pytest.mark.parametrize(
    "fixture,reference", [("simple_naca", 0.7), ("skewed_naca", 0.7), ("simple_naca", 0.65)]
)
def test_geom_propeller_reference_pitch_is_on_actual_loft(fixture, reference):
    """reference pitch and 0.7 R are measured on planar loft sections."""
    from scipy.interpolate import CubicSpline

    raw = definition(fixture)
    raw["propeller"]["pitch_reference_r_R"] = reference
    model = geometry.Propeller(raw)
    upper, lower = model.blade().patches[:2]
    table, prop = raw["stations"], raw["propeller"]
    law = CubicSpline(table["r_R"], table["beta_deg"])
    for r in (reference, 0.7):
        skew = raw["options"].get("skew_tip_deg", 0) * (
            (r - table["r_R"][0]) / (1 - table["r_R"][0])
        ) ** raw["options"].get("skew_power", 2)
        angle = np.deg2rad(skew)
        radial = np.array([0, np.sin(angle), np.cos(angle)])
        tangent = np.array([0, -np.cos(angle), np.sin(angle)])
        chord = (at(upper, 1, r) + at(lower, 1, r) - at(upper, 0, r) - at(lower, 0, r)) / 2
        beta = np.rad2deg(np.arctan2(chord[0], -chord @ tangent))
        expected = float(law(r) - law(reference) + prop["pitch_setting"])
        assert beta == pytest.approx(expected, abs=1e-8)
        for patch in (upper, lower):
            for u in (0, 0.3, 1):
                assert at(patch, u, r) @ radial == pytest.approx(prop["radius"] * r, abs=1e-8)
    actual = np.rad2deg(np.arctan2(chord[0], -chord @ tangent))
    assert model.report()["pitch"]["beta_0p7R_deg"] == pytest.approx(actual, abs=1e-10)


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("stations", "airfoil"), ["n9999"] * 7, r"unknown profile id 'n9999'.*defines n4412"),
        (("propeller", "rotation"), "clockwise", r"got 'clockwise', expected one of positive_x"),
        (("schema_version",), 2, r"schema_version: got 2, expected 1"),
        (("propeller", "misspelt"), 1, r"misspelt: unknown key; the keys are .*blades"),
    ],
)
def test_geom_propeller_refusals_name_the_value_and_the_choices(path, value, match):
    """API-004: a refusal names the received value and the valid choices or keys."""
    raw = definition()
    node = raw
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        geometry.Propeller(raw)

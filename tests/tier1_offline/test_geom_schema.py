"""Geometry: the keys of the geometry TOML, the three identities and the stage records."""

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from geom_inputs import stations as as_stations

import cfdbender
from cfdbender._errors import InputValueError
from cfdbender.geometry import load_config, resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._identity import build_id, parameter_id, short, surface_id
from cfdbender.geometry._patch import Patch
from cfdbender.geometry._records import (
    check_entry,
    exit_code,
    geometry_report,
    identity_view,
    stage_record,
    validate_stage_record,
    write_record,
)
from cfdbender.geometry.stages import write_config

GEOMETRY = Path(cfdbender.__file__).parent / "geometry"

DEMO = """
[aircraft]
name = "demo"

[components.hull]
type = "body"
length_mm = 8000.0
parts = [0.0, 0.15, 0.4, 0.8, 1.0]
sections = [{at = "middle", shape = "circle", radius_mm = 450.0}]
tail = {kind = "pole"}

[components.main]
type = "lifting_surface"
attach = {parent = "hull", station = 0.35}
planform = {sref_m2 = 9.0, aspect_ratio = 9.0, taper = 0.6, dihedral_deg = 3.0}
stations = [
  {eta = 0.0, section = {kind = "naca", code = "2412"}},
  {eta = 1.0, section = {kind = "naca", code = "0012"}},
]

[components.tailplane]
type = "lifting_surface"
attach = {parent = "hull", station = 0.9}
planform = {sref_m2 = 1.8, aspect_ratio = 5.0, taper = 0.7}

[components.main_blend]
type = "fillet"
hosts = ["main", "hull"]

[[dominance]]
dominant = "main"
dominated = "hull"

[[dominance]]
dominant = ["main", "hull"]
dominated = "main_blend"
operation = "project"

[configurations.full]
members = ["hull", "main", "tailplane", "main_blend"]
"""


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "demo.geom.toml"
    path.write_text(DEMO, encoding="utf-8")
    return path


def lifting(**extra):
    return {"components": {"w": {"type": "lifting_surface", **extra}}}


# 1. keys of section C: warn and skip, refuse with the key named


def test_unknown_keys_warn_and_are_skipped_at_every_level():
    cfg = resolve_config(
        {
            "mystery": 1,
            "aircraft": {"colour": "red"},
            "components": {
                "w": {"type": "lifting_surface", "whim": 2, "planform": {"zeta": 3}},
            },
        }
    )
    warnings = cfg["warnings"]
    for key in ("mystery", "aircraft.colour", "components.w.whim", "components.w.planform.zeta"):
        assert any(w.startswith(key + ":") for w in warnings), key
    plain = unpack(cfg)["components"]["w"]
    assert "whim" not in plain and "zeta" not in plain["planform"]


def test_demo_resolves_with_a_source_on_every_value_and_round_trips(demo, tmp_path):
    cfg = load_config(demo)
    assert cfg["aircraft"]["name"] == {"value": "demo", "unit": "1", "source": "given"}
    assert cfg["reader"]["profile"]["source"] == "default"
    assert cfg["acceptance"]["floor_mm"]["value"] == 0.5
    assert cfg["configurations"] == {"full": ["hull", "main", "tailplane", "main_blend"]}
    assert cfg["variants"]["clean"]["source"] == "default"
    rows = unpack(cfg)["dominance"]
    assert rows[1]["dominant"] == ["main", "hull"] and rows[1]["operation"] == "project"
    assert rows[0]["tolerance_m"] == 5e-5
    # every leaf of a component is a wrapper with a source
    stack = [cfg["components"], cfg["dominance"], cfg["registry"], cfg["acceptance"]]
    while stack:
        node = stack.pop()
        if isinstance(node, dict) and set(node) == {"value", "unit", "source"}:
            assert node["source"] in ("given", "default")
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    write_config(cfg, tmp_path / "demo.resolved.toml")
    assert load_config(tmp_path / "demo.resolved.toml") == cfg
    assert resolve_config(cfg) == cfg


def test_stating_a_default_explicitly_keeps_the_value_and_changes_the_source():
    plain = resolve_config(lifting())
    stated = resolve_config(lifting(trailing_edge="blunt"))
    assert unpack(plain)["components"] == unpack(stated)["components"]
    assert stated["components"]["w"]["trailing_edge"]["source"] == "given"
    assert plain["components"]["w"]["trailing_edge"]["source"] == "default"


@pytest.mark.parametrize(
    "raw,word",
    [
        ({"components": {"a": {}}}, "components.a.type"),
        ({"components": {"a": {"type": "wing"}}}, "unsupported"),
        ({"components": {"a": {"type": "body", "length_mm": float("inf")}}}, "length_mm"),
        ({"components": {"a": {"type": "body", "continuity": "G0"}}}, "continuity"),
        ({"components": {"a": {"type": "body", "tail": {"kind": "flat"}}}}, "tail"),
        ({"components": {"a": {"type": "body", "rings_file": "nowhere.csv"}}}, "rings_file"),
        (
            {"components": {"a": {"type": "lifting_surface", "sections": {"form": "cubic"}}}},
            "cubic",
        ),
        ({"components": {"a": {"type": "fillet"}}}, "hosts"),
        ({"components": {"a": {"type": "fillet", "hosts": ["a", "b"]}}}, "missing host"),
        ({"components": {"a": {"type": "blister", "host": ""}}}, "host"),
        ({"components": {"a": {"type": "lifting_surface", "version": "1.x"}}}, "version"),
        ({"components": {"a": {"type": "lifting_surface", "creases": [{"deg": 5}]}}}, "creases"),
        ({"components": {"a": {"type": "lifting_surface", "cap": {"type": "spiky"}}}}, "cap.type"),
        (
            {
                "components": {
                    "a": {
                        "type": "lifting_surface",
                        "sections": {"physical": {}},
                        "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
                    }
                }
            },
            "physical",
        ),
        ({"units": {"length": "ft"}, "components": {"a": {"type": "body"}}}, "units"),
        ({"acceptance": {"floor_mm": -1}, "components": {"a": {"type": "body"}}}, "floor_mm"),
    ],
)
def test_refusals_name_the_key(raw, word):
    with pytest.raises(InputValueError, match=word):
        resolve_config(raw)


def test_a_reader_profile_that_is_unknown_warns_and_uses_the_default():
    cfg = resolve_config({"reader": {"profile": "other"}, "components": {"a": {"type": "body"}}})
    assert unpack(cfg)["reader"]["profile"] == "flightstream"
    assert any(w.startswith("reader.profile") for w in cfg["warnings"])


def test_dominance_cycle_unknown_operation_and_unknown_component():
    base = {"components": {"a": {"type": "body"}, "b": {"type": "body"}}}
    cycle = [
        {"dominant": "a", "dominated": "b"},
        {"dominant": "b", "dominated": "a"},
    ]
    with pytest.raises(InputValueError, match="cycle"):
        resolve_config({**base, "dominance": cycle})
    with pytest.raises(InputValueError, match="operation"):
        resolve_config(
            {**base, "dominance": [{"dominant": "a", "dominated": "b", "operation": "x"}]}
        )
    cfg = resolve_config({**base, "dominance": [{"dominant": "a", "dominated": "ghost"}]})
    assert any("ghost" in w for w in cfg["warnings"])


def test_a_ribbon_on_a_cut_row_warns_and_is_skipped():
    base = {"components": {"a": {"type": "body"}, "b": {"type": "body"}}}
    row = {"dominant": "a", "dominated": "b", "operation": "cut", "ribbon": {"tail": 1}}
    cfg = resolve_config({**base, "dominance": [row]})
    assert "ribbon" not in unpack(cfg)["dominance"][0]
    assert any("ribbon" in w for w in cfg["warnings"])


def test_registry_extends_appends_overrides_and_refuses_a_cycle():
    comps = {"a": {"type": "body"}, "b": {"type": "body", "version": "2.0.0"}}
    cfg = resolve_config(
        {
            "components": comps,
            "variants": {"cs": []},
            "configurations": {
                "h": {"members": ["a"]},
                "v": {
                    "extends": "h",
                    "members": [{"component": "b", "version": "2.0.0"}],
                    "variants": ["cs", "nope"],
                },
            },
        }
    )
    registry = unpack(cfg)["registry"]
    assert [m["component"] for m in registry["v"]["members"]] == ["a", "b"]
    assert registry["h"]["variants"] == ["clean", "cs"]
    assert registry["v"]["variants"] == ["cs"]
    assert any("nope" in w for w in cfg["warnings"])
    assert resolve_config(cfg) == cfg
    loop = {"h": {"extends": "v", "members": ["a"]}, "v": {"extends": "h", "members": ["b"]}}
    with pytest.raises(InputValueError, match="cycle"):
        resolve_config({"components": comps, "configurations": loop})
    with pytest.raises(InputValueError, match="unknown component"):
        resolve_config({"components": comps, "configurations": {"x": ["a", "zz"]}})


def test_variants_naming_no_control_surface_warn_and_a_surface_resolves():
    surface = {"name": "aileron", "kind": "aileron", "eta": [0.6, 0.9]}
    cfg = resolve_config(
        {
            "components": {"main": {"type": "lifting_surface", "control": [surface]}},
            "variants": {"cs": ["main.aileron", "main.flap"]},
        }
    )
    assert any("main.flap" in w for w in cfg["warnings"])
    assert not any("main.aileron" in w for w in cfg["warnings"])
    assert unpack(cfg)["components"]["main"]["control_surfaces"][0]["chord_ratio"] == 0.25


@pytest.mark.parametrize(
    "surface,word",
    [
        ({"eta": [0.5, 0.4]}, "overlaps"),
        ({"eta": [0.1, 0.4], "chord_ratio": 0.9}, "chord_ratio"),
        ({"eta": [0.1, 0.4], "gap_span": 0.2}, "gap_span"),
        ({"eta": [0.1, 0.4], "gap_closure": "open"}, "gap_closure"),
        ({"eta": [0.1, 0.4], "kind": "slat"}, "kind"),
        ({"eta": [0.1, 0.4], "kind": "fowler", "cove": {}}, "cove.p3"),
        (
            {
                "eta": [0.1, 0.4],
                "kind": "fowler",
                "deflection_deg": [0, 20],
                "cove": {"p3": {"eta": 0.5, "z_over_c": -0.02}},
            },
            "kinematics",
        ),
    ],
)
def test_control_surface_refusals(surface, word):
    row = {"name": "f", **surface}
    with pytest.raises(InputValueError, match=word):
        resolve_config(lifting(control=[row]))


def test_duplicate_control_surface_names_are_refused():
    rows = [{"name": "f", "eta": [0.1, 0.2]}] * 2
    with pytest.raises(InputValueError, match="duplicate"):
        resolve_config(lifting(control=rows))


def test_fowler_extras_are_skipped_with_a_warning_on_a_plain_surface():
    row = {"name": "f", "eta": [0.1, 0.4], "cove": {"wall_x": 0.7}}
    cfg = resolve_config(lifting(control=[row]))
    assert any("cove" in w for w in cfg["warnings"])
    assert "cove" not in unpack(cfg)["components"]["w"]["control_surfaces"][0]


def test_section_items_set_stations_and_the_physical_form_is_exclusive():
    items = [
        {"eta": 0.0, "naca": "2412"},
        {"eta": 0.5, "naca": "0012"},
        {"eta": 1.0, "naca": "0010"},
    ]
    plain = unpack(resolve_config(lifting(stations=as_stations(items))))["components"]["w"][
        "sections"
    ]
    assert plain["eta"] == [0.0, 0.5, 1.0]
    assert plain["airfoils"] == ["NACA2412", "NACA0012", "NACA0010"]
    cst = {"eta": 0.0, "cst_upper": [0.2, 0.2], "cst_lower": [-0.1, -0.1], "te": 0.002}
    mixed = resolve_config(lifting(stations=as_stations([cst, {"eta": 1.0, "naca": "0012"}])))
    assert unpack(mixed)["components"]["w"]["sections"]["airfoils"] == ["", "NACA0012"]
    with pytest.raises(InputValueError, match="section.code"):
        resolve_config(
            lifting(stations=as_stations([{"eta": 0.0, "naca": "99"}, {"eta": 1.0, "naca": "12"}]))
        )
    with pytest.raises(InputValueError, match="stations"):
        resolve_config(
            lifting(
                stations=as_stations([{"eta": 0.5, "naca": "0012"}, {"eta": 0.5, "naca": "0012"}])
            )
        )


def test_trailing_edge_collapsed_warns_and_stays_blunt():
    cfg = resolve_config(lifting(trailing_edge="collapsed"))
    assert unpack(cfg)["components"]["w"]["trailing_edge"] == "blunt"
    assert any("trailing_edge" in w for w in cfg["warnings"])


def test_exact_loft_cap_resolves_and_a_nacelle_bound_is_checked():
    cfg = resolve_config(
        lifting(
            cap={"type": "exact_loft"},
            engine=[{"name": "n", "eta_axis": 0.25, "half_width_mm": 50.0}],
        )
    )
    cap = unpack(cfg)["components"]["w"]["tip_cap"]
    assert (cap["kind"], cap["sections"], cap["control_points"]) == ("exact_loft", 30, 31)
    bad = {"name": "n", "eta_axis": 0.35, "half_width_mm": -50.0}
    with pytest.raises(InputValueError, match="half_width_mm"):
        resolve_config(lifting(engine=[bad]))


def test_body_rear_boat_tail_and_blister_and_fillet_resolve():
    rear = {"boat_tail": {"x_cut": 0.8, "base_height": 0.1, "angle_limit_deg": 12.0}}
    tail = rear
    blister = {
        "type": "blister",
        "host": "hull",
        "stations": {"x": [0.2, 0.5]},
        "profile": {"depth": 0.1},
    }
    cfg = resolve_config(
        {
            "components": {
                "hull": {"type": "body", "tail": tail},
                "pod": blister,
                "fil": {"type": "fillet", "hosts": ["hull", "pod"]},
            }
        }
    )
    plain = unpack(cfg)["components"]
    assert plain["hull"]["rear"] == rear
    assert plain["pod"]["tips"] == "pole" and plain["fil"]["patches"] == 20
    for patch in ({"stations": {"x": [0.5, 0.2]}}, {"profile": {}}):
        with pytest.raises(InputValueError, match="stations|depth"):
            resolve_config({"components": {"hull": {"type": "body"}, "pod": {**blister, **patch}}})


def test_units_in_millimetres_and_radians_convert_to_metres_and_degrees():
    mm = resolve_config(
        {
            "units": {"length": "mm", "angle": "rad"},
            "components": {
                "w": {
                    "type": "lifting_surface",
                    "planform": {"semispan_mm": 4500.0, "incidence": 0.05},
                }
            },
        }
    )
    plain = unpack(mm)["components"]["w"]["planform"]
    assert plain["span"] == pytest.approx(9.0)
    assert plain["incidence"] == pytest.approx(0.05 * 180 / np.pi)
    assert mm["units"] == {"length": "m", "angle": "deg"}


def test_propeller_is_resolved_and_generation_is_deferred_with_a_warning():
    cfg = resolve_config({"type": "propeller"})
    assert any("0.58.0" in w for w in cfg["warnings"])


def test_open_findings_are_copied_into_the_findings_list():
    cfg = resolve_config(lifting(open_findings=["boom jumps not compared"]))
    assert cfg["findings"] == ["w: boom jumps not compared"]


# 2. no aircraft value in library code


def test_no_aircraft_token_in_the_geometry_library():
    tokens = re.compile(r"\b(FUS|WING|HTP|VTP|S[M]I|PA-?30)\b")
    hits = {
        f"{path.name}:{n}"
        for path in GEOMETRY.glob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if tokens.search(line)
    }
    assert not hits, sorted(hits)


def test_a_component_called_wing_or_fus_gets_no_special_default():
    cfg = resolve_config(
        {
            "components": {
                "WING": {"type": "lifting_surface", "attach": {"parent": "FUS"}},
                "FUS": {"type": "body"},
            }
        }
    )
    assert unpack(cfg)["components"]["WING"]["attach"]["station"] == 0.0
    with pytest.raises(InputValueError, match="hosts"):
        resolve_config({"components": {"x": {"type": "wing_fuselage_fairing"}}})


# 3. identities


def test_parameter_id_ignores_source_order_and_integer_or_float_spelling():
    a = resolve_config(lifting(planform={"semispan_mm": 4500.0, "taper": 0.5}))
    b = resolve_config(
        lifting(planform={"taper": 0.5, "semispan_mm": 4500.0}, trailing_edge="blunt")
    )
    assert parameter_id(a) == parameter_id(b)
    c = resolve_config(lifting(planform={"semispan_mm": 4500.00005, "taper": 0.5}))
    assert parameter_id(a) != parameter_id(c)
    assert len(short(parameter_id(a))) == 16


def test_parameter_id_ignores_component_table_order_and_the_aircraft_name(demo):
    cfg = load_config(demo)
    plain = unpack(cfg)
    flipped = {k: plain[k] for k in sorted(plain, reverse=True)}
    flipped["components"] = dict(reversed(list(plain["components"].items())))
    flipped["aircraft"] = {"name": "other"}
    assert parameter_id(flipped) == parameter_id(cfg)


def test_parameter_id_follows_the_content_of_a_data_file_not_its_path(tmp_path):
    (tmp_path / "rings.csv").write_text("0.1,0.2\n", encoding="utf-8")
    (tmp_path / "copy.csv").write_text("0.1,0.2\n", encoding="utf-8")
    raw = {"components": {"b": {"type": "body", "rings_file": "rings.csv"}}}
    first = resolve_config(raw, base=tmp_path)
    moved = copy.deepcopy(raw)
    moved["components"]["b"]["rings_file"] = "copy.csv"
    assert parameter_id(resolve_config(moved, base=tmp_path)) == parameter_id(first)
    (tmp_path / "copy.csv").write_text("0.1,0.3\n", encoding="utf-8")
    assert parameter_id(resolve_config(moved, base=tmp_path)) != parameter_id(first)


def patch(shift=0.0):
    controls = np.zeros((4, 4, 3))
    controls[..., 0] = np.arange(4)[:, None] + shift
    controls[..., 1] = np.arange(4)[None, :]
    return Patch("f", controls, (0.0, 1.0), (0.0, 1.0), (4, 4), (4, 4), 3, 3)


def test_surface_id_is_by_name_and_index_and_sees_a_one_ulp_move():
    one = surface_id({"b": [patch()], "a": [patch(), patch(1.0)]})
    assert one == surface_id({"a": [patch(), patch(1.0)], "b": [patch()]})
    assert one != surface_id({"a": [patch(1.0), patch()], "b": [patch()]})
    assert one != surface_id({"a": [patch(), patch(1.0 + 2e-16)], "b": [patch()]})
    assert one != surface_id({"a": [patch(), patch(1.0)], "c": [patch()]})


def test_build_id_depends_on_version_and_kernel():
    p = "a" * 64
    ids = {
        build_id(p, "0.57.0", "gmsh", "4.15"),
        build_id(p, "0.57.1", "gmsh", "4.15"),
        build_id(p, "0.57.0", "gmsh", "4.16"),
        build_id(p, "0.57.0", "other", "4.15"),
    }
    assert len(ids) == 4


def test_parameter_id_is_equal_across_fresh_processes(demo):
    code = (
        "import sys;from cfdbender.geometry import load_config;"
        "from cfdbender.geometry._identity import parameter_id;"
        "print(parameter_id(load_config(sys.argv[1])))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code, str(demo)],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ),
        )
        for _ in range(2)
    ]
    assert runs[0].stdout == runs[1].stdout
    assert runs[0].stdout.strip() == parameter_id(load_config(demo))


# 4. records


def good_check(**kw):
    return check_entry("split.identity", "passed", 3e-12, 1e-9, unit="m", where={"patch": 7}, **kw)


def test_stage_record_names_version_inputs_check_limit_and_measured(tmp_path):
    record = stage_record(
        "S7",
        "split",
        package_version="0.57.0",
        inputs_id="abc",
        parameter_id="p",
        checks=[good_check(), check_entry("split.corner_160", "skipped", note="no corners")],
        started="t",
        wall_s=1.5,
    )
    assert record["status"] == "passed" and record["package_version"] == "0.57.0"
    first = record["checks"][0]
    assert (first["id"], first["measured"], first["limit"], first["unit"]) == (
        "split.identity",
        3e-12,
        1e-9,
        "m",
    )
    validate_stage_record(record)
    write_record(tmp_path / "S7.json", record)
    assert json.loads((tmp_path / "S7.json").read_text(encoding="utf-8")) == record
    other = {**record, "started": "later", "wall_s": 9.0}
    assert identity_view(other) == identity_view(record)


def test_stage_status_follows_its_checks():
    kw = {"package_version": "v", "inputs_id": "i", "parameter_id": "p"}
    bad = check_entry("a", "failed", 2.0, 1.0, unit="deg")
    assert stage_record("S1", "x", checks=[good_check(), bad], **kw)["status"] == "failed"
    skipped = check_entry("a", "skipped", note="no original")
    assert stage_record("S1", "x", checks=[skipped], **kw)["status"] == "skipped"
    assert stage_record("S1", "x", **kw)["status"] == "skipped"


@pytest.mark.parametrize(
    "args",
    [
        ("a", "maybe", 1.0, 1.0),
        ("a", "passed", None, 1.0),
        ("a", "failed", 1.0, float("nan")),
        ("a", "skipped"),
    ],
)
def test_a_check_that_a_paper_cannot_cite_is_refused(args):
    with pytest.raises(InputValueError):
        check_entry(*args)


def test_validate_refuses_a_record_missing_what_a_paper_needs():
    record = stage_record(
        "S1", "x", package_version="v", inputs_id="i", parameter_id="p", checks=[good_check()]
    )
    for key in ("package_version", "inputs_id", "checks"):
        broken = {k: v for k, v in record.items() if k != key}
        with pytest.raises(InputValueError, match=key):
            validate_stage_record(broken)
    record["checks"][0]["measured"] = None
    with pytest.raises(InputValueError, match="measured"):
        validate_stage_record(record)


def test_report_status_and_exit_codes():
    accepted = geometry_report("accepted", "p", surface_id="s", build_id="b", stages=["S0: passed"])
    failed = geometry_report(
        "failed", "p", stage="S10", check="coverage.skin", measured=0.263, limit=0.001
    )
    refused = geometry_report("refused", "p", message="components: requires one")
    assert [exit_code(r) for r in (accepted, failed, refused)] == [0, 1, 2]
    assert (failed["stage"], failed["check"], failed["measured"]) == ("S10", "coverage.skin", 0.263)
    with pytest.raises(InputValueError, match="stage"):
        geometry_report("failed", "p")
    with pytest.raises(InputValueError, match="status"):
        geometry_report("done", "p")


# 5. the tip cap shape key


def test_tip_cap_shape_warns_is_skipped_and_is_listed_in_the_record():
    cfg = resolve_config(lifting(cap={"type": "rounded", "shape": {"depth": 0.1}}))
    assert "shape" not in unpack(cfg)["components"]["w"]["tip_cap"]
    assert any(w.startswith("w.cap.shape") for w in cfg["warnings"])
    assert any(f.startswith("w.cap.shape") for f in cfg["findings"])
    record = stage_record(
        "S0",
        "definition",
        package_version="v",
        inputs_id="i",
        parameter_id=parameter_id(cfg),
        checks=[check_entry("input.valid", "passed", 0.0, 0.0, unit="1")],
        findings=cfg["findings"],
        warnings=cfg["warnings"],
    )
    assert record["findings"] == cfg["findings"]


# The schema delta of the one-skin parametrization (key names provisional).

KNOTS = [0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0]
BSPLINE = {
    "degree": 2,
    "knots": KNOTS,
    "poles": [[1.0, 0.0], [0.5, 0.1], [0.0, 0.0], [0.5, -0.1], [1.0, 0.0]],
}


def plain_wing(tmp_path, text):
    path = tmp_path / "geometry.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    return cfg, unpack(cfg)["components"]["w"]


def test_minimal_wing_resolves_with_the_new_defaults_and_their_source(tmp_path):
    cfg, w = plain_wing(
        tmp_path,
        """
[aircraft]
name = "t"

[components.w]
type = "lifting_surface"
planform = {sref_m2 = 9.0, aspect_ratio = 9.0, taper = 0.6, dihedral_deg = 3.0}
spanwise = {law = "ruled"}
stations = [
  {eta = 0.0, section = {kind = "naca", code = "23015"}},
  {eta = 1.0, section = {kind = "naca", code = "0012"}},
]
cap = {type = "rounded"}
""",
    )
    s = w["sections"]
    assert (s["form"], s["degree"], s["continuity"], s["pin"]) == ("bspline", 5, "C2", [])
    assert s["tolerance"] == {"analytic": 2e-5, "file": 1e-4}
    assert (w["spanwise"], w["root"]) == ({"law": "ruled", "lines": []}, "open")
    assert "base_strip" not in w
    sources = cfg["components"]["w"]
    assert sources["spanwise"]["law"]["source"] == "given"
    assert sources["spanwise"]["lines"]["source"] == "default"
    assert sources["sections"]["continuity"]["source"] == "default"
    assert cfg["warnings"] == []


def test_every_new_key_given_keeps_its_value_and_source(tmp_path):
    cfg, w = plain_wing(
        tmp_path,
        """
[components.w]
type = "lifting_surface"
spanwise = {law = "smooth", lines = [0.3, 0.6]}
root = "flat"
sections.degree = 3
sections.continuity = "C1"
sections.tolerance = {analytic = 1e-5, file = 5e-5}
sections.pin = [0.25, 0.7]
""",
    )
    assert w["spanwise"] == {"law": "smooth", "lines": [0.3, 0.6]}
    assert (w["root"], w["sections"]["degree"], w["sections"]["continuity"]) == ("flat", 3, "C1")
    assert w["sections"]["tolerance"] == {"analytic": 1e-5, "file": 5e-5}
    assert w["sections"]["pin"] == [0.25, 0.7]
    src = cfg["components"]["w"]
    assert src["spanwise"]["lines"]["source"] == "given"
    assert src["sections"]["pin"]["source"] == "given"
    assert src["root"]["source"] == "given"
    assert src["sections"]["tolerance"]["file"]["source"] == "given"


def test_bspline_section_item_resolves_and_gives_no_designation():
    items = [{"eta": 0.0, "bspline": BSPLINE}, {"eta": 1.0, "naca": "0012"}]
    w = unpack(resolve_config(lifting(stations=as_stations(items))))["components"]["w"]["sections"]
    assert w["eta"] == [0.0, 1.0]
    assert w["airfoils"] == ["", "NACA0012"]
    assert w["items"][0]["bspline"]["knots"] == KNOTS


@pytest.mark.parametrize(
    ("mutate", "word"),
    [
        (lambda b: b.update(degree=0), "degree"),
        (lambda b: b.update(knots=KNOTS[:-1]), "knots"),
        (lambda b: b.update(knots=[0.0, 0.1, 0.2, 0.3, 0.7, 1.0, 1.0, 1.0]), "clamped"),
        (lambda b: b.update(knots=[0.0, 0.0, 0.0, 0.9, 0.5, 1.0, 1.0, 1.0]), "nondecreasing"),
        (lambda b: b.update(poles=[[1.0, 0.0, 0.0]] * 5), "poles"),
        (lambda b: b.pop("poles"), "poles"),
        (lambda b: b.update(knots=[0.0] * 8), "knots"),
    ],
)
def test_bad_bspline_item_is_refused_naming_the_key(mutate, word):
    spec = copy.deepcopy(BSPLINE)
    mutate(spec)
    items = [{"eta": 0.0, "bspline": spec}, {"eta": 1.0, "naca": "0012"}]
    with pytest.raises(InputValueError, match=rf"stations\[0\]\.section.*{word}"):
        resolve_config(lifting(stations=as_stations(items)))


def test_item_with_two_forms_or_a_misspelt_key_is_handled():
    # a station holds one section table whose kind picks the form, so two forms cannot be written
    no_kind = {"eta": 0.0, "section": {"code": "0012", **BSPLINE}}
    with pytest.raises(InputValueError, match="section.kind"):
        resolve_config(lifting(stations=[no_kind, *as_stations([{"eta": 1.0, "naca": "0012"}])]))
    typo = {"eta": 0.0, "section": {"kind": "bspline", **BSPLINE}, "bsplines": 1}
    cfg = resolve_config(lifting(stations=[typo, *as_stations([{"eta": 1.0, "naca": "0012"}])]))
    assert any("stations[0].bsplines: unknown key" in w for w in cfg["warnings"])


def test_cst_form_is_an_input_route_with_a_warning_and_cubic_stays_refused():
    cfg = resolve_config(lifting(sections={"form": "cst"}))
    assert any("sections.form" in w and "bspline" in w for w in cfg["warnings"])
    assert unpack(cfg)["components"]["w"]["sections"]["form"] == "bspline"
    with pytest.raises(InputValueError, match="sections.form.*cubic"):
        resolve_config(lifting(sections={"form": "cubic"}))


@pytest.mark.parametrize(
    ("extra", "word"),
    [
        ({"sections": {"continuity": "C3"}}, "sections.continuity"),
        ({"sections": {"tolerance": {"file": 0}}}, "sections.tolerance.file"),
        ({"sections": {"tolerance": {"analytic": "1e-5"}}}, "sections.tolerance.analytic"),
        ({"sections": {"pin": [0.5, 0.25]}}, "sections.pin"),
        ({"sections": {"pin": [0.0]}}, "sections.pin"),
        ({"sections": {"pin": 0.25}}, "sections.pin"),
        ({"spanwise": {"law": "ruledd"}}, "spanwise.law"),
        ({"spanwise": {"lines": [0.0]}}, "spanwise.lines"),
        ({"spanwise": {"lines": [0.6, 0.3]}}, "spanwise.lines"),
        ({"spanwise": {"lines": 0.5}}, "spanwise.lines"),
        ({"root": "closed"}, "root"),
    ],
)
def test_bad_lifting_values_are_refused_naming_the_key(extra, word):
    with pytest.raises(InputValueError, match=word):
        resolve_config(lifting(**extra))


def test_misspelt_keys_at_every_nesting_level_warn_and_are_skipped():
    cfg = resolve_config(
        lifting(
            spanwise={"law": "ruled", "line": [0.5]},
            sections={"tolerances": {"file": 1e-3}, "tolerance": {"files": 1e-3}},
            roots="flat",
        )
    )
    for key in (
        "components.w.spanwise.line",
        "components.w.sections.tolerances",
        "components.w.sections.tolerance.files",
        "components.w.roots",
    ):
        assert any(w.startswith(key + ":") for w in cfg["warnings"]), key
    w = unpack(cfg)["components"]["w"]
    assert w["spanwise"]["lines"] == [] and w["root"] == "open"


def test_base_strip_warns_and_is_skipped_at_the_load_verb(tmp_path):
    cfg, w = plain_wing(
        tmp_path,
        '[components.w]\ntype = "lifting_surface"\nbase_strip = "linear"\n',
    )
    assert any(text.startswith("components.w.base_strip:") for text in cfg["warnings"])
    assert "base_strip" not in w


def test_body_keys_resolve_with_defaults_and_are_checked():
    w = unpack(resolve_config({"components": {"b": {"type": "body"}}}))["components"]["b"]
    assert (w["regions"], w["nose_cap_fraction"], w["tolerance"], w["degree"]) == (
        [],
        0.004,
        1e-4,
        3,
    )
    given = {"regions": [0.3, 0.6], "nose_cap_fraction": 0.01, "tolerance": 2e-4, "degree": 5}
    cfg = resolve_config({"components": {"b": {"type": "body", **given}}})
    assert {k: unpack(cfg)["components"]["b"][k] for k in given} == given
    assert cfg["components"]["b"]["degree"]["source"] == "given"
    assert cfg["components"]["b"]["regions"]["source"] == "given"
    assert cfg["components"]["b"]["tolerance"]["source"] == "given"


@pytest.mark.parametrize(
    ("extra", "word"),
    [
        ({"regions": [0.0]}, "regions"),
        ({"regions": [0.6, 0.3]}, "regions"),
        ({"regions": [1.0]}, "regions"),
        ({"regions": "0.5"}, "regions"),
        ({"nose_cap_fraction": 0}, "nose_cap_fraction"),
        ({"nose_cap_fraction": 0.5}, "nose_cap_fraction"),
        ({"tolerance": -1e-4}, "tolerance"),
        ({"degree": 4}, "degree"),
        ({"degree": True}, "degree"),
    ],
)
def test_bad_body_values_are_refused_naming_the_key(extra, word):
    with pytest.raises(InputValueError, match=word):
        resolve_config({"components": {"b": {"type": "body", **extra}}})


def test_c7_flap_aileron_and_engine_are_planform_relative_inputs(tmp_path):
    cfg, w = plain_wing(
        tmp_path,
        """
[aircraft]
name = "t"

[components.w]
type = "lifting_surface"
planform = {sref_m2 = 16.0, aspect_ratio = 9.0, taper = 0.5}
stations = [
  {eta = 0.0, section = {kind = "naca", code = "23015"}},
  {eta = 1.0, section = {kind = "naca", code = "0012"}},
]
control = [
  {name = "flap", kind = "plain_flap", eta = [0.1, 0.55]},
  {name = "aileron", kind = "aileron", eta = [0.55, 0.95]},
]
engine = [{name = "engine", eta_axis = 0.31, half_width_mm = 180.0}]

[components.engine]
type = "body"
attach = {parent = "w", station = 0.31}
""",
    )
    flap, aileron = w["control_surfaces"]
    assert (flap["eta_in"], flap["eta_out"], aileron["eta_in"], aileron["eta_out"]) == (
        0.1,
        0.55,
        0.55,
        0.95,
    )
    assert w["sections"]["nacelle_bounds"][0]["eta_out"] == pytest.approx(0.34)
    assert cfg["warnings"] == []
    again = tmp_path / "geometry.toml"
    again.write_text(
        again.read_text().replace("sref_m2 = 16.0", "sref_m2 = 18.0"), encoding="utf-8"
    )
    other = unpack(load_config(again))["components"]["w"]
    assert other["control_surfaces"] == w["control_surfaces"]
    assert other["planform"]["span"] == pytest.approx((18.0 * 9.0) ** 0.5)
    for eta in (1.2, -0.1):
        bad = {"name": "flap", "kind": "plain_flap", "eta": [0.1, eta]}
        with pytest.raises(InputValueError, match="control.0..eta"):
            resolve_config(lifting(control=[bad]))


@pytest.mark.parametrize(
    ("spec", "section_degree", "word"),
    [
        ({"degree": 1, "knots": [0.0, 0.0, 0.5, 0.5, 0.7, 1.0, 1.0]}, 5, "interior knot"),
        ({}, 1, "degree"),
    ],
)
def test_a_bspline_item_the_section_form_cannot_hold_is_refused(spec, section_degree, word):
    """A repeated interior knot breaks the curve; a degree above the surface's cannot be lowered."""
    item = {**copy.deepcopy(BSPLINE), **spec}
    items = [{"eta": 0.0, "bspline": item}, {"eta": 1.0, "naca": "0012"}]
    raw = lifting(stations=as_stations(items), sections={"degree": section_degree})
    with pytest.raises(InputValueError, match=rf"stations\[0\]\.section.*{word}"):
        resolve_config(raw)


def test_c7_area_at_fixed_aspect_ratio_and_an_over_determined_planform_named():
    """C7: SREF varies at fixed AR from the area form; a third primary key names the rule."""

    def planform(**given):
        raw = {"components": {"w": {"type": "lifting_surface", "planform": given}}}
        return unpack(resolve_config(raw))["components"]["w"]["planform"]

    for area in (9.0, 12.0):
        p = planform(sref_m2=area, aspect_ratio=9.0, taper=0.6)
        assert p["area"] == pytest.approx(area) and p["aspect_ratio"] == pytest.approx(9.0)
    with pytest.raises(InputValueError) as refused:
        planform(semispan_mm=4500.0, aspect_ratio=9.0, taper=0.6, sref_m2=12.0)
    text = str(refused.value)
    assert "semispan_mm" in text and "sref_m2" in text and "redundant" in text


@pytest.mark.parametrize(
    ("kind", "given", "path"),
    [
        ("lifting_surface", {"spanwise": {"lines": [0.4]}}, "w.spanwise.lines"),
        ("lifting_surface", {"root": "flat"}, "w.root"),
        ("lifting_surface", {"sections": {"continuity": "C1"}}, "w.sections.continuity"),
        ("lifting_surface", {"sections": {"tolerance": {"file": 5e-5}}}, "w.sections.tolerance"),
        ("body", {"regions": [0.5]}, "w.regions"),
        ("body", {"nose_cap_fraction": 0.01}, "w.nose_cap_fraction"),
        ("body", {"tolerance": 2e-4}, "w.tolerance"),
        ("body", {"degree": 5}, "w.degree"),
    ],
)
def test_a_new_key_the_generator_does_not_apply_yet_is_warned(kind, given, path):
    """A value that resolves but changes no surface of this version says so (round 9's lesson)."""
    warnings = resolve_config({"components": {"w": {"type": kind, **given}}})["warnings"]
    assert [w for w in warnings if w.startswith(f"{path}:") and "does not apply" in w]
    quiet = resolve_config({"components": {"w": {"type": kind}}})["warnings"]
    assert not [w for w in quiet if "does not apply" in w]


@pytest.mark.parametrize(
    ("given", "path"),
    [
        ({"spanwise": {"law": "smooth"}}, "w.spanwise.law"),
        ({"sections": {"pin": [0.3]}}, "w.sections.pin"),
    ],
)
def test_a_key_the_generated_skin_applies_is_not_warned_as_unapplied(given, path):
    """the generated skin reads these keys (lifting_surface changes with them)."""
    raw = {"components": {"w": {"type": "lifting_surface", **given}}}
    warnings = resolve_config(raw)["warnings"]
    assert not [w for w in warnings if w.startswith(f"{path}:") and "does not apply" in w]

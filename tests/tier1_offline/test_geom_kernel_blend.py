"""G6R: the fillet and the blister on the one-skin lifting surface and on a body host.

Written from design row G6 (acceptance: both build on the demo and on a second fixture with
different hosts, tangent to 1 deg, with no module constant holding an aircraft value) and the
delta's row G6R (blister as two lofts trimmed by the host, fillet bases larger than the face).
"""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from kernel_support import kernel_or_skip

from cfdbender._errors import InputValueError
from cfdbender.geometry._blister import BodyHost, blister_model
from cfdbender.geometry._bodies import body_profile, body_regions, body_station_table
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._caps import lifting_model
from cfdbender.geometry._fillet import Surface, fillet_model
from cfdbender.geometry._generation import generate_components
from cfdbender.geometry._placement import wing_in_body_frame
from cfdbender.geometry.config import resolve_config

RADIUS = 0.45
DEMO = {
    "components": {
        "hull": {
            "type": "body",
            "length_mm": 8000.0,
            "parts": [0.0, 0.15, 0.4, 0.8, 1.0],
            "sections": [{"at": "middle", "shape": "circle", "radius_mm": 1e3 * RADIUS}],
            "tail": {"kind": "pole"},
        },
        "main": {
            "type": "lifting_surface",
            "attach": {"parent": "hull", "station": 0.35},
            "planform": {"sref_m2": 9.0, "aspect_ratio": 9.0, "taper": 0.6, "dihedral_deg": 3.0},
            "stations": [
                {"eta": 0.0, "section": {"kind": "naca", "code": "2412"}},
                {"eta": 1.0, "section": {"kind": "naca", "code": "0012"}},
            ],
        },
        "blend": {"type": "fillet", "hosts": ["main", "hull"]},
    }
}
#: A second aircraft: an elliptic, off-centre body with two exponents and a tail with anhedral,
#: blended with laws along the spine and the lower side held short of the trailing edge.
SECOND = {
    "components": {
        "pod": {
            "type": "body",
            "length_mm": 6000.0,
            "parts": [0.0, 0.2, 0.5, 0.8, 1.0],
            "sections": [
                {
                    "at": "middle",
                    "shape": "superellipse",
                    "half_width_mm": 500.0,
                    "half_height_mm": 350.0,
                    "n_top": 2.5,
                    "n_bottom": 2.0,
                    "centre_z_mm": 50.0,
                }
            ],
            "tail": {"kind": "pole"},
        },
        "tail": {
            "type": "lifting_surface",
            "attach": {"parent": "pod", "station": 0.55},
            "planform": {"sref_m2": 2.0, "aspect_ratio": 5.0, "taper": 0.7, "dihedral_deg": -2.0},
            "stations": [
                {"eta": 0.0, "section": {"kind": "naca", "code": "0012"}},
                {"eta": 1.0, "section": {"kind": "naca", "code": "0009"}},
            ],
        },
        "root": {
            "type": "fillet",
            "hosts": ["pod", "tail"],
            "extend_lower_to_te": False,
            "laws": {"radius": [0.03, 0.05, 0.02], "handle": 0.8, "setback": 0.2},
        },
    }
}
BLISTER = {
    "type": "blister",
    "host": "hull",
    "stations": {"x": [2.4, 3.0, 3.6, 4.4, 5.0]},
    "profile": {"depth": [0.0, 0.08, 0.15, 0.1, 0.0]},
    "sections": {"width": [0.0, 0.2, 0.3, 0.2, 0.0]},
}


def _plain(config):
    resolved = resolve_config(deepcopy(config))
    return resolved, unpack(resolved)["components"]


@pytest.fixture(scope="module")
def demo():
    resolved, components = _plain(DEMO)
    return resolved, components, fillet_model(components["blend"], components, "blend")


@pytest.fixture(scope="module")
def second():
    resolved, components = _plain(SECOND)
    return resolved, components, fillet_model(components["root"], components, "root")


def _skin(components, wing, body):
    return Surface(lifting_model(wing_in_body_frame(components, body, wing)).skin)


def _statuses(model):
    """The statuses of a fillet's records; on a body host ``fillet.contact_on_face`` is skipped."""
    out = {r["id"]: r["status"] for r in model.records}
    assert out.pop("fillet.contact_on_face") == "skipped"
    return set(out.values())


def _angle(a, b):
    cosine = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(cosine, 1.0))))


def test_geom_kernel_blend_fillet_on_the_demo_passes_every_record(demo):
    _, _, model = demo
    records = {r["id"]: r for r in model.records}
    assert _statuses(model) == {"passed"}
    assert records["closure.tangent_host"]["measured"] <= 1.0
    assert records["closure.tangent_wing"]["measured"] <= 1.0
    assert [t.face for t in model.trims] == ["lower", "upper", "trailing_edge"]


def test_geom_kernel_blend_fillet_meets_a_round_host_tangentially_by_its_radial_normal(demo):
    _, components, model = demo
    table = body_station_table(components["hull"])
    xs = model.contacts["host"][:, 0]
    widths = body_profile(components["hull"], table, xs / 8.0)[:, 0]
    assert np.allclose(widths, RADIUS, atol=1e-12)  # the round middle: normal is radial
    base = Surface(model.base)
    worst = 0.0
    for s in np.linspace(0.0, 1.0, 121):
        point = base(s, model.contacts["v_host"])
        radial = np.array([0.0, point[1], point[2]])
        assert abs(np.linalg.norm(radial) - RADIUS) < 1e-3
        worst = max(worst, _angle(base.normal(s, model.contacts["v_host"]), radial))
    assert worst <= 1.0


def test_geom_kernel_blend_fillet_meets_the_skin_tangentially_between_its_sections(demo):
    _, components, model = demo
    skin = _skin(components, "main", "hull")
    base = Surface(model.base)
    worst = 0.0
    s_values = model.contacts["wing"][:, 0]
    middles = (s_values[:-1] + s_values[1:]) / 2  # half way between sections, the worst place
    for s in middles:
        point = base(s, model.contacts["v_wing"])
        guess = float(np.interp(s, s_values, model.contacts["wing"][:, 2]))
        u, v = skin.closest(point, s, guess)
        assert np.linalg.norm(point - skin(u, v)) < 1e-3
        worst = max(worst, _angle(base.normal(s, model.contacts["v_wing"]), skin.normal(u, v)))
    assert worst <= 1.0


def test_geom_kernel_blend_setback_divides_the_two_contact_distances_in_its_ratio(second):
    _, _, model = second
    spine, wing_v = model.contacts["spine"], model.contacts["wing"]
    base = Surface(model.base)
    on_wing = np.array([base(s, model.contacts["v_wing"]) for s in wing_v[:, 0]])
    on_host = np.array([base(s, model.contacts["v_host"]) for s in wing_v[:, 0]])
    ratio = np.linalg.norm(on_wing - spine, axis=1) / np.linalg.norm(on_host - spine, axis=1)
    inner = slice(2, -2)  # the trailing-edge corners have no wedge to measure
    assert np.allclose(ratio[inner], (1 + 0.2) / (1 - 0.2), rtol=1e-6)


def test_geom_kernel_blend_lower_face_reaches_the_trailing_edge_unless_switched_off(demo, second):
    _, components, model = demo
    lower = model.trims[0]
    assert lower.loop[0][0][0] == 0.0
    skin = _skin(components, "main", "hull")
    start = Surface(model.base)(0.0, model.contacts["v_wing"])
    assert np.linalg.norm(start - skin(0.0, model.contacts["wing"][0, 2])) < 1e-9
    _, other, held = second
    u0 = held.trims[0].loop[0][0][0]
    assert 0.05 < u0 < 0.45
    tail = _skin(other, "tail", "pod")
    root_chord_x = [tail(0.0, 0.0)[0], tail(0.5, 0.0)[0]]
    point = tail(u0, 0.0)
    fraction = (point[0] - root_chord_x[1]) / (root_chord_x[0] - root_chord_x[1])
    assert 0.2 < fraction < 0.45  # the thickest point of a symmetric section, not the edge


def test_geom_kernel_blend_base_is_larger_than_its_faces_by_the_bands(demo):
    _, components, model = demo
    v_host, v_wing = model.contacts["v_host"], model.contacts["v_wing"]
    assert 0.0 < v_host < v_wing < 1.0
    chord = 1.25  # sref 9, AR 9, taper 0.6: 2 S / (b (1 + taper))
    base = Surface(model.base)
    beyond_host = [np.linalg.norm(base(s, 0.0) - base(s, v_host)) for s in np.linspace(0.1, 0.9, 9)]
    beyond_wing = [np.linalg.norm(base(s, 1.0) - base(s, v_wing)) for s in np.linspace(0.1, 0.9, 9)]
    band = 0.02 * chord
    assert 0.5 * band < np.mean(beyond_host) < 1.5 * band
    assert 0.5 * band < np.mean(beyond_wing) < 1.5 * band


def test_geom_kernel_blend_second_fixture_with_different_hosts_passes_every_record(second):
    _, _, model = second
    assert _statuses(model) == {"passed"}, [
        (r["id"], r["measured"]) for r in model.records if r["status"] != "passed"
    ]


def test_geom_kernel_blend_fillet_scales_with_the_aircraft_so_no_length_is_built_in(demo):
    _, _, model = demo
    scaled = deepcopy(DEMO)
    hull, main = scaled["components"]["hull"], scaled["components"]["main"]
    hull["length_mm"] = 16000.0
    hull["sections"][0]["radius_mm"] = 2e3 * RADIUS
    main["planform"]["sref_m2"] = 36.0
    _, components = _plain(scaled)
    big = fillet_model(components["blend"], components, "blend")
    assert np.allclose(big.base.controls, 2 * model.base.controls, rtol=0, atol=1e-6)


def test_geom_kernel_blend_fillet_between_two_bodies_is_refused_naming_the_hosts():
    config = deepcopy(DEMO)
    config["components"]["other"] = deepcopy(config["components"]["hull"])
    config["components"]["blend"]["hosts"] = ["hull", "other"]
    resolved, _ = _plain(config)
    with pytest.raises(InputValueError, match="one lifting surface and one body"):
        generate_components(resolved)


@pytest.mark.parametrize(
    ("laws", "match"),
    [({"setback": 1.0}, "setback"), ({"radius": [0.1, -0.1]}, "radius"), ({"handle": 0}, "handle")],
)
def test_geom_kernel_blend_a_law_out_of_range_is_refused(laws, match):
    config = deepcopy(DEMO)
    config["components"]["blend"]["laws"] = laws
    with pytest.raises(InputValueError, match=match):
        resolve_config(config)


def test_geom_kernel_blend_keys_with_no_effect_warn_at_resolve_and_a_misspelt_law_warns():
    config = deepcopy(DEMO)
    blend = config["components"]["blend"]
    blend["mesh_zone"] = {"across": 0.1}
    blend["spine"] = {"from": "hull"}
    blend["laws"] = {"radus": 0.1}
    config["components"]["pod"] = {**deepcopy(BLISTER), "sections_file": ""}
    config["components"]["pod"]["profile"]["plateau"] = 0.2
    warnings = resolve_config(config)["warnings"]
    assert any("blend.mesh_zone" in w and "skipped" in w for w in warnings)
    assert any("blend.spine" in w and "skipped" in w for w in warnings)
    assert any("radus" in w for w in warnings)
    assert any("pod.profile.plateau" in w and "not applied" in w for w in warnings)


# -- blister ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pod():
    config = deepcopy(DEMO)
    config["components"]["pod"] = deepcopy(BLISTER)
    _, components = _plain(config)
    return components, blister_model(components["pod"], components["hull"])


def test_geom_kernel_blend_blister_is_two_lofts_meeting_at_the_deepest_station(pod):
    _, model = pod
    assert set(model.bases) == {"front", "rear"}
    front, rear = model.bases["front"], model.bases["rear"]
    assert np.array_equal(front.controls[:, -1], rear.controls[:, 0])  # one shared edge curve
    assert np.allclose(front.controls[:, -1, 0], 3.6, atol=1e-12)  # planar at x of max depth
    records = {r["id"]: r for r in model.records}
    assert records["continuity.bluff_jump"]["measured"] <= 1.0
    assert {r["status"] for r in model.records} == {"passed"}


def test_geom_kernel_blend_blister_bottom_line_is_the_host_bottom_less_the_depth(pod):
    _, model = pod
    front = Surface(model.bases["front"])
    for v, x, depth in ((0.5, 3.0, 0.08), (1.0, 3.6, 0.15)):
        point = front(0.0, v)
        assert point[0] == pytest.approx(x, abs=1e-12)
        assert point[2] == pytest.approx(-RADIUS - depth, abs=1e-9)
        assert point[1] == pytest.approx(0.0, abs=1e-12)


def test_geom_kernel_blend_blister_face_ends_on_the_host_and_its_base_goes_inside(pod):
    components, model = pod
    host = BodyHost(components["hull"])
    for name, patch in model.bases.items():
        surface = Surface(patch)
        for u, v in model.crease[name]:
            point = surface(u, v)
            assert abs(np.hypot(point[1], point[2]) - RADIUS) < 1e-9  # on the round host
        for v in np.linspace(0.05, 0.95, 7):
            assert host.level(surface(1.0, v)) < 0.0  # the base runs on inside the host
        trim = next(t for t in model.trims if t.base == name)
        assert trim.loop[0][0] == trim.loop[-1][1]


def test_geom_kernel_blend_blister_ratio_stations_equal_the_absolute_ones(pod):
    components, model = pod
    ratio = deepcopy(components["pod"])
    ratio["stations"] = {"x": [x / 8.0 for x in BLISTER["stations"]["x"]], "of": "host_length"}
    ratio["profile"] = {
        "depth": [d / 8.0 for d in BLISTER["profile"]["depth"]],
        "of": "host_length",
    }
    other = blister_model(ratio, components["hull"])
    for name in model.bases:
        assert np.allclose(other.bases[name].controls, model.bases[name].controls, atol=1e-12)


def test_geom_kernel_blend_blister_wider_than_its_host_is_refused_by_its_check(pod):
    components, _ = pod
    wide = deepcopy(components["pod"])
    wide["sections"] = {"width": [0.0, 0.2, 0.6, 0.2, 0.0]}
    with pytest.raises(InputValueError, match="closure.base_present"):
        blister_model(wide, components["hull"])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"profile": {"depth": [0.0, 0.1, -0.1, 0.1, 0.0]}}, "profile.depth"),
        ({"sections": {"width": [0.1, 0.2]}}, "sections.width"),
    ],
)
def test_geom_kernel_blend_bad_blister_input_is_refused(change, match):
    config = deepcopy(DEMO)
    config["components"]["pod"] = {**deepcopy(BLISTER), **change}
    with pytest.raises(InputValueError, match=match):
        resolve_config(config)


def test_geom_kernel_blend_a_two_station_blister_warns_at_resolve_and_is_refused_at_generate():
    config = deepcopy(DEMO)
    config["components"]["pod"] = {
        **deepcopy(BLISTER),
        "stations": {"x": [2.0, 3.0]},
        "profile": {"depth": 0.1},
        "sections": {},
    }
    resolved, _ = _plain(config)
    assert any("pod.stations.x" in w and "three or more" in w for w in resolved["warnings"])
    with pytest.raises(InputValueError, match="three or more stations"):
        generate_components(resolved)


def test_geom_kernel_blend_a_blister_on_a_lifting_surface_is_refused_at_generate():
    config = deepcopy(DEMO)
    config["components"]["pod"] = {**deepcopy(BLISTER), "host": "main"}
    resolved, _ = _plain(config)
    with pytest.raises(InputValueError, match="a blister lies on a body"):
        generate_components(resolved)


def test_geom_kernel_blend_generate_writes_the_fillet_and_blister_and_reports_skipped_keys(
    tmp_path,
):
    kernel_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges

    config = deepcopy(DEMO)
    config["components"]["blend"]["mesh_zone"] = {"across": 0.1}
    config["components"]["pod"] = deepcopy(BLISTER)
    result = generate_iges(config, tmp_path / "demo.igs")
    assert result["faces"]["blend"] >= 2
    assert result["faces"]["pod"] >= 2
    assert any("blend.mesh_zone" in w and "skipped" in w for w in result["warnings"])


# -- the body's region plan (G4R), which the held wiring will make the written body -------------


def _distance_to(patches, point):
    """Return the distance from ``point`` to the nearest of ``patches`` (sampled, then Newton)."""
    best = np.inf
    for patch in patches:
        surface = Surface(patch)
        grid = np.linspace(0.0, 1.0, 9)
        samples = [(u, v, surface(u, v)) for u in grid for v in grid]
        u, v, _ = min(samples, key=lambda row: np.linalg.norm(row[2] - point))
        u, v = surface.closest(point, u, v)
        best = min(best, float(np.linalg.norm(surface(u, v) - point)))
    return best


def test_geom_kernel_blend_contacts_lie_on_the_region_plan_within_its_tolerance(demo, pod):
    _, components, model = demo
    hull = components["hull"]
    tolerance = 1e-4 * hull["length"]
    regions = [p for p in body_regions(hull) if p.role == "band"]
    for point in model.contacts["host"][::6]:
        assert _distance_to(regions, point) <= 2 * tolerance
    _, loft = pod
    surface = Surface(loft.bases["front"])
    for u, v in loft.crease["front"][2:-2:3]:
        assert _distance_to(regions, surface(u, v)) <= 2 * tolerance


# -- the checkable sub-steps of the run: build, trim by host, closure -------------------------


def _demo_with_pod():
    config = deepcopy(DEMO)
    config["components"]["pod"] = deepcopy(BLISTER)
    return config


def test_geom_kernel_blend_each_blend_sub_step_writes_its_record_with_inputs(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run

    report = _geom_run.run_geometry(_demo_with_pod(), out=tmp_path, until="S9")
    assert report["status"] == "incomplete", report.get("message")  # S1 to S7 are not built here
    steps = ["S8.B1", "S8.B2", "S8.B3", "S8.F1", "S8.F2", "S8.F3"]  # a blister may host a fillet
    assert [s["stage"] for s in report["stages"] if s["stage"].startswith("S8.")] == steps
    expected = {
        "S8.F1": {"fillet.spine", "closure.tangent_host", "closure.tangent_wing"},
        "S8.F2": {"closure.base_present", "fillet.host_footprint"},
        "S8.F3": {"closure.free_edges"},
        "S8.B2": {"trim.area_removed", "blister.crease_on_host"},
        "S8.B3": {"closure.free_edges"},
    }
    for step in steps:
        record = json.loads((tmp_path / f"{step}.json").read_text(encoding="utf-8"))
        statuses = {c["id"]: c["status"] for c in record["checks"]}
        on_face = statuses.pop("fillet.contact_on_face", "skipped")  # a body host: not trimmed
        assert set(statuses.values()) == {"passed"} and on_face == "skipped", step
        assert expected.get(step, set()) <= {c["id"] for c in record["checks"]}
        assert all(c["where"]["component"] in ("blend", "pod") for c in record["checks"])
        assert set(record["outputs"]["inputs"]) <= {"blend", "pod"}
    free = json.loads((tmp_path / "S8.F3.json").read_text(encoding="utf-8"))["checks"][0]
    assert (free["measured"], free["limit"]) == (0, 0)  # both trailing-edge ends closed


def test_geom_kernel_blend_a_too_coarse_fillet_stops_the_run_at_its_build_on_tangency(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run

    config = _demo_with_pod()
    config["components"]["blend"]["patches"] = 4  # 13 sections: the contacts leave the hosts
    report = _geom_run.run_geometry(config, out=tmp_path, until="S9")
    assert report["status"] == "failed"
    assert report["stage"] == "S8.F1"
    assert report["check"] in ("closure.tangent_host", "closure.tangent_wing")
    assert report["measured"] > report["limit"] == 1.0
    assert (tmp_path / "S8.F1.json").is_file() and not (tmp_path / "S8.F2.json").exists()


def test_geom_kernel_blend_a_crease_off_the_host_fails_the_area_removed(pod):
    from cfdbender.geometry._blister import AREA_REMOVED_LIMIT, area_removed_check

    components, model = pod
    host = BodyHost(components["hull"])
    assert area_removed_check(model.bases, model.crease, host)["status"] == "passed"
    moved = {name: rows + np.array([0.05, 0.0]) for name, rows in model.crease.items()}
    check = area_removed_check(model.bases, moved, host)
    assert check["status"] == "failed" and check["measured"] > AREA_REMOVED_LIMIT


def test_geom_kernel_blend_a_configuration_without_blends_skips_their_sub_steps_by_reason(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run

    config = deepcopy(DEMO)
    del config["components"]["blend"]
    report = _geom_run.run_geometry(config, out=tmp_path, until="S9")
    record = json.loads((tmp_path / "S8.F2.json").read_text(encoding="utf-8"))
    assert {c["status"] for c in record["checks"]} == {"skipped"}
    assert all("no fillet" in c["note"] for c in record["checks"])
    assert not any(item.startswith("S8.F") for item in report["not_run"])


def test_geom_kernel_blend_a_planted_free_edge_is_counted(pod):
    from cfdbender.geometry._blister import free_edge_check, host_gap
    from cfdbender.geometry._trims import iso_trim

    components, model = pod
    host = BodyHost(components["hull"])

    def on_host(point):
        return host_gap(host, point) <= 8e-4

    assert free_edge_check(model.bases, model.trims, on_host, expected=0)["measured"] == 0
    extra = iso_trim("loose", "front", (0.05, 0.2), (0.3, 0.6))  # a face nothing closes
    check = free_edge_check(model.bases, [*model.trims, extra], on_host, expected=0)
    assert check["status"] == "failed" and check["measured"] == 4


# -- PA-30 B3: blister sections from a file ----------------------------------------------------

#: A flat-bottomed section no superellipse gives: keel, a flat run, a chine, up into the host.
FLAT = [(0.0, -0.62), (0.12, -0.62), (0.2, -0.58), (0.26, -0.45), (0.28, -0.2)]


def _sections_csv(path, sections):
    lines = ["station,y,z"]
    for station, points in sections.items():
        lines += [f"{station},{y!r},{z!r}" for y, z in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _file_blister(tmp_path, sections):
    config = deepcopy(DEMO)
    config["components"]["pod"] = {
        "type": "blister",
        "host": "hull",
        "stations": {"x": [2.4, 3.0, 3.6, 4.4, 5.0]},
        "sections_file": str(_sections_csv(tmp_path / "pod.csv", sections)),
    }
    return config


def test_geom_kernel_blend_b3_a_sections_file_shapes_the_blister_and_no_longer_warns(tmp_path):
    config = _file_blister(tmp_path, {1: FLAT, 2: FLAT, 3: FLAT})
    resolved, components = _plain(config)
    assert not any("pod.sections_file" in w for w in resolved["warnings"])
    model = blister_model(components["pod"], components["hull"])
    assert {r["status"] for r in model.records} == {"passed"}, [
        (r["id"], r["measured"]) for r in model.records if r["status"] != "passed"
    ]
    records = {r["id"]: r for r in model.records}
    assert records["section.residual"]["measured"] < 2e-4  # every file point on the loft
    front = Surface(model.bases["front"])
    keel = front(0.0, 1.0)  # station 1, x = 3.0, the deepest: the file's first point
    assert keel[0] == pytest.approx(3.0, abs=1e-12)
    assert (keel[1], keel[2]) == pytest.approx((0.0, -0.62), abs=1e-9)
    chine = [front(u, 1.0) for u in np.linspace(0.0, 1.0, 4001)]
    nearest = min(chine, key=lambda p: np.hypot(p[1] - 0.2, p[2] + 0.58))
    assert np.hypot(nearest[1] - 0.2, nearest[2] + 0.58) < 2e-4  # the chine is kept


def test_geom_kernel_blend_b3_a_file_section_with_width_given_warns_the_width_is_skipped(tmp_path):
    config = _file_blister(tmp_path, {2: FLAT})
    config["components"]["pod"]["sections"] = {"width": 0.3}
    config["components"]["pod"]["profile"] = {"depth": 0.1}
    warnings = resolve_config(config)["warnings"]
    assert any("pod.sections.width" in w and "skipped" in w for w in warnings)
    assert any("pod.profile.depth" in w and "skipped" in w for w in warnings)


def test_geom_kernel_blend_b3_the_region_of_a_file_blister_is_its_loft(tmp_path):
    from cfdbender.geometry._generation import assembly_regions

    resolved, _ = _plain(_file_blister(tmp_path, {1: FLAT, 2: FLAT, 3: FLAT}))
    inside = assembly_regions(resolved, {})["pod"]
    assert inside([3.6, 0.1, -0.6]) and inside([3.6, 0.0, -0.5])
    assert not inside([3.6, 0.1, -0.65]) and not inside([3.6, 0.27, -0.6])


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("station,y\n1,0.0\n", "columns station, y, z"),
        ("station,y,z\n9,0.0,-0.6\n9,0.1,-0.6\n", "not one of the 5 stations"),
        ("station,y,z\n2,0.05,-0.6\n2,0.2,-0.3\n", "plane of symmetry"),
        ("station,y,z\n2,0.0,-0.6\n", "two or more points"),
        ("station,y,z\n2,a,-0.6\n", "not numbers"),
    ],
)
def test_geom_kernel_blend_b3_a_bad_sections_file_is_refused_by_line(tmp_path, text, match):
    config = _file_blister(tmp_path, {})
    (tmp_path / "pod.csv").write_text(text, encoding="utf-8")
    resolved, _ = _plain(config)
    with pytest.raises(InputValueError, match=match):
        generate_components(resolved)


# -- a fillet whose host is a blister (a low wing in a ventral fairing) ------------------------


def _low_wing_in_a_belly():
    config = deepcopy(DEMO)
    components = config["components"]
    components["main"]["placement"] = {"translation": [0.0, 0.0, -0.6]}  # under the hull
    components["belly"] = {
        "type": "blister",
        "host": "hull",
        "stations": {"x": [1.8, 2.6, 3.4, 4.4, 5.2]},
        "profile": {"depth": [0.0, 0.33, 0.35, 0.33, 0.0]},
        "sections": {"width": [0.0, 0.4, 0.42, 0.4, 0.0]},
    }
    components["blend"]["hosts"] = ["main", "belly"]
    return config


@pytest.fixture(scope="module")
def belly():
    resolved, components = _plain(_low_wing_in_a_belly())
    model = fillet_model(components["blend"], components, "blend")
    return resolved, components, model


def test_geom_kernel_blend_fillet_on_a_blister_passes_its_records_on_the_blister_face(belly):
    _, _, model = belly
    records = {r["id"]: r for r in model.records}
    assert {r["status"] for r in model.records} == {"passed"}, [
        (r["id"], r["measured"]) for r in model.records if r["status"] != "passed"
    ]
    assert records["fillet.contact_on_face"]["measured"] == 0  # every contact outside the hull


def test_geom_kernel_blend_fillet_on_a_blister_is_tangent_to_the_blister_loft(belly):
    _, components, model = belly
    loft = blister_model(components["belly"], components["hull"])
    surfaces = [Surface(p) for p in loft.bases.values()]
    base = Surface(model.base)
    worst = 0.0
    for s in np.linspace(0.02, 0.98, 41):
        point = base(s, model.contacts["v_host"])
        best = None
        for surface in surfaces:
            grid = np.linspace(0.0, 1.0, 17)
            u, v = min(
                ((a, b) for a in grid for b in grid),
                key=lambda uv, surface=surface: np.linalg.norm(surface(*uv) - point),
            )
            u, v = surface.closest(point, u, v)
            gap = float(np.linalg.norm(surface(u, v) - point))
            if best is None or gap < best[0]:
                best = (gap, surface.normal(u, v))
        assert best[0] < 1e-3
        worst = max(worst, _angle(base.normal(s, model.contacts["v_host"]), best[1]))
    assert worst <= 1.0


def test_geom_kernel_blend_generate_builds_a_fillet_on_a_blister(belly):
    resolved, _, _ = belly
    patches = generate_components(resolved)
    assert [p.family for p in patches["blend"]] == ["fillet_lower", "fillet_upper", "fillet_te"]


# -- the fillet's trailing-edge ends, closed -----------------------------------------------------


def _wing_te_in_hull_frame(components):
    from cfdbender.geometry._placement import move, placement_transform

    face = lifting_model(components["main"]).trailing_edge
    br, bt = placement_transform(components, "hull")
    wr, wt = placement_transform(components, "main")
    return Surface(move([face], br.T @ wr, br.T @ (wt - bt))[0])


def test_geom_kernel_blend_the_trailing_edge_ends_close_on_the_wing_te_face_and_the_host(demo):
    from cfdbender.geometry._skin import split_along

    _, components, model = demo
    records = {r["id"]: r for r in model.records}
    assert (records["closure.free_edges"]["measured"], records["closure.free_edges"]["limit"]) == (
        0,
        0,
    )
    te = model.trailing_edge
    assert te is not None and model.trims[-1].base == "te"
    middle = split_along(model.base, "v", [model.contacts["v_host"], model.contacts["v_wing"]])[1]
    assert np.array_equal(te.controls[:, 0][[0, -1]], middle.controls[[0, -1], 0])
    assert np.array_equal(te.controls[0], middle.controls[0])  # the lower end, shared exactly
    assert np.array_equal(te.controls[-1], middle.controls[-1])  # the upper end
    face, wing_te = Surface(te), _wing_te_in_hull_frame(components)
    for u in np.linspace(0.0, 1.0, 9):
        on_host = face(u, 0.0)
        assert abs(np.hypot(on_host[1], on_host[2]) - RADIUS) < 1e-4  # on the round hull
        point = face(u, 1.0)
        grid = np.linspace(0.0, 1.0, 41)
        a, b = min(
            ((p, q) for p in (0.0, 0.5, 1.0) for q in grid),
            key=lambda pq: np.linalg.norm(wing_te(*pq) - point),
        )
        a, b = wing_te.closest(point, a, b)
        assert np.linalg.norm(wing_te(a, b) - point) < 1e-4  # on the wing's blunt TE face


def test_geom_kernel_blend_a_fillet_held_short_closes_its_te_end_to_the_lower_corner(second):
    _, _, model = second
    records = {r["id"]: r for r in model.records}
    assert np.array_equal(
        model.trailing_edge.controls[0], np.repeat(model.contacts["spine"][:1], 4, 0)
    )
    assert (records["closure.free_edges"]["measured"], records["closure.free_edges"]["limit"]) == (
        1,
        1,
    )


# -- PA-30 B9: a fin fillet, the fin rolled upright on the second fixture's body ----------------

FIN = {
    "type": "lifting_surface",
    "surface": {"span_axis": "z"},
    "attach": {"parent": "pod", "station": 0.7},
    "planform": {
        "semispan_mm": 900.0,
        "root_chord_mm": 900.0,
        "tip_chord_mm": 500.0,
        "le_sweep_deg": 30.0,
    },
    "stations": [
        {"eta": 0.0, "section": {"kind": "naca", "code": "0010"}},
        {"eta": 1.0, "section": {"kind": "naca", "code": "0010"}},
    ],
}


def _second_with_fin():
    config = deepcopy(SECOND)
    config["components"]["fin"] = deepcopy(FIN)
    config["components"]["fin_root"] = {"type": "fillet", "hosts": ["fin", "pod"]}
    return config


@pytest.fixture(scope="module")
def fin():
    resolved, components = _plain(_second_with_fin())
    return resolved, components, fillet_model(components["fin_root"], components, "fin_root")


def test_geom_kernel_blend_b9_a_fin_fillet_passes_its_records_on_both_sides(fin):
    _, components, model = fin
    assert components["fin"]["placement"]["rotation"] == [90.0, 0.0, 0.0]  # rolled upright
    assert _statuses(model) == {"passed"}
    spine = model.contacts["spine"]
    assert np.allclose(spine[0] * [1, -1, 1], spine[-1], atol=1e-9)  # the two sides mirror
    assert spine[:, 2].min() > 0.3  # on the top of the body (centre 0.05, half height 0.35)


def test_geom_kernel_blend_b9_a_fin_fillet_is_tangent_to_the_body_s_region_faces(fin):
    _, components, model = fin
    regions = [Surface(p) for p in body_regions(components["pod"]) if p.role == "band"]
    base = Surface(model.base)
    worst = 0.0
    for s in np.linspace(0.05, 0.45, 9):  # the +y side, where the half body has faces
        point = base(s, model.contacts["v_host"])
        best = None
        for surface in regions:
            grid = np.linspace(0.0, 1.0, 9)
            u, v = min(
                ((a, b) for a in grid for b in grid),
                key=lambda uv, surface=surface: np.linalg.norm(surface(*uv) - point),
            )
            u, v = surface.closest(point, u, v)
            gap = float(np.linalg.norm(surface(u, v) - point))
            if best is None or gap < best[0]:
                best = (gap, surface.normal(u, v))
        assert best[0] < 2e-3
        worst = max(worst, _angle(base.normal(s, model.contacts["v_host"]), best[1]))
    assert worst <= 1.0


def test_geom_kernel_blend_b9_the_run_records_the_fin_fillet_sub_steps(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run

    _geom_run.run_geometry(_second_with_fin(), out=tmp_path, until="S9")
    for step in ("S8.F1", "S8.F2", "S8.F3"):
        record = json.loads((tmp_path / f"{step}.json").read_text(encoding="utf-8"))
        mine = [c for c in record["checks"] if c["where"].get("component") == "fin_root"]
        assert mine and {c["status"] for c in mine} <= {"passed", "skipped"}, step
        assert "fin_root" in record["outputs"]["inputs"]


# -- R22-003: a build that stops at a failed check stops the run there, as failed ---------------


def test_geom_kernel_blend_a_blister_off_its_host_fails_the_run_at_its_trim_sub_step(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run

    config = _demo_with_pod()
    config["components"]["pod"]["sections"] = {"width": [0.0, 0.2, 0.6, 0.2, 0.0]}
    report = _geom_run.run_geometry(config, out=tmp_path, until="S9")
    assert (report["status"], report["stage"], report["check"]) == (
        "failed",
        "S8.B2",
        "closure.base_present",
    )
    assert report["measured"] < report["limit"] == 0.0 and report["where"]["component"] == "pod"
    built = json.loads((tmp_path / "S8.B1.json").read_text(encoding="utf-8"))
    assert {c["status"] for c in built["checks"]} == {"skipped"}
    assert all("stopped at S8.B2 closure.base_present" in c["note"] for c in built["checks"])
    trim = json.loads((tmp_path / "S8.B2.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in trim["checks"] if c["status"] == "failed"] == ["closure.base_present"]
    log = (tmp_path / "geom-run.log").read_text(encoding="utf-8")
    assert "stopped at sub-step S8.B2, check closure.base_present" in log


def test_geom_kernel_blend_hosts_that_do_not_meet_fail_the_run_at_the_fillet_build(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run

    config = deepcopy(DEMO)
    config["components"]["main"]["placement"] = {"translation": [0.0, 0.0, 3.0]}  # above the hull
    report = _geom_run.run_geometry(config, out=tmp_path, until="S9")
    assert (report["status"], report["stage"], report["check"]) == (
        "failed",
        "S8.F1",
        "fillet.spine",
    )
    assert report["measured"] < report["limit"]
    assert (tmp_path / "S8.F1.json").is_file() and not (tmp_path / "S8.F2.json").exists()
    with pytest.raises(InputValueError, match="fillet.spine: the lifting surface"):
        _plain_fillet(config)


def _plain_fillet(config):
    _, components = _plain(config)
    return fillet_model(components["blend"], components, "blend")


def test_geom_kernel_blend_a_sections_file_under_other_length_units_is_warned_read_in_metres(
    tmp_path,
):
    """R22-006: [units] converts the keys, not the file: a file in millimetres builds 1000 times
    too large, so a sections file under a length unit other than metres is warned about."""
    config = _file_blister(tmp_path, {1: FLAT, 2: FLAT, 3: FLAT})
    assert not any("sections_file" in w for w in resolve_config(deepcopy(config))["warnings"])
    config["units"] = {"length": "mm"}
    hull = config["components"]["hull"]
    hull["length_mm"] = 8000.0  # unit-suffixed: millimetres whatever [units] says
    config["components"]["pod"]["stations"]["x"] = [2400.0, 3000.0, 3600.0, 4400.0, 5000.0]
    resolved, components = _plain(config)
    assert components["pod"]["stations"]["x"] == [2.4, 3.0, 3.6, 4.4, 5.0]
    (warning,) = [w for w in resolved["warnings"] if "pod.sections_file" in w]
    assert "read in meters" in warning and "'mm'" in warning
    model = blister_model(components["pod"], components["hull"])  # the file is in metres
    assert {r["status"] for r in model.records} == {"passed"}


def test_geom_kernel_blend_a_fillet_on_a_failing_blister_stops_the_run_at_the_blister(tmp_path):
    """R22-007: a blister hosts a fillet, so its sub-steps run first and its own check stops
    the run, naming the blister rather than the fillet built on it."""
    from cfdbender.geometry import pipeline as _geom_run

    config = _low_wing_in_a_belly()
    config["components"]["belly"]["sections"] = {"width": [0.0, 0.4, 0.9, 0.4, 0.0]}
    report = _geom_run.run_geometry(config, out=tmp_path, until="S9")
    assert (report["status"], report["stage"], report["check"]) == (
        "failed",
        "S8.B2",
        "closure.base_present",
    )
    assert report["where"]["component"] == "belly"
    stages = [s["stage"] for s in report["stages"] if s["stage"].startswith("S8.")]
    assert stages == ["S8.B1", "S8.B2"]


def test_geom_kernel_blend_the_log_lists_every_input_of_a_piece_whole_with_its_sections_file(
    tmp_path,
):
    """R22-008: the log's inputs line was cut at 160 characters and a blister's sections file
    was not among its inputs; each piece's inputs are written whole, the file named."""
    from cfdbender.geometry import pipeline as _geom_run

    config = _file_blister(tmp_path, {1: FLAT, 2: FLAT, 3: FLAT})
    config["components"]["pod"]["stations"]["x"] = [2.4, 3.0, 3.6, 4.4, 5.0]
    out = tmp_path / "run"
    _geom_run.run_geometry(config, out=out, until="S9")
    text = (out / "geom-run.log").read_text(encoding="utf-8")
    start = text.index("sub-step S8.B1")
    (line,) = [x for x in text[start:].splitlines() if x.startswith("  inputs of pod: ")][:1]
    inputs = json.loads(line.removeprefix("  inputs of pod: "))
    assert inputs["sections_file"].endswith("pod.csv")
    assert inputs["stations"]["x"] == [2.4, 3.0, 3.6, 4.4, 5.0] and "..." not in line
    record = json.loads((out / "S8.B1.json").read_text(encoding="utf-8"))
    assert record["outputs"]["inputs"]["pod"] == inputs  # the log derives from the record

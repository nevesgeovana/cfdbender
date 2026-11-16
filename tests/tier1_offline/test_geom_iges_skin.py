"""0.57.0 G5Rb: one skin per lifting surface, its faces as trims, the trailing-edge face, caps.

Expected values come from closed forms written here, not from the code under test: a
trapezoid's chord and leading edge at a station, the half span the published span gives, the
sagging of a chord rotated by a twist (1 - cos), and the area of a ruled panel.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

from cfdbender.geometry import resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._bspline import full_knots
from cfdbender.geometry._caps import lifting_model
from cfdbender.geometry._identity import surface_id
from cfdbender.geometry._lifting import layout_blocks, layout_record, place_points
from cfdbender.geometry._patch import Patch
from cfdbender.geometry._trims import iso_trim, trim_checks

SPAN, AR, TAPER = 8.0, 8.0, 0.6
SREF = SPAN**2 / AR


def wing(**extra):
    surface = {
        "type": "lifting_surface",
        "planform": {"sref_m2": SREF, "aspect_ratio": AR, "taper": TAPER},
        "stations": [
            {"eta": 0.0, "section": {"kind": "naca", "code": "2412"}},
            {"eta": 1.0, "section": {"kind": "naca", "code": "0012"}},
        ],
        **extra,
    }
    raw = {"components": {"W": surface, "N": {"type": "body"}}}
    return unpack(resolve_config(raw))["components"]["W"]


def _skin_share():
    """The share of the semi-span the skin covers under the rounded cap of ACCEPTANCE."""
    from cfdbender.geometry._lifting import cap_fraction

    return cap_fraction(wing(cap={"type": "rounded"}))


# Bounds 0.3 and 0.45 of the skin: a half width of 0.075 of the skin's span, a length in mm.
ACCEPTANCE = {
    "engine": [
        {"name": "N", "eta_axis": 0.375, "half_width_mm": 0.075 * SPAN / 2 * _skin_share() * 1000}
    ],
    "control": [{"name": "aileron", "eta": [0.6, 0.9], "chord_ratio": 0.25}],
    "cap": {"type": "rounded"},
}


@pytest.fixture(scope="module")
def accepted():
    component = wing(**ACCEPTANCE)
    return component, lifting_model(component)


def point(patch: Patch, u, v):
    tu = full_knots(patch.knots_u, patch.multiplicities_u)
    tv = full_knots(patch.knots_v, patch.multiplicities_v)
    along = BSpline(tv, np.asarray(patch.controls).transpose(1, 0, 2), patch.degree_v)(v)
    return BSpline(tu, along, patch.degree_u)(np.asarray(u, float))


def records(model):
    return {r["id"]: r for r in model.records if r["id"] != "continuity.internal"}


def test_geom_iges_skin_g5rb_acceptance_two_section_ruled_wing_passes_s6_to_s9_exactly(accepted):
    _, model = accepted
    found = records(model)
    checks = ("skin.section_exact", "closure.rim_shared", "trim.partition", "trim.closed",
              "trim.in_domain", "trim.iso_exact", "trim.outer_only", "law.ruled_twist")  # fmt: skip
    for check in checks:
        assert found[check]["status"] == "pass", found[check]
    assert found["skin.section_exact"]["value"] <= 1e-12
    assert found["closure.rim_shared"]["value"] <= 1e-12
    assert found["trim.partition"]["value"]["relative_area_error"] <= 1e-9
    assert not [r for r in model.records if r["status"] in ("fail", "warn")]
    internal = [r for r in model.records if r["id"] == "continuity.internal"]
    assert {r["base"] for r in internal} >= {"skin", "trailing_edge", "tip_cap_upper"}


def test_geom_iges_skin_g5rb_one_skin_degree_5_by_1_with_a_pole_row_per_defining_section(accepted):
    component, model = accepted
    skin = model.skin
    assert (skin.degree_u, skin.degree_v) == (5, 1)
    assert model.stations == (0.0, 1.0)
    assert np.asarray(skin.controls).shape[1] == 2
    # Row 1 is the tip section placed: the trapezoid's tip, where the cap starts.
    root = 2 * SPAN / AR / (1 + TAPER)
    start = model.fraction
    chord = root * (1 - (1 - TAPER) * start)
    tip_row = np.asarray(skin.controls)[:, 1]
    assert tip_row[:, 1] == pytest.approx(np.full(len(tip_row), SPAN / 2 * start), abs=1e-9)
    assert tip_row[[0, -1], 0] == pytest.approx([chord, chord], abs=1e-12)  # x of the TE ends


def test_geom_iges_skin_g5rb_published_span_includes_the_rounded_cap(accepted):
    component, model = accepted
    # Rounded, length 1: the reach is the tip's half thickness at its maximum, 0.06 chords for a
    # NACA 0012 (plus the base wedge there), and the cap ends on the half span:
    # f * b/2 + 0.06 c_r (1 - (1 - taper) f) = b/2.
    half, root = SPAN / 2, 2 * SPAN / AR / (1 + TAPER)
    expected = (half - 0.06 * root) / (half - 0.06 * root * (1 - TAPER))
    assert model.fraction == pytest.approx(expected, rel=2e-4)
    ridge = max(np.asarray(p.controls)[:, -1, 1].max() for p in model.caps.values())
    assert ridge == pytest.approx(SPAN / 2, abs=1e-9)
    flat = lifting_model(wing(cap={"type": "flat"}))
    assert flat.fraction == 1.0
    assert np.asarray(flat.skin.controls)[:, -1, 1] == pytest.approx(SPAN / 2, abs=1e-9)


def test_geom_iges_skin_g5rb_trailing_edge_face_shares_both_skin_edges(accepted):
    _, model = accepted
    te, skin = model.trailing_edge, model.skin
    v = np.linspace(0, 1, 23)
    for u_te, u_skin in ((0.0, 0.0), (1.0, 1.0)):
        a = np.array([point(te, [u_te], vv)[0] for vv in v])
        b = np.array([point(skin, [u_skin], vv)[0] for vv in v])
        assert np.abs(a - b).max() <= 1e-12
    assert (te.degree_u, len(np.asarray(te.controls))) == (1, 2)


def test_geom_iges_skin_g5rb_cap_rim_is_the_skin_tip_curve_sampled_independently(accepted):
    _, model = accepted
    t = np.linspace(0, 1, 101)
    for side, u in (("upper", 0.5 + 0.5 * t), ("lower", 0.5 - 0.5 * t)):
        cap = model.caps[f"tip_cap_{side}"]
        rim = np.array([point(cap, [tt], 0.0)[0] for tt in t])
        assert np.abs(rim - point(model.skin, u, 1.0)).max() <= 1e-12
    upper, lower = model.caps["tip_cap_upper"], model.caps["tip_cap_lower"]
    ridge = np.array([point(upper, [tt], 1.0)[0] - point(lower, [tt], 1.0)[0] for tt in t])
    assert np.abs(ridge).max() <= 1e-12


def test_geom_iges_skin_g5rb_faces_are_iso_trims_cut_at_the_nacelle_bounds(accepted):
    _, model = accepted
    faces = {t.face: t for t in model.trims}
    assert {"upper_0", "upper_1", "upper_2", "lower_0", "lower_1", "lower_2"} <= set(faces)
    vs = sorted({p[1] for t in model.trims if t.base == "skin" for piece in t.loop for p in piece})
    assert vs == pytest.approx([0.0, 0.3, 0.45, 1.0], abs=1e-9)
    us = sorted({p[0] for t in model.trims if t.base == "skin" for piece in t.loop for p in piece})
    assert us == [0.0, 0.5, 1.0]
    assert {t.base for t in model.trims} == {"skin", "trailing_edge", *model.caps}


def test_geom_iges_skin_g5rb_section_lines_are_iso_v_and_the_hinge_is_an_iso_u_knot(accepted):
    _, model = accepted
    v = model.lines["v"]
    found = sorted([v["nacelle_N_in"], v["nacelle_N_out"], v["aileron_in"], v["aileron_out"]])
    assert found == pytest.approx([0.3, 0.45, 0.6, 0.9], abs=1e-9)
    hinge = model.lines["u"]["hinge_aileron_upper"]
    assert hinge == pytest.approx(0.5 + 0.5 * np.sqrt(0.75), abs=1e-9)
    assert hinge in model.skin.knots_u
    assert model.lines["u"]["le"] == 0.5 and 0.5 in model.skin.knots_u


def test_geom_iges_skin_g5rb_partition_catches_a_missing_and_a_doubled_face(accepted):
    _, model = accepted
    bases = model.bases()
    kept = [t for t in model.trims if t.face != "upper_1"]
    assert {r["id"]: r for r in trim_checks(bases, kept)}["trim.partition"]["status"] == "fail"
    doubled = [*model.trims, iso_trim("extra", "skin", (0.5, 1.0), (0.3, 0.45))]
    assert {r["id"]: r for r in trim_checks(bases, doubled)}["trim.partition"]["status"] == "fail"
    skew = [*kept, iso_trim("upper_1", "skin", (0.5, 1.0), (0.3, 0.45))]
    shifted = skew[-1].loop[:1] + (((1.0, 0.3), (1.0, 0.5)), ((1.0, 0.5), (0.5, 0.45)))
    bent = [*kept, type(skew[-1])("upper_1", "skin", shifted + skew[-1].loop[3:], "upper")]
    checks = {r["id"]: r for r in trim_checks(bases, bent)}
    assert checks["trim.iso_exact"]["status"] == "fail"
    assert checks["trim.closed"]["status"] == "pass"  # a skew piece, but the loop closes
    assert checks["trim.partition"]["status"] == "fail"
    gap = (((0.5, 0.3), (1.0, 0.3)), ((1.0, 0.3), (1.0, 0.45)), ((1.0, 0.45), (0.5, 0.45)),
           ((0.5, 0.44), (0.5, 0.3)))  # fmt: skip
    open_loop = [*kept, type(skew[-1])("upper_1", "skin", gap, "upper")]
    checks = {r["id"]: r for r in trim_checks(bases, open_loop)}
    assert checks["trim.closed"]["status"] == "fail"
    assert checks["trim.closed"]["value"] == pytest.approx(0.01)
    outside = [*kept, iso_trim("upper_1", "skin", (0.5, 1.2), (0.3, 0.45))]
    assert {r["id"]: r for r in trim_checks(bases, outside)}["trim.in_domain"]["status"] == "fail"


def test_geom_iges_skin_g5rb_trimmed_areas_sum_to_the_ruled_panel_area():
    model = lifting_model(
        wing(stations=[{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}])
    )
    upper = sum(a for t in model.trims if t.role == "upper" for a in [_face_area(model, t)])
    # The upper surface of a trapezoid panel: arc length of the section times the mean chord.
    x = (1 - np.cos(np.linspace(0, np.pi, 4001))) / 2
    t = 0.12
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    arc = np.sum(np.hypot(np.diff(x), np.diff(yt)))
    root = 2 * SPAN / AR / (1 + TAPER)
    expected = arc * SPAN / 2 * root * (1 + TAPER) / 2
    assert upper == pytest.approx(expected, rel=2e-4)


def _face_area(model, trim):
    from cfdbender.geometry._trims import trim_areas

    return trim_areas(model.bases(), [trim])["faces"][trim.face]


def test_geom_iges_skin_g5rb_ruled_twist_inserts_sections_where_the_twist_sags_the_chord():
    straight = lifting_model(wing())
    assert records(straight)["law.ruled_twist"]["value"]["inserted"] == 0
    twisted_component = wing(
        planform={"sref_m2": SREF, "aspect_ratio": AR, "taper": TAPER, "tip_twist": -3.0}
    )
    twisted = lifting_model(twisted_component)
    found = records(twisted)["law.ruled_twist"]
    assert found["status"] == "pass" and found["value"]["inserted"] > 0
    # Between the two end sections alone, the ruled chord at mid span sags by 1 - cos(1.5 deg)
    # of the lever arm about the twist axis (0.75 chord to the trailing edge).
    sag = 0.75 * (1 - np.cos(np.radians(1.5)))
    assert sag > 2e-5  # above the analytic section tolerance, so a section had to be inserted
    # Every station of the twisted skin is the section placed by the law, exactly.
    assert records(twisted)["skin.section_exact"]["value"] <= 1e-12
    mid = 0.5 * (twisted.stations[1] + twisted.stations[2])
    shape = twisted.ends["tip"].poles
    exact = place_points(twisted_component, np.asarray(shape), mid, twisted.fraction)
    ruled = np.array([point(twisted.skin, [u], mid)[0] for u in (0.0,)])
    chord = 2 * SPAN / AR / (1 + TAPER)
    assert np.abs(ruled[0] - exact[0]).max() <= 2e-5 * chord


def test_geom_iges_skin_g5rb_layout_blocks_are_exact_pieces_of_the_skin_with_the_mesh_family_names(
    accepted,
):
    _, model = accepted
    blocks = layout_blocks(model)
    assert {b["family"] for b in blocks} == {
        "upper_fore", "upper_aft", "lower_fore", "lower_aft", "trailing_upper", "trailing_lower",
    }  # fmt: skip
    worst = 0.0
    for block in blocks:
        base = model.bases()[block["base"]]
        s = np.linspace(0, 1, 5)
        for vv in s:
            u = block["u"][0] + s * (block["u"][1] - block["u"][0])
            v = block["v"][0] + vv * (block["v"][1] - block["v"][0])
            worst = max(worst, np.abs(point(block["patch"], s, vv) - point(base, u, v)).max())
    assert worst <= 1e-12
    record = layout_record(model)
    area = sum(
        (b["u"][1] - b["u"][0]) * (b["v"][1] - b["v"][0])
        for b in record["blocks"] if b["base"] == "skin"
    )  # fmt: skip
    assert area == pytest.approx(1.0, abs=1e-9)


def test_geom_iges_skin_g5rb_surface_id_holds_the_trim_loops(accepted):
    _, model = accepted
    bases = {"W": list(model.bases().values())}
    plain = surface_id(bases)
    assert surface_id(bases, None) == plain
    with_loops = surface_id(bases, {"W": model.trims})
    assert with_loops != plain
    other = [t for t in model.trims if t.face != "upper_1"]
    assert surface_id(bases, {"W": other}) != with_loops
    assert surface_id(bases, {"W": list(reversed(model.trims))}) == with_loops


def test_geom_iges_skin_g5rb_smooth_law_interpolates_its_sections():
    component = wing(
        stations=[
            {"eta": 0.0, "section": {"kind": "naca", "code": "2412"}},
            {"eta": 1 / 3, "section": {"kind": "naca", "code": "2412"}},
            {"eta": 2 / 3, "section": {"kind": "naca", "code": "0012"}},
            {"eta": 1.0, "section": {"kind": "naca", "code": "0012"}},
        ],
        spanwise={"law": "smooth"},
    )
    model = lifting_model(component)
    assert model.skin.degree_v == 3
    assert records(model)["skin.section_exact"]["value"] <= 1e-12


def test_geom_iges_skin_g5rb_section_lines_are_not_face_edges_of_the_generated_faces():
    from cfdbender.geometry._lifting import lifting_surface

    patches = lifting_surface(wing(**ACCEPTANCE))
    assert len(patches) == 6  # one bay: the two defining sections, no split at the lines
    assert {p.bay for p in patches} == {(0.0, 1.0)}


def test_geom_iges_skin_g5rb_acceptance_wing_meshes_and_passes_the_tier1_spacing_rules():
    from kernel_support import kernel_or_skip

    from cfdbender.geometry._geometry_mesh import mesh_generated

    kernel_or_skip()
    surface = {
        "type": "lifting_surface",
        "planform": {"sref_m2": SREF, "aspect_ratio": AR, "taper": TAPER},
        **ACCEPTANCE,
    }
    config = {"components": {"W": surface, "N": {"type": "body"}}}
    record = mesh_generated(config, component="W").record
    assert record["accepted"] is True
    assert record["non_quad_count"] == 0 and record["cap_cells"] > 0
    rules = record["spacing_rules"]
    assert rules["passed"] is True and rules["checked_edges"] > 0, rules["failures"]

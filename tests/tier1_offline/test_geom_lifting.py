"""Geometry: the lifting surface, tip caps and root fillet, checked on the patches alone."""

import copy

import numpy as np
import pytest

from cfdbender.geometry import resolve_config
from cfdbender.geometry._airfoils import cst_airfoil
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._caps import (
    cap_taper,
    contour_segment,
    root_fillet,
    tip_cap,
)
from cfdbender.geometry._distributions import blended_section
from cfdbender.geometry._lifting import (
    cap_fraction,
    lifting_surface,
    planform_metrics,
    section_models,
    section_points,
    thickness_station,
)
from cfdbender.geometry._patch import Patch, evaluate_patch, interpolate_patch, patch_layout
from cfdbender.geometry._sections import ANALYTIC_LIMIT

PROFILE = {
    "stations": [0.0, 0.01, 0.09, 0.36, 0.81, 1.0],
    "start": [0.0, 0.0, 0.001, 0.004, 0.037, 0.03],
    "ridge": [0.0065, 0.027, 0.058, 0.074, 0.0775, 0.079],
    "exponent": [3.3, 2.5, 2.3, 2.4, 3.0, 5.0],
}


def component(raw):
    return unpack(resolve_config({"type": "lifting_surface", **raw}))["components"]["COMPONENT"]


def wing(kind="rounded", **extra):
    cap = {"type": kind, "length": 1.282, "exponent": 0.85} if kind == "rounded" else {"type": kind}
    if kind == "ridged":
        cap["profile"] = copy.deepcopy(PROFILE)
    return component(
        {
            "planform": {"taper": 0.7, "le_sweep_deg": 3.0},
            "stations": [
                {"eta": 0.0, "section": {"kind": "naca", "code": "23018"}},
                {"eta": 1.0, "section": {"kind": "naca", "code": "23012"}},
            ],
            "cap": cap,
            **extra,
        }
    )


def chord_and_twist(surfaces, eta_index):
    """Chord length and twist at lattice row ``eta_index``, from the leading and trailing edges."""
    le = surfaces["upper_fore"][0, eta_index]
    te = (surfaces["upper_aft"][-1, eta_index] + surfaces["lower_aft"][-1, eta_index]) / 2
    vector = te - le
    return float(np.linalg.norm(vector)), float(np.rad2deg(np.arctan2(-vector[2], vector[0])))


def lattice(patches, count):
    return {p.family: evaluate_patch(p, count) for p in patches}


def test_patch_layout_has_six_families_per_bay():
    layout = patch_layout([0.0, 1.0], breaks=[0.5])
    assert len(layout) == 12
    assert {e["family"] for e in layout} == {
        "upper_fore",
        "upper_aft",
        "lower_fore",
        "lower_aft",
        "trailing_upper",
        "trailing_lower",
    }


def test_patch_round_trips_through_plain_data_and_refuses_small_grids():
    u, v = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 6), indexing="ij")
    patch = interpolate_patch(np.stack((u, v, u * v), axis=-1), family="demo", bay=(0.0, 1.0))
    again = Patch.from_dict(patch.to_dict())
    np.testing.assert_allclose(again.evaluate(7), patch.evaluate(7), atol=1e-14)
    assert again.family == "demo" and again.bay == (0.0, 1.0)
    np.testing.assert_allclose(patch.evaluate(5)[2, 3], [0.5, 0.75, 0.375], atol=1e-12)
    with pytest.raises(ValueError, match="four by four"):
        interpolate_patch(np.zeros((3, 3, 3)))


def test_tapered_wing_chord_and_twist_follow_the_planform():
    c = component(
        {
            "planform": {"sref_m2": 50.0, "aspect_ratio": 8.0, "taper": 0.7, "incidence": 3.0},
            "stations": [
                {"eta": 0.0, "section": {"kind": "naca", "code": "23018"}},
                {"eta": 1.0, "section": {"kind": "naca", "code": "23012"}},
            ],
        }
    )
    patches = lifting_surface(c)
    assert len(patches) == 6
    metrics = planform_metrics(c)
    assert metrics["area"] == pytest.approx(50.0, abs=1e-10)
    assert metrics["mac"] == pytest.approx(2.526, abs=5e-4)
    surfaces = lattice(patches, 11)
    # The skin is built from the stored B-spline sections, whose leading-edge point lies within
    # the conversion limit of the analytic nose; chord and twist follow within that limit.
    for index, eta in enumerate(np.linspace(0, 1, 11)):
        chord, twist = chord_and_twist(surfaces, index)
        expected = (50 / 17) * (1 - 0.3 * eta)
        assert chord == pytest.approx(expected, abs=ANALYTIC_LIMIT * expected)
        assert twist == pytest.approx(3.0, abs=np.degrees(2 * ANALYTIC_LIMIT))
        assert surfaces["upper_fore"][0, index, 1] == pytest.approx(10 * eta, abs=1e-12)


def test_single_section_wing_is_a_constant_chord_rectangle():
    c = component({"stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}]})
    patches = lifting_surface(c)
    assert len(patches) == 6
    surfaces = lattice(patches, 5)
    for index in range(5):
        chord, twist = chord_and_twist(surfaces, index)
        assert chord == pytest.approx(0.125, abs=ANALYTIC_LIMIT * 0.125)
        assert twist == pytest.approx(0.0, abs=1e-10)
    assert planform_metrics(c)["area"] == pytest.approx(0.125)


def corners(surface):
    c = [surface[0, 0], surface[-1, 0], surface[-1, -1], surface[0, -1]]
    angles, sides = [], []
    for k in range(4):
        a, b = c[k - 1] - c[k], c[(k + 1) % 4] - c[k]
        cosine = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        sides.append(np.linalg.norm(b))
    return angles, sides


def normals(surface):
    n = np.cross(np.gradient(surface, axis=0), np.gradient(surface, axis=1))
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-300)


@pytest.fixture(scope="module")
def rounded():
    c = wing("rounded")
    return c, [evaluate_patch(p) for p in tip_cap(c)]


def test_rounded_cap_has_four_patches_with_sound_corners(rounded):
    c, surfaces = rounded
    assert len(surfaces) == 4
    te_half = c["sections"]["trailing_edge_thickness"] / 2 * c["planform"]["tip_chord"]
    for s in surfaces:
        angles, sides = corners(s)
        assert min(sides) > 0.9 * te_half
        assert 30.0 < min(angles) and max(angles) < 150.0


def test_rounded_cap_normals_point_outward(rounded):
    c, surfaces = rounded
    chord = c["planform"]["root_chord"]
    centre = np.concatenate([s.reshape(-1, 3) for s in surfaces]).mean(axis=0) - np.array(
        [0, 0.5 * chord, 0]
    )
    for s in surfaces:
        direction = s - centre
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        assert (np.einsum("ijk,ijk->ij", normals(s), direction)[2:-2, 1:-1] > 0).all()


def test_rounded_cap_adjacent_patches_agree_on_shared_edges(rounded):
    _, (u0, u1, l0, l1) = rounded
    worst = 0.0
    pairs = [
        (u0, u1, (-1, slice(None)), (0, slice(None))),
        (l0, l1, (-1, slice(None)), (0, slice(None))),
        (u0, l0, (slice(8, None), -1), (slice(8, None), 0)),
        (u1, l1, (slice(None, -2), -1), (slice(None, -2), 0)),
    ]
    for a, b, ia, ib in pairs:
        assert np.abs(a[ia] - b[ib]).max() < 1e-9
        na, nb = normals(a)[ia], normals(b)[ib]
        worst = max(worst, np.degrees(np.arccos(np.clip((na * nb).sum(-1), -1, 1))).max())
    assert worst < 25.0


def test_rounded_cap_is_closed_on_the_base_plane(rounded):
    c, (_, u1, _, l1) = rounded
    # The published span includes the cap, so the skin and the cap base end where the cap starts.
    base = c["planform"]["span"] / 2 * cap_fraction(c)
    assert base < c["planform"]["span"] / 2
    for s in (u1, l1):
        assert np.abs(s[-1][:, 1] - base).max() < 1e-12
    assert np.abs(u1[-1, -1] - l1[-1, 0]).max() < 1e-12


def test_rounded_cap_ridge_length_scales_with_the_half_thickness():
    c = component(
        {
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
            "cap": {"type": "rounded", "length": 1.7},
        }
    )
    models = section_models(c["sections"])
    s = blended_section(c["sections"], models, 1.0)
    x = thickness_station(c, models, 1.0)
    upper, lower = cst_airfoil(x, s["upper"], s["lower"], s["trailing_edge_thickness"])
    expected = 0.125 * (upper[1] - lower[1]) / 2 * 1.7
    patch = tip_cap(c)[0]
    # The published span includes the cap: the ridge ends on the half span, and the cap starts
    # one reach inboard of it (the section's half thickness within its conversion limit).
    start = 0.5 * cap_fraction(c)
    assert evaluate_patch(patch, 3)[-1, -1][1] == pytest.approx(0.5, abs=1e-12)
    assert 0.5 - start == pytest.approx(expected, rel=1e-3)


def test_flat_cap_is_one_patch_and_none_is_empty():
    c = component({"cap": {"type": "flat"}})
    flat = tip_cap(c)
    assert len(flat) == 1 and flat[0].family == "tip_cap"
    assert tip_cap(component({"cap": {"type": "none"}})) == []


@pytest.fixture(scope="module")
def ridged():
    c = wing("ridged")
    patches = tip_cap(c)
    return c, patches, [evaluate_patch(p) for p in patches]


def test_ridged_cap_has_four_caps_and_two_closures_without_degenerate_sides(ridged):
    _, patches, surfaces = ridged
    assert [p.family for p in patches] == ["tip_cap"] * 4 + ["tip_te_closure"] * 2
    for s in surfaces:
        c = [s[0, 0], s[-1, 0], s[-1, -1], s[0, -1]]
        assert min(np.linalg.norm(c[k] - c[k - 1]) for k in range(4)) > 1e-4


def test_ridged_cap_corner_angles_and_no_fold(ridged):
    c, _, surfaces = ridged
    centre = np.concatenate([s.reshape(-1, 3) for s in surfaces[:4]]).mean(axis=0)
    centre[1] = c["planform"]["span"] / 2 - 0.3
    for s in surfaces[:4]:
        angles, _ = corners(s)
        assert all(30 < a < 150 for a in angles)
        direction = s - centre
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        assert (np.einsum("ijk,ijk->ij", normals(s), direction)[2:-2, 1:-1] > 0).all()


def test_ridged_cap_reach_follows_the_profile_and_the_planform_taper(ridged):
    c, _, surfaces = ridged
    # The published span includes the cap: the cap reaches from its start to the half span.
    start = c["planform"]["span"] / 2 * cap_fraction(c)
    reach = max(s[..., 1].max() for s in surfaces) - start
    assert max(s[..., 1].max() for s in surfaces) == pytest.approx(c["planform"]["span"] / 2)
    chord = c["planform"]["root_chord"] + (
        c["planform"]["tip_chord"] - c["planform"]["root_chord"]
    ) * cap_fraction(c)
    assert reach == pytest.approx(max(PROFILE["ridge"]) * chord, rel=0.02)
    assert cap_taper(c) > 0
    assert surfaces[4][..., 0].min() < surfaces[4][..., 0].max()
    assert surfaces[1][:, 0][..., 1].max() == pytest.approx(start, abs=1e-9)


FILLET = {"root_fillet": {"enabled": True, "radii": [0.002, 0.003, 0.002]}}


def test_root_fillet_strips_join_end_to_end():
    c = component(
        {"stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}], **FILLET}
    )
    patches = root_fillet(c)
    assert len(patches) == 6 and all(p.family == "root_fillet" for p in patches)
    surfaces = [evaluate_patch(p, 31) for p in patches]
    gaps = [
        np.linalg.norm(a[-1] - surfaces[(i + 1) % 6][0], axis=-1).max()
        for i, a in enumerate(surfaces)
    ]
    assert max(gaps) < 1e-7


def test_root_fillet_touches_the_wing_skin_at_its_root_contact():
    c = component(
        {
            "planform": {"taper": 0.7},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
            **FILLET,
        }
    )
    patches = root_fillet(c)
    models = section_models(c["sections"])
    section = blended_section(c["sections"], models, 0.0)
    maximum = thickness_station(c, models, 0.0)
    surfaces = [evaluate_patch(p, 5) for p in patches]
    gaps = []
    for i, side in [(0, "upper"), (1, "upper"), (4, "lower"), (5, "lower")]:
        for k in (1, 2, 3):
            xyz = surfaces[i][k, 0]
            x = contour_segment(section, maximum, i, np.array([k / 4]))[0, 0]
            eta = 2 * xyz[1] / c["planform"]["span"]
            gaps.append(
                np.linalg.norm(xyz - section_points(c, models, eta, np.array([x]), side)[0])
            )
    assert max(gaps) < 1e-7


def test_root_fillet_on_a_curved_host_meets_it_tangentially():
    c = component(
        {
            "planform": {"taper": 0.7},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
            **FILLET,
        }
    )

    def distance(p):
        return np.hypot(p[1], p[2]) - 0.1

    def normal(p):
        return np.array([0.0, p[1], p[2]]) / np.hypot(p[1], p[2])

    patches = root_fillet(c, distance, normal)
    worst = 0.0
    for i in (0, 1, 4, 5):
        surface = evaluate_patch(patches[i], 201)
        n = np.cross(np.gradient(surface, axis=0), np.gradient(surface, axis=1))
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        for k in (50, 100, 150):
            xyz = surface[k, -1]
            assert abs(distance(xyz)) < 1e-7
            # At the host end the fillet surface normal is the host normal.
            cosine = abs(n[k, -1] @ normal(xyz))
            worst = max(worst, float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    assert worst < 0.5


def test_enabled_fillet_with_zero_radius_is_refused():
    c = component(
        {
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
            "root_fillet": {"enabled": True, "radii": [0.0, 0.0, 0.0]},
        }
    )
    with pytest.raises(ValueError, match="positive radii"):
        root_fillet(c)

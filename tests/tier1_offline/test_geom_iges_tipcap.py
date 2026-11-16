"""tip cap: one planar face bounded exactly by the tip section outline.

every tip cap, blade or lifting surface, is a single planar face whose boundary is
the suction side, the pressure side and, for a blunt trailing edge, the base segment, and it
shares those curves with the side surfaces. Sections are synthetic NACA profiles.
"""

import tomllib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline

import cfdbender.geometry as geometry
from cfdbender.geometry import resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._bspline import full_knots
from cfdbender.geometry._caps import cap_edge_gap, lifting_model, tip_cap
from cfdbender.geometry._patch import Patch

FIXTURES = Path(__file__).parent / "fixtures" / "propeller"
TOL = 1e-9


def point(patch: Patch, u, v):
    tu = full_knots(patch.knots_u, patch.multiplicities_u)
    tv = full_knots(patch.knots_v, patch.multiplicities_v)
    controls = np.asarray(patch.controls)
    if np.ndim(v) > 0:
        along = BSpline(tu, controls, patch.degree_u)(float(u))
        return BSpline(tv, along, patch.degree_v)(np.asarray(v, float))
    along = BSpline(tv, controls.transpose(1, 0, 2), patch.degree_v)(v)
    return BSpline(tu, along, patch.degree_u)(np.asarray(u, float))


def plane_distance(points):
    """Largest distance of the points to their best-fit plane."""
    points = np.asarray(points, float).reshape(-1, 3)
    centred = points - points.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return float(np.abs(centred @ normal).max())


def wing_model(te="blunt"):
    thickness = 0.00252 if te == "blunt" else 0.0
    raw = {
        "components": {
            "wing": {
                "type": "lifting_surface",
                "cap": {"type": "flat"},
                "sections": {"trailing_edge_thickness": thickness},
            }
        }
    }
    component = unpack(resolve_config(raw))["components"]["wing"]
    return component, lifting_model(component)


@pytest.mark.parametrize("te", ["blunt", "sharp"])
def test_geom_iges_tipcap_tipcap_wing_cap_is_one_face_on_the_tip_outline(te):
    """P0570-TIPCAP-WING: one face, rims equal the skin tip curves, base segment closes it."""
    component, model = wing_model(te)
    chord = float(component["planform"]["tip_chord"])
    assert list(model.caps) == ["tip_cap"]
    cap = model.caps["tip_cap"]
    t = np.linspace(0, 1, 201)
    upper_skin = point(model.skin, 0.5 + 0.5 * t, 1.0)
    lower_skin = point(model.skin, 0.5 - 0.5 * t, 1.0)
    assert np.abs(point(cap, t, 1.0) - upper_skin).max() <= TOL * chord
    assert np.abs(point(cap, t, 0.0) - lower_skin).max() <= TOL * chord
    base = point(cap, 1.0, np.linspace(0, 1, 51))
    straight = lower_skin[-1] + np.linspace(0, 1, 51)[:, None] * (upper_skin[-1] - lower_skin[-1])
    assert np.abs(base - straight).max() <= TOL * chord
    nose = point(cap, 0.0, np.linspace(0, 1, 51))
    assert np.abs(nose - upper_skin[0]).max() <= TOL * chord


@pytest.mark.parametrize("te", ["blunt", "sharp"])
def test_geom_iges_tipcap_tipcap_wing_cap_is_planar(te):
    """P0570-TIPCAP-PLANAR: every sample of the cap lies on its best-fit plane."""
    component, model = wing_model(te)
    chord = float(component["planform"]["tip_chord"])
    grid = model.caps["tip_cap"].evaluate(61)
    assert plane_distance(grid) <= TOL * chord
    # the cap lies in the plane of the skin's tip section curve as well
    assert plane_distance(np.vstack([grid.reshape(-1, 3), model.skin.evaluate(5)[:, -1]])) <= (
        TOL * chord
    )


@pytest.mark.parametrize("te", ["blunt", "sharp"])
def test_geom_iges_tipcap_tipcap_wing_cap_is_watertight_with_the_side_surfaces(te):
    """P0570-TIPCAP-WATERTIGHT: the cap rims and the trailing-edge face end meet with no gap."""
    _, model = wing_model(te)
    assert cap_edge_gap(model) <= 1e-12
    record = next(r for r in model.records if r["id"] == "closure.rim_shared")
    assert record["status"] == "pass" and record["value"] <= 1e-12
    cap = model.caps["tip_cap"]
    if te == "blunt":
        face_end = np.asarray(model.trailing_edge.controls)[:, -1]
        edge = np.asarray(cap.controls)[-1]
        assert np.abs(np.sort(face_end, axis=0) - np.sort(edge, axis=0)).max() <= 1e-12


@pytest.mark.parametrize("te", ["blunt", "sharp"])
def test_geom_iges_tipcap_tipcap_wing_legacy_patches_are_one_planar_face(te):
    """P0570-TIPCAP-LEGACY: the patch list the mesher reads holds one planar cap patch."""
    component, model = wing_model(te)
    chord = float(component["planform"]["tip_chord"])
    patches = tip_cap(component)
    assert len(patches) == 1 and patches[0].family == "tip_cap"
    t = np.linspace(0, 1, 101)
    assert np.abs(point(patches[0], t, 1.0) - point(model.skin, 0.5 + 0.5 * t, 1.0)).max() <= (
        1e-7 * chord
    )
    assert plane_distance(patches[0].evaluate(61)) <= TOL * chord


def blade(te):
    supplied = tomllib.loads((FIXTURES / "simple_naca.toml").read_text(encoding="utf-8"))
    supplied = deepcopy(supplied)
    supplied["options"]["trailing_edge"] = te
    return geometry.Propeller(supplied).blade()


@pytest.mark.parametrize("te", ["blunt", "sharp"])
def test_geom_iges_tipcap_tipcap_blade_cap_is_one_planar_face_on_the_tip_outline(te):
    """P0570-TIPCAP-BLADE: the blade tip is one planar face bounded by its tip section."""
    model = blade(te)
    upper, lower = model.patches[0], model.patches[1]
    caps = [p for p in model.patches if p.family == "tip"]
    assert len(caps) == 1
    cap = caps[0]
    chord = float(np.ptp(np.asarray(upper.controls)[:, -1], axis=0).max())
    assert plane_distance(cap.evaluate(61)) <= TOL * chord
    t = np.linspace(0, 1, 201)
    assert np.abs(point(cap, t, 1.0) - point(upper, t, 1.0)).max() <= TOL * chord
    assert np.abs(point(cap, t, 0.0) - point(lower, t, 1.0)).max() <= TOL * chord
    rung = point(cap, 1.0, np.linspace(0, 1, 21))
    ends = (point(lower, 1.0, 1.0), point(upper, 1.0, 1.0))
    straight = ends[0] + np.linspace(0, 1, 21)[:, None] * (ends[1] - ends[0])
    assert np.abs(rung - straight).max() <= TOL * chord
    if te == "sharp":
        assert np.linalg.norm(ends[1] - ends[0]) <= TOL * chord
    else:
        assert np.linalg.norm(ends[1] - ends[0]) > 1e-6 * chord
        closing = [p for p in model.patches if p.family == "trailing_edge"]
        assert len(closing) == 1
        assert np.abs(np.asarray(closing[0].controls)[:, -1] - [ends[0], ends[1]]).max() <= (
            TOL * chord
        )


WB = {
    "components": {
        "FUS": {"type": "body"},
        "WING": {
            "type": "lifting_surface",
            "attach": {"parent": "FUS", "station": 0.4},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        },
        "FAIRING": {
            "type": "wing_fuselage_fairing",
            "hosts": ["FUS", "WING"],
            "attach": {"parent": "FUS", "station": 0.0},
        },
    },
    "configurations": {"WB": ["FUS", "WING", "FAIRING"]},
}
ALONE = {"components": {"WING": {"type": "lifting_surface"}}}


def test_geom_iges_tipcap_tipcap_assembly_sources_hold_one_cap_patch():
    """P0570-TIPCAP-SOURCES: the legacy patch lists carry the one-face cap, never nine patches."""
    from cfdbender.geometry.stages import generate_components

    for raw in (ALONE, WB):
        sources = generate_components(resolve_config(raw))
        caps = [p for p in sources["WING"] if p.family == "tip_cap"]
        assert len(caps) == 1 and caps[0].degree_v == 1


@pytest.mark.parametrize("raw", [ALONE, WB], ids=["wing-alone", "wing-body"])
def test_geom_iges_tipcap_tipcap_generated_iges_sews_audits_and_passes_the_reader_profile(
    tmp_path, raw
):
    """P0570-TIPCAP-IGES: one cap face is written, sewn, audited clean and read back."""
    from kernel_support import kernel_or_skip

    kernel_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges

    result = generate_iges(raw, tmp_path / "out.igs", configuration="WB" if raw is WB else None)
    audit = result["audit"]
    assert audit["free_edges_off_symmetry"] == []
    assert audit["nonmanifold_edges"] == []
    assert audit["non_ring_nonquadrilateral_faces"] == []
    # Only the fairing's own surfaces (degree 6, an open (u, v) loop by 1e-4) draw reader-profile
    # notes, identically with the nine-patch cap; the wing alone draws none.
    fairing = ("profile.degree", "profile.loop_open") if raw is WB else ()
    notes = [
        w for w in result["warnings"] if "reader profile" in w and not any(k in w for k in fairing)
    ]
    assert not notes, notes
    from cfdbender.geometry._geom_gate import check_profile

    if raw is ALONE:  # the fairing's own open loops fail the gate on a wing body, cap or not
        assert check_profile(tmp_path / "out.igs").passed


def test_geom_iges_tipcap_tipcap_the_mesher_closes_the_one_face_cap_with_a_planar_mesh():
    """P0570-TIPCAP-MESH: the cap meshes as one planar face; all quad, accepted."""
    from kernel_support import kernel_or_skip

    kernel_or_skip()
    from cfdbender.geometry._geometry_mesh import mesh_generated

    for raw, component in ((WB, "WING"), (ALONE, "WING")):
        record = mesh_generated(raw, component=component).record
        assert record["accepted"] is True
        assert record["non_quad_count"] == 0
        assert record["cap_cells"] > 0
    whole = mesh_generated(WB).record
    assert whole["non_manifold_count"] == 0
    assert whole["free_edges_off_y0_count"] == 0

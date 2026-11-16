"""Geometry: bodies, fairing, placement and regions, checked on the patches alone.

Checked on ``evaluate_patch`` so none of them needs a geometry kernel; the kernel-bound
checks of the same parts come with the kernel stages.
"""

import numpy as np
import pytest
from geom_inputs import body_input

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._bodies import (
    axisymmetric_radius,
    body,
    body_points,
    body_station_table,
)
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._fairings import host_y, wing_fuselage_fairing
from cfdbender.geometry._lifting import lifting_surface
from cfdbender.geometry._patch import Patch, evaluate_patch
from cfdbender.geometry._placement import (
    attachment_point,
    body_junctions,
    place,
    placement_transform,
    rotation_matrix,
    wing_in_body_frame,
)
from cfdbender.geometry._regions import body_region, inside_triangles, lifting_region

SUPERELLIPSE = {
    "at": "middle",
    "shape": "superellipse",
    "width": 0.2,
    "height": 0.1,
    "n_top": 4.0,
    "n_bottom": 2.0,
}


def make_body(**extra):
    raw = {
        "type": "body",
        "length": 20.0,
        "sections": [{"at": "middle", "shape": "circle", "radius": 1.2}],
        **extra,
    }
    return unpack(resolve_config(body_input(raw)))["components"]["COMPONENT"]


def wing_body():
    return unpack(
        resolve_config(
            {
                "components": {
                    "FUS": {"type": "body"},
                    "WING": {
                        "type": "lifting_surface",
                        "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
                        "distributions": {
                            "leading_edge_x": {"stations": [0.0, 1.0], "values": [0.4, 0.4]}
                        },
                    },
                    "FAIRING": {"type": "wing_fuselage_fairing", "hosts": ["FUS", "WING"]},
                }
            }
        )
    )["components"]


def wb_components():
    return unpack(
        resolve_config(
            {
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
                }
            }
        )
    )["components"]


def test_geom_body_middle_radius_and_patch_families():
    patches = body(make_body())
    assert all(isinstance(p, Patch) for p in patches)
    assert sorted({p.family for p in patches}) == ["front", "middle", "nose", "rear"]
    errors = []
    for patch in patches:
        if patch.family == "middle":
            p = evaluate_patch(patch, 9)
            errors.append(np.max(abs(np.hypot(p[..., 1], p[..., 2]) - 1.2)))
    assert errors
    assert max(errors) < 1e-6


def test_geom_body_minimum_counts():
    with pytest.raises(InputValueError, match="azimuths"):
        resolve_config({"type": "body", "azimuths": [0, 90, 180]})
    # fewer than three interior part boundaries is warned and skipped, not refused
    cfg = resolve_config({"type": "body", "parts": [0.0, 0.2, 0.8, 1.0]})
    assert any(w.startswith("COMPONENT.parts:") for w in cfg["warnings"])


def test_geom_body_axisymmetric_reduction():
    c = make_body()
    axial = np.linspace(0, 1, 31)
    radius = axisymmetric_radius(c, axial)
    table = body_station_table(c)
    assert table["axisymmetric"]
    for theta in [0.1, 0.8, 1.7, 2.8]:
        xyz = body_points(c, table, axial, theta)
        np.testing.assert_allclose(np.hypot(xyz[:, 1], xyz[:, 2]), radius, atol=1e-14)
    np.testing.assert_allclose(radius[[0, -1]], 0, atol=1e-14)


def test_geom_body_axisymmetric_radius_refuses_a_superellipse():
    c = make_body(sections=[SUPERELLIPSE])
    with pytest.raises(InputValueError, match="not axisymmetric"):
        axisymmetric_radius(c, 0.5)


def test_geom_body_superellipse_and_base():
    c = make_body(rear="base", base_radius=0.05, sections=[{**SUPERELLIPSE, "centre_height": 0.03}])
    table = body_station_table(c)
    assert not table["axisymmetric"]
    p = body_points(c, table, 0.5, np.pi / 4)
    assert (p[1] / 0.2) ** 4 + ((p[2] - 0.03) / 0.1) ** 4 == pytest.approx(1.0, abs=1e-14)
    assert body_points(c, table, 1.0, np.pi / 2)[1] == pytest.approx(0.05)


def test_geom_body_base_rear_has_base_patches_and_pole_rear_has_none():
    assert "base" in {p.family for p in body(make_body(rear="base", base_radius=0.3))}
    assert "base" not in {p.family for p in body(make_body(rear="pole"))}


def test_geom_body_junction_stations_split_the_bands():
    plain = [p for p in body(make_body()) if p.family == "middle"]
    split = [p for p in body(make_body(), junctions=(0.3, 0.6)) if p.family == "middle"]
    assert len(split) > len(plain)
    edges = {x for p in split for x in p.bay}
    assert {0.3, 0.6} <= edges
    assert not {0.3, 0.6} & {x for p in plain for x in p.bay}


def test_geom_body_profile_refuses_a_negative_radius():
    c = make_body()
    c["sections"][0]["radius"] = -1.0
    with pytest.raises(InputValueError, match="invalid"):
        body_points(c, body_station_table(c), 0.5, 0.3)


@pytest.fixture(scope="module")
def fairing_case():
    data = wing_body()
    patches = wing_fuselage_fairing(data["FAIRING"], data["FUS"], data["WING"])
    return data, patches


def test_geom_fairing_has_twelve_patches_body_then_wing_per_segment(fairing_case):
    _, patches = fairing_case
    assert [p.family for p in patches] == ["fairing_body", "fairing_wing"] * 6


def test_geom_fairing_touches_the_body_and_stays_outside_it(fairing_case):
    _, patches = fairing_case
    radius = 0.1
    gaps, clearance = [], []
    for patch in patches:
        grid = evaluate_patch(patch, 17)
        r = np.hypot(grid[..., 1], grid[..., 2])
        clearance.append(np.min(r) - radius)
        if patch.family == "fairing_body":
            gaps.append(np.max(abs(r[:, 0] - radius)))
    assert max(gaps) < 1e-7
    assert min(clearance) > -1e-7


def test_geom_fairing_wing_split_lines_continue(fairing_case):
    _, patches = fairing_case
    wing = [p for p in patches if p.family == "fairing_wing"]
    gaps = []
    for i in range(6):
        a = evaluate_patch(wing[i], 31)[-1]
        b = evaluate_patch(wing[(i + 1) % 6], 31)[0]
        gaps.append(np.max(np.linalg.norm(a - b, axis=1)))
    assert max(gaps) < 1e-7


def test_geom_host_y_refuses_a_point_outside_the_body(fairing_case):
    data, _ = fairing_case
    table = body_station_table(data["FUS"])
    with pytest.raises(InputValueError, match="outside body axial range"):
        host_y(data["FUS"], table, -1.0, 0.0)
    with pytest.raises(InputValueError, match="beyond body height"):
        host_y(data["FUS"], table, 0.5 * data["FUS"]["length"], 100.0)


def test_geom_rotation_matrix_is_a_rotation_applied_z_after_y_after_x():
    r = rotation_matrix([10.0, 20.0, 30.0])
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(r) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation_matrix([0, 0, 90.0]) @ [1.0, 0, 0], [0, 1.0, 0], atol=1e-15)


def test_geom_attachment_point_refuses_a_fairing_as_a_parent():
    with pytest.raises(InputValueError, match="unsupported attachment"):
        attachment_point({"type": "wing_fuselage_fairing"}, 0.5)


def test_geom_parent_motion_moves_child():
    components = wb_components()
    patches = lifting_surface(components["WING"])
    original = place(patches, components, "WING")
    components["FUS"]["placement"]["translation"] = [2.0, 3.0, 4.0]
    moved = place(patches, components, "WING")
    for a, b in zip(original, moved, strict=True):
        np.testing.assert_allclose(
            b.controls - a.controls, np.broadcast_to([2.0, 3.0, 4.0], a.controls.shape), atol=1e-14
        )
    np.testing.assert_allclose(placement_transform(components, "WING")[0], np.eye(3))


def test_geom_wing_in_body_frame_moves_the_leading_edge_and_refuses_bad_roots():
    components = wb_components()
    wing = wing_in_body_frame(components, "FUS", "WING")
    expected = 0.4 * components["FUS"]["length"] / wing["planform"]["span"]
    assert wing["distributions"]["leading_edge_x"]["values"][0] == pytest.approx(expected)
    components["WING"]["placement"]["translation"] = [0.0, 0.5, 0.0]
    with pytest.raises(InputValueError, match="symmetry plane"):
        wing_in_body_frame(components, "FUS", "WING")
    components["WING"]["placement"]["translation"] = [0.0, 0.0, 0.0]
    components["WING"]["placement"]["rotation"] = [0.0, 5.0, 0.0]
    with pytest.raises(InputValueError, match="relative rigid rotation"):
        wing_in_body_frame(components, "FUS", "WING")


def test_geom_body_junctions_are_the_wing_root_and_fairing_ends():
    components = wb_components()
    assert body_junctions(components, "WING") == []
    for exponent in (1.0, 2.0):
        components["FAIRING"].update(start_exponent=exponent, end_exponent=exponent)
        junctions = body_junctions(components, "FUS")
        assert len(junctions) == 4 and all(0.0 < j < 1.0 for j in junctions)


def test_geom_body_junctions_leave_out_a_tangent_fairing_end():
    """Below exponent 1 the footprint end is tangent to the ring there: no station (no cusp)."""
    components = wb_components()
    assert components["FAIRING"]["start_exponent"] == 0.5  # the default rounded nose
    root = [0.4, 0.525]  # wing leading and trailing edge at the root, fractions of the body
    assert body_junctions(components, "FUS") == pytest.approx(root)
    components["FAIRING"]["end_exponent"] = 1.0
    assert body_junctions(components, "FUS") == pytest.approx([*root, 0.55625])


def test_geom_regions_classify_points():
    inside = body_region(make_body())
    assert inside((10.0, 0.0, 0.0))
    assert not inside((10.0, 0.0, 1.3))
    assert not inside((-1.0, 0.0, 0.0))
    components = wb_components()
    wing = lifting_region(components["WING"])
    root_chord = components["WING"]["planform"]["root_chord"]
    assert wing((root_chord / 2, 0.0, 0.0))
    assert not wing((root_chord / 2, 0.0, 5.0))
    assert not wing((root_chord / 2, 1e6, 0.0))


def test_geom_inside_triangles_is_the_winding_test_of_a_closed_surface():
    corners = np.array(
        [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    )
    faces = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]  # fmt: skip
    triangles = corners[np.array(faces)]
    assert inside_triangles([0.1, 0.1, 0.1], triangles)
    assert not inside_triangles([2.0, 0.0, 0.0], triangles)

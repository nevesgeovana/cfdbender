"""Blade sections on the analytic QPROP test blade, on planes and on cylinders.

The blade is the analytic QPROP test blade (see qprop_support): chord and pitch follow known
laws, so every recovered station is compared with the law evaluated at its own radius. The
plane cut of the propeller slicer gives the same sections as the plain plane cut it replaced.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qprop_support import R_ROOT, R_TIP, beta_law, blade_obj, chord_law

from cfdbender.geometry._cylinder_cut import cylinder_sections
from cfdbender.geometry._propeller_slice import ROUND_TIE, edges_and_sides
from cfdbender.geometry.blade_sections import (
    BladeSliceError,
    parse_obj,
    radial_extent,
    slice_blade,
    station_radii,
)

#: Plane cuts of this blade agree with the laws to 0.001 percent chord and 0.001 degree; the
#: skewed plane of the propeller slicer moves them by at most the values below.
CHORD_REL, BETA_DEG = 1e-4, 0.01
#: Measured on this blade: the unrolled cylinder bends the section, 0.44 percent chord and
#: 0.06 degree beta.
TOLERANCES = {"plane": (CHORD_REL, BETA_DEG), "cylinder": (0.006, 0.1)}


def _side(shape, upper):
    points = np.asarray(shape)
    middle = int(np.argmin(points[:, 0]))
    side = points[: middle + 1][::-1] if upper else points[middle:]
    return side[np.argsort(side[:, 0], kind="stable")]


def check_laws(sections, count, cut="plane"):
    chord_rel, beta_deg = TOLERANCES[cut]
    assert len(sections) == count
    radii = [s.radius for s in sections]
    assert radii == sorted(radii)
    assert R_ROOT <= radii[0] and radii[-1] <= R_TIP
    for s in sections:
        assert s.chord == pytest.approx(chord_law(s.radius), rel=chord_rel)
        assert abs(s.beta_deg - beta_law(s.radius)) < beta_deg


def _same_sections(new, old, *, shape_tol):
    assert len(new) == len(old)
    grid = np.linspace(0.05, 0.95, 19)
    for a, b in zip(new, old, strict=True):
        assert a.radius == pytest.approx(b.radius, rel=1e-4)
        assert a.chord == pytest.approx(b.chord, rel=CHORD_REL * 10)
        assert abs(a.beta_deg - b.beta_deg) < BETA_DEG
        for upper in (True, False):
            mine, theirs = _side(a.shape, upper), _side(b.shape, upper)
            ours = np.interp(grid, mine[:, 0], mine[:, 1])
            ref = np.interp(grid, theirs[:, 0], theirs[:, 1])
            assert np.max(np.abs(ours - ref)) < shape_tol


@pytest.mark.parametrize("spacing", ["cosine", "uniform"])
def test_geom_blade_sections_plane_cut_matches_the_plain_plane_cut(spacing):
    old = cylinder_sections(blade_obj(), n_sections=9, spacing=spacing, cut="plane")
    new = slice_blade(blade_obj(), n_sections=9, spacing=spacing, cut="plane")
    _same_sections(new, old, shape_tol=2e-3)


@pytest.mark.parametrize("cut", ["plane", "cylinder"])
def test_geom_blade_sections_clockwise_offset_rotor(cut):
    obj = blade_obj(mirror=True, shift=(0.3, -0.2, 0.1))
    arguments = {"n_sections": 6, "rotation": "cw_from_front", "hub_center": (0.3, -0.2, 0.1)}
    check_laws(slice_blade(obj, cut=cut, **arguments), 6, cut)


def _lens(sharp_nose):
    """A unit-chord section, leading edge at x = 1, moving toward +x; round nose if not sharp."""
    u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 81)))
    if sharp_nose:
        half = 0.06 * 4 * u * (1 - u)
    else:
        half = 0.06 * np.sqrt(np.clip(1 - u, 0, None)) * u * 2
    x = 1 - u
    upper = np.column_stack((x, half))
    lower = np.column_stack((x[::-1], -half[::-1]))[1:-1]
    return np.vstack((upper, lower))


@pytest.mark.parametrize("ahead", [(1.0, 0.0), (-1.0, 0.0)])
def test_geom_blade_sections_a_tie_goes_to_the_end_ahead(ahead):
    """Two equally sharp ends: the end ahead in the direction of motion is the leading edge."""
    section = _lens(sharp_nose=True)
    le, te, _ = edges_and_sides(section, ahead=ahead)
    assert np.sign(le[0] - te[0]) == np.sign(ahead[0])


def test_geom_blade_sections_a_round_nose_leads_whatever_the_motion():
    """Ends whose widths differ by more than the tie keep the rounder-end rule."""
    section = _lens(sharp_nose=False)
    for ahead in ((1.0, 0.0), (-1.0, 0.0), None):
        le, te, _ = edges_and_sides(section, ahead=ahead)
        assert le[0] < te[0], ahead
    assert ROUND_TIE == 0.05


@pytest.mark.parametrize("spacing", ["cosine", "uniform"])
@pytest.mark.parametrize("count", [1, 3, 9])
def test_geom_blade_sections_recover_chord_and_beta(spacing, count):
    check_laws(slice_blade(blade_obj(), n_sections=count, spacing=spacing, cut="plane"), count)


#: Trailing-edge ordinate of a written shape: closed on the cylinder cut, within a few
#: 1e-5 chord on the plane cut (its half gap and the turn of a section about mid-chord).
TE_Y = {"cylinder": 1e-6, "plane": 5e-5}


@pytest.mark.parametrize("cut", ["cylinder", "plane"])
def test_geom_blade_sections_shape_is_unit_chord_with_upper_side_first(cut):
    grid = np.linspace(0.05, 0.95, 19)
    for s in slice_blade(blade_obj(), n_sections=5, cut=cut):
        xs = [x for x, _ in s.shape]
        assert s.shape[0] == pytest.approx((1.0, 0.0), abs=TE_Y[cut])
        assert s.shape[-1] == pytest.approx((1.0, 0.0), abs=TE_Y[cut])
        assert min(xs) == pytest.approx(0.0, abs=1e-6)
        assert max(xs) == pytest.approx(1.0, abs=1e-6)
        upper, lower = _side(s.shape, True), _side(s.shape, False)
        assert upper[:, 1].sum() > 0 > lower[:, 1].sum()
        assert upper[:, 1].max() - lower[:, 1].min() == pytest.approx(0.12, abs=0.02)
        # RG-3: the camber of the analytic blade (2 percent chord) pins the shape's ordinates.
        mean = (np.interp(grid, *upper.T) + np.interp(grid, *lower.T)) / 2
        assert mean.max() == pytest.approx(0.02, abs=0.006)


@pytest.mark.parametrize("spacing", ["cosine", "uniform"])
def test_geom_blade_sections_cylinder_cut_recovers_chord_and_beta(spacing):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sections = slice_blade(blade_obj(), n_sections=9, spacing=spacing, cut="cylinder")
    check_laws(sections, 9, "cylinder")


def test_geom_blade_sections_pick_the_largest_component():
    first = blade_obj()
    count = sum(1 for line in first.splitlines() if line.startswith("v "))
    small = [f"v {a} {b} {c}" for a, b, c in ((1, 1, 1), (1.1, 1, 1), (1, 1.1, 1))]
    small.append(f"f {count + 1} {count + 2} {count + 3}")
    check_laws(slice_blade(first + "\n".join(small) + "\n", n_sections=4, cut="plane"), 4)


def test_geom_blade_sections_radial_extent_spans_root_to_tip():
    low, high = radial_extent(blade_obj())
    assert low == pytest.approx(R_ROOT, abs=2e-3)
    assert high == pytest.approx(R_TIP, abs=2e-3)
    moved = radial_extent(blade_obj(shift=(0.3, -0.2, 0.1)), hub_center=(0.3, -0.2, 0.1))
    assert moved == pytest.approx((low, high), abs=1e-9)
    with pytest.raises(BladeSliceError, match="axis"):
        radial_extent(blade_obj(), axis="w")


def test_geom_blade_sections_station_spacing_laws():
    assert station_radii(0.0, 1.0, 4, "uniform") == pytest.approx([0.125, 0.375, 0.625, 0.875])
    cosine = station_radii(0.0, 1.0, 4, "cosine")
    assert cosine == sorted(cosine) and 0.0 < cosine[0] < 0.125 < 0.875 < cosine[-1] < 1.0
    assert cosine[1] - cosine[0] < cosine[2] - cosine[1]


def test_geom_blade_sections_parse_obj_polygons_and_references():
    vertices, faces = parse_obj(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1/1 2/2/2 3 4\nf -4 -3 -2\n"
    )
    assert len(vertices) == 4
    assert faces == [(0, 1, 2), (0, 2, 3), (0, 1, 2)]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"spacing": "tan"}, "spacing"),
        ({"cut": "sphere"}, "cut"),
        ({"axis": "w"}, "axis"),
        ({"rotation": "left"}, "rotation"),
        ({"n_sections": 0}, "n_sections"),
    ],
)
def test_geom_blade_sections_refuse_unknown_choices(kwargs, message):
    with pytest.raises(BladeSliceError, match=message):
        slice_blade(blade_obj(rings=5), **{"n_sections": 3, **kwargs})


def test_geom_blade_sections_refuse_empty_and_unsliceable_input():
    with pytest.raises(BladeSliceError, match="no faces"):
        slice_blade("v 0 0 0\n", n_sections=3)
    with pytest.raises(BladeSliceError, match="radial"):
        slice_blade("v 1 0 0\nv 2 0 0\nv 3 0 0\nf 1 2 3\n", n_sections=3)


def test_geom_blade_sections_default_cut_is_the_cylinder_of_earlier_releases():
    """A call or a QPROP setup that states no cut gets the cylinder sections of 0.5.x."""
    from cfdbender.solvers.qprop._config import setup

    default = slice_blade(blade_obj(), n_sections=9)
    earlier = cylinder_sections(blade_obj(), n_sections=9, cut="cylinder")
    assert default == earlier
    assert setup({"blade": "inputs/mesh/blade.obj", "n_blades": 3}).cut == "cylinder"
    assert setup({"blade": "inputs/mesh/blade.obj", "n_blades": 3, "cut": "plane"}).cut == "plane"


def _soup(text):
    """Return the OBJ with every face on its own vertices, as an STL conversion writes it."""
    vertices, faces = parse_obj(text)
    lines = [f"v {x!r} {y!r} {z!r}" for face in faces for x, y, z in (vertices[i] for i in face)]
    lines += [f"f {3 * k + 1} {3 * k + 2} {3 * k + 3}" for k in range(len(faces))]
    return "\n".join(lines) + "\n"


def test_geom_blade_sections_plane_cut_welds_a_blade_written_face_by_face():
    """RG-1: the plane cut sees the welded blade, as the propeller slicer does."""
    from cfdbender.solvers.qprop._config import setup
    from cfdbender.solvers.qprop._sections import resolve

    shared = blade_obj()
    soup = _soup(shared)
    assert slice_blade(soup, n_sections=5, cut="plane") == slice_blade(
        shared, n_sections=5, cut="plane"
    )
    assert radial_extent(soup, cut="plane") == pytest.approx(radial_extent(shared), abs=1e-12)
    chosen = setup({"blade": "inputs/mesh/blade.obj", "n_blades": 3, "cut": "plane"})
    radii = [resolve(chosen, 5, lambda _, t=t: t.encode()).radius for t in (soup, shared)]
    assert radii[0] == pytest.approx(radii[1], abs=1e-12)
    with pytest.raises(BladeSliceError, match="cut"):
        radial_extent(shared, cut="sphere")

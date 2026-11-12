"""P0000: slice a synthetic rotor blade and recover its chord, pitch and shape.

The blade is analytic (see qprop_support): chord and pitch follow known laws, so
every recovered station is compared with the law evaluated at its own radius.
"""

from __future__ import annotations

import pytest
from qprop_support import R_ROOT, R_TIP, beta_law, blade_obj, chord_law

from cfdbender.geometry.blade_slicing import (
    BladeSliceError,
    parse_obj,
    radial_extent,
    slice_blade,
    station_radii,
)

# Measured on the synthetic blade: cylinder cut 0.44 percent chord and 0.06 degree
# beta (the unrolled cylinder bends the section); plane cut 0.001 percent and 0.001 degree.
TOLERANCES = {"cylinder": (0.006, 0.1), "plane": (1e-4, 0.01)}


def check_laws(sections, count, cut="cylinder"):
    chord_tolerance, beta_tolerance = TOLERANCES[cut]
    assert len(sections) == count
    radii = [s.radius for s in sections]
    assert radii == sorted(radii)
    assert R_ROOT <= radii[0] and radii[-1] <= R_TIP
    for s in sections:
        assert s.chord == pytest.approx(chord_law(s.radius), rel=chord_tolerance)
        assert abs(s.beta_deg - beta_law(s.radius)) < beta_tolerance


@pytest.mark.parametrize("cut", ["cylinder", "plane"])
@pytest.mark.parametrize("spacing", ["cosine", "uniform"])
def test_p0000_qprop_slicing_recovers_chord_and_beta(cut, spacing):
    sections = slice_blade(blade_obj(), n_sections=9, spacing=spacing, cut=cut)
    check_laws(sections, 9, cut)


def test_p0000_qprop_slicing_shape_is_unit_chord_with_upper_side_first():
    sections = slice_blade(blade_obj(), n_sections=5)
    for s in sections:
        xs = [x for x, _ in s.shape]
        ys = [y for _, y in s.shape]
        assert s.shape[0] == pytest.approx((1.0, 0.0), abs=1e-6)
        assert s.shape[-1] == pytest.approx((1.0, 0.0), abs=1e-6)
        assert min(xs) == pytest.approx(0.0, abs=1e-6)
        assert max(xs) == pytest.approx(1.0, abs=1e-6)
        middle = xs.index(min(xs))
        upper, lower = s.shape[: middle + 1], s.shape[middle:]
        assert sum(y for _, y in upper) > 0 > sum(y for _, y in lower)
        assert max(ys) - min(ys) == pytest.approx(2 * 0.06 / 1.0 * 1.0, abs=0.02)
        camber = max((u[1] + w[1]) / 2 for u, w in zip(upper[::-1], lower, strict=False))
        assert camber == pytest.approx(0.02, abs=0.006)


def test_p0000_qprop_slicing_clockwise_rotor_and_offset_hub():
    obj = blade_obj(mirror=True, shift=(0.3, -0.2, 0.1))
    sections = slice_blade(obj, n_sections=6, rotation="cw_from_front", hub_center=(0.3, -0.2, 0.1))
    check_laws(sections, 6)


def test_p0000_qprop_slicing_picks_the_largest_component():
    first = blade_obj()
    lines = first.splitlines()
    count = sum(1 for line in lines if line.startswith("v "))
    small = [f"v {a} {b} {c}" for a, b, c in ((1, 1, 1), (1.1, 1, 1), (1, 1.1, 1))]
    small.append(f"f {count + 1} {count + 2} {count + 3}")
    sections = slice_blade(first + "\n".join(small) + "\n", n_sections=4)
    check_laws(sections, 4)


def test_p0000_qprop_slicing_radial_extent_spans_root_to_tip():
    low, high = radial_extent(blade_obj())
    assert low == pytest.approx(R_ROOT, abs=2e-3)
    assert high == pytest.approx(R_TIP, abs=2e-3)
    moved = radial_extent(blade_obj(shift=(0.3, -0.2, 0.1)), hub_center=(0.3, -0.2, 0.1))
    assert moved == pytest.approx((low, high), abs=1e-9)
    with pytest.raises(BladeSliceError, match="axis"):
        radial_extent(blade_obj(), axis="w")


def test_p0000_qprop_slicing_station_spacing_laws():
    uniform = station_radii(0.0, 1.0, 4, "uniform")
    assert uniform == pytest.approx([0.125, 0.375, 0.625, 0.875])
    cosine = station_radii(0.0, 1.0, 4, "cosine")
    assert cosine == sorted(cosine) and 0.0 < cosine[0] < 0.125 < 0.875 < cosine[-1] < 1.0
    assert cosine[1] - cosine[0] < cosine[2] - cosine[1]


def test_p0000_qprop_slicing_parse_obj_polygons_and_references():
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
def test_p0000_qprop_slicing_refuses_unknown_choices(kwargs, message):
    arguments = {"n_sections": 3, **kwargs}
    with pytest.raises(BladeSliceError, match=message):
        slice_blade(blade_obj(rings=5), **arguments)


def test_p0000_qprop_slicing_refuses_empty_and_unsliceable_input():
    with pytest.raises(BladeSliceError, match="no faces"):
        slice_blade("v 0 0 0\n", n_sections=3)
    flat = "v 1 0 0\nv 2 0 0\nv 3 0 0\nf 1 2 3\n"
    with pytest.raises(BladeSliceError, match="radial"):
        slice_blade(flat, n_sections=3)

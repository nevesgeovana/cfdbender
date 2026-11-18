"""The plane cut: stations, ordering, Cp rows, chord, families, frames and misses."""

from __future__ import annotations

import numpy as np
import pytest
from post_sections_support import cylinder, plate, wing

from cfdbender.post.sections import (
    Distribution,
    Frame,
    Reference,
    SectionError,
    Surface,
    compute_sections,
)

REF = Reference(p_inf=100.0, q_ref=50.0)


def _one(result):
    assert len(result.blocks) == 1
    return result.blocks[0].stations


def test_plate_stations_sit_at_strip_centres_and_the_cut_is_open_from_the_leading_edge():
    surf = plate(2.0, 3.0, 4, 6, p0=100.0, slope=10.0)
    res = compute_sections(surf, REF, Distribution(planes=("XZ",), count=3))
    st = _one(res)
    assert [s.offset for s in st] == pytest.approx([0.5, 1.5, 2.5])
    cp = st[0].cp
    # one open polyline from x = 0 to x = c; its points are the cell edges crossed
    assert cp["Section_direction_value"][0] == pytest.approx(0.0)
    assert cp["Section_direction_value"][-1] == pytest.approx(2.0)
    assert np.all(np.diff(cp["L"]) > 0)
    assert cp["L"][0] == 0.0
    # Cp = (p - p_inf) / q_ref with p = 100 + 10 x, so Cp = 10 x / 50
    np.testing.assert_allclose(cp["Cp"], 10.0 * cp["X"] / 50.0, atol=1e-12)
    np.testing.assert_allclose(cp["Y"], 0.5)
    np.testing.assert_allclose(cp["nz"], 1.0)


def test_quadrilateral_cells_give_the_same_cut_as_triangles():
    a = compute_sections(plate(2.0, 3.0, 4, 6, slope=7.0), REF, Distribution(count=3))
    b = compute_sections(plate(2.0, 3.0, 4, 6, slope=7.0, quads=True), REF, Distribution(count=3))
    for sa, sb in zip(_one(a), _one(b), strict=True):
        assert sa.fz == pytest.approx(sb.fz, rel=1e-12)
        assert sa.chord == pytest.approx(sb.chord)
        np.testing.assert_allclose(sa.cp["Cp"].sum(), sb.cp["Cp"].sum(), atol=1e-12)


def test_cylinder_section_is_a_closed_counterclockwise_loop_from_the_lowest_plot_value():
    sides, radius = 36, 0.5
    surf = cylinder(radius, 2.0, sides, 2, 0.0, 1.0)
    res = compute_sections(surf, Reference(0.0, 1.0), Distribution(planes=("XZ",), count=1))
    (st,) = _one(res)
    cp = st.cp
    assert len(cp["L"]) == sides
    # starts at x = -radius (theta = pi), then turns counterclockwise in (x, z): z first falls
    assert cp["X"][0] == pytest.approx(-radius)
    assert cp["Z"][1] < 0.0
    edge = 2 * radius * np.sin(np.pi / sides)
    assert cp["L"][-1] + edge == pytest.approx(sides * edge)
    # outward normals are radial at the nodes
    np.testing.assert_allclose(cp["nx"], cp["X"] / radius, atol=1e-12)
    np.testing.assert_allclose(cp["nz"], cp["Z"] / radius, atol=1e-12)
    # chord of a closed section is its largest width
    assert st.chord == pytest.approx(2 * radius)


def test_plot_direction_two_runs_along_the_second_axis():
    surf = cylinder(0.5, 2.0, 36, 2, 0.0, 1.0)
    res = compute_sections(
        surf, Reference(0.0, 1.0), Distribution(planes=("XZ",), count=1, plot_direction=2)
    )
    (st,) = _one(res)
    np.testing.assert_allclose(st.cp["Section_direction_value"], st.cp["Z"], atol=1e-12)
    assert st.cp["Z"][0] == pytest.approx(-0.5)


def test_swept_tapered_wing_chord_and_quarter_chord_per_station():
    root, tip, span, tan = 2.0, 1.0, 4.0, 0.5
    surf = wing(root, tip, span, tan, 0.0, 0.0)
    res = compute_sections(surf, Reference(0.0, 1.0), Distribution(planes=("XZ",), count=8))
    for s in _one(res):
        y = s.offset
        chord = root + (tip - root) * y / span
        lead = tan * y
        assert s.chord == pytest.approx(chord, rel=1e-12)
        assert s.x_qc == pytest.approx(lead + 0.25 * chord, rel=1e-12)
        assert s.z_qc == pytest.approx(0.0, abs=1e-12)


def test_a_plane_that_misses_the_surface_is_left_out_with_a_reason():
    a = plate(1.0, 1.0, 2, 2)
    b = plate(1.0, 1.0, 2, 2, y0=2.0, family=1)
    joined = Surface(
        points=np.concatenate([a.points, b.points]),
        triangles=np.concatenate([a.triangles, b.triangles + len(a.points)]),
        quads=np.zeros((0, 4), dtype=np.int64),
        pressure=np.zeros(len(a.points) + len(b.points)),
        triangle_family=np.concatenate([a.triangle_family, b.triangle_family]),
        families=("A", "B"),
    )
    res = compute_sections(joined, Reference(0.0, 1.0), Distribution(count=6))
    assert [round(s.offset, 6) for s in _one(res)] == [0.25, 0.75, 2.25, 2.75]
    assert [s.section for s in _one(res)] == [1, 2, 5, 6]
    assert len(res.skips) == 2
    assert all("no cut" in text for text in res.skips)
    only_b = compute_sections(joined, Reference(0.0, 1.0), Distribution(families=("B",), count=2))
    assert [round(s.offset, 6) for s in _one(only_b)] == [2.25, 2.75]
    assert only_b.family_label == "B"


def test_unknown_family_frame_plane_and_options_are_refused_by_name():
    surf = plate(1.0, 1.0, 2, 2)
    with pytest.raises(SectionError, match="'Wing'"):
        compute_sections(surf, REF, Distribution(families=("Wing",)))
    with pytest.raises(SectionError, match="'RMRP'"):
        compute_sections(surf, REF, Distribution(frame="RMRP"))
    with pytest.raises(SectionError, match="'AB'"):
        compute_sections(surf, REF, Distribution(planes=("AB",)))
    with pytest.raises(SectionError, match="include_symmetry"):
        compute_sections(surf, REF, Distribution(), include_symmetry=True)
    with pytest.raises(SectionError, match="plot_direction"):
        compute_sections(surf, REF, Distribution(plot_direction=3))


def test_a_declared_frame_places_the_planes_and_the_chord_axes():
    # frame x = reference y, frame y = -reference x: the plate below, written in the
    # frame as x in [0, 2], y in [0, 3], lies at reference y in [0, 2], x in [-3, 0]
    frame = Frame(axes=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    base = plate(2.0, 3.0, 4, 6)
    pts = np.column_stack([-base.points[:, 1], base.points[:, 0], base.points[:, 2]])
    surf = Surface(
        points=pts,
        triangles=base.triangles,
        quads=base.quads,
        pressure=base.pressure,
    )
    res = compute_sections(surf, REF, Distribution(frame="LOCAL", count=3), frames={"LOCAL": frame})
    st = _one(res)
    assert [s.offset for s in st] == pytest.approx([0.5, 1.5, 2.5])
    assert st[0].chord == pytest.approx(2.0)
    assert st[0].x_qc == pytest.approx(0.5)
    # X, Y, Z of the Cp rows stay in reference coordinates: the cut lies at reference x = -0.5
    np.testing.assert_allclose(st[0].cp["X"], -0.5)


def test_fields_are_interpolated_along_the_cut_edges():
    surf0 = plate(2.0, 3.0, 4, 6)
    mach = 0.1 * surf0.points[:, 0] + 0.02 * surf0.points[:, 1]
    surf = plate(2.0, 3.0, 4, 6, fields={"Mach": mach})
    res = compute_sections(surf, REF, Distribution(count=3), fillable={"Mach", "X", "Y", "Offset"})
    cp = _one(res)[0].cp
    np.testing.assert_allclose(cp["Mach"], 0.1 * cp["X"] + 0.02 * cp["Y"], atol=1e-12)

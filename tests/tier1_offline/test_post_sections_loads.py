"""Sectional loads, strips, integrated totals, absent columns and the written files.

Every expected number is computed here from the closed form of the case, never from
the output under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from post_sections_support import cylinder, plate, wing

from cfdbender.post.sections import (
    CP_COLUMNS,
    INTEGRATED_COLUMNS,
    SLOADS_COLUMNS,
    Distribution,
    Frame,
    Reference,
    SectionError,
    Surface,
    compute_sections,
    distribution_names,
    strip_lengths,
    write_section_files,
)

NONE_REF = Reference(p_inf=0.0, q_ref=1.0)


def test_plate_loads_follow_the_closed_form():
    chord, p0, k = 2.0, 30.0, 5.0
    surf = plate(chord, 3.0, 8, 6, p0=p0, slope=k)
    res = compute_sections(surf, NONE_REF, Distribution(planes=("XZ",), count=3))
    mean_p = p0 + k * chord / 2.0
    my = p0 * chord**2 / 2.0 + k * chord**3 / 3.0
    for s in res.blocks[0].stations:
        assert s.fx == pytest.approx(0.0, abs=1e-10)
        assert s.fz == pytest.approx(-mean_p * chord, rel=1e-12)
        assert s.moment == pytest.approx(my, rel=1e-12)
        assert s.chord == pytest.approx(chord)
        assert s.x_qc == pytest.approx(0.25 * chord)


def test_moment_is_about_the_stated_point():
    chord, p0 = 2.0, 30.0
    surf = plate(chord, 3.0, 8, 6, p0=p0)
    fz = -p0 * chord
    # M_y = z Fx - x Fz about the point; the pressure acts at the mid chord
    for xm, zm in ((0.0, 0.0), (0.5, 0.0), (1.0, 0.7)):
        res = compute_sections(surf, NONE_REF, Distribution(count=1), moment_point=(xm, 0.0, zm))
        s = res.blocks[0].stations[0]
        expected = p0 * ((chord**2 / 2.0) - xm * chord)  # integral of (x - xm) p0 dx
        assert s.moment == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert s.fz == pytest.approx(fz, rel=1e-12)


def test_the_normal_sign_flips_the_force():
    surf = plate(1.0, 1.0, 2, 2, p0=10.0)
    flipped = Surface(
        points=surf.points,
        triangles=surf.triangles,
        quads=surf.quads,
        pressure=surf.pressure,
        normal_sign=-1,
    )
    a = compute_sections(surf, NONE_REF, Distribution(count=1)).blocks[0].stations[0]
    b = compute_sections(flipped, NONE_REF, Distribution(count=1)).blocks[0].stations[0]
    assert a.fz == pytest.approx(-10.0)
    assert b.fz == pytest.approx(10.0)


def test_reference_pressure_is_subtracted_from_the_load_and_the_cp():
    surf = plate(1.0, 1.0, 2, 2, p0=110.0)
    res = compute_sections(surf, Reference(100.0, 20.0), Distribution(count=1))
    s = res.blocks[0].stations[0]
    assert s.fz == pytest.approx(-10.0)
    np.testing.assert_allclose(s.cp["Cp"], 0.5)


def test_cylinder_section_force_and_moment_match_the_closed_form():
    radius, p0, p1, sides = 0.5, 7.0, 3.0, 360
    surf = cylinder(radius, 2.0, sides, 2, p0, p1)
    mp = (0.2, 0.0, 0.3)
    res = compute_sections(surf, NONE_REF, Distribution(planes=("XZ",), count=1), moment_point=mp)
    s = res.blocks[0].stations[0]
    # continuous: Fx = -integral (p0 + p1 cos t) cos t R dt = -p1 pi R, Fz = 0
    fx_exact = -p1 * math.pi * radius
    # the polygon differs from the circle by chord length sin(d/2)/(d/2) = 1 - d^2/24,
    # trapezoid mean cos(d/2) = 1 - d^2/8 and the mid angle: below d^2/6 relative;
    # the bound allows d^2/4
    delta = 2.0 * math.pi / sides
    assert s.fx == pytest.approx(fx_exact, rel=delta**2 / 4.0)
    assert abs(s.fz) < 1e-9
    # the polygon itself, by its own closed form, to rounding
    th = 2.0 * math.pi * np.arange(sides) / sides
    th2 = np.roll(th, -1) + np.where(np.arange(sides) == sides - 1, 2.0 * math.pi, 0.0)
    mid = (th + th2) / 2.0
    length = 2.0 * radius * np.sin(delta / 2.0)
    pbar = p0 + p1 * (np.cos(th) + np.cos(th2)) / 2.0
    assert s.fx == pytest.approx(float(-(pbar * np.cos(mid) * length).sum()), rel=1e-11)
    # about the stated point: M_y = z Fx - x Fz of the net force (the pressure is normal to a
    # circle about the centre, so its moment about the centre is zero)
    assert s.moment == pytest.approx(-mp[2] * fx_exact + mp[0] * 0.0, rel=delta**2, abs=1e-6)
    assert s.chord == pytest.approx(2 * radius)


def test_swept_tapered_wing_integrated_total_within_the_midpoint_bound():
    root, tip, span, tan, p0, k, count = 2.0, 1.0, 4.0, 0.5, 20.0, 3.0, 8
    surf = wing(root, tip, span, tan, p0, k)
    res = compute_sections(
        surf, NONE_REF, Distribution(planes=("XZ",), count=count, integrate=True)
    )
    stations = res.blocks[0].stations
    total = sum(s.fz * s.strip_length for s in stations)
    # F(y) = -(p0 c + k (c x_le + c^2 / 2)) is quadratic in y; integrate it exactly
    y = np.polynomial.Polynomial([0.0, 1.0])
    c = root + (tip - root) / span * y
    x_le = tan * y
    f = -(p0 * c + k * (c * x_le + c * c / 2.0))
    exact = float(f.integ()(span) - f.integ()(0.0))
    # midpoint rule on a quadratic: error = h^2/24 (F'(b) - F'(0)), h = span / count
    h = span / count
    bound = h * h / 24.0 * abs(float(f.deriv()(span) - f.deriv()(0.0)))
    assert abs(total - exact) <= 1.0001 * bound + 1e-12
    assert abs(total - exact) == pytest.approx(bound, rel=1e-9)
    assert sum(s.strip_length for s in stations) == pytest.approx(span)


def test_uniform_plate_integrated_total_is_the_pressure_load_exactly():
    surf = plate(2.0, 3.0, 4, 6, p0=12.0)
    res = compute_sections(surf, NONE_REF, Distribution(count=5, integrate=True))
    total = sum(s.fz * s.strip_length for s in res.blocks[0].stations)
    assert total == pytest.approx(-12.0 * 2.0 * 3.0, rel=1e-12)


def test_strip_rule_on_uneven_offsets():
    assert strip_lengths([0.0, 1.0, 3.0]) == pytest.approx([1.0, 1.5, 2.0])
    assert strip_lengths([3.0, 1.0, 0.0]) == pytest.approx([2.0, 1.5, 1.0])
    with pytest.raises(SectionError, match="at least two"):
        strip_lengths([1.0])
    with pytest.raises(SectionError, match="monotonic"):
        strip_lengths([0.0, 2.0, 1.0])


def test_integration_never_blocks_the_loads_it_omits_the_columns():
    surf = plate(2.0, 3.0, 4, 6, p0=1.0)
    res = compute_sections(surf, NONE_REF, Distribution(count=1, integrate=True))
    assert res.sloads_columns == SLOADS_COLUMNS
    assert any("#integration" in t for t in res.skips)
    assert res.blocks[0].stations[0].strip_length is None


def test_a_rotated_frame_gives_the_loads_in_the_frame_axes():
    frame = Frame(axes=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    base = plate(2.0, 3.0, 4, 6, p0=30.0, slope=5.0)
    pts = np.column_stack([-base.points[:, 1], base.points[:, 0], base.points[:, 2]])
    rot = Surface(points=pts, triangles=base.triangles, quads=base.quads, pressure=base.pressure)
    a = compute_sections(base, NONE_REF, Distribution(count=3)).blocks[0].stations
    b = (
        compute_sections(
            rot, NONE_REF, Distribution(frame="LOCAL", count=3), frames={"LOCAL": frame}
        )
        .blocks[0]
        .stations
    )
    for sa, sb in zip(a, b, strict=True):
        assert sb.fx == pytest.approx(sa.fx, abs=1e-10)
        assert sb.fz == pytest.approx(sa.fz, rel=1e-12)
        assert sb.moment == pytest.approx(sa.moment, rel=1e-12)


def test_columns_the_solver_cannot_fill_are_absent_not_empty(tmp_path):
    fields = {"Mach": np.full(35, 0.2), "CF": np.full(35, 0.003)}
    surf = plate(2.0, 3.0, 4, 6, p0=5.0, fields=fields)
    can = set(CP_COLUMNS) - {"Theta", "Delta*", "Delta", "H", "vx", "vy", "vz", "vtot"}
    can |= set(SLOADS_COLUMNS)
    res = compute_sections(surf, NONE_REF, Distribution(count=2), fillable=can)
    cp, sl = write_section_files(res, tmp_path, "P1", "Wing", leading={"POL": "P1"})
    header = cp.read_text().splitlines()[0].split(",")
    assert header == ["POL", "SECTION", *[c for c in CP_COLUMNS if c in can]]
    for banned in ("Theta", "Delta*", "Delta", "H", "vtot"):
        assert banned not in header
    for line in cp.read_text().splitlines()[1:]:
        cells = line.split(",")
        assert len(cells) == len(header) and all(cells)
    assert sl.read_text().splitlines()[0] == "POL," + ",".join(SLOADS_COLUMNS)


def test_a_fillable_column_with_no_data_is_refused_by_name():
    surf = plate(1.0, 1.0, 2, 2)
    with pytest.raises(SectionError, match="'Theta'"):
        compute_sections(surf, NONE_REF, Distribution(), fillable={"Theta", "Cp"})


def test_files_have_pyflightstreams_names_columns_and_leading_identity(tmp_path):
    surf = plate(2.0, 3.0, 4, 6, p0=5.0)
    dist = Distribution(planes=("XZ", "YZ"), count=2, integrate=True)
    res = compute_sections(surf, NONE_REF, dist)
    lead = {"POL": "P7", "FAMILY": "x", "PLANE": "x", "ROTOR": "NA"}
    cp, sl = write_section_files(res, tmp_path, "P7", "Wing", leading=lead)
    assert cp == tmp_path / "sections" / "P7_cp_Wing.csv"
    assert sl == tmp_path / "sections" / "P7_sloads_Wing.csv"
    rows = [line.split(",") for line in sl.read_text().splitlines()]
    assert rows[0] == ["POL", "FAMILY", "PLANE", "ROTOR", *SLOADS_COLUMNS, *INTEGRATED_COLUMNS]
    assert {r[1] for r in rows[1:]} == {"all"}
    assert {r[2] for r in rows[1:]} == {"XZ", "YZ"}
    assert len(rows) == 1 + 2 + 2
    # My_int is Moment times Strip_length, Fx_int is Fx times Strip_length
    first = dict(zip(rows[0], rows[1], strict=True))
    assert float(first["My_int"]) == pytest.approx(
        float(first["Moment"]) * float(first["Strip_length"])
    )
    assert float(first["Fx_int"]) == pytest.approx(
        float(first["Fx"]) * float(first["Strip_length"])
    )


def test_the_same_input_gives_the_same_bytes(tmp_path):
    surf = plate(2.0, 3.0, 4, 6, p0=5.0, slope=1.3)
    dist = Distribution(planes=("XZ",), count=4, integrate=True)
    one = write_section_files(compute_sections(surf, NONE_REF, dist), tmp_path / "a", "P", "W")
    two = write_section_files(compute_sections(surf, NONE_REF, dist), tmp_path / "b", "P", "W")
    for first, second in zip(one, two, strict=True):
        assert first.read_bytes() == second.read_bytes()
        assert b"\r" not in first.read_bytes()


def test_file_name_words_follow_the_declared_families_and_collisions():
    assert distribution_names(["blades", ["Blade1", "Blade2"], "a:b "]) == [
        "blades",
        "Blade1-Blade2",
        "a_b",
    ]
    assert distribution_names(["Wing", "wing", "Tail"]) == ["Wing_1", "wing_2", "Tail"]

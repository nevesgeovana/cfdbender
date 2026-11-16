"""P0600 airfoil-2d: the section the 2D mesh is made from, and its trailing edge.

A section is a NACA 4 or 5 digit designation or a coordinates file in Selig order (trailing
edge, upper surface, leading edge, lower surface, trailing edge). It is normalised to a unit
chord from the leading edge to the trailing-edge midpoint. The trailing edge is sharp by
default (the gap is collapsed in the mesh, and the gap removed is recorded) or blunt (the gap
is kept as a base of its own).
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.geometry._airfoils import normalized_naca
from cfdbender.mesh import _airfoil2d as af

#: The published NACA 0012 trailing-edge gap, 2 x 5 x 0.12 x (sum of the thickness coefficients).
NACA0012_GAP = 2 * 5 * 0.12 * (0.2969 - 0.1260 - 0.3516 + 0.2843 - 0.1015)


def _selig(path, upper, lower, *, title="test section"):
    rows = [*upper[::-1], *lower[1:]]
    path.write_text(
        title + "\n" + "".join(f"{x:.17g} {y:.17g}\n" for x, y in rows), encoding="utf-8"
    )
    return path


def test_p0600_naca_section_is_normalised_and_blunt_as_published():
    section = af.naca_section("0012")
    assert section.upper[0] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert section.lower[0] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert section.upper[-1][0] == pytest.approx(1.0, abs=1e-9)
    assert section.lower[-1][0] == pytest.approx(1.0, abs=1e-9)
    assert section.gap == pytest.approx(NACA0012_GAP, rel=0.02)
    assert section.source == {"naca": "0012"}
    assert max(section.upper[:, 1]) == pytest.approx(0.06, rel=0.01)


def test_p0600_five_digit_and_prefixed_designations_are_read():
    assert af.naca_section("NACA 23012").source == {"naca": "23012"}
    cambered = af.naca_section("naca2412")
    assert cambered.upper[:, 1].max() > -cambered.lower[:, 1].min()


def test_p0600_sharp_trailing_edge_collapses_the_gap_and_records_it():
    shaped = af.shape_trailing_edge(af.naca_section("0012"), "sharp")
    assert shaped.upper[-1] == pytest.approx([1.0, 0.0], abs=1e-12)
    assert shaped.lower[-1] == pytest.approx([1.0, 0.0], abs=1e-12)
    assert shaped.gap == 0.0
    assert shaped.gap_removed == pytest.approx(NACA0012_GAP, rel=0.02)
    assert shaped.upper[0] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_p0600_blunt_trailing_edge_keeps_the_gap():
    shaped = af.shape_trailing_edge(af.naca_section("0012"), "blunt")
    assert shaped.gap == pytest.approx(NACA0012_GAP, rel=0.02)
    assert shaped.gap_removed == 0.0
    assert shaped.upper[-1][1] > 0 > shaped.lower[-1][1]


def test_p0600_unknown_trailing_edge_is_refused():
    with pytest.raises(InputValueError, match="sharp or blunt"):
        af.shape_trailing_edge(af.naca_section("0012"), "round")


def test_p0600_blunt_needs_a_gap(tmp_path):
    upper, lower, _ = normalized_naca("0012", count=201)
    upper[-1], lower[-1] = [1.0, 0.0], [1.0, 0.0]
    section = af.read_coordinates(_selig(tmp_path / "closed.dat", upper, lower))
    assert section.gap == 0.0
    with pytest.raises(InputValueError, match="closed trailing edge"):
        af.shape_trailing_edge(section, "blunt")


def test_p0600_file_and_designation_give_the_same_section(tmp_path):
    upper, lower, _ = normalized_naca("0012")
    section = af.read_coordinates(_selig(tmp_path / "n0012.dat", upper, lower))
    named = af.naca_section("0012")
    assert section.gap == pytest.approx(named.gap, rel=1e-6)
    for side in ("upper", "lower"):
        theirs, ours = getattr(section, side), getattr(named, side)
        assert len(theirs) == len(ours)
        np.testing.assert_allclose(theirs, ours, atol=1e-9)
    assert section.source["coordinates"].endswith("n0012.dat")


def test_p0600_file_is_normalised_to_unit_chord_and_clockwise_is_accepted(tmp_path):
    upper, lower, _ = normalized_naca("2412", count=301)
    scale = np.array([[2.0, 2.0]])
    shift = np.array([[0.5, -0.25]])
    reversed_rows = [*lower[::-1], *upper[1:]]
    path = tmp_path / "scaled.dat"
    path.write_text(
        "".join(f"{x:.17g} {y:.17g}\n" for x, y in np.asarray(reversed_rows) * scale + shift),
        encoding="utf-8",
    )
    section = af.read_coordinates(path)
    assert section.upper[0] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert (section.upper[-1][0] + section.lower[-1][0]) / 2 == pytest.approx(1.0, abs=1e-9)
    assert section.transform["chord"] == pytest.approx(2.0, rel=1e-6)
    assert section.upper[:, 1].max() > 0 > section.lower[:, 1].min()


def test_p0600_unknown_designation_is_refused():
    with pytest.raises(InputValueError, match="NACA 4 or 5 digit"):
        af.naca_section("9")


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("name\n1 0\n0 0\n", "at least"),
        ("name\n1 0\n0.5 x\n0 0\n0.5 -0.1\n1 0\n", "line 3"),
        ("name\n61. 61.\n0 0\n1 0\n", "Selig"),
    ],
)
def test_p0600_unreadable_file_is_refused(tmp_path, text, named):
    path = tmp_path / "bad.dat"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InputArtifactError, match=named):
        af.read_coordinates(path)


def test_p0600_missing_file_is_refused(tmp_path):
    with pytest.raises(InputArtifactError, match="not found"):
        af.read_coordinates(tmp_path / "absent.dat")


def test_p0600_open_contour_is_refused(tmp_path):
    upper, lower, _ = normalized_naca("0012", count=201)
    path = _selig(tmp_path / "open.dat", upper, lower[: len(lower) // 2])
    with pytest.raises(InputValueError, match="open"):
        af.read_coordinates(path)


def test_p0600_self_crossing_contour_is_refused(tmp_path):
    t = np.linspace(0.0, 1.0, 40)
    upper = np.stack([t, 0.05 * np.sin(2 * np.pi * t)], axis=1)
    lower = np.stack([t, -0.05 * np.sin(2 * np.pi * t)], axis=1)
    path = _selig(tmp_path / "eight.dat", upper, lower)
    with pytest.raises(InputValueError, match="crosses itself"):
        af.read_coordinates(path)


def test_ra2_coincident_surfaces_are_refused(tmp_path):
    path = tmp_path / "flat.dat"
    path.write_text("1 0\n0.5 0\n0 0\n0.5 0\n1 0\n", encoding="ascii")
    with pytest.raises(InputValueError, match="crosses itself"):
        af.read_coordinates(path)

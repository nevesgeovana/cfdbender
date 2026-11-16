"""Geometry: reverse identification of a lifting surface from section cuts.

The numerical half (``geometry._reverse``) is pure and runs on synthetic cuts of a linearly
tapered wing whose planform is known; the half that cuts an IGES skin needs the geometry kernel
and comes with the kernel stages.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry._airfoils import normalized_naca
from cfdbender.geometry._reverse import (
    StationRejectedError,
    clean_band,
    closed_contour,
    contour_features,
    cosine,
    fit_section,
    leading_extension,
    reference_planform,
    refine_tip,
)

SPAN = np.array([0.0, 1.0, 0.0])
ROOT_CHORD, TIP_CHORD, SEMI, SWEEP = 0.4, 0.2, 1.5, 20.0


def chord_at(y):
    return ROOT_CHORD + (TIP_CHORD - ROOT_CHORD) * y / SEMI


def le_x(y):
    return math.tan(math.radians(SWEEP)) * y


def cut(y, designation="NACA0012"):
    """The closed section contour of the known wing at span station y, as one segment."""
    upper, lower, _ = normalized_naca(designation, 401)
    x = np.r_[upper[::-1, 0], lower[1:, 0], upper[-1, 0]]
    z = np.r_[upper[::-1, 1], lower[1:, 1], upper[-1, 1]]
    c = chord_at(y)
    return np.column_stack([le_x(y) + c * x, np.full(len(x), y), c * z])


def stations(count=41):
    return 0.02 + (SEMI - 0.1) * cosine(count)


@pytest.fixture(scope="module")
def scan():
    s = stations()
    features = [contour_features(closed_contour([cut(y)]), SPAN) for y in s]
    return s, features


def test_geom_reverse_cosine_spacing_runs_from_zero_to_one_and_refuses_fewer_than_three():
    values = cosine(5)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(np.diff(values) > 0)
    with pytest.raises(InputValueError, match="three"):
        cosine(2)


def test_geom_reverse_closed_contour_chains_segments_in_any_direction():
    ring = cut(0.5)
    half = len(ring) // 2
    chained = closed_contour([ring[:half], ring[half - 1 :][::-1][::-1]])
    assert np.allclose(chained[0], chained[-1])
    assert len(chained) == len(ring)
    flipped = closed_contour([ring[:half][::-1], ring[half - 1 :]])
    assert np.allclose(flipped[0], flipped[-1])


def test_geom_reverse_closed_contour_rejects_a_gap_and_an_empty_cut():
    ring = cut(0.5)
    with pytest.raises(StationRejectedError, match="open"):
        closed_contour([ring[: len(ring) // 2]])
    with pytest.raises(StationRejectedError, match="open or multiple"):
        closed_contour([ring[:100], ring[200:]])
    with pytest.raises(StationRejectedError, match="empty"):
        closed_contour([])


def test_geom_reverse_contour_features_measure_chord_leading_edge_and_incidence():
    y = 0.6
    feature = contour_features(closed_contour([cut(y)]), SPAN)
    assert feature["chord"] == pytest.approx(chord_at(y), rel=1e-4)
    assert feature["le"][0] == pytest.approx(le_x(y), abs=2e-5)
    assert feature["incidence"] == pytest.approx(0.0, abs=1e-3)
    assert feature["thickness_ratio"] == pytest.approx(0.12, abs=2e-3)
    assert feature["extrema"] == 2


def test_geom_reverse_contour_features_measure_a_rotated_section():
    ring = cut(0.6)
    le = np.array([le_x(0.6), 0.6, 0.0])
    angle = math.radians(4.0)  # nose up, a rotation about the span axis
    x, z = ring[:, 0] - le[0], ring[:, 2]
    turned = ring.copy()
    turned[:, 0] = le[0] + x * math.cos(angle) + z * math.sin(angle)
    turned[:, 2] = -x * math.sin(angle) + z * math.cos(angle)
    feature = contour_features(closed_contour([turned]), SPAN)
    assert abs(feature["incidence"]) == pytest.approx(4.0, abs=0.01)
    assert feature["chord"] == pytest.approx(chord_at(0.6), rel=1e-4)


def test_geom_reverse_a_station_that_is_not_one_airfoil_is_rejected_with_a_reason():
    ring = cut(0.5)
    figure_eight = np.vstack([ring, ring[::-1] * [1.0, 1.0, -1.0]])
    with pytest.raises(StationRejectedError):
        contour_features(closed_contour([figure_eight]), SPAN)
    flat = ring.copy()
    flat[:, 2] = 0.0
    with pytest.raises(StationRejectedError, match="thickness|collapsed|extremum"):
        contour_features(closed_contour([flat]), SPAN)


def test_geom_reverse_clean_band_takes_every_clean_station_and_drops_a_distorted_one(scan):
    s, features = scan
    run, mask, diagnostics = clean_band(s, features)
    assert len(run) == len(s)
    assert mask.all()
    assert set(diagnostics) == {"chord", "thickness_ratio"}
    damaged = list(features)
    damaged[-3:] = [None] * 3  # a tip cap that does not cut as one closed airfoil
    damaged[-4] = dict(features[-4], chord=features[-4]["chord"] * 1.02)
    run, mask, _ = clean_band(s, damaged)
    assert int(run[-1]) == len(s) - 5
    assert not mask[-4:].any()


def test_geom_reverse_clean_band_refuses_when_too_few_stations_cut(scan):
    s, features = scan
    with pytest.raises(InputValueError, match="insufficient"):
        clean_band(s, [f if i < 5 else None for i, f in enumerate(features)])


def test_geom_reverse_reference_planform_recovers_the_known_trapezoid(scan):
    s, features = scan
    band = np.arange(len(s))
    result = reference_planform(s, features, band, SPAN, root_station=0.0, tip_station=SEMI)
    assert result["semi_span_m"] == pytest.approx(SEMI)
    assert result["root_chord_m"] == pytest.approx(ROOT_CHORD, rel=1e-4)
    assert result["tip_chord_m"] == pytest.approx(TIP_CHORD, rel=1e-4)
    assert result["taper"] == pytest.approx(TIP_CHORD / ROOT_CHORD, rel=1e-4)
    assert result["sweep_le_deg"] == pytest.approx(SWEEP, abs=0.01)
    assert result["area_m2"] == pytest.approx(2 * SEMI * (ROOT_CHORD + TIP_CHORD) / 2, rel=1e-4)
    assert result["span_m"] == pytest.approx(2 * SEMI)
    assert result["dihedral_deg"] == 0.0
    assert result["tip_twist_deg"] == pytest.approx(0.0, abs=1e-3)
    one_side = reference_planform(
        s, features, band, SPAN, root_station=0.0, tip_station=SEMI, mirrored=False
    )
    assert one_side["area_m2"] == pytest.approx(result["area_m2"] / 2)


def test_geom_reverse_reference_planform_finds_the_root_where_a_boundary_vanishes(scan):
    s, features = scan
    band = np.arange(len(s))
    found = reference_planform(
        s, features, band, SPAN, boundary=lambda p: p[1] - 0.01, root_bracket=(-0.5, 0.2)
    )
    assert found["root_station_m"] == pytest.approx(0.01, abs=1e-8)


def test_geom_reverse_reference_planform_refuses_a_root_that_does_not_precede_the_tip(scan):
    s, features = scan
    band = np.arange(len(s))
    with pytest.raises(InputValueError, match="precede"):
        reference_planform(s, features, band, SPAN, root_station=SEMI + 1.0)
    with pytest.raises(InputValueError, match="root"):
        reference_planform(s, features, band, SPAN)


def test_geom_reverse_the_leading_edge_extension_continues_linearly_past_the_band(scan):
    s, features = scan
    curve = leading_extension(s, features, np.arange(len(s)))
    assert curve(0.0)[0] == pytest.approx(le_x(0.0), abs=3e-5)
    assert curve(SEMI + 0.3)[0] == pytest.approx(le_x(SEMI + 0.3), abs=3e-4)


def test_geom_reverse_refine_tip_keeps_a_clean_endpoint_between_the_last_clean_and_first_bad(scan):
    s, features = scan
    band = np.flatnonzero(s <= 1.2)

    def evaluate(y):
        if y > 1.2:
            raise StationRejectedError("a cap")
        return contour_features(closed_contour([cut(y)]), SPAN)

    stations_out, features_out, band_out, bracket = refine_tip(
        s, list(features), band, evaluate, tolerance=1e-3
    )
    assert bracket[0] <= 1.2 <= bracket[1] + 1e-12
    assert bracket[1] - bracket[0] < 1e-3
    assert stations_out[band_out[-1]] == pytest.approx(bracket[0])
    assert len(features_out) == len(stations_out)
    again = refine_tip(s, list(features), np.arange(len(s)), evaluate)
    assert again[3] == [float(s[-1]), float(s[-1])]


def test_geom_reverse_fit_section_meets_its_tolerance_on_the_section_it_was_cut_from():
    feature = contour_features(closed_contour([cut(0.8)]), SPAN)
    fit = fit_section(feature["contour"], SPAN, "NACA0012")
    assert fit["accepted"]
    assert fit["max_residual_m"] <= fit["tolerance_m"]
    assert fit["chord_m"] == pytest.approx(chord_at(0.8), rel=1e-4)
    assert fit["measured_thickness_ratio"] == pytest.approx(0.12, abs=2e-3)
    assert len(fit["upper"]) == fit["degree"] + 1 == len(fit["lower"])


def test_geom_reverse_fit_section_says_when_a_section_missed_its_tolerance():
    feature = contour_features(closed_contour([cut(0.8, "NACA4412")]), SPAN)
    fit = fit_section(
        feature["contour"], SPAN, "NACA0012", degree=2, max_degree=2, tolerance_m=1e-9
    )
    assert not fit["accepted"]
    assert fit["max_residual_m"] > fit["tolerance_m"]
    assert fit["degree"] == 2


def test_geom_reverse_a_left_half_measures_as_its_mirror_image():
    """a span axis toward -y keeps the incidence sign, the upper side and the dihedral."""
    tilt = math.radians(3.0)
    right_span = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    left_span = right_span * [1.0, -1.0, 1.0]
    angle = math.radians(3.0)
    ring = cut(0.6, "NACA4412")
    x, z = ring[:, 0], ring[:, 2]
    turned = np.column_stack(
        [
            x * math.cos(angle) + z * math.sin(angle),
            ring[:, 1],
            -x * math.sin(angle) + z * math.cos(angle),
        ]
    )
    right = contour_features(closed_contour([turned]), right_span)
    left = contour_features(closed_contour([turned * [1.0, -1.0, 1.0]]), left_span)
    assert right["incidence"] > 0
    assert left["incidence"] == pytest.approx(right["incidence"], abs=1e-9)
    assert left["unit_contour"][:, 1] == pytest.approx(right["unit_contour"][:, 1], abs=1e-9)
    fit_right = fit_section(right["contour"], right_span, "NACA4412")
    fit_left = fit_section(left["contour"], left_span, "NACA4412")
    assert fit_left["upper"] == pytest.approx(fit_right["upper"], abs=1e-6)
    s = stations()
    for span, sign in ((right_span, 1.0), (left_span, -1.0)):
        features = [contour_features(closed_contour([cut(y) * [1.0, sign, 1.0]]), span) for y in s]
        planform = reference_planform(s, features, np.arange(len(s)), span, root_station=0.0)
        assert planform["dihedral_deg"] == pytest.approx(3.0)


def test_geom_reverse_fit_section_refuses_a_start_degree_above_its_limit():
    """the degree loop never ran and the fit died on an unbound name."""
    feature = contour_features(closed_contour([cut(0.8)]), SPAN)
    with pytest.raises(InputValueError, match="max_degree"):
        fit_section(feature["contour"], SPAN, "NACA0012", degree=6, max_degree=5)


def test_geom_reverse_a_root_the_leading_edge_never_reaches_is_refused_by_name(scan):
    """the bracket holds no crossing; the message names the root, not the root finder."""
    s, features = scan
    with pytest.raises(InputValueError, match="root: the leading edge does not cross"):
        reference_planform(
            s,
            features,
            np.arange(len(s)),
            SPAN,
            boundary=lambda p: p[1] + 5.0,
            root_bracket=(-0.5, 0.2),
        )


# --- the workspace half, on a generated IGES -------------------------------------------------

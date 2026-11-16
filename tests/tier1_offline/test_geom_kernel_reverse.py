"""geometry: reverse identification of a lifting surface from an original IGES skin.

The numerical half (``geometry.reverse``) is pure and runs everywhere on synthetic cuts of a
linearly tapered wing whose planform is known; the workspace half cuts a generated IGES with
the geometry kernel and skips where the optional extra is absent. The comparison with the
research package it was ported from is tier 2: it skips unless ``PYFS_RESEARCH_GEOMREV`` names
the folder that holds the research ``geomrev`` package (its sibling ``geomgen`` is read too).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from kernel_support import kernel_or_skip

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
from cfdbender.mesh._extras import MissingExtraError

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


def test_geom_kernel_reverse_cosine_spacing_runs_from_zero_to_one_and_refuses_fewer_than_three():
    values = cosine(5)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(np.diff(values) > 0)
    with pytest.raises(InputValueError, match="three"):
        cosine(2)


def test_geom_kernel_reverse_closed_contour_chains_segments_in_any_direction():
    ring = cut(0.5)
    half = len(ring) // 2
    chained = closed_contour([ring[:half], ring[half - 1 :][::-1][::-1]])
    assert np.allclose(chained[0], chained[-1])
    assert len(chained) == len(ring)
    flipped = closed_contour([ring[:half][::-1], ring[half - 1 :]])
    assert np.allclose(flipped[0], flipped[-1])


def test_geom_kernel_reverse_closed_contour_rejects_a_gap_and_an_empty_cut():
    ring = cut(0.5)
    with pytest.raises(StationRejectedError, match="open"):
        closed_contour([ring[: len(ring) // 2]])
    with pytest.raises(StationRejectedError, match="open or multiple"):
        closed_contour([ring[:100], ring[200:]])
    with pytest.raises(StationRejectedError, match="empty"):
        closed_contour([])


def test_geom_kernel_reverse_contour_features_measure_chord_leading_edge_and_incidence():
    y = 0.6
    feature = contour_features(closed_contour([cut(y)]), SPAN)
    assert feature["chord"] == pytest.approx(chord_at(y), rel=1e-4)
    assert feature["le"][0] == pytest.approx(le_x(y), abs=2e-5)
    assert feature["incidence"] == pytest.approx(0.0, abs=1e-3)
    assert feature["thickness_ratio"] == pytest.approx(0.12, abs=2e-3)
    assert feature["extrema"] == 2


def test_geom_kernel_reverse_contour_features_measure_a_rotated_section():
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


def test_geom_kernel_reverse_a_station_that_is_not_one_airfoil_is_rejected_with_a_reason():
    ring = cut(0.5)
    figure_eight = np.vstack([ring, ring[::-1] * [1.0, 1.0, -1.0]])
    with pytest.raises(StationRejectedError):
        contour_features(closed_contour([figure_eight]), SPAN)
    flat = ring.copy()
    flat[:, 2] = 0.0
    with pytest.raises(StationRejectedError, match="thickness|collapsed|extremum"):
        contour_features(closed_contour([flat]), SPAN)


def test_geom_kernel_reverse_clean_band_takes_every_clean_station_and_drops_a_distorted_one(scan):
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


def test_geom_kernel_reverse_clean_band_refuses_when_too_few_stations_cut(scan):
    s, features = scan
    with pytest.raises(InputValueError, match="insufficient"):
        clean_band(s, [f if i < 5 else None for i, f in enumerate(features)])


def test_geom_kernel_reverse_reference_planform_recovers_the_known_trapezoid(scan):
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


def test_geom_kernel_reverse_reference_planform_finds_the_root_where_a_boundary_vanishes(scan):
    s, features = scan
    band = np.arange(len(s))
    found = reference_planform(
        s, features, band, SPAN, boundary=lambda p: p[1] - 0.01, root_bracket=(-0.5, 0.2)
    )
    assert found["root_station_m"] == pytest.approx(0.01, abs=1e-8)


def test_geom_kernel_reverse_reference_planform_refuses_a_root_that_does_not_precede_the_tip(scan):
    s, features = scan
    band = np.arange(len(s))
    with pytest.raises(InputValueError, match="precede"):
        reference_planform(s, features, band, SPAN, root_station=SEMI + 1.0)
    with pytest.raises(InputValueError, match="root"):
        reference_planform(s, features, band, SPAN)


def test_geom_kernel_reverse_the_leading_edge_extension_continues_linearly_past_the_band(scan):
    s, features = scan
    curve = leading_extension(s, features, np.arange(len(s)))
    assert curve(0.0)[0] == pytest.approx(le_x(0.0), abs=3e-5)
    assert curve(SEMI + 0.3)[0] == pytest.approx(le_x(SEMI + 0.3), abs=3e-4)


def test_geom_kernel_reverse_refine_tip_keeps_a_clean_endpoint_between_the_last_clean_and_first_bad(
    scan,
):
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


def test_geom_kernel_reverse_fit_section_meets_its_tolerance_on_the_section_it_was_cut_from():
    feature = contour_features(closed_contour([cut(0.8)]), SPAN)
    fit = fit_section(feature["contour"], SPAN, "NACA0012")
    assert fit["accepted"]
    assert fit["max_residual_m"] <= fit["tolerance_m"]
    assert fit["chord_m"] == pytest.approx(chord_at(0.8), rel=1e-4)
    assert fit["measured_thickness_ratio"] == pytest.approx(0.12, abs=2e-3)
    assert len(fit["upper"]) == fit["degree"] + 1 == len(fit["lower"])


def test_geom_kernel_reverse_fit_section_says_when_a_section_missed_its_tolerance():
    feature = contour_features(closed_contour([cut(0.8, "NACA4412")]), SPAN)
    fit = fit_section(
        feature["contour"], SPAN, "NACA0012", degree=2, max_degree=2, tolerance_m=1e-9
    )
    assert not fit["accepted"]
    assert fit["max_residual_m"] > fit["tolerance_m"]
    assert fit["degree"] == 2


def test_geom_kernel_reverse_r6_a_left_half_measures_as_its_mirror_image():
    """R6-002: a span axis toward -y keeps the incidence sign, the upper side and the dihedral."""
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


def test_geom_kernel_reverse_r6_fit_section_refuses_a_start_degree_above_its_limit():
    """R6-008: the degree loop never ran and the fit died on an unbound name."""
    feature = contour_features(closed_contour([cut(0.8)]), SPAN)
    with pytest.raises(InputValueError, match="max_degree"):
        fit_section(feature["contour"], SPAN, "NACA0012", degree=6, max_degree=5)


def test_geom_kernel_reverse_r6_a_root_the_leading_edge_never_reaches_is_refused_by_name(scan):
    """R6-009: the bracket holds no crossing; the message names the root, not the root finder."""
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

WING = {
    "components": {
        "WING": {
            "type": "lifting_surface",
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
            "planform": {
                "sref_m2": 0.24,
                "aspect_ratio": 6.0,
                "taper": 0.5,
                "le_sweep_deg": 15.0,
                "incidence": 1.0,
                "tip_twist": -2.0,
            },
        }
    }
}


def gmsh_or_skip():
    kernel_or_skip()
    pytest.importorskip("contourpy")


@pytest.fixture(scope="module")
def generated_wing(tmp_path_factory):
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges

    path = tmp_path_factory.mktemp("reverse") / "wing.igs"
    generate_iges(WING, path)
    return path


@pytest.fixture(scope="module")
def identified(generated_wing):
    from cfdbender.geometry._identify import identify_lifting_surface

    return identify_lifting_surface(generated_wing, "WING", stations=61, grid=129)


def test_geom_kernel_reverse_identify_recovers_the_planform_a_wing_was_generated_from(identified):
    planform = identified.planform
    assert identified.accepted
    assert identified.root_kind == "symmetry plane y = 0"
    assert planform["span_m"] == pytest.approx(1.2, abs=1e-3)
    assert planform["root_chord_m"] == pytest.approx(1.2 / 6.0 / 0.75 * 1.0, abs=1e-3)
    assert planform["taper"] == pytest.approx(0.5, abs=2e-3)
    assert planform["sweep_le_deg"] == pytest.approx(15.0, abs=0.05)
    assert planform["incidence_deg"] == pytest.approx(1.0, abs=0.02)
    assert planform["tip_twist_deg"] == pytest.approx(-2.0, abs=0.05)
    assert identified.convergence["accepted"]
    assert all(fit["accepted"] for fit in identified.sections + [identified.midspan])


def test_geom_kernel_reverse_identify_result_rebuilds_through_load_config(identified):
    from cfdbender.geometry import resolve_config

    entry = identified.to_component()
    resolved = resolve_config({"components": {"WING": entry}})
    span = resolved["components"]["WING"]["planform"]["span"]["value"]
    assert span == pytest.approx(1.2, abs=1e-3)
    assert identified.to_dict()["component"] == "WING"


def test_geom_kernel_reverse_r6_the_configuration_rebuilds_a_twisted_wing_inside_the_limit(
    identified, generated_wing
):
    """R6-003: incidence and twist turn about c/4, so the measured leading edge moved."""
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry._identify import check_against_original

    entry = identified.to_component()
    assert entry["placement"]["translation"] == pytest.approx([0.0, 0.0, 0.0], abs=2e-4)
    assert entry["planform"]["le_sweep_deg"] == pytest.approx(15.0, abs=0.05)
    assert entry["planform"]["dihedral_deg"] == pytest.approx(0.0, abs=0.02)
    again = generated_wing.parent / "again.igs"
    generate_iges({"components": {"WING": entry}}, again)
    record = check_against_original(
        again,
        generated_wing,
        {"chord_m": 0.27, "forward_count": 9, "reverse_count": 9, "edge_count": 21},
    )
    assert record["accepted"], record["forward"]


@pytest.fixture(scope="module")
def plain_wing(tmp_path_factory):
    """A wing with no incidence or twist: its first span estimate is already right."""
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges

    path = tmp_path_factory.mktemp("plain") / "wing.igs"
    planform = {"sref_m2": 0.24, "aspect_ratio": 6.0, "taper": 0.5, "le_sweep_deg": 15.0}
    stations = [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}]
    wing = {"type": "lifting_surface", "stations": stations, "planform": planform}
    generate_iges({"components": {"WING": wing}}, path)
    return path


def test_geom_kernel_reverse_r6_the_fine_pass_takes_the_stations_asked_when_the_span_is_settled(
    plain_wing,
):
    """R6-004: a settled first estimate stopped at the 81 stations of the first scan."""
    from cfdbender.geometry._identify import identify_lifting_surface

    found = identify_lifting_surface(plain_wing, "WING", stations=85, grid=65, converge=False)
    assert found.scan["stations"] == 85
    assert len(found.scan["station_m"]) >= 85


def test_geom_kernel_reverse_r6_a_rejected_station_keeps_its_own_label_after_the_tip_is_refined(
    tmp_path,
):
    """R6-005: the station refine_tip inserted shifted every rejected label after it."""
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry._identify import identify_lifting_surface

    wing = {
        "type": "lifting_surface",
        "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        "planform": {"sref_m2": 0.24, "aspect_ratio": 6.0, "taper": 0.5, "le_sweep_deg": 15.0},
        "cap": {"type": "rounded"},
    }
    path = tmp_path / "rounded.igs"
    generate_iges({"components": {"WING": wing}}, path)
    found = identify_lifting_surface(path, "WING", stations=41, grid=65, converge=False)
    outermost = found.scan["station_m"][-1]
    assert len(found.scan["station_m"]) == 42  # the tip bisection inserted a clean endpoint
    assert f"{outermost:.6g}" in found.scan["rejected"]


def test_geom_kernel_reverse_r6_a_boundary_root_measures_its_extent_along_the_settled_span_axis(
    tmp_path,
):
    """R6-006: the extent was taken along the first estimate and read against the settled axis."""
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry._identify import identify_lifting_surface

    wing = {
        "type": "lifting_surface",
        "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        "planform": {"sref_m2": 0.24, "aspect_ratio": 6.0, "taper": 0.5, "dihedral_deg": 6.0},
    }
    path = tmp_path / "dihedral.igs"
    generate_iges({"components": {"WING": wing}}, path)
    found = identify_lifting_surface(
        path, "WING", root="boundary", stations=41, grid=65, converge=False
    )
    assert found.planform["dihedral_deg"] == pytest.approx(6.0, abs=0.05)
    assert found.planform["semi_span_m"] >= 0.6 / math.cos(math.radians(6.0)) - 5e-4


def test_geom_kernel_reverse_r6_one_quantity_past_its_tolerance_breaks_the_agreement_of_the():
    """R6-015: no test had the passes disagree, so the guard was never seen to say no."""
    from cfdbender.geometry._geometry_reverse import CONVERGENCE_TOLERANCES, convergence

    fine = dict.fromkeys(CONVERGENCE_TOLERANCES, 1.0)
    assert convergence(dict(fine), fine)["accepted"] is True
    for key, tolerance in CONVERGENCE_TOLERANCES.items():
        coarse = dict(fine, **{key: 1.0 + 2 * tolerance})
        result = convergence(coarse, fine)
        assert result["accepted"] is False
        assert result["checks"][key]["passed"] is False


def test_geom_kernel_reverse_identify_refuses_a_bad_argument_before_any_cut(generated_wing):
    from cfdbender.geometry._identify import identify_lifting_surface

    for bad, match in (
        ({"root": "air"}, "root"),
        ({"cut": "x"}, "cut"),
        ({"stations": 3}, "stations"),
        ({"root": "host"}, "hosts"),
        ({"span_axis": (0, 0, 0)}, "span_axis"),
        ({"section_tolerance_mm": -1.0}, "section_tolerance_mm"),  # R6-008
        ({"host_tolerance": -1.0}, "host_tolerance"),  # R6-008
    ):
        with pytest.raises(InputValueError, match=match):
            identify_lifting_surface(generated_wing, "WING", **bad)


def test_geom_kernel_reverse_identify_names_the_components_found_when_one_is_unknown(
    generated_wing,
):
    from cfdbender.geometry._identify import identify_lifting_surface

    with pytest.raises((InputValueError, ValueError), match="NOPE|WING"):
        identify_lifting_surface(generated_wing, "NOPE", stations=21, grid=33)


def test_geom_kernel_reverse_the_reverse_module_refuses_without_gmsh(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "gmsh" or name.startswith("gmsh."):
            raise ImportError("no gmsh")
        return real(name, *args, **kwargs)

    from cfdbender.geometry._identify import identify_lifting_surface

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(MissingExtraError, match=r"gmsh>=4\.15\.1"):
        identify_lifting_surface(Path("x.igs"), "WING")


# --- tier 2: the research package ---------------------------------------------------------


def _same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _same(x, y)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, np.asarray(b))
    else:
        assert a == b


def test_geom_kernel_reverse_reverse_equals_the_research_reference_to_the_last_bit(
    monkeypatch, scan
):
    """The ported numerics equal the research ``geomrev``, value by value (max difference 0)."""
    folder = os.environ.get("PYFS_RESEARCH_GEOMREV")
    if not folder:
        pytest.skip("needs PYFS_RESEARCH_GEOMREV, the folder of the research geomrev package")
    kernel_or_skip()
    monkeypatch.syspath_prepend(str(Path(folder).parent / "geomgen"))
    monkeypatch.syspath_prepend(folder)
    from geomrev import junctions, reference, sections

    s, features = scan
    old_features = [reference.contour_features(reference.closed_contour([cut(y)]), SPAN) for y in s]
    _same(old_features, features)

    old_run, old_mask, old_diagnostics = reference.clean_band(s, old_features)
    run, mask, diagnostics = clean_band(s, features)
    _same((old_run, old_mask, old_diagnostics), (run, mask, diagnostics))

    kwargs = {"root_station": 0.0, "tip_station": SEMI}
    _same(
        reference.reference_planform(s, old_features, old_run, SPAN, **kwargs),
        reference_planform(s, features, run, SPAN, **kwargs),
    )
    _same(
        reference.reference_planform(
            s, old_features, old_run, SPAN, boundary=lambda p: p[1] - 0.01, root_bracket=(-0.5, 0.2)
        ),
        reference_planform(
            s, features, run, SPAN, boundary=lambda p: p[1] - 0.01, root_bracket=(-0.5, 0.2)
        ),
    )
    old_curve = junctions.leading_extension(s, old_features, old_run)
    curve = leading_extension(s, features, run)
    for at in np.linspace(-0.2, SEMI + 0.4, 17):  # the research curve takes one station at a time
        _same(old_curve(at), curve(at))
    assert curve(np.linspace(-0.2, SEMI + 0.4, 17)).shape == (17, 3)
    contour = features[20]["contour"]
    _same(sections.fit_section(contour, SPAN, "NACA0012"), fit_section(contour, SPAN, "NACA0012"))

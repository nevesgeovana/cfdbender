"""Geometry: the stored section is a direct B-spline; conversions meet their limits.

Oracles are closed forms: the NACA thickness equation, the published nose radius 1.1019 t^2,
the exact poles of a curve that was read, and the base a file or an input states.
"""

from __future__ import annotations

import numpy as np
import pytest
from geom_inputs import stations as as_stations

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._sections import (
    BSplineSection,
    SectionForm,
    bspline_sections,
    section_checks,
    section_forms,
    section_from_bspline,
    section_from_stored,
    shape_unchanged,
)


def table(items, te=0.00252, **extra):
    return {
        "form": "bspline",
        "degree": 5,
        "trailing_edge_thickness": te,
        "eta": [float(i) for i in range(len(items))],
        "airfoils": [""] * len(items),
        "items": items,
        **extra,
    }


def half_thickness(x, t=0.12, closed=False):
    last = -0.1036 if closed else -0.1015
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + last * x**4)


def write_file(path, closed=False, count=201):
    x = (1 - np.cos(np.linspace(0, np.pi, count))) / 2
    yt = half_thickness(x, closed=closed)
    pts = np.vstack([np.stack((x[::-1], yt[::-1]), 1), np.stack((x[1:], -yt[1:]), 1)])
    path.write_text("test\n" + "\n".join(f"{a:.10f} {b:.10f}" for a, b in pts) + "\n")


def statuses(section):
    return {c["id"]: c["status"] for c in section_checks(section)}


@pytest.fixture(scope="module")
def common():
    items = [
        {"naca": "0012"},
        {"naca": "2412"},
        {"naca": "23015"},
        {"naca": "22112"},
        {"cst_upper": [0.2, 0.25, 0.2], "cst_lower": [-0.2, -0.15, -0.1], "te": 0.002},
    ]
    return bspline_sections(table(items))


def test_every_section_is_a_degree_5_c2_bspline_on_one_knot_vector(common):
    assert len({s.knots for s in common}) == 1
    section = common[0]
    assert isinstance(section, SectionForm)
    assert section.degree == 5
    knots = np.asarray(section.knots)
    interior = knots[6:-6]
    _, counts = np.unique(interior, return_counts=True)
    assert set(counts) == {3}
    spans = len(counts) + 1
    assert len(section.poles) == 3 * spans + 3
    assert 0.5 in interior


def test_the_conversions_meet_their_limit_and_checks(common):
    for section in common:
        assert section.max_residual <= 2e-5
        assert statuses(section)["section.residual"] == "pass"
        assert "fail" not in statuses(section).values()


def test_the_fitted_curve_follows_the_naca_thickness_equation():
    (section,) = bspline_sections(table([{"naca": "0012"}], te=0.0021))
    x = np.array([0.02, 0.1, 0.3, 0.6, 0.9])
    upper, lower = section.evaluate(x)
    wanted = half_thickness(x) + x * (0.0021 / 2 - 0.00126)
    assert np.max(np.abs(upper[:, 1] - wanted)) < 3e-5
    assert np.max(np.abs(lower[:, 1] + wanted)) < 3e-5


def test_the_nose_radius_of_a_symmetric_section_is_the_published_one():
    (section,) = bspline_sections(table([{"naca": "0012"}]))
    assert section.nose_radius() == pytest.approx(1.1019 * 0.12**2, rel=0.02)
    assert statuses(section)["section.one_nose_radius"] == "pass"


def test_the_leading_edge_is_parameter_one_half_and_the_ends_are_the_base(common):
    section = common[1]
    from scipy.interpolate import BSpline

    spline = BSpline(np.asarray(section.knots), np.asarray(section.poles), 5)
    assert spline(0.5)[0] == pytest.approx(0.0, abs=2e-6)
    assert np.all(np.diff(spline(np.linspace(0.5, 1, 50))[:, 0]) > 0)
    assert np.all(np.diff(spline(np.linspace(0.5, 0, 50))[:, 0]) > 0)
    assert section.te_thickness() == pytest.approx(0.00252, abs=1e-9)


def test_a_pinned_chordwise_line_is_a_knot_at_its_own_parameter():
    pins = [0.25, 0.7]
    sections = bspline_sections(table([{"naca": "0012"}, {"naca": "2412"}], pin=pins))
    knots = np.asarray(sections[0].knots)
    for x in pins:
        for u in (0.5 - 0.5 * np.sqrt(x), 0.5 + 0.5 * np.sqrt(x)):
            assert np.min(np.abs(knots - u)) < 1e-12
    from scipy.interpolate import BSpline

    for section in sections:
        spline = BSpline(np.asarray(section.knots), np.asarray(section.poles), 5)
        for x in pins:
            assert spline(0.5 + 0.5 * np.sqrt(x))[0] == pytest.approx(x, abs=2e-4)


def test_the_spans_rise_only_until_the_limit_is_met(tmp_path):
    write_file(tmp_path / "foil.dat")
    path = str(tmp_path / "foil.dat")
    loose = bspline_sections(table([{"file": path, "tolerance": 1e-3}]))[0]
    normal = bspline_sections(table([{"file": path}]))[0]
    assert loose.max_residual <= 1e-3
    assert len(loose.poles) < len(normal.poles)
    assert len(loose.poles) >= 27  # eight spans is where the search starts


def test_a_file_converts_within_its_stated_tolerance(tmp_path):
    write_file(tmp_path / "foil.dat")
    (section,) = bspline_sections(table([{"file": str(tmp_path / "foil.dat")}]))
    assert section.limit == 1e-4
    assert section.max_residual <= 1e-4
    x = np.array([0.05, 0.4, 0.8])
    upper, _ = section.evaluate(x)
    assert np.max(np.abs(upper[:, 1] - half_thickness(x))) < 3e-4
    (tight,) = bspline_sections(table([{"file": str(tmp_path / "foil.dat"), "tolerance": 2e-5}]))
    assert tight.limit == 2e-5
    assert len(tight.poles) >= len(section.poles)


def test_a_bspline_input_round_trips_exactly(common):
    stored = common[2].to_stored()
    again = section_from_stored(stored)
    assert isinstance(again, BSplineSection)
    assert again.poles == common[2].poles
    (back,) = bspline_sections(table([{"bspline": stored}]))
    assert back.poles == common[2].poles
    assert back.knots == common[2].knots
    assert back.max_residual == 0.0
    assert statuses(back)["section.residual"] == "pass"


def test_a_bspline_file_is_read_and_united_with_a_finer_vector(tmp_path):
    import json

    (coarse,) = bspline_sections(table([{"naca": "0012"}], te=0.0))
    poles = np.asarray(coarse.poles)
    # a different knot vector from the fit: degree 5, two C2 knots only
    knots = [0.0] * 6 + [0.3] * 3 + [0.7] * 3 + [1.0] * 6
    rng = np.random.default_rng(1)
    own = np.cumsum(rng.uniform(0.01, 0.1, size=(len(knots) - 6, 2)), axis=0) * [1.0, 0.1]
    own[:, 0] /= own[-1, 0]
    path = tmp_path / "curve.json"
    path.write_text(json.dumps({"poles": own.tolist(), "knots": knots, "degree": 5}))
    one, two = bspline_sections(table([{"naca": "0012"}, {"bspline": str(path)}]))
    assert one.knots == two.knots
    from scipy.interpolate import BSpline

    mine = BSpline(np.asarray(knots, float), own, 5)
    union = BSpline(np.asarray(two.knots), np.asarray(two.poles), 5)
    u = np.linspace(0, 1, 301)
    assert np.max(np.abs(mine(u) - union(u))) < 1e-10
    assert poles.shape[0] > 0


def test_a_low_degree_bspline_is_raised_without_changing_it():
    from scipy.interpolate import BSpline

    poles = [[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]]
    knots = [0, 0, 0, 1, 1, 1]
    section = section_from_bspline({"poles": poles, "knots": knots, "degree": 2})
    assert section.degree == 5
    u = np.linspace(0, 1, 50)
    raised = BSpline(np.asarray(section.knots), np.asarray(section.poles), 5)(u)
    assert (
        np.max(np.abs(raised - BSpline(np.asarray(knots, float), np.asarray(poles), 2)(u))) < 1e-12
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"poles": [[0, 0]], "knots": [0, 1], "degree": 5}, "knots"),
        ({"poles": [[0, 0]] * 6, "knots": [0] * 6 + [1] * 6, "degree": 7}, "degree"),
        ({"poles": [[0, 0]] * 6, "knots": [0] * 6 + [1] * 6}, "poles, knots and degree"),
        ({"poles": [[0, 0, 0]] * 6, "knots": [0] * 6 + [1] * 6, "degree": 5}, "poles"),
        ({"poles": [[0, np.nan]] * 6, "knots": [0] * 6 + [1] * 6, "degree": 5}, "poles"),
        ({"poles": [[0, 0]] * 6, "knots": [0] * 6 + [2] * 6, "degree": 5}, "knots"),
    ],
)
def test_a_bad_bspline_input_is_refused_with_its_key(source, message):
    with pytest.raises(InputValueError, match=message):
        section_from_bspline(source)


def test_an_empty_or_missing_bspline_file_is_refused(tmp_path):
    (tmp_path / "empty.json").write_text("")
    with pytest.raises(InputValueError, match="cannot read"):
        section_from_bspline(str(tmp_path / "empty.json"))
    with pytest.raises(InputValueError, match="cannot read"):
        section_from_bspline(str(tmp_path / "none.json"))


def test_a_misplaced_pin_is_refused():
    for pins in ([0.0], [1.0], [0.5, 0.4], [float("nan")]):
        with pytest.raises(InputValueError, match="sections.pin:"):
            bspline_sections(table([{"naca": "0012"}], pin=pins))


def test_shape_unchanged_holds_for_the_bspline_form():
    record = shape_unchanged(table([{"naca": "0012"}, {"naca": "2412"}]))
    assert record["status"] == "pass"
    assert record["value"] < 1e-9


def test_section_forms_follows_the_named_form():
    forms = section_forms(table([{"naca": "0012"}]))
    assert isinstance(forms[0], BSplineSection)
    old = {**table([{"naca": "0012"}]), "form": "cst"}
    assert not isinstance(section_forms(old)[0], BSplineSection)


def test_a_fit_above_its_limit_is_reported_not_refused(tmp_path):
    write_file(tmp_path / "foil.dat", count=121)
    (section,) = bspline_sections(table([{"file": str(tmp_path / "foil.dat"), "tolerance": 1e-12}]))
    record = {c["id"]: c for c in section_checks(section)}["section.residual"]
    assert record["status"] == "fail"
    assert record["value"] == section.max_residual > 1e-12


# ---------------------------------------------------------------------------------------------
# r15-002: a file with a sharp trailing edge takes the blunt base, with a warning
# ---------------------------------------------------------------------------------------------


def raw_wing(items):
    return {
        "type": "lifting_surface",
        "planform": {"semispan_mm": 600.0, "taper": 0.5, "le_sweep_deg": 0.0},
        "stations": as_stations(items),
    }


def test_a_sharp_file_takes_the_blunt_base_and_the_input_says_so(tmp_path):
    write_file(tmp_path / "sharp.dat", closed=True)
    resolved = resolve_config(raw_wing([{"eta": 0.0, "file": "sharp.dat"}]), base=tmp_path)
    comp = unpack(resolved)["components"]["COMPONENT"]
    used = comp["sections"]["items"][0]["trailing_edge_used"]
    assert used == pytest.approx(comp["sections"]["trailing_edge_thickness"])
    assert used > 1e-3
    assert any(
        "sharp trailing edge" in w and "stations[0].section.file" in w for w in resolved["warnings"]
    )
    (section,) = bspline_sections(comp["sections"] | {"form": "bspline"})
    assert section.te_thickness() == pytest.approx(used, abs=1e-9)


def test_the_generator_names_the_station_as_the_user_wrote_it(tmp_path):
    """A generator message names ``stations``, never the resolver's ``sections.items``."""
    from cfdbender._errors import InputValueError
    from cfdbender.geometry._sections import section_from_item

    write_file(tmp_path / "sharp.dat", closed=True)
    resolved = resolve_config(raw_wing([{"eta": 0.0, "file": "sharp.dat"}]), base=tmp_path)
    comp = unpack(resolved)["components"]["COMPONENT"]
    warnings = []
    bspline_sections(comp["sections"] | {"form": "bspline"}, warnings)
    assert warnings and all("sections.items" not in w for w in warnings)
    assert any(w.startswith("stations[eta=0].section.file: sharp") for w in warnings)
    bspline = {"degree": 1, "knots": [0.0, 0.0, 1.0, 1.0], "poles": [[1.0, 0.0], [0.0, 0.0]]}
    with pytest.raises(InputValueError, match=r'^stations\[eta=0\.5\]\.section\.kind: "bspline"'):
        section_from_item({"eta": 0.5, "bspline": bspline})


def test_a_blunt_file_keeps_its_own_base_and_does_not_warn(tmp_path):
    write_file(tmp_path / "blunt.dat", closed=False)
    resolved = resolve_config(raw_wing([{"eta": 0.0, "file": "blunt.dat"}]), base=tmp_path)
    comp = unpack(resolved)["components"]["COMPONENT"]
    assert comp["sections"]["items"][0]["trailing_edge_used"] == pytest.approx(0.00252, abs=2e-5)
    assert not any("sharp trailing edge" in w for w in resolved["warnings"])
    (section,) = bspline_sections(comp["sections"] | {"form": "bspline"})
    assert section.te_thickness() == pytest.approx(0.00252, abs=2e-5)


def test_a_declared_zero_base_keeps_a_sharp_file_sharp(tmp_path):
    write_file(tmp_path / "sharp.dat", closed=True)
    sections = table([{"file": str(tmp_path / "sharp.dat")}], te=0.0)
    warnings = []
    (section,) = bspline_sections(sections, warnings)
    assert section.te_thickness() < 1e-4
    assert warnings == []


def test_a_bspline_row_in_the_cst_form_is_refused_with_the_reason():
    from cfdbender.geometry._sections import fit_or_read

    stored = bspline_sections(table([{"naca": "0012"}]))[0].to_stored()
    sections = {**table([{"bspline": stored}]), "form": "cst"}
    with pytest.raises(InputValueError, match=r'section\.kind: "bspline" resolves, but'):
        fit_or_read(sections)


def test_the_proposed_section_table_is_read_like_the_row_keys():
    stored = bspline_sections(table([{"naca": "2412"}]))[0].to_stored()
    body = {"degree": 5, "knots": stored["knots"], "poles": stored["poles"]}
    (via,) = bspline_sections(table([{"section": {"kind": "bspline", **body}}]))
    (plain,) = bspline_sections(table([{"bspline": body}]))
    assert via.poles == plain.poles
    (naca,) = bspline_sections(table([{"section": {"kind": "naca", "naca": "0012"}}]))
    assert naca.nose_radius() == pytest.approx(1.1019 * 0.12**2, rel=0.02)
    with pytest.raises(InputValueError, match="section.kind"):
        bspline_sections(table([{"section": {"kind": "nope"}}]))

"""Geometry: nacelle bounding sections (requirement 4) and the exact_loft tip cap.

The loft is checked against the closed form it is built from: section k is the tip section with
its thickness scaled by sqrt(1 - f^(2/e)) at the extension f of the cap length.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._airfoils import cst_airfoil
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._caps import cap_patch_grids, tip_cap
from cfdbender.geometry._distributions import blended_section, planform_values
from cfdbender.geometry._lifting import (
    bound_stations,
    cap_fraction,
    lifting_surface,
    section_models,
    thickness_station,
)


def configuration(**wing):
    data = {
        "components": {
            "N": {"type": "body"},
            "W": {
                "type": "lifting_surface",
                "planform": {"semispan_mm": 600.0, "taper": 0.5},
                **wing,
            },
        }
    }
    resolved = resolve_config(data)
    return resolved["warnings"], unpack(resolved)["components"]["W"]


def bays(component):
    return sorted({tuple(patch.bay) for patch in lifting_surface(component)})


def test_nacelle_bounds_add_two_sections_and_keep_the_shape():
    # the two bounds are iso-v lines of the one skin and cut lines of its faces' trims,
    # no longer edges of the generated faces (design delta C.1.2 and C.1.3).
    from cfdbender.geometry._caps import lifting_model

    _, plain = configuration()
    bound = {"name": "N", "eta_axis": 0.425, "half_width_mm": 75.0}
    warnings, bounded = configuration(engine=[bound])
    assert not [w for w in warnings if "nacelle_bounds" in w]
    assert bound_stations(bounded) == pytest.approx([0.3, 0.55])
    assert bays(bounded) == bays(plain)
    model = lifting_model(bounded)
    assert {model.lines["v"]["nacelle_N_in"], model.lines["v"]["nacelle_N_out"]} == {0.3, 0.55}
    cuts = {p[1] for t in model.trims if t.base == "skin" for piece in t.loop for p in piece}
    assert cuts == {0.0, 0.3, 0.55, 1.0}
    eta = np.linspace(0, 1, 9)
    for key in ("chord", "leading_edge_x"):
        assert planform_values(bounded, eta)[key] == pytest.approx(planform_values(plain, eta)[key])
    assert section_models(bounded["sections"]) == section_models(plain["sections"])


def test_nacelle_bound_on_no_component_is_reported_by_the_verb_and_skipped():
    bound = {"name": "pod", "eta_axis": 0.425, "half_width_mm": 75.0}
    warnings, component = configuration(engine=[bound])
    assert any(".engine.name" in text and "pod" in text for text in warnings)
    assert bound_stations(component) == []


def test_nacelle_bounds_bad_input_is_refused():
    for bad in (
        {"name": "N", "eta_axis": 0.55, "half_width_mm": -30.0},
        {"name": "N", "eta_axis": 0.2, "half_width_mm": 180.0},
        {"name": "N", "eta_axis": 0.85, "half_width_mm": 390.0},
    ):
        with pytest.raises(InputValueError, match="engine"):
            configuration(engine=[bad])


def test_nacelle_bounds_given_twice_share_a_station():
    pair = [
        {"name": "N", "eta_axis": 0.4, "half_width_mm": 60.0},
        {"name": "N", "eta_axis": 0.6, "half_width_mm": 60.0},
    ]
    _, component = configuration(engine=pair)
    assert bound_stations(component) == pytest.approx([0.3, 0.5, 0.7])


def loft_component(exponent=1.0):
    cap = {"kind": "exact_loft", "length": 1.0, "exponent": exponent}
    return configuration(cap={**cap, "sections": 8, "control_points": 15})[1]


def test_exact_loft_matches_its_closed_form():
    component = loft_component()
    models = section_models(component["sections"])
    section = blended_section(component["sections"], models, 1.0)
    maximum = thickness_station(component, models, 1.0)
    grids = [g for _, g in cap_patch_grids(component, section, maximum)]
    assert len(grids) == 2
    upper, lower = grids
    assert upper.shape == (15, 9, 3)
    # the published span includes the cap, so the cap starts at the skin's tip station.
    start = cap_fraction(component)
    p = planform_values(component, start)
    chord = float(p["chord"])
    top, bottom = cst_airfoil(
        maximum, section["upper"], section["lower"], section["trailing_edge_thickness"]
    )
    length = float((top[1] - bottom[1]) / 2)
    fraction = np.linspace(0, 1, 9)
    ratio = (upper[:, :, 2] - lower[:, :, 2]) / chord
    x = (upper[:, 0, 0] - float(p["leading_edge_x"])) / chord
    tip_up, tip_low = cst_airfoil(
        x, section["upper"], section["lower"], section["trailing_edge_thickness"]
    )
    assert ratio[:, 0] == pytest.approx(tip_up[:, 1] - tip_low[:, 1], abs=1e-9)
    for k in range(1, 9):
        assert ratio[:, k] == pytest.approx(ratio[:, 0] * np.sqrt(1 - fraction[k] ** 2), abs=1e-9)
    extension = (upper[:, :, 1] - 1.2 / 2 * start) / chord
    assert extension[0] == pytest.approx(length * fraction, abs=1e-12)
    assert upper[:, -1] == pytest.approx(lower[:, -1], abs=1e-12)


def test_exact_loft_cap_is_two_patches_and_a_wrong_exponent_changes_it():
    component = loft_component()
    patches = tip_cap(component)
    assert len(patches) == 2
    assert {p.family for p in patches} == {"tip_cap"}
    other = loft_component(exponent=2.0)
    models = section_models(component["sections"])
    section = blended_section(component["sections"], models, 1.0)
    maximum = thickness_station(component, models, 1.0)
    one = cap_patch_grids(component, section, maximum)[0][1]
    two = cap_patch_grids(other, section, maximum)[0][1]
    assert float(np.max(np.abs(one[:, 4] - two[:, 4]))) > 1e-4
    assert one[:, 0] == pytest.approx(two[:, 0], abs=1e-12)


def test_exact_loft_counts_come_from_the_input_and_bad_counts_are_refused():
    component = loft_component()
    assert component["tip_cap"]["sections"] == 8
    with pytest.raises(InputValueError, match="cap.sections"):
        configuration(cap={"type": "exact_loft", "sections": 1})
    with pytest.raises(InputValueError, match="cap.type"):
        configuration(cap={"type": "exact-loft"})


def trailing(component):
    return [p for p in lifting_surface(component) if p.family.startswith("trailing_")]


def test_base_strip_is_removed_warns_and_leaves_the_bicubic_strips():
    _, plain = configuration()
    warnings, stated = configuration(base_strip="linear")
    assert any(text.startswith("components.W.base_strip:") for text in warnings)
    assert "base_strip" not in stated
    assert {p.degree_u for p in trailing(stated)} == {3}
    assert len(trailing(stated)) == len(trailing(plain))
    warnings, _ = configuration(base_strips="linear")
    assert any("base_strips" in text for text in warnings)


def test_tip_cap_extent_sets_the_printed_reach_for_rounded_and_loft_caps():
    from cfdbender.geometry._boundary import unpack  # noqa: F401

    for kind in ("rounded", "exact_loft"):
        _, component = configuration(cap={"type": kind, "extent": 0.08})
        models = section_models(component["sections"])
        section = blended_section(component["sections"], models, 1.0)
        maximum = thickness_station(component, models, 1.0)
        grids = [g for _, g in cap_patch_grids(component, section, maximum)]
        # the published span includes the cap: it reaches 0.08 from its start to b / 2.
        top = max(float(np.max(g[..., 1])) for g in grids)
        assert top == pytest.approx(1.2 / 2, abs=1e-6)
        assert top - 1.2 / 2 * cap_fraction(component) == pytest.approx(0.08, abs=1e-6)


def test_tip_cap_extent_bad_input():
    with pytest.raises(InputValueError, match="together with cap.length"):
        configuration(cap={"type": "rounded", "extent": 0.08, "length": 2.0})
    with pytest.raises(InputValueError, match="extent"):
        configuration(cap={"type": "rounded", "extent": -0.1})
    warnings, component = configuration(cap={"type": "flat", "extent": 0.08})
    assert any("cap.extent" in text for text in warnings)
    assert "extent" not in component["tip_cap"]


@pytest.mark.parametrize("kind", ["flat", "rounded", "ridged", "none"])
def test_loft_counts_on_another_cap_are_warned_and_skipped(kind):
    warnings, _ = configuration(cap={"type": kind, "sections": 12, "control_points": 9})
    assert any("cap.sections" in text and kind in text for text in warnings)
    assert any("cap.control_points" in text for text in warnings)
    warnings, _ = configuration(cap={"type": "exact_loft", "sections": 12})
    assert not [text for text in warnings if "cap.sections" in text]

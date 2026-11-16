"""Geometry: body regions of the fuselage and nacelle layouts.

Oracles: the radius law of a body of revolution (a circle section has radius from the stations),
the tangent continuity of a surface that is one law, and counts that follow from the layout rule.
"""

import numpy as np
import pytest
from geom_inputs import body_input

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._bodies import (
    axisymmetric_radius,
    body_regions,
    region_plan,
)
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._continuity import audit_jumps
from cfdbender.geometry._patch import evaluate_patch

CIRCLE = {"shape": "circle", "radius": 1.0}
TAIL = {"boat_tail": {"x_cut": 0.97, "base_height": 0.3, "angle_limit_deg": 45.0}}


def fuselage(**extra):
    """A fuselage with a constant run from 0.2 to 0.8, a boat-tail and a base."""
    sections = [{"at": at, **CIRCLE} for at in (0.2, 0.5, 0.8)]
    return make_body(
        length=10.0,
        sections=sections,
        stations=[0.2, 0.5, 0.8],
        rear=TAIL,
        nose_exponent=0.5,
        **extra,
    )


def test_fuselage_is_cut_at_the_constant_run_the_boat_tail_and_the_base():
    plan = region_plan(fuselage())
    spans = [(round(r["x0"], 6), round(r["x1"], 6), r["constant"]) for r in plan["regions"]]
    assert [s[1:] for s in spans] == [(0.2, False), (0.8, True), (0.97, False)]
    assert spans[0][0] == pytest.approx(0.0, abs=1e-5)  # the cap takes the rest of the nose
    assert plan["degree"] == 3 and plan["rear"] == "boat_tail"


def test_fuselage_patches_follow_the_region_count():
    patches = body_regions(fuselage())
    roles = [p.role for p in patches]
    assert (roles.count("nose_cap"), roles.count("base"), roles.count("band")) == (3, 3, 6)
    constant = [p for p in patches if p.role == "band" and p.bay[0] == 0.2]
    assert all(p.degree_v == 1 and p.controls.shape[1] == 2 for p in constant)
    assert all(p.degree_u == 3 for p in patches)


def test_a_whole_body_has_four_faces_per_region():
    patches = body_regions(fuselage(symmetry="full"))
    roles = [p.role for p in patches]
    assert (roles.count("nose_cap"), roles.count("base"), roles.count("band")) == (6, 6, 12)


def test_the_skin_holds_the_law_within_the_tolerance_off_the_samples():
    component = fuselage()
    tolerance = 1e-4 * component["length"]
    worst = 0.0
    for patch in body_regions(component):
        if patch.role == "base":
            continue
        points = evaluate_patch(patch, 37)
        law = axisymmetric_radius(component, points[..., 0] / component["length"])
        worst = max(worst, float(np.max(np.abs(np.hypot(points[..., 1], points[..., 2]) - law))))
    assert 0.0 < worst <= 2 * tolerance


def test_a_tighter_tolerance_takes_more_spans_and_an_unreachable_one_is_refused():
    component = fuselage()
    loose = sum(p.controls.size for p in body_regions(component, tolerance=1e-2))
    tight = sum(p.controls.size for p in body_regions(component, tolerance=1e-5))
    assert tight > loose
    with pytest.raises(InputValueError, match="tolerance"):
        body_regions(component, tolerance=1e-13)


def test_seams_are_tangent_except_the_declared_crease_at_the_base():
    result = audit_jumps(body_regions(fuselage()))
    kinds = result["by_kind"]
    assert kinds["band|band"] < 0.05 and kinds["band|nose_cap"] < 0.05
    assert kinds["nose_cap|nose_cap"] < 0.05
    assert 45.0 < kinds["band|base"] < 90.0  # the boat-tail meets the flat base at a real angle
    assert result["max_jump_deg"] == kinds["band|base"] and result["unpaired_edges"] == []
    crease = [{"at": "band|base", "deg": kinds["band|base"]}]
    assert audit_jumps(body_regions(fuselage()), creases=crease)["failed"] == []


def test_the_cap_size_comes_from_the_law_not_from_the_first_station():
    sizes = []
    for first in (0.1, 0.2):
        sections = [{"at": at, **CIRCLE} for at in (first, 0.5, 0.8)]
        component = make_body(length=10.0, sections=sections, stations=[first, 0.5, 0.8])
        plan = region_plan(component, cap_fraction=0.01)
        sizes.append(float(axisymmetric_radius(component, plan["regions"][0]["x0"])))
    assert sizes == pytest.approx([0.01, 0.01], rel=1e-6)


def test_a_nacelle_is_cut_in_bays_of_degree_five_and_audited_without_error():
    component = fuselage()
    plan = region_plan(component, "nacelle")
    assert len(plan["regions"]) == 4 and not any(r["constant"] for r in plan["regions"])
    patches = body_regions(component, "nacelle")
    bands = [p for p in patches if p.role == "band"]
    assert len(bands) == 8 and all(p.degree_u == p.degree_v == 5 for p in bands)
    result = audit_jumps(patches)
    assert result["by_kind"]["band|band"] < 0.05


def test_listed_stations_cut_a_region_and_bad_input_is_refused():
    component = fuselage()
    assert len(region_plan(component, regions=[0.35])["regions"]) == 4
    with pytest.raises(InputValueError, match="regions"):
        region_plan(component, regions=[0.99])
    with pytest.raises(InputValueError, match="layout"):
        region_plan(component, "fusalage")
    with pytest.raises(InputValueError, match="cap_fraction"):
        region_plan(component, cap_fraction=0.0)
    with pytest.raises(InputValueError, match="tolerance"):
        body_regions(component, tolerance=-1.0)


def make_body(**extra):
    raw = body_input(
        {"type": "body", "length": 20.0, "sections": [{"at": "middle", **CIRCLE}], **extra}
    )
    return unpack(resolve_config(raw))["components"]["COMPONENT"]


@pytest.mark.parametrize(("top", "bottom"), [(4.0, 3.0), (6.0, 2.0)])
def test_a_squarish_superellipse_body_is_fitted_within_its_tolerance(top, bottom):
    """an exponent above 2 has an infinite slope in the superellipse angle.

    The faces follow the polar angle instead, so the fit converges and every face point lies on
    the section's superellipse within the tolerance.
    """
    from cfdbender.geometry._bodies import body_profile, body_station_table

    section = {"shape": "superellipse", "width": 0.25, "height": 0.2, "n_top": top,
               "n_bottom": bottom}  # fmt: skip
    component = make_body(
        length=12.0,
        sections=[{"at": at, **section} for at in (0.2, 0.5, 0.8)],
        stations=[0.2, 0.5, 0.8],
        rear={"boat_tail": {"x_cut": 0.97, "base_height": 0.1, "angle_limit_deg": 25.0}},
        nose_exponent=0.5,
    )
    tolerance = 1e-4 * 12.0
    patches = body_regions(component, tolerance=tolerance)
    table = body_station_table(component)
    for patch in (p for p in patches if p.role == "band"):
        points = evaluate_patch(patch, 9).reshape(-1, 3)
        width, height, centre, n_top, n_bottom = np.moveaxis(
            body_profile(component, table, points[:, 0] / 12.0), -1, 0
        )
        dz = points[:, 2] - centre
        n = np.where(dz >= 0, n_top, n_bottom)
        level = np.abs(points[:, 1] / width) ** n + np.abs(dz / height) ** n
        # a distance d off the curve moves the level by about n d / (half size)
        assert np.max(np.abs(level - 1.0)) < 4 * top * tolerance / 0.1

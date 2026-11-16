"""Geometry: bluff skin, jump audit and body features.

Oracles are closed forms (a sphere, two planes at a known angle, a cone of known half angle) and
the definition of the requirement (surface equals unsplit surface, tangent plane continuous across
every shared edge), never the output under test.
"""

import numpy as np
import pytest
from geom_inputs import body_input

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._bodies import body, body_points, body_profile, body_station_table
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._continuity import audit_jumps, bluff_jump_check, edge_jumps
from cfdbender.geometry._patch import (
    Patch,
    evaluate_patch,
    interpolate_patch,
    interpolate_polynomial_strip,
)
from cfdbender.geometry._skin import bay_cuts, skin, split_along

CIRCLE = {"at": "middle", "shape": "circle", "radius": 1.2}
#: The turn at the rim of the flat base of ``boom_like``, as the audit measures it, declared.
BASE_CREASE = [{"at": "band|base", "deg": 87.0}]


def resolved(raw, base=None):
    data = body_input({"type": "body", "length": 20.0, "sections": [CIRCLE], **raw})
    out = resolve_config(data, base=base)
    return out


def make_body(**extra):
    return unpack(resolved(extra))["components"]["COMPONENT"]


def three_stations(**extra):
    """A fuselage with three stations of different size and a rounded rear."""
    sections = [
        {"at": 0.15, "shape": "circle", "radius": 0.8},
        {"at": 0.45, "shape": "circle", "radius": 1.3},
        {"at": 0.75, "shape": "circle", "radius": 0.9},
    ]
    return make_body(
        sections=sections, stations=[0.15, 0.45, 0.75], **{"rear_exponent": 0.5, **extra}
    )


def boom_like(**extra):
    """A long slender boom, squarish in section, with a boat-tail rear."""
    boom = {
        "shape": "superellipse",
        "width": 0.25,
        "height": 0.2,
        "n_top": 4.0,
        "n_bottom": 3.0,
    }
    sections = [{"at": 0.2, **boom}, {"at": 0.5, **boom}, {"at": 0.8, **boom, "width": 0.2}]
    tail = {"boat_tail": {"x_cut": 0.97, "base_height": 0.1, "angle_limit_deg": 25.0}}
    return make_body(
        length=12.0,
        sections=sections,
        stations=[0.2, 0.5, 0.8],
        nose_exponent=0.5,
        rear=tail,
        **extra,
    )


def test_skin_pieces_equal_the_unsplit_surface_to_a_nanometre():
    component = three_stations()
    table = body_station_table(component)
    axial = np.concatenate([np.linspace(0.15, 0.45, 17)[:-1], np.linspace(0.45, 0.75, 17)])
    theta = np.linspace(0.0, np.pi / 3, 17)
    points = body_points(component, table, axial[None, :], theta[:, None])
    parameters = (axial - axial[0]) / (axial[-1] - axial[0])
    whole = skin(points, parameters_v=parameters)
    pieces = split_along(whole, "v", bay_cuts([0.15, 0.45, 0.75]))
    assert len(pieces) == 2
    cut = float(bay_cuts([0.15, 0.45, 0.75])[0])
    count = 81
    reference = evaluate_patch(whole, 2 * count - 1)
    for piece, lo, hi in [(pieces[0], 0.0, cut), (pieces[1], cut, 1.0)]:
        sampled = evaluate_patch(piece, count)
        # the piece parameter s sits at lo + s (hi - lo) of the whole
        index = np.round((lo + np.linspace(0, 1, count) * (hi - lo)) * (2 * count - 2)).astype(int)
        error = np.max(np.abs(sampled - reference[::2][:, index]))
        assert error < 1e-9


def test_skin_of_a_cylinder_is_uniform_in_the_axial_coordinate_after_the_cut():
    x = np.concatenate([np.linspace(0, 1, 17)[:-1], np.linspace(1, 4, 17)])
    angle = np.linspace(0, np.pi / 2, 9)
    grid = np.stack(
        [np.stack([x, np.full_like(x, np.cos(a)), np.full_like(x, np.sin(a))], -1) for a in angle]
    )
    whole = skin(grid, parameters_v=x / x[-1])
    assert np.max(np.abs(evaluate_patch(whole, 41)[0, :, 0] - np.linspace(0, 4, 41))) < 1e-12
    first, second = split_along(whole, "v", bay_cuts([0, 1, 4]))
    assert np.max(np.abs(evaluate_patch(first, 11)[0, :, 0] - np.linspace(0, 1, 11))) < 1e-12
    assert np.max(np.abs(evaluate_patch(second, 11)[0, :, 0] - np.linspace(1, 4, 11))) < 1e-12


def test_split_refuses_cuts_out_of_order_or_outside():
    patch = interpolate_patch(np.random.default_rng(2).normal(size=(6, 9, 3)))
    for cuts in ([0.6, 0.4], [0.0], [1.2], [0.5, 0.5]):
        with pytest.raises(InputValueError):
            split_along(patch, "v", cuts)


def test_a_body_with_three_stations_passes_the_jump_audit_under_one_degree():
    patches = body(three_stations())
    result = audit_jumps(patches, limit_deg=1.0)
    assert result["failed"] == []
    assert result["max_jump_deg"] < 1.0
    assert bluff_jump_check(result)["status"] == "passed"
    # the band seams are the same surface: the skin is smooth across the cut
    assert result["by_kind"]["band|band"] < 1e-3


def test_a_boat_tail_and_boom_like_body_passes_with_the_base_rim_declared():
    patches = body(boom_like())
    result = audit_jumps(patches, limit_deg=1.0, creases=BASE_CREASE)
    assert result["failed"] == []
    assert result["unmatched_creases"] == []
    assert result["max_jump_deg"] < 1.0
    # the rim of a flat base is a crease by design and the table says how large
    assert 60.0 < result["by_kind"]["band|base"] < 120.0
    undeclared = audit_jumps(patches, limit_deg=1.0)
    assert undeclared["failed"]
    assert bluff_jump_check(undeclared)["status"] == "failed"


def test_a_cone_apex_fails_the_audit_which_is_what_it_is():
    cone = three_stations(rear_exponent=1.0)
    result = audit_jumps(body(cone), limit_deg=1.0)
    assert result["failed"]
    assert result["max_jump_deg"] > 20.0


def make_planes(angle_deg):
    """Two unit squares meeting along the line x = 1 with a fold of ``angle_deg``."""
    s = np.linspace(0, 1, 6)
    t = np.linspace(0, 1, 6)
    first = np.stack([*np.meshgrid(s, t, indexing="ij"), np.zeros((6, 6))], -1)
    a = np.radians(angle_deg)
    second = np.stack(
        [
            1 + s[:, None] * np.cos(a) + 0 * t[None, :],
            np.tile(t, (6, 1)),
            s[:, None] * np.sin(a) + 0 * t[None, :],
        ],
        -1,
    )
    return [
        Patch.from_dict({**interpolate_patch(first).to_dict(), "role": "x"}),
        Patch.from_dict({**interpolate_patch(second).to_dict(), "role": "y"}),
    ]


@pytest.mark.parametrize("angle", [0.0, 10.0, 30.0, 90.0, 150.0])
def test_the_audit_measures_a_fold_of_known_angle(angle):
    result = audit_jumps(make_planes(angle), limit_deg=1.0, creases=[{"at": "x|y", "deg": angle}])
    assert len(result["edges"]) == 1
    assert result["edges"][0]["jump_deg"] == pytest.approx(angle, abs=1e-6)
    assert result["failed"] == []


def test_a_fold_over_the_limit_fails_and_a_crease_at_the_right_angle_allows_it():
    planes = make_planes(30.0)
    assert audit_jumps(planes)["failed"] == [0]
    assert audit_jumps(planes, creases=[{"at": "x|y", "deg": 28.5}])["failed"] == [0]
    allowed = audit_jumps(planes, creases=[{"at": "x|y", "deg": 30.5}])
    assert allowed["failed"] == []
    assert allowed["edges"][0]["allowed_deg"] == 30.5


@pytest.mark.parametrize("declared", [180.0, 90.0])
def test_a_crease_cannot_declare_away_a_larger_or_smaller_fold(declared):
    # a crease states the turn it has: a 60 degree fold under a 180 or 90 degree crease fails
    result = audit_jumps(make_planes(60.0), limit_deg=1.0, creases=[{"at": "x|y", "deg": declared}])
    assert result["failed"] == [0]
    assert bluff_jump_check(result)["status"] == "failed"


def test_a_crease_on_smooth_seams_does_not_hide_them():
    # declaring every band seam a fold must not pass a body: its smooth seams are not folds
    result = audit_jumps(body(three_stations()), creases=[{"at": "band|band", "deg": 180.0}])
    assert result["failed"]
    check = bluff_jump_check(result)
    assert check["status"] == "failed" and check["measured"] > 179.0


def test_a_named_shared_edge_the_audit_cannot_pair_fails():
    # the second plane runs along the shared edge at another parameter speed: same curve, a
    # 60 degree fold, and no sample of one side meets the other at an equal parameter
    first = make_planes(60.0)[0]
    s = np.linspace(0, 1, 6)
    a = np.radians(60.0)
    across, along = np.meshgrid(s, s**2, indexing="ij")
    grid = np.stack([1 + across * np.cos(a), along, across * np.sin(a)], -1)
    warped = Patch.from_dict(
        {**interpolate_patch(grid).to_dict(), "role": "y", "edge_ids": ["seam", "e1", "e2", "e3"]}
    )
    named = Patch.from_dict({**first.to_dict(), "edge_ids": ["f0", "seam", "f2", "f3"]})
    result = audit_jumps([named, warped], limit_deg=1.0)
    assert result["edges"] == []
    assert result["unpaired_edges"] == ["seam"]
    check = bluff_jump_check(result)
    assert check["status"] == "failed" and check["where"] == {"edge": "seam"}
    # the same pair with matching speeds is measured and fails on its turn
    measured = audit_jumps(make_planes(60.0), limit_deg=1.0)
    assert measured["unpaired_edges"] == [] and measured["failed"] == [0]


@pytest.mark.parametrize("samples", [3, 5, 6])
def test_the_audit_reads_the_degree_of_each_patch(samples):
    # a trailing-edge strip is cubic across and of degree samples - 1 along: a fold of 30
    # degrees between two such strips reads 30, whatever the degree
    s, t = np.linspace(0, 1, 6), np.linspace(0, 1, samples)
    across, along = np.meshgrid(s, t, indexing="ij")
    a = np.radians(30.0)
    grids = [
        np.stack([across, along, 0 * across], -1),
        np.stack([1 + across * np.cos(a), along, across * np.sin(a)], -1),
    ]
    strips = [
        Patch.from_dict({**interpolate_polynomial_strip(g).to_dict(), "role": r})
        for g, r in zip(grids, "xy", strict=True)
    ]
    assert strips[0].degree_v == samples - 1
    result = audit_jumps(strips, limit_deg=1.0)
    assert [round(e["jump_deg"], 6) for e in result["edges"]] == [30.0]


def test_a_crease_that_matches_no_edge_is_reported():
    result = audit_jumps(make_planes(0.0), creases=[{"at": "nowhere", "deg": 5.0}])
    assert result["unmatched_creases"] == [{"at": "nowhere", "deg": 5.0}]


@pytest.mark.parametrize(
    "bad", [{"at": 3, "deg": 5}, {"deg": 5}, {"at": "x|y"}, {"at": "x|y", "deg": "5"}]
)
def test_the_audit_refuses_a_malformed_crease(bad):
    with pytest.raises(InputValueError, match="creases"):
        audit_jumps(make_planes(0.0), creases=[bad])


def test_the_audit_refuses_bad_limits_and_counts():
    planes = make_planes(0.0)
    with pytest.raises(InputValueError):
        audit_jumps(planes, limit_deg=-1.0)
    with pytest.raises(InputValueError):
        edge_jumps(planes, count=2)
    with pytest.raises(InputValueError):
        edge_jumps(planes, tolerance=0.0)
    assert edge_jumps([]) == []


def test_a_disjoint_pair_has_no_shared_edge():
    first, second = make_planes(0.0)
    shifted = Patch.from_dict(
        {**second.to_dict(), "controls": (np.asarray(second.controls) + 5.0).tolist()}
    )
    assert edge_jumps([first, shifted]) == []


def test_edge_names_agree_with_the_geometry_everywhere():
    patches = body(three_stations())
    matched = edge_jumps(patches)
    seen = {}
    for record in matched:
        a = patches[record["a"]].edge_ids["u0 u1 v0 v1".split().index(record["edge_a"])]
        b = patches[record["b"]].edge_ids["u0 u1 v0 v1".split().index(record["edge_b"])]
        assert a == b, record
        assert seen.setdefault(a, (record["a"], record["b"])) == (record["a"], record["b"])
    names = [n for p in patches for n in p.edge_ids]
    shared = {n for n in names if names.count(n) == 2}
    assert shared == set(seen)
    assert all(names.count(n) in (1, 2) for n in names)
    assert all(p.role in {"nose_cap", "rear_cap", "band"} for p in patches)


def test_patch_role_and_edge_ids_survive_the_plain_data_form():
    patch = body(three_stations())[0]
    again = Patch.from_dict(patch.to_dict())
    assert (again.role, again.edge_ids) == (patch.role, patch.edge_ids)
    assert patch.role == "nose_cap" and len(patch.edge_ids) == 4
    legacy = {k: v for k, v in patch.to_dict().items() if k not in ("role", "edge_ids")}
    assert Patch.from_dict(legacy).role == ""


def height_at(component, axial, theta=0.0):
    table = body_station_table(component)
    return body_points(component, table, np.asarray(axial, float), np.asarray(theta, float))[..., 2]


def test_the_boat_tail_closes_on_its_base_height_within_the_angle_limit():
    component = boom_like()
    length, last, cut = component["length"], 0.8, 0.97
    x = np.linspace(last, cut, 4001)
    top = height_at(component, x)
    # the last station holds height 0.2 and the base half-height is 0.1
    assert top[0] == pytest.approx(0.2, abs=1e-12)
    assert top[-1] == pytest.approx(0.1, abs=1e-12)
    slope = np.abs(np.diff(top) / np.diff(x * length))
    assert np.degrees(np.arctan(slope.max())) <= 25.0 + 1e-6
    # tangent to the straight run at the join, so the slope there is the run's own slope
    run = height_at(component, np.array([0.8 - 1e-4, 0.8]))
    closure = (top[1] - top[0]) / ((x[1] - x[0]) * length)
    assert abs(closure - np.diff(run)[0] / (1e-4 * length)) < 1e-3
    base = [p for p in body(component) if p.role == "base"]
    assert base
    assert all(np.allclose(np.asarray(p.controls)[..., 0], cut * length, atol=1e-9) for p in base)


def test_a_boat_tail_angle_limit_too_small_is_refused_at_resolve():
    tail = {"boat_tail": {"x_cut": 0.97, "base_height": 0.1, "angle_limit_deg": 1.0}}
    with pytest.raises(InputValueError, match="angle_limit_deg"):
        resolved({"rear": tail, "length": 12.0})


def test_the_angle_limit_holds_on_the_side_of_a_wide_body():
    # width 1.0 over height 0.3: the crown closes at about 12.5 degrees, the side at about 36.5
    flat = {"shape": "superellipse", "width": 1.0, "height": 0.3, "n_top": 2.0, "n_bottom": 2.0}
    sections = [{"at": x, **flat} for x in (0.2, 0.5, 0.8)]
    tail = {"boat_tail": {"x_cut": 0.95, "base_height": 0.05, "angle_limit_deg": 20.0}}
    raw = {"length": 10.0, "sections": sections, "stations": [0.2, 0.5, 0.8], "rear": tail}
    with pytest.raises(InputValueError, match="angle_limit_deg"):
        resolved(raw)
    tail["boat_tail"]["angle_limit_deg"] = 40.0
    component = unpack(resolved(raw))["components"]["COMPONENT"]
    x = np.linspace(0.8, 0.95, 3001)
    side = body_points(component, body_station_table(component), x, np.pi / 2)[..., 1]
    slope = np.max(np.abs(np.diff(side) / np.diff(x * 10.0)))
    assert 30.0 < np.degrees(np.arctan(slope)) <= 40.0 + 1e-6


@pytest.mark.parametrize("x_cut", [0.5, 0.75])
def test_a_boat_tail_must_start_beyond_the_last_station(x_cut):
    tail = {"boat_tail": {"x_cut": x_cut, "base_height": 0.1, "angle_limit_deg": 80.0}}
    with pytest.raises(InputValueError, match="x_cut"):
        resolved({"rear": tail})


def test_boat_tail_forms_that_are_not_a_table_are_refused():
    for rear in ("sphere", {"boat_tail": 3}):
        with pytest.raises(InputValueError, match="tail"):
            resolved({"rear": rear})
    # an unknown key inside the tail table is warned and skipped, not refused
    assert any("tail.tail: unknown key" in w for w in resolved({"rear": {"tail": {}}})["warnings"])


RINGS = (
    "x,width,height,centre_height,n_top,n_bottom\n"
    "0.2,0.25,0.2,0,4,3\n0.5,0.25,0.2,0,4,3\n0.8,0.2,0.2,0,4,3\n"
)


def test_a_rings_file_gives_the_sections_and_stations(tmp_path):
    (tmp_path / "rings.csv").write_text(RINGS, encoding="utf-8")
    result = resolved({"rings_file": "rings.csv", "length": 12.0}, base=tmp_path)
    from_file = unpack(result)["components"]["COMPONENT"]
    inline_sections = [
        {"at": x, "shape": "superellipse", "width": w, "height": 0.2, "n_top": 4.0, "n_bottom": 3.0}
        for x, w in [(0.2, 0.25), (0.5, 0.25), (0.8, 0.2)]
    ]
    inline = make_body(length=12.0, sections=inline_sections, stations=[0.2, 0.5, 0.8])
    assert from_file["stations"] == [0.2, 0.5, 0.8]
    got, want = body(from_file), body(inline)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert np.max(np.abs(a.controls - b.controls)) < 1e-12


def test_a_rings_file_replacing_given_sections_is_warned(tmp_path):
    (tmp_path / "rings.csv").write_text(RINGS, encoding="utf-8")
    sections = [{"at": "middle", "shape": "circle", "radius": 3.0}]
    result = resolved({"rings_file": "rings.csv", "sections": sections}, base=tmp_path)
    assert any("replaced by rings_file" in w for w in result["warnings"])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "x,width\n0.2,0.1\n0.4,0.1\n0.6,0.1\n",
        "x,width,height\n0.2,0.1,abc\n0.4,0.1,0.1\n0.6,0.1,0.1\n",
        "x,width,height\n0.2,0.1,0.1\n0.4,0.1,0.1\n",
        "x,width,height\n0.4,0.1,0.1\n0.2,0.1,0.1\n0.6,0.1,0.1\n",
        "x,width,height\n0.0,0.1,0.1\n0.4,0.1,0.1\n0.6,0.1,0.1\n",
    ],
)
def test_a_bad_rings_file_is_warned_and_skipped_at_resolve(tmp_path, text):
    (tmp_path / "rings.csv").write_text(text, encoding="utf-8")
    result = resolved({"rings_file": "rings.csv"}, base=tmp_path)
    assert any(w.startswith("rings_file") and "skipped" in w for w in result["warnings"])
    assert unpack(result)["components"]["COMPONENT"]["sections"] == [CIRCLE]


def test_a_missing_rings_file_is_refused():
    with pytest.raises(InputValueError, match="rings_file"):
        resolved({"rings_file": "no_such_rings.csv"})


def test_a_whole_body_closes_about_its_axis_and_has_no_free_edge():
    component = make_body(symmetry="full", rear_exponent=0.5)
    assert component["azimuths"][-1] == 360.0
    table = body_station_table(component)
    theta = np.linspace(0.1, 3.0, 9)
    up = body_points(component, table, np.array(0.5), theta)
    down = body_points(component, table, np.array(0.5), 2 * np.pi - theta)
    assert np.allclose(up[:, 1], -down[:, 1]) and np.allclose(up[:, 2], down[:, 2])
    assert np.all(up[:, 1] > 0)
    patches = body(component)
    result = audit_jumps(patches, limit_deg=1.0)
    assert result["free_edges"] == 0
    assert result["failed"] == []
    ys = np.concatenate([np.asarray(p.controls)[..., 1].ravel() for p in patches])
    assert ys.min() < -1.0 and ys.max() > 1.0


def test_a_straight_angle_corner_is_not_read_as_a_fold():
    """the region caps of a whole body meet at the pole on a straight-angle corner.

    The inward tangent there runs along the edge, so the turn is undefined; the audit read it as
    a 180 deg fold. Every other sample of those seams is under 0.06 deg (measured).
    """
    from cfdbender.geometry._bodies import body_regions

    patches = body_regions(make_body(symmetry="full", rear_exponent=0.5))
    result = audit_jumps(patches, limit_deg=1.0)
    assert result["failed"] == []
    assert result["max_jump_deg"] < 0.5
    # a real fold of the same set is still a fold
    folded = make_planes(90.0)
    assert audit_jumps(folded, limit_deg=1.0)["failed"]


def test_an_off_centre_whole_body_keeps_its_offset():
    section = {
        "at": "middle",
        "shape": "superellipse",
        "width": 0.3,
        "height": 0.2,
        "centre_height": 0.7,
        "n_top": 2.0,
        "n_bottom": 2.0,
    }
    component = make_body(symmetry="full", sections=[section], rear_exponent=0.5)
    table = body_station_table(component)
    ring = body_points(component, table, np.array(0.5), np.linspace(0, 2 * np.pi, 73))
    assert ring[:, 2].min() == pytest.approx(0.5, abs=1e-9)
    assert ring[:, 2].max() == pytest.approx(0.9, abs=1e-9)
    assert ring[:, 1].min() == pytest.approx(-0.3, abs=1e-9)
    assert audit_jumps(body(component))["failed"] == []


def test_a_half_body_stays_on_one_side_of_the_plane():
    patches = body(make_body())
    ys = np.concatenate([np.asarray(p.controls)[..., 1].ravel() for p in patches])
    assert ys.min() > -1e-6


@pytest.mark.parametrize(
    "extra",
    [
        {"symmetry": "quarter"},
        {"symmetry": "full", "azimuths": [0.0, 60.0, 120.0, 180.0]},
        {"azimuths": [0.0, 90.0, 180.0, 270.0, 360.0]},
        {"nose": "sphere"},
    ],
)
def test_symmetry_and_nose_misuse_is_refused(extra):
    with pytest.raises(InputValueError):
        resolved(extra)


def test_an_open_end_has_no_cap_and_its_rim_is_free():
    component = make_body(nose="open", rear="open")
    patches = body(component)
    assert {p.role for p in patches} == {"band"}
    assert audit_jumps(patches)["failed"] == []
    names = [n for p in patches for n in p.edge_ids]
    rims = [n for n in names if n.startswith("ring") and names.count(n) == 1]
    halves = 2 * (len(component["azimuths"]) - 1)
    assert len(rims) == 2 * halves


def test_an_open_front_shares_one_rim_with_its_parent():
    parent = make_body(rear="open")
    child = make_body(nose="open")
    first, second = body(parent), body(child)
    # slide the child so its first ring sits on the parent's last ring
    shift = (0.75 - 0.15) * 20.0
    moved = []
    for p in second:
        controls = np.asarray(p.controls).copy()
        controls[..., 0] += shift
        moved.append(Patch.from_dict({**p.to_dict(), "controls": controls.tolist()}))
    pairs = [r for r in edge_jumps(first + moved) if (r["a"] < len(first)) != (r["b"] < len(first))]
    assert len(pairs) == 2 * (len(parent["azimuths"]) - 1)
    assert all(r["kind"] == "band|band" for r in pairs)


def test_a_boat_tail_after_a_sloping_run_has_no_kink_at_the_join():
    tail = {"boat_tail": {"x_cut": 0.95, "base_height": 0.3, "angle_limit_deg": 40.0}}
    component = three_stations(rear=tail)
    rim = [{"at": "band|base", "deg": 81.5}]
    result = audit_jumps(body(component), limit_deg=1.0, creases=rim)
    assert result["failed"] == []
    assert result["by_kind"]["band|band"] < 0.5
    run = height_at(component, np.array([0.75 - 1e-5, 0.75]))
    assert np.diff(run)[0] < -1e-6  # the run is sloping, so a zero start slope would be a kink
    before = np.diff(height_at(component, np.array([0.75 - 2e-5, 0.75])))[0]
    after = np.diff(height_at(component, np.array([0.75, 0.75 + 2e-5])))[0]
    assert abs(after - before) < 1e-4 * abs(before)


def test_a_boat_tail_is_tangent_in_width_and_centre_too():
    # the run narrows and drops at the last station while its height holds: a law that scales
    # width and centre by the height ratio starts them flat and kinks the side and the crown
    section = {"shape": "superellipse", "height": 1.0, "n_top": 2.0, "n_bottom": 2.0}
    sections = [
        {"at": 0.2, **section, "width": 1.0, "centre_height": 0.0},
        {"at": 0.5, **section, "width": 1.0, "centre_height": 0.0},
        {"at": 0.8, **section, "width": 0.6, "centre_height": -0.3},
    ]
    tail = {"boat_tail": {"x_cut": 0.97, "base_height": 0.2, "angle_limit_deg": 45.0}}
    component = make_body(length=10.0, sections=sections, stations=[0.2, 0.5, 0.8], rear=tail)
    table = body_station_table(component)
    step = 2e-6
    left = body_profile(component, table, np.array([0.8 - step, 0.8]))
    right = body_profile(component, table, np.array([0.8, 0.8 + step]))
    before, after = np.diff(left, axis=0)[0] / step, np.diff(right, axis=0)[0] / step
    for index in (0, 1, 2):  # width, height, centre height
        assert abs(after[index] - before[index]) < 1e-3, (index, before, after)
    assert abs(before[0]) > 0.1 and abs(before[2]) > 0.1  # a real slope, so flat would be a kink

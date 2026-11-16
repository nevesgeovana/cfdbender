"""Geometry: the measured checks of the junction sub-steps, pure part.

Written from the junction stages C, D and E and the S10 row of the stage table, before the
code. Each test holds a YES fixture that passes at the stated value and a NO fixture built
synthetically from the defect the check guards; a NO that passed would be a defect of the
check. Every fixture is made here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfdbender.geometry import _junctions as jn
from cfdbender.geometry.stage_table import STAGE_ROWS

# ---- the stage table: every sub-step of C, D and E is a row with its named checks ----------

#: Fixture id -> (stage table row, the named check of that row).
HANDOFF = {
    "BJ1": ("S10.C1", "join.shared_pairs"),
    "BJ2": ("S10.C2", "join.written_pairs"),
    "BJ3": ("S10.C3", "join.edge_on_surface"),
    "BJ4": ("S10.C4", "join.group_free_edges"),
    "BJ5": ("S10.C5", "piece.regenerated"),
    "BJ6": ("S10.C6", "piece.symmetry_edges"),
    "BJ7": ("S10.C7", "junction.free_edges_classified"),
    "BJ8": ("S10.C8", "join.rim_inside"),
    "BJ9": ("S10.C9", "join.opening_on_partner"),
    "S1": ("S10.D1", "assembly.face_count"),
    "S2": ("S10.D2", "reimport.faces"),
    "S3": ("S10.D3", "join.crease_gap"),
    "S4": ("S10.D4", "junction.curve_gap"),
    "S5": ("S10.D5", "junction.projected_length"),
    "S6": ("S10.D6", "junction.on_host_loops"),
    "S8": ("S10.D8", "removal.area"),
    "S9": ("S10.D9", "write.read_back"),
    "S10": ("S10.D10", "sew.free_introduced"),
    "T1": ("S10.E1", "group.junction_length"),
    "T2": ("S10.E2", "assembly.clearance"),
}


def test_geom_every_handoff_substep_is_a_row_of_the_stage_table_with_its_check():
    rows = {row.stage: row for row in STAGE_ROWS}
    for handoff, (stage, check) in HANDOFF.items():
        assert stage in rows, handoff
        assert rows[stage].kernel, stage
        assert check in {cid for cid, _ in rows[stage].checks}, (handoff, stage)


def test_geom_no_check_id_is_folded_under_two_substeps():
    """A check id names one sub-step only, so a failure names the step (stage table rule)."""
    seen = {}
    for row in STAGE_ROWS:
        if not row.stage.startswith("S10."):
            continue
        for cid, _ in row.checks:
            assert cid not in seen, (cid, seen.get(cid), row.stage)
            seen[cid] = row.stage


def test_geom_the_substeps_run_in_the_order_of_execution():
    order = [row.stage for row in STAGE_ROWS if row.stage.startswith("S10.")]
    # pieces first (load, regenerate, symmetry, rims, openings, alone), then the cut and the
    # group, the projection, the split, the removal and its coverage, the join written and sewn,
    # the whole written and sewn, and the clearance of the separate pieces
    expected = [
        "S10.D1", "S10.C5", "S10.C6", "S10.C8", "S10.C9", "S10.D2", "S10.C7", "S10.D3",
        "S10.D4", "S10.E1", "S10.D5", "S10.D6", "S10.D8", "S10.D8b", "S10.C1", "S10.C2",
        "S10.C3", "S10.C4", "S10.D9", "S10.D10", "S10.E2",
        "S10.FW1", "S10.FW2", "S10.FW3", "S10.FW4",
        "S10.F0", "S10.F1", "S10.F2", "S10.F3", "S10.F4", "S10.F5", "S10.F6", "S10.F7",
    ]  # fmt: skip
    assert order == expected


# ---- the verdict of a measured check -------------------------------------------------------


def test_geom_a_failed_check_names_its_class_and_cause():
    ok = jn.measured("junction.curve_gap", 4.59e-6, jn.JUNCTION_GAP_M, unit="m", cause="BUG: x")
    assert ok["status"] == "passed"
    bad = jn.measured("junction.curve_gap", 6e-6, jn.JUNCTION_GAP_M, unit="m", cause="BUG: off")
    assert bad["status"] == "failed"
    assert "class BUG: off" in bad["note"]
    minimum = jn.measured("group.junction_length", 0.01, 0.1, unit="m", bound="min", cause="x")
    assert minimum["status"] == "failed"


# ---- distances and chains (S4, S5, S6) -----------------------------------------------------


def _arc(n, radius=1.0):
    t = np.linspace(0.0, math.pi / 2, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.zeros(n)])


def test_geom_s6_a_point_on_the_curve_is_on_it_measured_point_to_polyline():
    """T-S6 YES: a J end on the boundary is at 0 from the exact curve (dense polyline).

    NO (the bug of run 2): the same end tested against a coarse polyline of the boundary reads
    far off, so the check must use the exact curve; and a J truly off the loop must fail.
    """
    exact = _arc(20001, radius=5.0)
    on = _arc(7, radius=5.0)[2:5]
    assert jn.distance_to_polyline(on, exact).max() <= jn.J_ON_LOOP_M
    coarse = _arc(4, radius=5.0)
    assert jn.distance_to_polyline(on, coarse).max() > jn.J_ON_LOOP_M
    off = on * (1.0 + 1e-5)
    assert jn.distance_to_polyline(off, exact).max() > jn.J_ON_LOOP_M


def test_geom_s6_nearest_sample_is_not_point_to_polyline():
    """Run 4: nearest-sample distance read 0.05 mm where point to polyline reads 0."""
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    point = np.array([[0.5, 0.0, 0.0]])
    assert jn.distance_to_polyline(point, line)[0] == pytest.approx(0.0, abs=1e-15)


def test_geom_s4_a_closed_chain_has_no_lonely_end_and_an_open_one_has_two():
    square = [
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1e-6)),
    ]
    chains, lonely = jn.chain(square, jn.CHAIN_END_GAP_M)
    assert len(chains) == 1 and lonely == []
    broken = [*square[:3], ((0, 1, 0), (0, 0.001, 0))]  # an end 1 mm from its partner
    _, lonely = jn.chain(broken, jn.CHAIN_END_GAP_M)
    assert len(lonely) == 2


def test_geom_s4_loop_closure_measures_the_largest_gap_between_consecutive_curves():
    loop = [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)), ((1, 1, 0), (0, 0, 0))]
    assert jn.loop_closure(loop) == 0.0
    loop[1] = ((1, 0, 3e-6), (1, 1, 0))
    assert jn.loop_closure(loop) == pytest.approx(3e-6)


def test_geom_polyline_length():
    assert jn.polyline_length(_arc(20001, 2.0)) == pytest.approx(math.pi, rel=1e-8)


# ---- inside tests: parity and its fallback (A3, BJ8, coverage) -----------------------------


def _box_triangles(lo, hi, skip=()):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = {
        "xmin": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        "xmax": [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
        "ymin": [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],
        "ymax": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        "zmin": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        "zmax": [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],
    }
    out = []
    for name, (a, b, c, d) in quads.items():
        if name not in skip:
            out += [(a, b, c), (a, c, d)]
    return np.asarray(out, float)


def test_geom_a3_parity_tells_inside_from_outside_of_a_closed_hull():
    hull = _box_triangles((0, 0, 0), (2, 1, 1))
    points = np.array([[1.0, 0.5, 0.5], [3.0, 0.5, 0.5], [1.0, 0.5, 1.2], [0.5, 0.5, 0.5]])
    assert list(jn.parity_inside(points, hull)) == [True, False, False, True]


def test_geom_parity_fallback_where_the_closest_dominant_face_is_gone():
    """The PA-30 note: a point within reach of the hull is decided by the closest face of the
    dominant's current faces; where that face is gone the closest point is far and orients
    nothing, so beyond ``near`` the hull parity decides. NO: with the fallback disabled (near
    infinite), the point outside is voted inside."""
    hull = _box_triangles((0, 0, 0), (2, 1, 1))
    current = _box_triangles((0, 0, 0), (2, 1, 1), skip=("zmax",))
    outside = np.array([[1.0, 0.5, 1.0005]])  # 0.5 mm above the removed top face
    assert not jn.dominant_inside(outside, current, hull, near=1e-4)[0]
    assert jn.dominant_inside(outside, current, hull, near=math.inf)[0]
    inside = np.array([[1.0, 0.5, 0.99999]])
    assert jn.dominant_inside(inside, current, hull, near=1e-4)[0]


def test_geom_bj8_a_rim_inside_the_body_has_no_point_outside():
    """T-BJ8: a partner's rim lies inside the body (0 outside); NO: one point outside by 1 mm."""
    hull = _box_triangles((0, -1, -1), (4, 1, 1))
    rim = np.column_stack([np.linspace(1, 3, 800), np.full(800, 0.5), np.full(800, 0.9)])
    assert jn.points_outside(rim, hull, margin=jn.RIM_MARGIN_M) == 0
    rim[400, 2] = 1.001
    assert jn.points_outside(rim, hull, margin=jn.RIM_MARGIN_M) == 1


# ---- free edge classes (A2, A3, S2, BJ7, BJ9) ----------------------------------------------


def _segment(a, b, n=50):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def test_geom_free_edges_are_classified_symmetry_opening_buried_or_unexplained():
    """T-A2 and T-A3: an opening within 0.02 mm of a partner, a buried edge, a symmetry edge;
    NO: an opening at 0.05 mm from every partner is unexplained, and so is an edge outside."""
    partner = _box_triangles((0, -1, -1), (4, 1, 1))

    def distance(points):
        return jn.distance_to_triangles(points, partner)

    def inside(points):
        return jn.parity_inside(points, partner)

    edges = [
        _segment((1, 0, 1.5), (2, 0, 1.5)),  # on y = 0, outside the partner
        _segment((1, 0.5, 1.0 + 1.7e-5), (2, 0.5, 1.0 + 1.7e-5)),  # 0.017 mm off the partner
        _segment((1, 0.5, 0.5), (2, 0.5, 0.5)),  # inside the partner
        _segment((1, 0.5, 1.0 + 5e-5), (2, 0.5, 1.0 + 5e-5)),  # 0.05 mm off: unexplained
        _segment((1, 3.0, 3.0), (2, 3.0, 3.0)),  # far outside: unexplained
    ]
    classes = jn.classify_free_edges(edges, distance=distance, inside=inside, plane_axis=1)
    assert classes == ["symmetry", "opening", "buried", "unexplained", "unexplained"]
    assert jn.classify_free_edges(edges[:1], distance=distance, inside=inside, plane_axis=None) == [
        "unexplained"
    ]


# ---- shared curves of a join (BJ1, BJ2) ----------------------------------------------------


def test_geom_bj1_crease_pairs_are_found_and_their_gap_measured():
    """T-BJ1: two curves of different pieces joining the same vertices within 0.1 mm are a pair,
    with the gap before unification; NO: a crease 0.2 mm apart is not paired (A2 NO)."""
    a = _segment((0, 0, 0), (1, 0, 0))
    b = a + np.array([0.0, 0.0, 2.05e-5])
    far = a + np.array([0.0, 0.0, 2e-4])
    curves = [("Body", a), ("Blister", b), ("Blister", far)]
    pairs = jn.pair_curves(curves, near=jn.SHARED_PAIR_NEAR_M, end_gap=jn.SHARED_PAIR_END_M)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert pairs[0][2] == pytest.approx(2.05e-5)


def test_geom_bj2_written_pair_distance_is_zero_for_one_curve_and_not_for_an_offset():
    a = _segment((0, 0, 0), (1, 0.2, 0), 401)
    assert jn.curve_distance(a, a.copy()) <= jn.WRITTEN_PAIR_M
    assert jn.curve_distance(a, a + np.array([0, 0, 2.05e-5])) > jn.WRITTEN_PAIR_M


# ---- symmetry and regeneration (BJ5, BJ6) --------------------------------------------------


def test_geom_bj6_symmetry_edges_on_the_plane_and_the_mirror_match():
    """T-BJ6: plane edges at |y| <= 0.001 mm; NO: a half with a y offset of 0.01 mm."""
    plane = _segment((0, 0, 0), (1, 0, 0.3))
    assert jn.plane_offset([plane], axis=1) <= jn.SYMMETRY_M
    assert jn.plane_offset([plane + np.array([0, 1e-5, 0])], axis=1) > jn.SYMMETRY_M
    left = plane + np.array([0, 4e-7, 0])
    right = plane * np.array([1, -1, 1]) + np.array([0, -4e-7, 0])
    assert jn.mirror_match([left], [right], axis=1) <= jn.SYMMETRY_M
    assert jn.mirror_match([left], [right + np.array([0, 0, 1e-5])], axis=1) > jn.SYMMETRY_M


def test_geom_bj5_pole_change_of_a_regenerated_piece():
    """T-BJ5: a piece regenerated from its own inputs moves no pole (below 1e-6 mm); NO: a
    section moved by 1e-6 m reads as a change."""
    from cfdbender.geometry._patch import Patch

    def patch(controls):
        n, m = controls.shape[:2]
        return Patch("p", controls, (0.0, 1.0), (0.0, 1.0), (n, n), (m, m), n - 1, m - 1)

    controls = np.random.default_rng(1).normal(size=(4, 4, 3))
    assert jn.pole_change([patch(controls)], [patch(controls.copy())]) <= jn.REGENERATE_M
    moved = controls.copy()
    moved[2] += 1e-6
    assert jn.pole_change([patch(controls)], [patch(moved)]) > jn.REGENERATE_M
    with pytest.raises(ValueError, match="pole net"):
        jn.pole_change([patch(controls)], [patch(controls[:3])])


# ---- removal (S8), slivers and introduced edges (S10) --------------------------------------


def test_geom_s8_flood_stops_at_barriers_and_reaches_through_seams():
    """Faces 0-1-2 joined by seams, 2-3 by a junction: a seed at 0 keeps 0, 1, 2 only."""
    neighbours = {0: [(10, 1)], 1: [(10, 0), (11, 2)], 2: [(11, 1), (12, 3)], 3: [(12, 2)]}
    assert jn.flood(neighbours, seeds=[0], barriers={12}) == {0, 1, 2}
    assert jn.flood(neighbours, seeds=[0], barriers=set()) == {0, 1, 2, 3}


def test_geom_s8_zero_width_regions_are_dropped_and_counted():
    """T-S8 NO: a zero-width region (the fillet lower edge on the TE line) kept gives a zero-area
    face; regions under 0.05 mm mean width are dropped and logged with their count."""
    widths = [jn.mean_width(1.3916, 6.0), jn.mean_width(1e-8, 4.0), jn.mean_width(2e-7, 8.0)]
    keep, dropped = jn.drop_zero_width([0, 1, 2], widths)
    assert keep == [0] and dropped == [1, 2]


def test_geom_s10_free_edges_introduced_are_the_total_minus_the_inputs():
    assert jn.introduced(96, 96) == 0
    assert jn.introduced(104, 96) == 8
    assert jn.introduced(90, 96) == 0


def test_geom_slivers_are_counted_by_mean_width():
    widths = [jn.mean_width(0.019, 10.0), jn.mean_width(1.0, 4.0)]  # 3.8 mm and 0.5 m
    assert jn.slivers(widths) == [0]


# ---- the dominance table, applied to the members of one configuration ----------------------


def test_geom_dominance_plan_orders_rows_and_warns_and_skips_absent_members():
    rows = [
        {"dominant": ["F"], "dominated": ["H", "W"], "operation": "project"},
        {"dominant": ["B"], "dominated": ["W"], "operation": "cut"},
        {"dominant": ["B"], "dominated": ["N"], "operation": "cut"},
    ]
    warnings = []
    plan = jn.dominance_plan(rows, ["B", "W", "F", "H"], warnings)
    assert [(p.operation, p.dominant, p.dominated) for p in plan] == [
        ("cut", "B", "W"),
        ("project", "F", "H"),
        ("project", "F", "W"),
    ]
    assert warnings == ["dominance[2]: N is not a member of this configuration, row skipped"]
    assert jn.dominance_plan([], ["B"], []) == []


def test_geom_t2_the_stage_table_states_the_clearance_floor_the_check_uses():
    """assembly.clearance refuses a separate piece within the sewing tolerance
    (SEW_M); the stage table states that floor rather than merely "above 0"."""
    row = next(r for r in STAGE_ROWS if r.stage == "S10.E2")
    text = dict(row.checks)["assembly.clearance"]
    assert "SEW_M" in text and "1e-5 m" in text, text


def test_geom_s10_the_stage_table_says_new_slivers_fail():
    """sew.slivers fails on slivers the assembly introduces
    and lists those the pieces bring; the table said only "counted and listed"."""
    row = next(r for r in STAGE_ROWS if r.stage == "S10.D10")
    text = dict(row.checks)["sew.slivers"]
    assert "none beyond the pieces' own" in text, text

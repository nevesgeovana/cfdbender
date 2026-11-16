"""strategy 2 of , the full quad structured grid.

Pure block planning on synthetic blocks (topologies, ring counts, carry rows, valence 3 poles,
azimuth by count, E07 to E11 on planted input), the route wiring, and one kernel run on the
committed WB fixture of ``test_p0570_geometry_mesh.py`` (skipped without gmsh).
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry._mesh_blocks import (
    Block,
    azimuth_split,
    block_from_walk,
    carry_pairs_growth,
    carry_plan,
    check_poles,
    check_triangles,
    component_topology,
    ring_count,
    solve_counts,
)

# ---- sub-step 1: topologies -------------------------------------------------------------------


def test_geom_mesh_strategy2_g12_topologies_default_by_type_and_take_the_blocks_table():
    body = component_topology("FUS", "body")
    assert body["topology"] == "O" and body["poles"] == {"nose": "valence3", "base": "valence3"}
    assert component_topology("WING", "lifting_surface")["topology"] == "C"
    assert component_topology("FAIRING", "wing_fuselage_fairing")["topology"] == "collar"
    opened = component_topology("NAC", "body", {"poles": {"nose": "open"}})
    assert opened["poles"] == {"nose": "open", "base": "valence3"}
    assert component_topology("P", "lifting_surface", {"topology": "H"})["topology"] == "H"


def test_geom_mesh_strategy2_g12_e11_trips_on_a_planted_valence_4_pole_and_an_unknown_topology():
    with pytest.raises(InputValueError, match="E11"):
        component_topology("SPIN", "body", {"poles": {"nose": "valence4"}})
    with pytest.raises(InputValueError, match="topology"):
        component_topology("X", "body", {"topology": "Y"})


def test_geom_mesh_strategy2_g12_a_face_cut_by_a_junction_becomes_a_block_with_a_composite_side():
    """YES: six edges, four sharp corners, the cut junction curve stays one side. NO: two."""
    walk = [1, 2, 3, 4, 5, 6]
    turns = [90.0, 90.0, 5.0, 4.0, 90.0, 90.0]
    block = block_from_walk("FUS.9", "FUS", walk, turns)
    assert block.sides == ((1,), (2, 3, 4), (5,), (6,))
    assert block_from_walk("SLIVER", "F", [7, 8], [175.0, 176.0]) is None
    cusp = [90.0, 175.0, 5.0, 90.0]
    assert block_from_walk("CAP", "W", [1, 2, 3, 4], cusp, cusps=False) is None
    assert block_from_walk("CAP", "W", [1, 2, 3, 4], cusp).sides == ((1,), (2,), (3,), (4,))


# ---- sub-step 2: ring counts ------------------------------------------------------------------


def _strip():
    """Two blocks side by side: a (1, 2, 3, 4) and b (3, 5, 6, 7); 3 is shared."""
    a = Block("a", "W", ((1,), (2,), (3,), (4,)), (0, 1, 2, 3))
    b = Block("b", "F", ((3,), (5,), (6,), (7,)), (0, 1, 2, 3))
    lengths = dict.fromkeys(range(1, 8), 1.0)
    return [a, b], lengths


def test_geom_mesh_strategy2_g12_counts_hold_opposite_sides_and_the_ring_total():
    blocks, lengths = _strip()
    counts = solve_counts(blocks, lengths, dict.fromkeys(lengths, 0.25), totals=[([1, 6], 10)])
    assert counts[1] == counts[3] == counts[6] and counts[2] == counts[4]
    assert counts[5] == counts[7] and counts[1] + counts[6] == 10
    assert ring_count(20, 2) == 42


def test_geom_mesh_strategy2_g12_e09_trips_on_planted_opposite_counts():
    blocks, lengths = _strip()
    with pytest.raises(InputValueError, match="E09.*a sides 0 and 2: 4 and 6"):
        solve_counts(blocks, lengths, {}, fixed={1: 4, 3: 6})


def test_geom_mesh_strategy2_g12_e08_trips_on_a_planted_ring_that_cannot_hold():
    blocks, lengths = _strip()
    # Edges 1 and 6 are opposite sides of one row: their total is always even.
    with pytest.raises(InputValueError, match="E08"):
        solve_counts(blocks, lengths, {}, totals=[([1, 6], 9)])


def test_geom_mesh_strategy2_g12_a_presolve_that_calls_a_ring_infeasible_is_not_believed(
    monkeypatch,
):
    """The HiGHS presolve of scipy 1.15 (the floor) called the WB ring rows infeasible."""
    import scipy.optimize

    solve, presolved = scipy.optimize.milp, []

    def floor_presolve(cost, **kwargs):
        if kwargs.get("options", {}).get("presolve", True):
            presolved.append(True)
            return scipy.optimize.OptimizeResult(x=None, status=2, success=False)
        return solve(cost, **kwargs)

    monkeypatch.setattr(scipy.optimize, "milp", floor_presolve)
    blocks, lengths = _strip()
    counts = solve_counts(blocks, lengths, dict.fromkeys(lengths, 0.25), totals=[([1, 6], 10)])
    assert presolved and counts[1] + counts[6] == 10
    with pytest.raises(InputValueError, match="E08"):
        solve_counts(blocks, lengths, {}, totals=[([1, 6], 9)])


# ---- sub-step 3: carry rows -------------------------------------------------------------------


def test_geom_mesh_strategy2_g12_carry_rows_keep_the_growth_and_e10_trips_on_too_few_rows():
    plan = carry_plan(0.01, 0.05, 1.2)
    assert plan["rows"] == 9 and plan["growth"] <= 1.2 and plan["widths"][-1] == pytest.approx(0.05)
    with pytest.raises(InputValueError, match="E10.*over 3 rows"):
        carry_plan(0.01, 0.05, 1.2, rows=3)


def test_geom_mesh_strategy2_g12_carry_pairs_are_graded_by_the_solver_and_e10_trips_when_pinned():
    blocks, lengths = _strip()
    lengths[2], lengths[5] = 0.1, 1.0  # one grid line: edge 2 of a continues as edge 5 of b
    targets = {**dict.fromkeys(lengths, 0.5), 2: 0.01}
    counts = solve_counts(blocks, lengths, targets, carry=[(2, 5)], max_growth=1.2)
    assert carry_pairs_growth(counts, lengths, [(2, 5)]) <= 1.2
    with pytest.raises(InputValueError, match="E10"):
        solve_counts(blocks, lengths, targets, fixed={2: 10, 5: 2}, carry=[(2, 5)])


# ---- sub-step 4: valence 3 --------------------------------------------------------------------


def _star(k):
    """``k`` quads around node 0 on a ring of ``2 k`` nodes: node 0 is interior, valence ``k``."""
    angles = np.linspace(0.0, 2 * np.pi, 2 * k, endpoint=False)
    xyz = np.vstack([[0.0, 0.0, 1.0], np.c_[np.cos(angles), np.sin(angles), np.ones(2 * k)]])
    ring = lambda i: 1 + i % (2 * k)  # noqa: E731
    cells = [(0, ring(2 * q), ring(2 * q + 1), ring(2 * q + 2)) for q in range(k)]
    return xyz, cells


def test_geom_mesh_strategy2_g12_e11_trips_on_an_undeclared_irregular_node_and_passes_a_declared():
    xyz, cells = _star(4)
    assert check_poles(xyz, cells, [], tolerance_m=1e-9) == {"valence3_poles": 0, "irregular": 0}
    xyz, cells = _star(3)
    pole = xyz[0].tolist()
    assert check_poles(xyz, cells, [pole], tolerance_m=1e-9)["valence3_poles"] == 1
    with pytest.raises(InputValueError, match="E11.*valence 3"):
        check_poles(xyz, cells, [], tolerance_m=1e-9)
    xyz, cells = _star(5)
    with pytest.raises(InputValueError, match="E11.*valence 5"):
        check_poles(xyz, cells, [xyz[0].tolist()], tolerance_m=1e-9)


# ---- sub-step 5: azimuth by count -------------------------------------------------------------


def test_geom_mesh_strategy2_g12_azimuth_count_is_split_over_the_sectors_by_arc():
    assert azimuth_split(26, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) == [5, 5, 4, 4, 4, 4]
    split = azimuth_split(24, [3.0, 1.0])
    assert sum(split) == 24 and split == [18, 6]
    with pytest.raises(InputValueError, match="cannot give"):
        azimuth_split(2, [1.0, 1.0, 1.0])


# ---- E07 --------------------------------------------------------------------------------------


def test_geom_mesh_strategy2_g12_e07_trips_on_a_planted_triangle_and_a_quad_dominant_fill():
    from cfdbender.geometry._geometry_mesh_s2 import plan_strategy2

    with pytest.raises(InputValueError, match="E07.*FAIRING"):
        check_triangles([(0, 1, 2, 3), (3, 4, 5)], ["WING", "FAIRING"])
    assert check_triangles([(0, 1, 2, 3)], ["WING"]) is None
    controls = _wb_controls()
    controls["components"]["FAIRING"] = {
        "role": "follower",
        "follows": "FUS",
        "fill": "quad_dominant",
    }
    with pytest.raises(InputValueError, match="E07"):
        plan_strategy2(_resolved(), controls)


# ---- plan and route ---------------------------------------------------------------------------

WB = {
    "components": {
        "FUS": {"type": "body"},
        "WING": {
            "type": "lifting_surface",
            "attach": {"parent": "FUS", "station": 0.4},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        },
        "FAIRING": {
            "type": "wing_fuselage_fairing",
            "hosts": ["FUS", "WING"],
            "attach": {"parent": "FUS", "station": 0.0},
        },
    },
}


def _resolved():
    from cfdbender.geometry import resolve_config

    return resolve_config(WB)


def _wb_controls():
    return {
        "mesh": {"strategy": 2, "variant": "clean"},
        "defaults": {"junction": {"max_growth": 1.2}},
        "components": {
            "FUS": {
                "type": "body",
                "role": "primary",
                "x_step_m": 0.0125,
                "azimuth_rule": {"mode": "aspect", "target_aspect": 1.0},
            },
            "WING": {
                "type": "lifting_surface",
                "role": "primary",
                "chordwise": {"cells": 20, "law": "constant"},
                "spanwise": {"cells": 16, "law": "constant"},
            },
        },
    }


def test_geom_mesh_strategy2_g12_plan_needs_strategy_2_and_a_dry_run_opens_no_kernel():
    from cfdbender.geometry._geometry_mesh_s2 import mesh_full_grid, plan_strategy2

    controls = _wb_controls()
    controls["mesh"]["strategy"] = 1
    with pytest.raises(InputValueError, match="strategy 2"):
        plan_strategy2(_resolved(), controls)
    record = mesh_full_grid(WB, _wb_controls(), dry_run=True).record
    assert record["dry_run"] and record["members"] == ["FUS", "WING", "FAIRING"]
    assert record["components"]["FAIRING"]["blocks"]["topology"] == "collar"
    assert record["components"]["FUS"]["blocks"]["poles"]["nose"] == "valence3"


@pytest.fixture(scope="module")
def wb_run():
    from kernel_support import kernel_or_skip

    kernel_or_skip()
    from cfdbender.geometry._geometry_mesh_s2 import mesh_full_grid

    return mesh_full_grid(WB, _wb_controls())


def _checks(record):
    return {c["id"]: c for c in record["checks"]}


def test_geom_mesh_strategy2_g12_wb_rings_are_equal_and_opposite_sides_hold(wb_run):
    checks = _checks(wb_run.record)
    ring = checks["E08.ring_counts"]
    assert ring["status"] == "passed" and ring["where"]["WING"] == ring["where"]["FAIRING"]
    assert ring["planned"] == ring["where"]["WING"] == 42
    assert checks["E09.opposite_sides"]["status"] == "passed"


def test_geom_mesh_strategy2_g12_wb_carry_growth_is_under_the_limit(wb_run):
    carry = _checks(wb_run.record)["E10.carry_growth"]
    assert carry["status"] == "passed" and carry["measured"] <= 1.2


def test_geom_mesh_strategy2_g12_wb_every_block_is_all_quad_and_triangles_only_on_named_faces(
    wb_run,
):
    """Triangles are counted; every one lies on a face reported as not full grid."""
    mesh, record = wb_run.mesh, wb_run.record
    named = {int(item.rsplit(".", 1)[1]) for item in record["not_full_grid"]}
    on = {s for c, s in zip(mesh.cells, mesh.surfaces, strict=True) if len(c) == 3 and s != -1}
    assert on <= named
    assert _checks(record)["E07.triangles"]["measured"] == 0
    # The one-face tip cap and a body uncut at the tangent fairing ends leave 66 faces (the
    # legacy fixture had 87); every one is a block but the fairing's tangent sliver.
    assert record["not_full_grid"] == ["FAIRING.88"]
    assert record["blocks"] > 60 and record["components"]["FUS"]["azimuth_count"] >= 24


def test_geom_mesh_strategy2_g12e_wb_tip_cap_is_quad_dominant_and_reported_apart(wb_run):
    """The one-face cap is meshed free and quad-dominant, as strategy 1 does (design section 8).

    Its cells carry the cap surface -1 and are reported apart; no cell is inverted or
    degenerate, and no triangle lies outside the cap.
    """
    mesh, record = wb_run.mesh, wb_run.record
    checks = _checks(record)
    cap = [i for i, s in enumerate(mesh.surfaces) if s == -1]
    assert len(cap) > 10 and {mesh.components[i] for i in cap} == {"WING"}
    quads = sum(len(mesh.cells[i]) == 4 for i in cap)
    assert quads > len(cap) // 2
    assert checks["E07.triangles"]["where"]["tip_cap"] == len(cap) - quads
    assert checks["mesh.inverted"]["measured"] == 0, checks["mesh.inverted"]
    assert checks["mesh.degenerate"]["measured"] == 0, checks["mesh.degenerate"]
    assert not any(f.endswith(".61") for f in record["not_full_grid"])


def test_geom_mesh_strategy2_g12_wb_nodes_lie_on_their_sources_and_the_planar_cap_on_its_plane(
    wb_run,
):
    """The one-face cap is measured on its plane: no node of the base 0.125 m off its source.

    The kernel's closest point on the cap, whose leading-edge side is a collapsed edge, runs to
    the leading edge from the base midpoint; that edge also bounds no cells and takes no count.
    """
    check = _checks(wb_run.record)["mesh.source_distance"]
    assert check["status"] == "passed" and check["measured"] < 1e-6


def test_geom_mesh_strategy2_g12_wb_poles_are_valence_3_at_the_body_closures(wb_run):
    poles = _checks(wb_run.record)["E11.valence3_poles"]
    assert poles["measured"] == 6


def test_geom_mesh_strategy2_r27_e11_measures_the_valence_of_a_declared_pole():
    xyz, cells = _star(4)
    assert check_poles(xyz, cells, [], tolerance_m=1e-9)["irregular"] == 0
    with pytest.raises(InputValueError, match="E11.*valence 4"):
        check_poles(xyz, cells, [xyz[0].tolist()], tolerance_m=1e-9)

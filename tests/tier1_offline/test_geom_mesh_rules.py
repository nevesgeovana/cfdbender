"""Geometry mesh: spacing laws, rules, seeds, orientation, checks and OBJ, no kernel.

Covers spacing, rules, orientation, checks and writers. The
module under test imports no gmsh, so every test here runs on the base install.
"""

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry import _mesh_rules as mesh_rules
from cfdbender.geometry import _mesh_topology as mesh_topology
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._mesh_rules import (
    DEFAULT_CONTROLS,
    constant,
    count_classes,
    geometric_cap,
    insert_seed,
    load_mesh_controls,
    tip_refinement,
)
from cfdbender.geometry._mesh_topology import (
    Mesh,
    area_vectors,
    check_mesh,
    corner_quality,
    flip_slivers,
    inconsistent_edges,
    orient_mesh,
    oriented_source_normals,
    paired_triangles,
    read_obj,
    repair_inverted,
    write_obj,
)

CONTROLS = {"tip_cells": 6, "tip_ratio": 1.3, "chord_growth": 1.15}


def test_constant():
    x = constant(16)
    assert len(x) == 17 and x[0] == 0 and x[-1] == 1
    np.testing.assert_allclose(np.diff(x), 1 / 16)


def test_growth_ratios_and_cap():
    d = np.diff(geometric_cap(20, 1.2, 8))
    np.testing.assert_allclose(d[1:9] / d[:8], 1.2)
    np.testing.assert_allclose(d[8:], d[8])


def test_midchord_plateau():
    x = geometric_cap(40)
    d = np.diff(x)
    start = np.flatnonzero(np.isclose(d, d[-1]))[0]
    assert abs(x[start] - 0.5) <= 2 * max(d)


def test_tip_refinement_shrinks_the_last_cells_by_the_ratio():
    x = tip_refinement(20, 5, 1.4)
    d = np.diff(x)
    assert len(x) == 21
    np.testing.assert_allclose(d[:15], d[0])
    np.testing.assert_allclose(d[15:] / d[14:-1], 1 / 1.4)


@pytest.mark.parametrize("count", [0, -1, 2.5, True])
def test_a_bad_cell_count_is_refused(count):
    for law in (constant, geometric_cap, tip_refinement):
        with pytest.raises(InputValueError):
            law(count)


@pytest.mark.parametrize("ratio", [0.9, float("nan"), float("inf")])
def test_a_bad_ratio_is_refused(ratio):
    with pytest.raises(InputValueError):
        geometric_cap(20, ratio)
    with pytest.raises(InputValueError):
        tip_refinement(20, ratio=ratio)


def test_default_controls_are_returned_and_overridden_by_table():
    base = load_mesh_controls()
    assert base == DEFAULT_CONTROLS and base is not DEFAULT_CONTROLS
    over = load_mesh_controls({"body": {"x_cells": 40}})
    assert over["body"] == {"x_cells": 40, "azimuth_cells": 24}
    assert DEFAULT_CONTROLS["body"]["x_cells"] == 80


def test_controls_from_a_toml_file(tmp_path):
    path = tmp_path / "controls.toml"
    path.write_text("[lifting_surface]\nspan_cells = 12\n", encoding="utf-8")
    assert load_mesh_controls(path)["lifting_surface"]["span_cells"] == 12


def test_unknown_controls_are_refused_naming_the_available_ones():
    with pytest.raises(InputValueError, match="unknown table"):
        load_mesh_controls({"wing": {}})
    with pytest.raises(InputValueError, match="x_cells"):
        load_mesh_controls({"body": {"x_cell": 4}})


def test_opposite_edges_share_one_count_and_conflicts_are_refused():
    loops = {1: (1, 2, -3, -4), 2: (3, 5, -6, -2)}
    counts = count_classes(loops, {}, default_cells=7)
    assert counts[1] == counts[3] == counts[6] and counts[2] == counts[4] == counts[5]
    seeds = {1: constant(4)}
    assert count_classes(loops, seeds, default_cells=7)[3] == 4
    with pytest.raises(InputValueError, match="conflict"):
        count_classes(loops, {1: constant(4), 3: constant(5)})


def test_free_classes_take_the_rounded_length_over_the_target():
    loops = {1: (1, 2, -3, -4)}
    lengths = {1: 1.0, 3: 1.0, 2: 0.52, 4: 0.52}
    counts = count_classes(loops, {}, lengths=lengths, target=0.1)
    assert counts == {1: 10, 2: 5, 3: 10, 4: 5}


def test_seed_walked_backwards_is_mirrored_and_a_different_seed_refused():
    seeds = {}
    nodes = tip_refinement(10, 3, 1.3)
    insert_seed(seeds, -7, nodes)
    np.testing.assert_allclose(seeds[7], 1 - nodes[::-1])
    insert_seed(seeds, 7, 1 - nodes[::-1])
    with pytest.raises(InputValueError, match="inconsistent"):
        insert_seed(seeds, 7, constant(10))


def span_stations(count, tip_cells=6, ratio=1.3, half=10.0):
    y = tip_refinement(count, tip_cells, ratio) * half
    return np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)


def test_spanwise_rule_accepts_the_law_and_refuses_growth():
    ok, detail = mesh_rules.spanwise(span_stations(32), CONTROLS, 10.0)
    assert ok and detail["shrinking_steps"] == 6 and detail["reaches_tip"]
    squared = np.stack([np.zeros(33), 10 * np.linspace(0, 1, 33) ** 2, np.zeros(33)], axis=1)
    assert not mesh_rules.spanwise(squared, CONTROLS, 10.0)[0]


def test_spanwise_rule_refuses_a_missing_tip_refinement():
    y = np.linspace(0, 10, 20)
    points = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    assert not mesh_rules.spanwise(points, CONTROLS, 10.0)[0]


def wing_component():
    from cfdbender.geometry import resolve_config

    raw = {
        "type": "lifting_surface",
        "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
    }
    return unpack(resolve_config(raw))["components"]["COMPONENT"]


def chord_points(component, fractions, eta=0.5):
    from cfdbender.geometry._distributions import planform_values

    p = planform_values(component, eta)
    theta = np.deg2rad(p["twist"])
    dx = (np.asarray(fractions) - 0.25) * p["chord"]
    return np.stack(
        [
            p["leading_edge_x"] + 0.25 * p["chord"] + dx * np.cos(theta),
            np.full(len(dx), eta * component["planform"]["span"] / 2),
            p["leading_edge_z"] - dx * np.sin(theta),
        ],
        axis=1,
    )


def test_chord_fraction_inverts_the_chord_station_of_a_point():
    component = wing_component()
    fractions = np.linspace(0, 1, 9)
    np.testing.assert_allclose(
        mesh_rules.chord_fraction(component, chord_points(component, fractions)),
        fractions,
        atol=1e-12,
    )


def test_chordwise_rule_accepts_growth_then_plateau_and_refuses_reversal():
    component = wing_component()
    law = geometric_cap(40, 1.15, growth_until=0.5)
    ok, detail = mesh_rules.chordwise(component, chord_points(component, law), CONTROLS)
    assert ok and detail["growth_steps"] > 0
    assert not mesh_rules.chordwise(component, chord_points(component, 1 - law[::-1]), CONTROLS)[0]
    uniform = chord_points(component, np.linspace(0, 1, 41))
    assert mesh_rules.chordwise(component, uniform, CONTROLS)[0]


def test_body_rules():
    angle = np.deg2rad(np.linspace(0, 30, 5))
    arc = np.stack([np.zeros(5), 1.2 * np.cos(angle), 1.2 * np.sin(angle)], axis=1)
    assert mesh_rules.body_edge(arc, "azimuth", 0.25, 7.5)[0]
    skew = angle**1.3
    skewed = np.stack([np.zeros(5), 1.2 * np.cos(skew), 1.2 * np.sin(skew)], axis=1)
    assert not mesh_rules.body_edge(skewed, "azimuth", 0.25, 7.5)[0]
    line = np.stack([np.linspace(0, 1, 5), np.zeros(5), np.ones(5)], axis=1)
    assert mesh_rules.body_edge(line, "axial", 0.25, 7.5)[0]
    assert not mesh_rules.body_edge(line, "axial", 0.5, 7.5)[0]


def test_check_lifting_skips_trailing_strips_and_reports_failing_edges():
    half = 10.0
    component = wing_component()
    component = {**component, "planform": {**component["planform"], "span": 2 * half}}
    good = span_stations(32, half=half)
    items = [
        {"family": "upper_fore", "surface": 1, "u": [], "v": [5]},
        {"family": "trailing_upper", "surface": 2, "u": [], "v": [6]},
    ]
    bad = good.copy()
    bad[:, 1] = bad[:, 1] ** 2 / half
    rows, failures = mesh_rules.check_lifting(component, CONTROLS, items, {5: good, 6: bad})
    assert [r["edge"] for r in rows] == [5] and not failures
    rows, failures = mesh_rules.check_lifting(component, CONTROLS, items, {5: bad, 6: bad})
    assert [f["edge"] for f in failures] == [5]


def test_check_body_classifies_band_and_pole_edges():
    controls = {"x_cells": 4, "azimuth_cells": 24}
    component = {"length": 1.0}
    line = np.stack([np.linspace(0, 1, 5), np.zeros(5), np.ones(5)], axis=1)
    angle = np.deg2rad(np.linspace(0, 30, 5))
    arc = np.stack([np.zeros(5), 1.2 * np.cos(angle), 1.2 * np.sin(angle)], axis=1)
    band = {"family": "middle", "surface": 1, "u": [1], "v": [2], "axial": True}
    rows, failures = mesh_rules.check_body(component, controls, [band], {1: arc, 2: line})
    assert [r["role"] for r in rows] == ["azimuth", "axial"] and not failures
    pole = {"family": "nose", "surface": 3, "u": [2], "v": [2], "axial": False}
    rows, _ = mesh_rules.check_body(component, controls, [pole], {2: line})
    assert {r["role"] for r in rows} == {"pole"}


def two_patch_body():
    xyz = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1]], float)
    return Mesh(xyz, [(0, 1, 4, 3), (4, 5, 2, 1)], ["X"] * 2, [1, 2], np.arange(6))


def test_orientation_of_two_opposite_body_patches():
    mesh = two_patch_body()
    assert len(inconsistent_edges(mesh.cells)) == 1
    result, record = orient_mesh(mesh, "body")
    assert record["inconsistent_shared_edge_directions"] == 0
    assert np.all(area_vectors(result)[:, 1] > 0)
    assert np.array_equal(result.xyz, mesh.xyz)
    normals = oriented_source_normals(result, np.array([[0, -1, 0], [0, 1, 0]], float))
    assert corner_quality(result.xyz, result.cells, normals) == ([], [])


def half_box():
    xyz = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        float,
    )
    cells = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5), (3, 2, 6, 7)]
    return Mesh(xyz, cells, ["X"] * 5, list(range(5)), np.arange(8))


def test_mirrored_half_box_is_oriented_outward_and_accepted():
    mesh = half_box()
    result, record = orient_mesh(mesh)
    assert record["positive_scores"][0] == pytest.approx(2.0)
    assert not inconsistent_edges(result.cells)
    normals = area_vectors(result)
    centers = np.array([mesh.xyz[list(c)].mean(axis=0) for c in result.cells])
    assert np.all(np.sum(normals * (centers - [0.5, 0, 0.5]), axis=1) > 0)
    assert check_mesh(result, normals, np.zeros(8))["accepted"]


def test_a_fold_is_not_hidden_by_the_orientation():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0.2, 0.2, 0], [0, 1, 0]], float)
    mesh = Mesh(xyz, [(0, 1, 2, 3)], ["X"], [1], np.arange(4))
    result, _ = orient_mesh(mesh, "reference", reference=[0, 0, 1])
    normals = oriented_source_normals(result, np.array([[0, 0, -1.0]]))
    assert corner_quality(result.xyz, result.cells, normals)[0] == [0]


def test_orientation_refusals():
    mesh = half_box()
    with pytest.raises(InputValueError, match="unknown outward sense"):
        orient_mesh(mesh, "inward")
    open_mesh = Mesh(
        mesh.xyz + [0, 0.5, 0], mesh.cells, mesh.components, mesh.surfaces, mesh.node_tags
    )
    with pytest.raises(InputValueError, match="free edges on y=0"):
        orient_mesh(open_mesh)
    triple = Mesh(
        two_patch_body().xyz,
        [(0, 1, 4, 3), (0, 1, 4, 3), (0, 1, 4, 3)],
        ["X"] * 3,
        [1] * 3,
        np.arange(6),
    )
    with pytest.raises(InputValueError, match="nonmanifold"):
        orient_mesh(triple)
    flat = Mesh(two_patch_body().xyz, [(0, 1, 1, 0)], ["X"], [1], np.arange(6))
    with pytest.raises(InputValueError, match="indeterminate"):
        orient_mesh(flat, "reference", reference=[0, 0, 1])


def unit_square(cells):
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    return Mesh(xyz, cells, ["X"] * len(cells), [1] * len(cells), np.arange(4))


@pytest.mark.parametrize(
    ("cells", "inverted", "degenerate", "nonmanifold", "nonquad"),
    [
        ([(0, 1, 2, 3)], 0, 0, 0, 0),
        ([(3, 2, 1, 0)], 1, 0, 0, 0),
        ([(0, 1, 1, 3)], 0, 1, 0, 0),
        ([(0, 1, 2, 3)] * 3, 0, 0, 4, 0),
        ([(0, 1, 2)], 0, 0, 0, 1),
        ([(0, 2, 1, 3)], 1, 0, 0, 0),
    ],
)
def test_defects_are_counted(cells, inverted, degenerate, nonmanifold, nonquad):
    mesh = unit_square(cells)
    record = check_mesh(mesh, np.tile([0, 0, 1], (len(cells), 1)), np.zeros(4))
    assert record["inverted_count"] == inverted
    assert record["degenerate_count"] == degenerate
    assert record["non_manifold_count"] == nonmanifold
    assert record["non_quad_count"] == nonquad
    assert not record["accepted"]


def test_free_edges_off_the_symmetry_plane_are_counted():
    record = check_mesh(unit_square([(0, 1, 2, 3)]), [[0, 0, 1]], np.zeros(4))
    assert record["free_edges_off_y0_count"] == 3


def test_source_distance_decides_acceptance_and_bad_metrics_are_refused():
    mesh = half_box()
    result, _ = orient_mesh(mesh)
    normals = area_vectors(result)
    far = np.zeros(8)
    far[3] = 0.01
    record = check_mesh(result, normals, far)
    assert record["source_distance_max_m"] == 0.01 and not record["accepted"]
    with pytest.raises(InputValueError, match="one reference normal per cell"):
        check_mesh(result, normals[:2], np.zeros(8))
    with pytest.raises(InputValueError, match="finite"):
        check_mesh(result, normals, -np.ones(8))


def test_triangles_are_allowed_only_on_transition_surfaces():
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mesh = Mesh(xyz, [(0, 1, 2)], ["X"], [9], np.arange(3))
    normal = [[0, 0, 1]]
    assert check_mesh(mesh, normal, np.zeros(3))["forbidden_non_quad_cells"] == [0]
    assert (
        check_mesh(mesh, normal, np.zeros(3), transition_surfaces=[9])["forbidden_non_quad_cells"]
        == []
    )


def test_coincident_nodes_name_a_junction_that_is_not_node_to_node():
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 1e-12]])
    assert mesh_topology.coincident_nodes(xyz, 1e-9) == [(1, 2)]


def test_repair_replaces_the_wrong_diagonal_of_a_dart_quadrilateral():
    # A dart with the reflex vertex at node 2; the diagonal 1-3 leaves triangle (1, 2, 3) inverted.
    xyz = np.array([[0.0, 0, 0], [2, -1, 0], [0.5, 0, 0], [2, 1, 0]])
    mesh = Mesh(xyz, [(0, 1, 3), (1, 2, 3)], ["X"] * 2, [1, 1], np.arange(4))
    normals = np.tile([0, 0, 1.0], (2, 1))
    inverted = corner_quality(mesh.xyz, mesh.cells, normals)[0]
    assert inverted == [1]
    repaired, flips = repair_inverted(mesh, normals, inverted)
    assert flips == 1
    assert repaired.cells == [(0, 1, 2), (3, 0, 2)][::-1] or repaired.cells == [
        (0, 1, 2),
        (3, 0, 2),
    ]
    assert corner_quality(repaired.xyz, repaired.cells, normals) == ([], [])
    untouched, none = repair_inverted(mesh, normals, [])
    assert none == 0 and untouched.cells == mesh.cells


def test_a_sliver_triangle_is_flipped_when_that_raises_the_worst_quality():
    xyz = np.array([[0.0, 0, 0], [10, 0, 0], [5, 0.05, 0], [5, -4, 0]])
    mesh = Mesh(xyz, [(0, 1, 2), (1, 0, 3)], ["X"] * 2, [1, 1], np.arange(4))
    result, flips = flip_slivers(mesh)
    assert flips == 1
    assert result.cells == [(0, 3, 2), (3, 1, 2)]
    assert corner_quality(xyz, result.cells, np.tile([0, 0, 1.0], (2, 1))) == ([], [])
    again, more = flip_slivers(result)
    assert more == 0 and again.cells == result.cells


def test_quadrilaterals_pair_into_reciprocal_triangles_with_one_based_partners():
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]])
    mesh = Mesh(xyz, [(0, 1, 2, 3), (1, 4, 2)], ["A", "B"], [1, 2], np.arange(5))
    triangles, partners, slots, owners = paired_triangles(mesh)
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
    assert partners == [2, 1, 0] and slots == [3, 1, 0] and owners == ["A", "A", "B"]
    with pytest.raises(InputValueError, match="triangles and quadrilaterals"):
        paired_triangles(Mesh(xyz, [(0, 1, 2, 3, 4)], ["A"], [1], np.arange(5)))


def test_obj_round_trip_keeps_nodes_cells_and_component_groups(tmp_path):
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0.1]])
    mesh = Mesh(xyz, [(0, 1, 2, 3), (1, 4, 2)], ["A", "B"], [1, 2], np.arange(5))
    path = tmp_path / "m.obj"
    write_obj(path, mesh)
    back = read_obj(path)
    assert np.array_equal(back.xyz, mesh.xyz)
    assert back.cells == mesh.cells and back.components == mesh.components
    bad = Mesh(xyz, mesh.cells, ["A", "has space"], [1, 2], np.arange(5))
    with pytest.raises(InputValueError, match="no whitespace"):
        write_obj(tmp_path / "bad.obj", bad)
    (tmp_path / "orphan.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="ascii")
    with pytest.raises(InputValueError, match="no component group"):
        read_obj(tmp_path / "orphan.obj")


def test_the_pure_modules_import_no_geometry_kernel():
    import os
    import subprocess
    import sys

    code = (
        "import sys, cfdbender.geometry._mesh_rules, cfdbender.geometry._mesh_topology;"
        "sys.exit(1 if 'gmsh' in sys.modules else 0)"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    assert subprocess.run([sys.executable, "-c", code], check=False, env=env).returncode == 0


@pytest.mark.parametrize(
    ("table", "key", "value"),
    [
        ("body", "x_cells", 0),
        ("body", "azimuth_cells", 2.5),
        ("lifting_surface", "span_cells", "32"),
        ("lifting_surface", "tip_ratio", float("nan")),
        ("lifting_surface", "chord_growth", -1.0),
        ("checks", "distance_tolerance_m", 0.0),
    ],
)
def test_geom_mesh_controls_refuse_a_value_outside_its_domain(table, key, value):
    with pytest.raises(InputValueError, match=rf"controls\.{table}\.{key}"):
        load_mesh_controls({table: {key: value}})


def test_geom_mesh_controls_refuse_a_table_that_is_not_a_table():
    with pytest.raises(InputValueError, match="controls.body"):
        load_mesh_controls({"body": 5})


def _line(widths):
    x = np.r_[0.0, np.cumsum(widths)]
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


@pytest.mark.parametrize(
    ("step", "accepted"), [(0.4, False), (0.5, True), (1.0, True), (3.0, True), (3.5, False)]
)
def test_geom_a_pole_edge_step_must_stay_within_the_band_of_the_x_step(step, accepted):
    ok, detail = mesh_rules.body_edge(_line([step] * 6), "pole", 1.0, 15.0)
    assert ok is accepted
    assert detail["step_over_x_step"] == pytest.approx(step)


def test_geom_a_pole_edge_refuses_a_cell_width_spread_beyond_the_limit():
    mean = 1.0
    uneven = [mean * 0.6, mean * 1.4, mean * 0.6, mean * 1.4]
    assert mesh_rules.body_edge(_line(uneven), "pole", 1.0, 15.0)[0] is False
    nearly = [mean * 0.9, mean * 1.1, mean * 0.9, mean * 1.1]
    assert mesh_rules.body_edge(_line(nearly), "pole", 1.0, 15.0)[0] is True

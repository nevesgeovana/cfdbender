"""requirement fixtures for mesh controls and settled surface rules."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_mesh import mesh_generated


def _controls():
    return {
        "mesh": {"strategy": 1, "variant": "clean"},
        "components": {
            "foil": {
                "type": "lifting_surface",
                "role": "primary",
                "chordwise": {"cells": 12, "law": "constant"},
                "spanwise": {"cells": 8, "law": "constant"},
            }
        },
    }


def test_geom_mesh_gm_e15_is_named_before_kernel_loading():
    """YES has both distribution tables; NO omits the required spanwise law."""
    controls = _controls()
    del controls["components"]["foil"]["spanwise"]["law"]
    with pytest.raises(InputValueError, match="E15.*spanwise"):
        mesh_generated({"components": {"foil": {"type": "lifting_surface"}}}, controls=controls)


@pytest.mark.skip(reason="waiting for the tip-cap rule")
def test_geom_mesh_gm_tip_cap_rule():
    """The open tip-cap rule has no passing implementation fixture."""


def test_geom_mesh_gm_schema_yes_and_no():
    from cfdbender.geometry._mesh_schema import mesh_controls

    good = _controls()
    assert mesh_controls(good, members={"foil"}, names={"foil"})["mesh"]["strategy"] == 1
    bad = deepcopy(good)
    bad["mesh"]["variant"] = "typo"
    with pytest.raises(InputValueError, match="E16"):
        mesh_controls(bad, members={"foil"}, names={"foil"})


@pytest.mark.parametrize(
    ("code", "change"),
    [
        ("E01", {"role": "follower", "follows": "missing"}),
        ("E03", {"role": "follower", "follows": "foil"}),
        ("E05", {"x_cells": 8, "x_step_m": 0.1}),
        ("E06", {"role": "follower"}),
        ("E11", {"blocks": {"poles": {"nose": "valence4"}}}),
        ("E12", {"transition": {"max_aspect": 8}}),
        ("E13", {"azimuth_cells": 12, "azimuth_rule": {"mode": "aspect"}}),
        ("E15", {"spanwise": {"cells": 8}}),
    ],
)
def test_geom_mesh_gm_schema_error_yes_no(code, change):
    from cfdbender.geometry._mesh_schema import mesh_controls

    good = _controls()
    assert mesh_controls(good, members={"foil"}, names={"foil"})["order"] == ["foil"]
    bad = deepcopy(good)
    bad["components"]["foil"].update(change)
    with pytest.raises(InputValueError, match=code):
        mesh_controls(bad, members={"foil"}, names={"foil"})


def test_geom_mesh_gm_e04_and_w01_distinguish_geometry_from_configuration():
    from cfdbender.geometry._mesh_schema import mesh_controls

    assert mesh_controls(_controls(), members={"foil"}, names={"foil"})["components"]
    with pytest.raises(InputValueError, match="E04"):
        mesh_controls(_controls(), members={"foil"}, names=set())
    ignored = mesh_controls(_controls(), members=set(), names={"foil"})
    assert ignored["components"] == {}
    assert "W01" in ignored["warnings"][0]


def test_geom_mesh_gm_e07_yes_no():
    from cfdbender.geometry._mesh_schema import mesh_controls

    controls = _controls()
    controls["mesh"]["strategy"] = 2
    assert mesh_controls(controls, members={"foil"}, names={"foil"})["components"]
    controls["components"]["foil"]["fill"] = "quad_dominant"
    with pytest.raises(InputValueError, match="E07"):
        mesh_controls(controls, members={"foil"}, names={"foil"})


@pytest.mark.parametrize(
    ("code", "yes", "no", "limit"),
    [
        ("E02", 0, 1, 0),
        ("E08", [8, 8], [8, 9], None),
        ("E09", [4, 4], [4, 5], None),
        ("E10", 1.1, 1.3, 1.2),
        ("E14", 1.0, 1.6, [0.67, 1.5]),
    ],
)
def test_geom_mesh_gm_realized_errors_yes_no(code, yes, no, limit):
    from cfdbender.geometry._mesh_schema import mesh_measurements

    assert mesh_measurements(code, "synthetic", yes, limit)["passed"]
    with pytest.raises(InputValueError, match=code):
        mesh_measurements(code, "synthetic", no, limit)


def test_geom_mesh_gm_followers_copy_shared_ids_with_first_neighbour_priority():
    from cfdbender.geometry._geometry_mesh_s1 import follower_edges

    controls = {
        "components": {
            "first": {"role": "primary", "follows": []},
            "second": {"role": "primary", "follows": []},
            "blend": {"role": "follower", "follows": ["first", "second"]},
        },
        "order": ["first", "second", "blend"],
    }
    given = {"first": {"rim": [0, 1, 2]}, "second": {"rim": [8, 9, 10], "host": [3, 4, 5]}}
    edges = follower_edges(controls, given)
    assert edges["blend"] == {"rim": [0, 1, 2], "host": [3, 4, 5]}
    assert given == {"first": {"rim": [0, 1, 2]}, "second": {"rim": [8, 9, 10], "host": [3, 4, 5]}}
    with pytest.raises(InputValueError, match="follower.*distribution"):
        follower_edges(controls, {**given, "blend": {"rim": [7, 8]}})


def _grid(nx=2, ny=2, *, component="patch"):
    from cfdbender.geometry._mesh_topology import Mesh

    xyz = np.array([(i, j, 0.0) for j in range(ny + 1) for i in range(nx + 1)])
    ids = np.arange(len(xyz)).reshape(ny + 1, nx + 1)
    cells = [
        (int(ids[j, i]), int(ids[j, i + 1]), int(ids[j + 1, i + 1]), int(ids[j + 1, i]))
        for j in range(ny)
        for i in range(nx)
    ]
    return Mesh(xyz, cells, [component] * len(cells), [1] * len(cells), np.arange(len(xyz))), ids


@pytest.mark.parametrize("kind", ["lifting_surface", "control_surface", "fillet", "fuselage"])
def test_geom_mesh_gm_full_structured_grid_yes_no(kind):
    from cfdbender.geometry._geometry_mesh_s1 import structured_check

    mesh, ids = _grid()
    assert structured_check(mesh, ids, "patch", kind)["status"] == "passed"
    bad = deepcopy(mesh)
    bad.cells[0] = bad.cells[0][:3]
    assert structured_check(bad, ids, "patch", kind)["status"] == "failed"
    bad = deepcopy(mesh)
    bad.cells[0] = tuple(reversed(bad.cells[0]))
    assert structured_check(bad, ids, "patch", kind)["status"] == "failed"


@pytest.mark.parametrize("kind", ["lifting_surface", "fillet"])
def test_geom_mesh_gm_constant_y_from_root_with_no_blend_yes_no(kind):
    from cfdbender.geometry._geometry_mesh_s1 import row_check

    mesh, ids = _grid()
    assert row_check(mesh.xyz, ids, kind)["measured"] == 0
    bad = mesh.xyz.copy()
    bad[ids[0, 1], 1] += 0.02
    assert row_check(bad, ids, kind)["status"] == "failed"


def test_geom_mesh_gm_blunt_te_one_quad_across_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import trailing_edge_check

    assert trailing_edge_check(np.arange(6).reshape(3, 2))["status"] == "passed"
    assert trailing_edge_check(np.arange(9).reshape(3, 3))["status"] == "failed"


def test_geom_mesh_gm_fuselage_uniform_longitudinal_spacing_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import axial_check

    mesh, ids = _grid()
    # The axial stations index grid columns; 150 mm is fixture data, not a default.
    mesh.xyz[:, 0] *= 0.15
    assert axial_check(mesh.xyz, ids.T, 0.15)["status"] == "passed"
    mesh.xyz[ids[1, 1], 0] += 0.01
    assert axial_check(mesh.xyz, ids.T, 0.15)["status"] == "failed"


@pytest.mark.parametrize(
    ("region", "intersects", "yes"),
    [
        ("fairing", [], True),
        ("aft_patch", ["pylon"], True),
        ("aft_patch", ["vertical_tail"], True),
        ("aft_patch", [], False),
        ("fuselage", ["pylon"], False),
        ("fillet", [], False),
        ("lifting_surface", [], False),
    ],
)
def test_geom_mesh_gm_quad_dominant_only_allowed_regions(region, intersects, yes):
    from cfdbender.geometry._geometry_mesh_s1 import transition_checks

    mesh, _ = _grid()
    mesh.cells[0] = mesh.cells[0][:3]
    checks = transition_checks(
        mesh,
        "patch",
        region,
        {"max_triangle_fraction": 0.5, "max_aspect": 3, "max_skew_deg": 90},
        intersects=intersects,
    )
    assert (checks[0]["status"] == "passed") is yes


def test_geom_mesh_gm_transition_limits_measured_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import transition_checks

    mesh, _ = _grid()
    limits = {"max_triangle_fraction": 0.1, "max_aspect": 2, "max_skew_deg": 50}
    assert all(c["status"] == "passed" for c in transition_checks(mesh, "patch", "fairing", limits))
    mesh.cells[0] = mesh.cells[0][:3]
    assert any(
        c["id"] == "mesh.transition.triangle_fraction" and c["status"] == "failed"
        for c in transition_checks(mesh, "patch", "fairing", limits)
    )


def test_geom_mesh_gm_fillet_uses_wing_root_refinement_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import fillet_checks

    mesh, ids = _grid()
    root, next_row = ids[0], ids[1]
    assert all(
        c["status"] == "passed"
        for c in fillet_checks(mesh.xyz, ids, root, next_row, tolerance_m=1e-10)
    )
    bad = ids.copy()
    bad[0] = bad[0, ::-1]
    assert fillet_checks(mesh.xyz, bad, root, next_row, tolerance_m=1e-10)[0]["status"] == "failed"
    bad_xyz = mesh.xyz.copy()
    bad_xyz[ids[-1], 1] += 0.4
    assert fillet_checks(bad_xyz, ids, root, next_row, tolerance_m=1e-10)[1]["status"] == "failed"


def test_geom_mesh_gm_fairing_keel_on_fuselage_stations_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import keel_check

    assert keel_check([0, 0.15, 0.3], [0, 0.15, 0.3])["status"] == "passed"
    assert keel_check([0, 0.16, 0.3], [0, 0.15, 0.3])["status"] == "failed"


def test_geom_mesh_gm_small_fairing_cells_only_boundary_row_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import fairing_boundary_check

    mesh, ids = _grid(3, 3)
    stations = np.array([0, 0.1, 1.1, 1.2])
    mesh.xyz[:, 0] = stations[mesh.xyz[:, 0].astype(int)]
    mesh.xyz[:, 1] = stations[mesh.xyz[:, 1].astype(int)]
    rim = [
        *zip(ids[0, :-1], ids[0, 1:], strict=True),
        *zip(ids[-1, :-1], ids[-1, 1:], strict=True),
        *zip(ids[:-1, 0], ids[1:, 0], strict=True),
        *zip(ids[:-1, -1], ids[1:, -1], strict=True),
    ]
    assert fairing_boundary_check(mesh, "patch", rim, 0.2)["status"] == "passed"
    mesh.xyz[ids[2], 1] = 0.2
    mesh.xyz[ids[:, 2], 0] = 0.2
    assert fairing_boundary_check(mesh, "patch", rim, 0.2)["status"] == "failed"


def test_geom_mesh_gm_aft_fairing_growth_strictly_below_1_2_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import aft_growth_checks

    good = np.cumsum([0, 0.03, 0.035, 0.04])
    checks = aft_growth_checks([good], first_m=0.03, far_m=0.04)
    assert all(c["status"] == "passed" for c in checks)
    for widths in ([0.03, 0.036, 0.04], [0.03, 0.04, 0.05]):
        assert any(
            c["status"] == "failed"
            for c in aft_growth_checks([np.cumsum([0, *widths])], first_m=0.03, far_m=0.04)
        )


def test_geom_mesh_gm_te_cut_continues_over_fairing_and_fuselage_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import cut_check

    mesh, ids = _grid()
    mesh.components = ["fairing", "fuselage", "fairing", "fuselage"]
    assert cut_check(mesh, ids[:, 1], 1, 0, required={"fairing", "fuselage"})["status"] == "passed"
    assert (
        cut_check(mesh, ids[[0, 2], 1], 1, 0, required={"fairing", "fuselage"})["status"]
        == "failed"
    )
    assert cut_check(mesh, ids[:, 0], 1, 0, required={"fairing", "fuselage"})["status"] == "failed"


def test_geom_mesh_gm_fillet_tip_omitted_from_mesh_only_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import omission_checks

    mesh, _ = _grid()
    assert all(
        c["status"] == "passed" for c in omission_checks(mesh, {9}, "sourcehash", "sourcehash")
    )
    assert omission_checks(mesh, {1}, "sourcehash", "sourcehash")[0]["status"] == "failed"
    assert omission_checks(mesh, {9}, "before", "after")[1]["status"] == "failed"


def test_geom_mesh_gm_surfaces_only_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import surface_only_check

    assert surface_only_check([2, 2])["status"] == "passed"
    assert surface_only_check([2, 3])["status"] == "failed"


def test_geom_mesh_gm_azimuth_rule_ceil_and_ring_floor_yes_no():
    from cfdbender.geometry._mesh_schema import azimuth_count

    stations = [[0.0, 2.0, 0.2], [1.0, 3.01, 0.25]]
    assert azimuth_count(stations, {"target_aspect": 1.0, "reference": "max_perimeter"}) == (
        13,
        1.0,
    )
    assert azimuth_count(stations, {"target_aspect": 1.0, "reference": 0.0}, ring_count=16) == (
        16,
        0.0,
    )
    with pytest.raises(InputValueError, match="reference.*station"):
        azimuth_count(stations, {"reference": 0.5})
    with pytest.raises(InputValueError, match="positive"):
        azimuth_count([[0, 3, 0]], {})


def test_geom_mesh_gm_body_aspect_is_measured_on_realized_stations_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import body_aspect_checks

    mesh, ids = _grid()
    assert all(c["passed"] for c in body_aspect_checks(mesh.xyz, ids, [0.67, 1.5]))
    mesh.xyz[:, 0] *= 2
    with pytest.raises(InputValueError, match="E14"):
        body_aspect_checks(mesh.xyz, ids, [0.67, 1.5])


def test_geom_mesh_gm_hinge_rows_exact_and_cell_count_preserved_yes_no():
    from cfdbender.geometry._mesh_schema import hinge_nodes

    table = {"cells": 12, "law": "geometric_cap", "ratio": 1.1, "hinge_rows": [0.72, 0.73]}
    nodes = hinge_nodes(table)
    assert len(nodes) == 13
    assert np.all(np.diff(nodes) > 0)
    assert 0.72 in nodes and 0.73 in nodes
    with pytest.raises(InputValueError, match="hinge.*cells"):
        hinge_nodes({**table, "cells": 2})


def test_geom_mesh_gm_control_surface_defaults_to_parent_follower_yes_no():
    from cfdbender.geometry._mesh_schema import mesh_controls

    controls = _controls()
    controls["mesh"]["variant"] = "cs"
    controls["components"]["flap"] = {"parent": "foil"}
    resolved = mesh_controls(controls, members={"foil", "flap"}, names={"foil", "flap"})
    assert resolved["components"]["flap"]["role"] == "follower"
    assert resolved["components"]["flap"]["follows"] == ["foil"]
    assert resolved["components"]["flap"]["fill"] == "full_quad"
    controls["mesh"]["variant"] = "clean"
    with pytest.raises(InputValueError, match="E16"):
        mesh_controls(controls, members={"foil", "flap"}, names={"foil", "flap"})


def test_geom_mesh_gm_collapse_te_is_mesh_only_and_distance_bounded_yes_no():
    from cfdbender.geometry._geometry_mesh_s1 import collapse_trailing_edge
    from cfdbender.geometry._mesh_topology import Mesh

    xyz = np.array(
        [
            (1, 0, 0.01),
            (1, 1, 0.01),
            (1, 0, -0.01),
            (1, 1, -0.01),
            (0, 0, 0.1),
            (0, 1, 0.1),
            (0, 0, -0.1),
            (0, 1, -0.1),
        ]
    )
    source = Mesh(
        xyz, [(4, 0, 1, 5), (2, 6, 7, 3), (0, 2, 3, 1)], ["foil"] * 3, [1, 2, 3], np.arange(8)
    )
    before = deepcopy(source)
    collapsed, checks = collapse_trailing_edge(source, [(0, 2), (1, 3)], {2}, tolerance_m=0.011)
    assert len(collapsed.xyz) == 6 and len(collapsed.cells) == 2
    assert all(len(c) == 4 for c in collapsed.cells)
    assert all(c["status"] == "passed" for c in checks)
    assert checks[0]["measured"] == pytest.approx(0.01)
    assert np.array_equal(source.xyz, before.xyz) and source.cells == before.cells
    with pytest.raises(InputValueError, match="collapse.*distance"):
        collapse_trailing_edge(source, [(0, 2), (1, 3)], {2}, tolerance_m=0.009)
    with pytest.raises(InputValueError, match="base"):
        collapse_trailing_edge(source, [(0, 2), (1, 3)], set(), tolerance_m=0.011)


def test_geom_mesh_gm_chordwise_plateau_and_spanwise_runs():
    from cfdbender.geometry._mesh_schema import spacing_nodes

    chord = spacing_nodes(
        {"cells": 24, "law": "geometric_cap", "ratio": 1.1}, direction="chordwise"
    )
    widths = np.diff(chord)
    assert len(chord) == 25 and chord[0] == 0 and chord[-1] == 1
    assert np.min(np.diff(widths)) >= -1e-14
    assert widths[-1] == pytest.approx(widths[-3])
    table = {
        "cells": 16,
        "law": "constant",
        "root_cells": 3,
        "tip_cells": 3,
        "root_ratio": 1.2,
        "tip_ratio": 1.4,
    }
    widths = np.diff(spacing_nodes(table, direction="spanwise"))
    assert widths[1] / widths[0] == pytest.approx(1.2)
    assert widths[-2] / widths[-1] == pytest.approx(1.4)
    assert np.ptp(widths[3:-3]) < 1e-14
    with pytest.raises(InputValueError, match="constant interior"):
        spacing_nodes({**table, "cells": 6}, direction="spanwise")


def test_geom_mesh_gm_coarsening_recomputes_laws_instead_of_removing_strips():
    from cfdbender.geometry._mesh_schema import spacing_nodes

    refined = spacing_nodes(
        {"cells": 24, "law": "geometric_cap", "ratio": 1.1}, direction="chordwise"
    )
    coarse = spacing_nodes(
        {"cells": 12, "law": "geometric_cap", "ratio": 1.16}, direction="chordwise"
    )
    assert len(coarse) == 13
    assert not np.allclose(coarse, refined[::2])
    with pytest.raises(InputValueError, match="law"):
        spacing_nodes({"cells": 12, "law": "remove_strips"}, direction="chordwise")


def test_geom_mesh_gm_first_cell_is_honoured_by_the_law():
    from cfdbender.geometry._mesh_schema import spacing_nodes

    nodes = spacing_nodes(
        {"cells": 48, "law": "geometric_cap", "first_cell": 0.0033, "growth_until": 0.515},
        direction="chordwise",
    )
    widths = np.diff(nodes)
    assert widths[0] == pytest.approx(0.0033, abs=1e-14)
    assert np.all(widths > 0) and widths[-1] == pytest.approx(widths[-2])
    with pytest.raises(InputValueError, match="first_cell"):
        spacing_nodes(
            {"cells": 8, "law": "geometric_cap", "first_cell": 0.2}, direction="chordwise"
        )

"""Point location and linear interpolation in an unstructured volume grid.

Every expected value comes from the analytic field, never from the sampler.
"""

from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pytest
from post_volume_support import (
    HEX,
    big_hex_grid,
    box_grid,
    distort,
    mixed_grid,
    with_fields,
)

from cfdbender.post.volume import VolumeError, VolumeSampler
from cfdbender.post.vtu import UnstructuredGrid

KINDS = ("hex", "tet", "wedge", "pyramid")


def linear(p: np.ndarray) -> np.ndarray:
    return 1.5 + 2.0 * p[:, 0] - 3.0 * p[:, 1] + 0.5 * p[:, 2]


def vector(p: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [p[:, 0] - p[:, 2], 4.0 * p[:, 1] + 1.0, 2.0 * p[:, 0] + p[:, 1] - p[:, 2]]
    )


def inside_points(n: int, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.02, 0.98, (n, 3))


@pytest.mark.parametrize("kind", KINDS)
def test_linear_field_is_reproduced_in_every_cell_type(kind: str) -> None:
    grid = with_fields(box_grid(kind), linear, vector)
    points = inside_points(400)
    result = VolumeSampler(grid).sample(points)
    assert result.n_missing == 0
    np.testing.assert_allclose(result.values["f"], linear(points), atol=1e-12)
    np.testing.assert_allclose(result.values["v"], vector(points), atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_linear_field_is_reproduced_in_distorted_cells(kind: str) -> None:
    grid = with_fields(distort(box_grid(kind, (4, 4, 4)), 0.03), linear)
    # points inside the distorted unit-sized box, away from its boundary
    points = np.random.default_rng(5).uniform(0.15, 0.85, (400, 3))
    result = VolumeSampler(grid).sample(points)
    assert result.n_missing == 0
    np.testing.assert_allclose(result.values["f"], linear(points), atol=1e-11)


def test_linear_field_is_reproduced_across_cell_types_of_a_mixed_grid() -> None:
    grid = with_fields(mixed_grid(), linear, vector)
    assert {10, 12, 13, 14} >= set(VolumeSampler(grid).cell_counts)
    assert {12, 13, 14} <= set(VolumeSampler(grid).cell_counts)
    rng = np.random.default_rng(11)
    points = np.column_stack(
        [rng.uniform(0.02, 2.98, 600), rng.uniform(0.02, 0.98, 600), rng.uniform(0.02, 0.98, 600)]
    )
    result = VolumeSampler(grid).sample(points)
    assert result.n_missing == 0
    np.testing.assert_allclose(result.values["f"], linear(points), atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_points_on_the_faces_between_cells_are_found_with_the_exact_value(kind: str) -> None:
    grid = with_fields(box_grid(kind, (3, 3, 3)), linear)
    rng = np.random.default_rng(2)
    planes = np.array([1.0 / 3.0, 2.0 / 3.0])
    points = []
    for axis in range(3):
        for plane in planes:
            q = rng.uniform(0.05, 0.95, (30, 3))
            q[:, axis] = plane
            points.append(q)
    points_all = np.vstack(points)
    result = VolumeSampler(grid).sample(points_all)
    assert result.n_missing == 0
    np.testing.assert_allclose(result.values["f"], linear(points_all), atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_edges_and_nodes_are_found_once_with_the_nodal_value(kind: str) -> None:
    grid = with_fields(box_grid(kind, (2, 2, 2)), linear)
    sampler = VolumeSampler(grid)
    nodes = grid.points[:27]  # lattice nodes come first for every kind
    result = sampler.sample(nodes)
    assert result.location.cell.shape == (27,)
    assert np.all(result.location.cell >= 0)
    np.testing.assert_allclose(result.values["f"], linear(nodes), atol=1e-12)
    # edge midpoints along x between lattice nodes
    mids = 0.5 * (nodes[:-9] + nodes[9:])
    again = sampler.sample(mids)
    assert again.n_missing == 0
    np.testing.assert_allclose(again.values["f"], linear(mids), atol=1e-12)


def test_the_weights_of_a_found_point_sum_to_one_and_a_missing_point_has_none() -> None:
    grid = with_fields(box_grid("hex"), linear)
    sampler = VolumeSampler(grid)
    location = sampler.locate(np.array([[0.3, 0.3, 0.3], [5.0, 5.0, 5.0]]))
    assert location.weights[0].sum() == pytest.approx(1.0, abs=1e-14)
    assert location.cell[1] == -1
    assert location.weights[1].sum() == 0.0


@pytest.mark.parametrize("kind", ["hex", "wedge"])
def test_a_quadratic_field_converges_at_second_order(kind: str) -> None:
    def quad(p: np.ndarray) -> np.ndarray:
        return p[:, 0] ** 2 + p[:, 1] * p[:, 2] + np.sin(p[:, 0] + p[:, 2])

    points = np.random.default_rng(9).uniform(0.05, 0.95, (3000, 3))
    errors = []
    for n in (8, 16, 32):
        grid = with_fields(box_grid(kind, (n, n, n)), quad)
        result = VolumeSampler(grid).sample(points)
        errors.append(float(np.max(np.abs(result.values["f"] - quad(points)))))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(1.8 < order < 2.3 for order in orders), (errors, orders)


def test_a_quadratic_field_converges_at_second_order_on_tetrahedra_and_pyramids() -> None:
    def quad(p: np.ndarray) -> np.ndarray:
        return p[:, 0] ** 2 + p[:, 1] * p[:, 2]

    points = np.random.default_rng(10).uniform(0.05, 0.95, (3000, 3))
    for kind in ("tet", "pyramid"):
        errors = []
        for n in (6, 12, 24):
            grid = with_fields(box_grid(kind, (n, n, n)), quad)
            errors.append(
                float(np.max(np.abs(VolumeSampler(grid).sample(points).values["f"] - quad(points))))
            )
        orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
        assert all(1.8 < order < 2.3 for order in orders), (kind, errors, orders)


def test_points_outside_the_domain_get_no_value() -> None:
    grid = with_fields(box_grid("hex"), linear)
    points = np.array([[1.2, 0.5, 0.5], [-0.01, 0.5, 0.5], [0.5, 0.5, 3.0], [0.5, 0.5, 0.5]])
    result = VolumeSampler(grid).sample(points)
    assert result.n_found == 1 and result.n_missing == 3
    assert np.isnan(result.values["f"][:3]).all()
    assert result.values["f"][3] == pytest.approx(linear(points[3:4])[0], abs=1e-12)


def test_points_in_a_hole_get_no_value_and_are_not_snapped_to_a_node() -> None:
    full = box_grid("hex", (6, 6, 6))
    starts = np.concatenate(([0], full.offsets[:-1]))
    centres = np.array([full.points[full.connectivity[s : s + 8]].mean(axis=0) for s in starts])
    keep = ~np.all(np.abs(centres - 0.5) < 0.2, axis=1)  # a hole of 2 by 2 by 2 cells
    conn = np.concatenate(
        [full.connectivity[s : s + 8] for s, k in zip(starts, keep, strict=True) if k]
    )
    n = int(keep.sum())
    holed = UnstructuredGrid(
        full.points,
        conn,
        np.arange(1, n + 1, dtype=np.int64) * 8,
        np.full(n, HEX, dtype=np.uint8),
        {"f": linear(full.points)},
    )
    sampler = VolumeSampler(holed)
    in_hole = np.array([[0.5, 0.5, 0.5], [0.45, 0.55, 0.5], [0.52, 0.48, 0.55]])
    around = np.array([[0.1, 0.1, 0.1], [0.9, 0.5, 0.5]])
    result = sampler.sample(np.vstack([in_hole, around]))
    assert result.n_missing == 3 and result.n_found == 2
    assert np.isnan(result.values["f"][:3]).all()
    np.testing.assert_allclose(result.values["f"][3:], linear(around), atol=1e-12)


def test_the_exact_search_finds_points_when_the_nearest_centres_are_not_enough() -> None:
    xs = np.cumsum(np.r_[0.0, 1e-3 * 1.9 ** np.arange(14)])  # cells grow by a large ratio
    xs = xs / xs[-1]
    grid = with_fields(box_grid("hex", (len(xs) - 1, 2, 2), xs=xs), linear)
    rng = np.random.default_rng(4)
    points = np.column_stack(
        [rng.uniform(0.0, 1.0, 800), rng.uniform(0.0, 1.0, 800), rng.uniform(0.0, 1.0, 800)]
    )
    narrow = VolumeSampler(grid, neighbours=1).sample(points)
    wide = VolumeSampler(grid).sample(points)
    assert narrow.n_missing == 0 and wide.n_missing == 0
    np.testing.assert_allclose(narrow.values["f"], linear(points), atol=1e-11)
    np.testing.assert_array_equal(narrow.values["f"], wide.values["f"])


def test_cells_that_are_not_searched_are_counted() -> None:
    base = box_grid("hex", (1, 1, 1))
    grid = UnstructuredGrid(
        base.points,
        np.concatenate([base.connectivity, [0, 1, 2]]),
        np.array([8, 11]),
        np.array([12, 5], dtype=np.uint8),
        {"f": linear(base.points)},
    )
    sampler = VolumeSampler(grid)
    assert sampler.skipped_cells == 1
    assert sampler.cell_counts == {5: 1, 12: 1}


def test_an_unknown_field_name_is_refused_by_name() -> None:
    sampler = VolumeSampler(with_fields(box_grid("hex"), linear))
    with pytest.raises(VolumeError, match="'g'"):
        sampler.sample(np.zeros((1, 3)) + 0.5, ["g"])


def test_a_grid_without_searched_cells_misses_every_point() -> None:
    base = box_grid("hex", (1, 1, 1))
    grid = UnstructuredGrid(
        base.points, np.array([0, 1, 2]), np.array([3]), np.array([5], dtype=np.uint8), {}
    )
    result = VolumeSampler(grid).sample(np.array([[0.5, 0.5, 0.5]]))
    assert result.n_found == 0 and result.n_missing == 1


def test_a_request_that_is_not_n_by_3_is_refused() -> None:
    with pytest.raises(VolumeError, match="shape"):
        VolumeSampler(box_grid("hex")).locate(np.zeros((4, 2)))


def test_fifty_thousand_cells_locate_a_thousand_points() -> None:
    grid = big_hex_grid(37)
    sampler = VolumeSampler(grid)
    points = np.random.default_rng(1).uniform(0.0, 1.0, (1000, 3))
    result = sampler.sample(points)
    assert result.n_missing == 0
    np.testing.assert_allclose(
        result.values["f"], points[:, 0] + 2 * points[:, 1] - points[:, 2], atol=1e-12
    )


@pytest.mark.skipif(
    os.environ.get("CFDBENDER_SLOW") != "1", reason="set CFDBENDER_SLOW=1 for the million-cell run"
)
def test_a_million_cells_are_indexed_and_a_hundred_thousand_points_located() -> None:
    grid = big_hex_grid(100)
    assert grid.n_cells == 1_000_000
    points = np.random.default_rng(2).uniform(0.0, 1.0, (100_000, 3))
    tracemalloc.start()
    t0 = time.perf_counter()
    sampler = VolumeSampler(grid)
    t_index = time.perf_counter() - t0
    after_index, peak_index = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    result = sampler.sample(points)
    t_locate = time.perf_counter() - t0
    peak_locate = tracemalloc.get_traced_memory()[1] - after_index
    tracemalloc.stop()
    print(
        f"million cells: index {t_index:.2f} s, peak {peak_index / 2**20:.0f} MiB; "
        f"locate 100000 points {t_locate:.2f} s, extra peak {peak_locate / 2**20:.0f} MiB"
    )
    assert t_index < 60.0 and t_locate < 120.0
    assert result.n_missing == 0
    np.testing.assert_allclose(
        result.values["f"], points[:, 0] + 2 * points[:, 1] - points[:, 2], atol=1e-11
    )

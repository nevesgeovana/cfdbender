"""Collect must cut only the requested family of a separable surface."""

import csv
import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_wiring_products import _su2_window

from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import _cell
from cfdbender.post.vtu import read_vtu


def test_separated_surface_does_not_include_other_families(tmp_path, monkeypatch):
    root, point, adapter = _su2_window(tmp_path)
    inputs = adapter.product_inputs(point)
    grid = read_vtu(root / inputs.surface[0].path)
    # Two coincident manufactured wings, with independent nodes and double pressure
    # variation on B2: all-family loading is 300, B1 alone must remain 100 N/m.
    n = len(grid.points)
    merged = replace(
        grid,
        points=np.concatenate([grid.points, grid.points]),
        connectivity=np.concatenate([grid.connectivity, grid.connectivity + n]),
        offsets=np.concatenate([grid.offsets, grid.offsets + len(grid.connectivity)]),
        types=np.tile(grid.types, 2),
        point_data={
            key: np.concatenate(
                [value, value if key != "Pressure" else 2 * value - inputs.free_stream.p_inf]
            )
            for key, value in grid.point_data.items()
        },
        cell_data={"family": np.repeat([0, 1], len(grid.types))},
    )
    monkeypatch.setattr("cfdbender.matrix._point_products.read_vtu", lambda path: merged)
    supplied = replace(
        inputs, families=("B1", "B2"), families_separable=True, family_field="family"
    )
    proxy = SimpleNamespace(PRODUCTS=adapter.PRODUCTS, product_inputs=lambda point: supplied)
    found = point_products(
        root,
        "7001",
        "DP",
        point,
        proxy,
        lead={},
        moment_point=(0, 0, 0),
        cell=_cell,
        target=lambda p: root / p,
    )
    loads = next(f for f in found.files.values() if f["product"] == "sections_sloads")
    rows = list(csv.DictReader(io.StringIO(loads["text"])))
    assert [float(row["Fz"]) for row in rows] == pytest.approx([100] * 8)

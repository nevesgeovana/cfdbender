"""Undefined dynamic pressure cannot turn a friction coefficient into reported loads."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_wiring_products import _su2_window

from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import _cell


@pytest.mark.parametrize("q_ref", [0.0, float("nan"), float("inf")])
def test_invalid_dynamic_pressure_skips_sections(tmp_path, q_ref):
    root, point, adapter = _su2_window(tmp_path)
    inputs = adapter.product_inputs(point)
    supplied = replace(inputs, free_stream=replace(inputs.free_stream, q_ref=q_ref))
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
    assert not any(f["product"].startswith("sections_") for f in found.files.values())
    assert any("sloads" in product for product, reason in found.skipped)
    assert any(f["product"] == "probes" for f in found.files.values())

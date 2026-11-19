"""A missing family field skips the surface products without losing probes."""

from dataclasses import replace
from types import SimpleNamespace

from test_wiring_products import _su2_window

from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import _cell


def test_missing_cell_family_field_is_isolated(tmp_path):
    root, point, adapter = _su2_window(tmp_path)
    inputs = replace(adapter.product_inputs(point), families_separable=True, family_field="family")
    proxy = SimpleNamespace(PRODUCTS=adapter.PRODUCTS, product_inputs=lambda point: inputs)
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
    assert any(item["product"] == "probes" for item in found.files.values())
    assert not any(item["product"] == "sections_sloads" for item in found.files.values())
    assert any("family" in reason and "field" in reason for _, reason in found.skipped)

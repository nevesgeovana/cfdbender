"""A partial export window must name its missing sources in the collect report."""

from test_wiring_products import _su2_window

from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import _cell


def test_partial_window_names_missing_surface_and_volume(tmp_path):
    root, point, adapter = _su2_window(tmp_path)
    folder = root / point.point.directory
    for name in ("flow_00034.vtu", "surface_flow_00034.vtu"):
        (folder / name).unlink()
    found = point_products(
        root,
        "7001",
        "DP",
        point,
        adapter,
        lead={},
        moment_point=(0, 0, 0),
        cell=_cell,
        target=lambda p: root / p,
    )
    assert found.files
    for name in ("flow_00034.vtu", "surface_flow_00034.vtu"):
        assert any(
            name in product and "did not write" in reason for product, reason in found.skipped
        ), name

"""Empty PyFR output is failed evidence and must remain collectable."""

import pytest
from test_conformance_workspace import build, prepared


@pytest.mark.parametrize("rows", ["", "4,0.1,0.3,0,0\n"])
def test_no_samples_in_the_pyfr_window_returns_a_failed_result(tmp_path, rows):
    root = build(tmp_path, "pyfr")
    adapter, point = prepared(root, "pyfr")
    adapter.stage(point)
    (root / point.point.directory / "forces.csv").write_text("t,px,py,vx,vy\n" + rows)
    result = adapter.collect(point)
    assert result.classification == "failed"
    assert not result.coefficients

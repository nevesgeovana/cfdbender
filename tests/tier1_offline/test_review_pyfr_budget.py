"""An unfinished PyFR output cannot establish that its time budget was exhausted."""

import pytest
from test_conformance_workspace import build, prepared


@pytest.mark.parametrize("rows", ["0,0.1,0.3,0,0\n", "0,0.1,0.3,0,0\n1,0.1,0.8,0,0\n"])
def test_pyfr_does_not_call_truncated_output_an_iteration_limit(tmp_path, rows):
    root = build(tmp_path, "pyfr")
    adapter, point = prepared(root, "pyfr")
    adapter.stage(point)
    (root / point.point.directory / "forces.csv").write_text("t,px,py,vx,vy\n" + rows)
    assert adapter.collect(point).classification == "failed"

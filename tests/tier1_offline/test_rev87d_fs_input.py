"""R2-05: growing a matrix must not exempt other tracked .fs files from digest checks."""

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("operation", ["translate", "register", "start"])
def test_changed_fs_geometry_is_still_stale(tmp_path, operation):
    workspace(tmp_path)
    geometry = tmp_path / "inputs/mesh/shape.fs"
    geometry.parent.mkdir()
    geometry.write_text("planned geometry")
    (tmp_path / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SETUP|REF|PPROC|MESH\n1001|1|001|001|001|shape.fs\n"
    )
    plan = plan_workspace(tmp_path, "study")
    point = load_point(tmp_path, plan.points[0], solver="mock")
    registry = RunRegistry(tmp_path)
    if operation == "start":
        registry.record(plan)
    geometry.write_text("changed geometry")
    with pytest.raises(ValueError, match="input changed"):
        if operation == "translate":
            from cfdbender.solvers import mock

            mock.translate(point)
        elif operation == "register":
            registry.record(plan)
        else:
            registry.transition(plan.points[0].point_id, "RUNNING")

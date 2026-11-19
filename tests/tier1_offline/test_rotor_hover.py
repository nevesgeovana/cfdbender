"""A turning rotor at zero free-stream speed still has dimensional loads and CT."""

from __future__ import annotations

from test_runtypes_b1_workflows import workspace

from cfdbender.core.adapter import load_point
from cfdbender.solvers.su2._config import config
from cfdbender.solvers.su2._rotor import rotor_point
from cfdbender.workspace.planning import plan_workspace


def test_hover_keeps_rotor_coefficients(tmp_path):
    workspace(tmp_path, workflow="qsteady_rotor", condition="MACH:0, RPM:1000")
    plan = plan_workspace(tmp_path, "rt")
    point = load_point(tmp_path, plan.points[0], solver="su2")
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )
    result = rotor_point(
        values,
        point.reference,
        dict.fromkeys(("CFx", "CFy", "CFz", "CMx", "CMy", "CMz"), 1.0),
        condition=point.point.condition,
    )
    assert result is not None
    assert result["row"]["J"] == 0.0
    assert result["row"]["CT"] > 0.0
    assert result["row"]["ETA"] == "NA"

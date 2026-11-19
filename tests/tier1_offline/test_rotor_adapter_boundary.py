"""Neutral rotor axes are accepted; SU2 states its own plan-time restriction."""

import pytest
from test_runtypes_b1_workflows import ROTOR, workspace

from cfdbender.core.rotor import periodic_state, rotor_block
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("shaft,zero", [("X", "Z"), ("Y", "Z"), ("Z", "Y")])
@pytest.mark.parametrize("table", [False, True])
def test_neutral_rotor_accepts_declared_axes(shaft, zero, table):
    vector = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[shaft]
    axis = {"shaft": shaft, "vector_wrt_global": vector} if table else shaft
    rotor = rotor_block(
        {
            "R": {
                "kind": "rotor",
                "axis": axis,
                "diameter_m": 1,
                "blade1": {"azimuth_deg": 0, "zero": zero},
            }
        }
    )
    assert rotor is not None and rotor.axis == tuple(vector)


@pytest.mark.parametrize("shaft,zero", [("Y", "Z"), ("Z", "Y")])
def test_su2_refuses_other_shafts_at_plan_time(tmp_path, shaft, zero):
    rotor = ROTOR.replace('axis = "X"', f'axis = "{shaft}"').replace(
        'zero = "Z"', f'zero = "{zero}"'
    )
    workspace(tmp_path, workflow="qsteady_rotor", rotor=rotor)
    with pytest.raises(ValueError, match=f"shaft {shaft}: the SU2 adapter"):
        plan_workspace(tmp_path, "rt")


def test_neutral_symmetry_error_names_no_solver():
    with pytest.raises(ValueError) as caught:
        periodic_state({"SYMMETRY": "MIRROR"})
    assert "SU2" not in str(caught.value)
    assert "NONE or PERIODIC" in str(caught.value)

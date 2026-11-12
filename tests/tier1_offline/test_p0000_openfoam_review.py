"""Classification uses the final logged operating iteration."""

import json

import pytest
from test_p0000_openfoam_adapter import api, case, point


@pytest.mark.parametrize("mode", ["stale_force", "early_message", "stale_residual", "stale_limit"])
def test_p0000_openfoam_final_iteration_evidence(tmp_path, mode):
    plan = case(tmp_path, sweep="4")
    folder = tmp_path / plan.points[0].directory
    force = folder / "postProcessing/forceCoeffs/0/coefficient.dat"
    force.parent.mkdir(parents=True)
    row_time = 4 if mode == "stale_limit" else 2
    force.write_text(f"# Time Cd Cl CmPitch\n{row_time} 0.03 0.47 -0.02\n")
    (folder / "execution.json").write_text(json.dumps({"exit_code": 0}))
    low = "\n".join(
        f"Solving for {field}, Initial residual = 1e-8, Final residual = 1e-9"
        for field in ("Ux", "Uy", "p", "nuTilda")
    )
    logs = {
        "stale_force": "Time = 3\nSIMPLE solution converged in 3 iterations\n",
        "early_message": "Time = 1\nSIMPLE solution converged in 1 iterations\nTime = 2\n",
        "stale_residual": f"Time = 2\n{low}\nTime = 3\n",
        "stale_limit": "Time = 5\n",
    }
    (folder / "log.simpleFoam").write_text(logs[mode])
    assert api().collect(point(plan)).classification == "failed"

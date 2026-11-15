"""SU2INV: a RANS deck without CFL keys gets the recipe that converged the TMR NACA 0012.

Measured on SU2 8.6: the alpha 0 plateau of SU2 on
the TMR 225 x 65 grid is pseudo-time stiffness in the wake-cut cells (aspect ratio up to 1e7),
removed only by a CFL near 1e5. The recipe holds CFL 1000 for 300 iterations, then raises it by
20 percent per iteration to 1e5 and never lowers it, so SU2's flip-flop test cannot pull it back on
the plateau. 50 linear iterations keep the linear residual below SU2's reset level; the unweighted
least-squares reconstruction gradient keeps CFL 1000 stable without a limiter on the cut cells.
"""

from __future__ import annotations

import test_p0000_item3d_config as su2_case

RANS_RAMP = "CFL_ADAPT_PARAM= ( 1, 1.2, 1000, 100000, 1, 300 )"


def _lines(root, extra="", solver="RANS"):
    """Translate the fixture deck as `solver`, without its CFL_NUMBER so the default applies."""
    su2_case.workspace(root)
    setup = root / "inputs/setups/s001.toml"
    kept = [
        f'SOLVER = "{solver}"' if line.startswith("SOLVER") else line
        for line in setup.read_text(encoding="utf-8").splitlines()
        if not line.startswith("CFL")
    ]
    setup.write_text("\n".join([*kept, extra, ""]), encoding="utf-8")
    return su2_case.api().translate(su2_case.prepared(root)).splitlines()


def test_rans_default_ramp_is_the_naca0012_recipe(tmp_path):
    lines = _lines(tmp_path)
    assert "CFL_NUMBER= 1000" in lines
    assert "CFL_ADAPT= YES" in lines
    assert RANS_RAMP in lines
    assert "LINEAR_SOLVER_ITER= 50" in lines
    assert "NUM_METHOD_GRAD_RECON= LEAST_SQUARES" in lines


def test_rans_user_keys_win(tmp_path):
    lines = _lines(
        tmp_path, 'LINEAR_SOLVER_ITER = 20\nNUM_METHOD_GRAD_RECON = "WEIGHTED_LEAST_SQUARES"'
    )
    assert "LINEAR_SOLVER_ITER= 20" in lines
    assert "NUM_METHOD_GRAD_RECON= WEIGHTED_LEAST_SQUARES" in lines
    assert RANS_RAMP in lines


def test_rans_native_cfl_key_switches_the_ramp_and_linear_default_off(tmp_path):
    lines = _lines(tmp_path, "CFL_NUMBER = 25")
    assert "CFL_NUMBER= 25" in lines
    assert not any(line.startswith("CFL_ADAPT") for line in lines)
    assert not any(line.startswith("LINEAR_SOLVER_ITER=") for line in lines)


def test_rans_ramp_keys_keep_the_recipe_slots(tmp_path):
    lines = _lines(tmp_path, "cfl_max = 50000")
    assert "CFL_NUMBER= 1000" in lines
    assert "CFL_ADAPT_PARAM= ( 1, 1.2, 1000, 50000, 1, 300 )" in lines


def test_euler_default_ramp_is_unchanged(tmp_path):
    lines = _lines(tmp_path, solver="EULER")
    assert "CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 25 )" in lines
    assert not any(line.startswith("NUM_METHOD_GRAD_RECON=") for line in lines)
    assert not any(line.startswith("LINEAR_SOLVER_ITER=") for line in lines)

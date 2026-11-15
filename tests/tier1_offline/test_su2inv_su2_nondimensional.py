"""SU2INV: a compressible SU2 deck is scaled by the free stream unless the user states otherwise.

SU2's default REF_DIMENSIONALIZATION is DIMENSIONAL. On the TMR NACA 0012 RANS case at CFL 1000
that default alone turns SU2's official deck, which converges in 688 iterations, into a plateau at
rms[Rho] -4.39 for 1500 iterations: the energy rows of a
dimensional system are about 1e5 times the others and the linear solver no longer solves them.
"""

from __future__ import annotations

import pytest
import test_p0000_item3d_config as su2_case


def _lines(root, solver="EULER", extra=""):
    su2_case.workspace(root)
    setup = root / "inputs/setups/s001.toml"
    kept = [
        f'SOLVER = "{solver}"' if line.startswith("SOLVER") else line
        for line in setup.read_text(encoding="utf-8").splitlines()
    ]
    setup.write_text("\n".join([*kept, extra, ""]), encoding="utf-8")
    return su2_case.api().translate(su2_case.prepared(root)).splitlines()


@pytest.mark.parametrize("solver", ["EULER", "NAVIER_STOKES", "RANS"])
def test_compressible_deck_is_scaled_by_the_free_stream(tmp_path, solver):
    lines = _lines(tmp_path, solver)
    assert "REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE" in lines


def test_user_dimensionalization_wins(tmp_path):
    lines = _lines(tmp_path, "RANS", 'REF_DIMENSIONALIZATION = "DIMENSIONAL"')
    assert "REF_DIMENSIONALIZATION= DIMENSIONAL" in lines
    assert sum(line.startswith("REF_DIMENSIONALIZATION=") for line in lines) == 1


def test_incompressible_deck_gets_no_compressible_scaling(tmp_path):
    lines = _lines(tmp_path, "INC_NAVIER_STOKES")
    assert not any(line.startswith("REF_DIMENSIONALIZATION=") for line in lines)

"""The QPROP non-converged-section policy against a real XFOIL run.

tests/data/qprop_xfoil_unconverged holds a captured section run (the xfoil.in that was sent,
the polar.txt XFOIL saved and the part of stdout.log around the failure): the viscous solution
did not converge at 14 degrees, XFOIL printed ``VISCAL:  Convergence failed`` and saved no row
for that angle, and the sweep went on to 15 and 16.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfdbender.solvers.qprop._execute import _fit_folder, _missing_angles, _unconverged

DATA = Path(__file__).resolve().parents[1] / "data" / "qprop_xfoil_unconverged"
SETUP = SimpleNamespace(alpha=(-8.0, 16.0, 1.0))


def test_the_captured_stdout_shows_the_failure_at_the_missing_angle():
    text = (DATA / "stdout.log").read_text(encoding="ascii")
    failed_at = text.index("VISCAL:  Convergence failed")
    assert "a = 14.000" in text[:failed_at].rsplit("rms:", 3)[-1]
    assert "ASEQ -8 16 1" in (DATA / "xfoil.in").read_text(encoding="ascii")


def test_the_polar_lacks_exactly_the_angle_xfoil_failed_at():
    assert _missing_angles(DATA, SETUP, narrow=False) == [14.0]
    assert _fit_folder(DATA) is not None


def test_warn_drops_the_angle_and_names_it():
    label = "section 5/10 (r = 0.1)"
    with pytest.warns(UserWarning, match=r"section 5/10.*did not converge at alpha 14;") as seen:
        _unconverged({"on_unconverged_section": "warn"}, label, [14.0])
    assert "dropped from this section's polar" in str(seen[0].message)


def test_fail_refuses_the_point_and_says_how_to_run_on_the_rest():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"did not converge at alpha 14.*set it to 'warn'"):
            _unconverged({"on_unconverged_section": "fail"}, "section 5/10", [14.0])

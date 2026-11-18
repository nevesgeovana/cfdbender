"""The rotor coefficient formulas live in one module; every other caller reads them there."""

from __future__ import annotations

import inspect
import math
import re

import pytest

import cfdbender.core.rotor as rotor
import cfdbender.core.rotor_coefficients as home
import cfdbender.solvers.qprop._output as qprop

CASES = [
    # thrust N, torque N m, rps signed, diameter m, density kg/m3, speed m/s
    (72.18, 3.398, 107.34, 0.4064, 1.22498, 18.0017),
    (12.5, -0.7, -48.2, 1.1, 1.18, 30.0),
    (0.3, 0.01, 3.7, 0.2, 0.9, 0.0),
    (1500.0, 210.0, 25.0, 2.5, 1.225, 61.0),
]


def _old_rotor(thrust, torque, rps, diameter, density, speed):
    """The definitions of record: J, CT, CQ, CP, ETA with n in signed rev/s."""
    rate = abs(rps)
    ct = thrust / (density * rate * rate * diameter**4)
    cq = torque / (density * rate * rate * diameter**4 * diameter)
    cp = 2.0 * math.pi * cq * (1.0 if rps > 0 else -1.0)
    j = speed / (rate * diameter)
    return j, ct, cq, cp


@pytest.mark.parametrize("case", CASES)
def test_the_rotor_coefficients_keep_their_values(case):
    thrust, torque, rps, diameter, density, speed = case
    got = rotor.rotor_coefficients(
        thrust_n=thrust,
        torque_nm=torque,
        rps=rps,
        diameter_m=diameter,
        density_kg_m3=density,
        speed_m_s=speed,
    )
    j, ct, cq, cp = _old_rotor(thrust, torque, rps, diameter, density, speed)
    assert got["J"] == pytest.approx(j, rel=1e-14)
    assert got["CT"] == pytest.approx(ct, rel=1e-14)
    assert got["CQ"] == pytest.approx(cq, rel=1e-14)
    assert got["CP"] == pytest.approx(cp, rel=1e-14)
    if speed == 0.0:
        assert got["ETA"] == "NA"
    else:
        assert got["ETA"] == pytest.approx(j * ct / cp, rel=1e-13)


def test_the_qprop_coefficients_keep_their_values():
    row = {
        "V(m/s)": 20.0,
        "rpm": 6000.0,
        "T(N)": 30.0,
        "Q(N-m)": 1.2,
        "Pshaft(W)": 753.98,
        "CT": 0.1,
        "CP": 0.05,
        "effprop": 0.7,
        "eff": 0.6,
    }
    got = qprop.coefficients(row, rho=1.2, radius=0.15)
    n, d = 100.0, 0.3
    assert got["J"] == pytest.approx(20.0 / (n * d), rel=1e-14)
    assert got["CT"] == pytest.approx(30.0 / (1.2 * n**2 * d**4), rel=1e-14)
    assert got["CQ"] == pytest.approx(1.2 / (1.2 * n**2 * d**5), rel=1e-14)
    assert got["CP"] == pytest.approx(753.98 / (1.2 * n**3 * d**5), rel=1e-14)
    assert (got["EFFICIENCY"], got["T"], got["P"]) == (0.7, 30.0, 753.98)


def test_one_function_is_the_formula_and_the_others_call_it():
    assert rotor.rotor_coefficients is home.rotor_coefficients
    for module in (rotor, qprop):
        source = re.sub(r"\s+", "", inspect.getsource(module))
        for power in ("**4", "**5", "**3"):
            assert power not in source, f"{module.__name__} computes a normalisation itself"
    frame = re.sub(r"\s+", "", inspect.getsource(home.rotor_frame_coefficients))
    assert "**4" not in frame and "force_scale(" in frame

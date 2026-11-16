"""P058-12: post keeps rotor loads when a size is missing and writes NA for its ratios.

A failed rotor point may have no diameter. Rendering its run block and RCOEF must
preserve the row; a rotor at rest has zero shaft power even without a measured torque.
"""

import math

import pytest

from cfdbender.core.axes import frame_rotations, rotor_frame
from cfdbender.core.rotor_coefficients import rotor_frame_coefficients
from cfdbender.matrix._run_block import RotorRun, run_block_row


@pytest.mark.parametrize("diameter", [0.0, -1.0, math.nan, math.inf])
def test_p058_missing_diameter_keeps_loads_and_marks_ratios_na(diameter):
    frame = rotor_frame((-1.0, 0.0, 0.0))
    try:
        row = rotor_frame_coefficients(
            force_n=(-100.0, 0.0, 0.0),
            moment_hub_nm=(-5.0, 0.0, 0.0),
            frame=frame,
            rotations=frame_rotations(0, 0, [frame]),
            rpm=600.0,
            diameter_m=diameter,
            density_kg_m3=1.0,
            vinf_m_s=10.0,
            sref_m2=2.0,
            cref_m=0.5,
            bref_m=4.0,
        )
        block = run_block_row(
            pol="1001",
            point_id=1,
            point="DP-test",
            run_type="steady",
            condition={"VINF": 10.0},
            rotors=[RotorRun(0, diameter, 0, 0, rpm=600)],
        )
    except (ValueError, ZeroDivisionError) as error:
        pytest.fail(f"a missing rotor size must keep the point row: {error}")
    assert row["T"] == 100.0 and row["Q"] == 5.0
    assert all(row[key] == "NA" for key in ("CT", "CQ", "CP", "ETA", "TC", "QC", "ETAW"))
    assert row["TCA"] == 1.0
    assert all(block[key] == "NA" for key in ("D0", "ADISC0", "J0"))


def test_p058_zero_speed_has_zero_power_without_a_torque_measurement():
    frame = rotor_frame((-1.0, 0.0, 0.0))
    row = rotor_frame_coefficients(
        force_n=(-100.0, math.nan, math.nan),
        moment_hub_nm=(math.nan,) * 3,
        frame=frame,
        rotations=frame_rotations(0, 0, [frame]),
        rpm=0,
        diameter_m=1,
        density_kg_m3=1,
        vinf_m_s=0,
        sref_m2=None,
        cref_m=None,
        bref_m=None,
    )
    assert row["P"] == 0.0
    assert row["Q"] == row["CT"] == row["ETA"] == "NA"

"""P058-4: the canonical run block opens every table, in the stated order, NA where absent."""

from __future__ import annotations

import pytest

from cfdbender.matrix._run_block import RotorRun, run_block_columns, run_block_row

#: The order stated for the block: fixed columns, identifiers after POINT, rotor groups.
FIXED = [
    "POL", "POINT_ID", "POINT", "RUN_TYPE", "ALPHA", "BETA", "MACH", "RE", "VINF", "VREF",
    "J", "J_CLOCK", "RPM_CLOCK", "THRUST", "ALT", "RHO", "TEMP", "MU", "SREF", "CREF", "BREF",
    "XMOM", "YMOM", "ZMOM", "FIRST_STEP", "LAST_STEP", "STEPS", "WINDOW_REVS",
    "WINDOW_PASSAGES",
]  # fmt: skip


def test_without_rotors_or_identifiers_the_block_is_the_fixed_columns():
    assert list(run_block_columns()) == FIXED


def test_identifiers_follow_point_and_rotors_are_grouped_by_id():
    columns = list(run_block_columns(("BETA07",), (1, 0)))
    assert columns[:5] == ["POL", "POINT_ID", "POINT", "BETA07", "RUN_TYPE"]
    vref = columns.index("VREF")
    assert columns[vref + 1 : vref + 9] == [
        "RPM0", "J0", "MTIP0", "MHEL0", "RPM1", "J1", "MTIP1", "MHEL1",
    ]  # fmt: skip
    zmom = columns.index("ZMOM")
    assert columns[zmom + 1 : zmom + 5] == ["D0", "ADISC0", "PITCH0", "TOE0"]
    with pytest.raises(ValueError, match="ALPHA"):
        run_block_columns(("ALPHA",))
    with pytest.raises(ValueError, match="distinct"):
        run_block_columns((), (0, 0))


def test_a_steady_row_states_no_window_and_writes_mu_in_scientific_notation():
    row = run_block_row(
        pol="1001",
        point_id=2,
        point="DP-AL+020",
        run_type="steady",
        condition={"ALPHA": 2.0, "VINF": 40.0, "MU": 1.789e-05, "FIRST_STEP": 3},
        identifiers={"BETA07": 30.0},
        rotors=[RotorRun(0, 0.4, 0.0, 0.0, rpm=6000.0)],
    )
    assert list(row) == list(run_block_columns(("BETA07",), (0,)))
    assert row["POINT_ID"] == 2 and row["BETA07"] == 30.0 and row["RUN_TYPE"] == "steady"
    assert row["MU"] == "1.78900e-05" and row["FIRST_STEP"] == "NA" and row["RHO"] == "NA"
    assert row["J0"] == pytest.approx(40.0 / (100.0 * 0.4))
    assert row["ADISC0"] == pytest.approx(3.141592653589793 * 0.04)
    with pytest.raises(ValueError, match="run type"):
        run_block_row(pol="1", point_id=1, point="p", run_type="quasi", condition={})
    with pytest.raises(ValueError, match="1-based"):
        run_block_row(pol="1", point_id=0, point="p", run_type="steady", condition={})

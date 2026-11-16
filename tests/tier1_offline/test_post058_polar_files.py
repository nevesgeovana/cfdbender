"""P058-6: collect writes FAM-DIM, FAM-ACOEF and FAM-RCOEF per polar and one set per group.

The expected values come from the stated rules: in the body frame CD, CY and CL are the
geometry-frame force over q SREF along aft, right and up, the DIM force is that force in
newtons, a group of one family carries that family's values, and QPROP's thrust and torque
are the R0 thrust and torque of a forward shaft.
"""

from __future__ import annotations

import csv
import math

import pytest
import qprop_workspace as qw
from test_matrix_selection import three
from test_p0000_item3c_adapter import workspace

from cfdbender.matrix._run_block import run_block_columns
from cfdbender.matrix.cli import main


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def collected(root, matrix="study"):
    assert main(["run", matrix, "--workspace", str(root)]) == 0
    assert main(["collect", matrix, "--workspace", str(root)]) == 0
    return root / "post" / matrix / "polars"


def test_an_airframe_matrix_writes_family_and_group_files_per_polar(tmp_path):
    three(tmp_path)
    polars = collected(tmp_path)
    expected = {
        name
        for pol in ("1001", "1002", "1003")
        for name in (
            f"FAM-DIM-P{pol}.csv",
            f"FAM-ACOEF-P{pol}.csv",
            f"P{pol}-TOTAL-DIM.csv",
            f"P{pol}-TOTAL-ACOEF.csv",
            f"P{pol}-AIRFRAME-DIM.csv",
            f"P{pol}-AIRFRAME-ACOEF.csv",
        )
    }
    assert {p.name for p in polars.iterdir()} == expected
    family = read(polars / "FAM-ACOEF-P1001.csv")
    assert [r["POINT"] for r in family] == ["DP-AL+000", "DP-AL+020"]
    assert [r["POINT_ID"] for r in family] == ["1", "2"]
    assert {r["FAMILY"] for r in family} == {"BODY"} and {r["FAMILY_ID"] for r in family} == {"1"}
    header = list(family[0])
    block = list(run_block_columns())
    assert header[: len(block)] == block
    first = family[0]
    # ALPHA 0: the wind, stability and body frames coincide; the mock's CX, CZ and Cm are
    # its drag, lift and pitching moment.
    for frame in "WSB":
        assert float(first[f"CD{frame}"]) == pytest.approx(float(first["CX"]))
        assert float(first[f"CL{frame}"]) == pytest.approx(float(first["CZ"]))
        assert float(first[f"Cm{frame}"]) == pytest.approx(float(first["Cm"]))
    assert first["CYW"] == "NA", "the mock states no side force in a frame"
    total = read(polars / "P1001-TOTAL-ACOEF.csv")
    assert [r["GROUP"] for r in total] == ["TOTAL", "TOTAL"] and total[0]["GROUP_ID"] == "0"
    for name in ("CDW", "CLW", "CmB", "CX"):
        assert [r[name] for r in total] == [r[name] for r in family]
    assert read(polars / "P1001-AIRFRAME-DIM.csv")[0]["GROUP_ID"] == "1"


def test_the_dimensional_loads_are_the_body_coefficients_times_q_sref(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        "1001|1|001|001|001|MACH:0.2, ALTFT:0, ALPHA:sweep|4\n",
        encoding="ascii",
    )
    polars = collected(tmp_path)
    dim, acoef = read(polars / "FAM-DIM-P1001.csv")[0], read(polars / "FAM-ACOEF-P1001.csv")[0]
    q = 0.5 * float(dim["RHO"]) * float(dim["VINF"]) ** 2
    q_area = q * float(dim["SREF"])
    assert dim["VREF"] == dim["VINF"]
    assert float(dim["FX_MRP"]) / q_area == pytest.approx(float(acoef["CDB"]))
    assert float(dim["FZ_MRP"]) / q_area == pytest.approx(float(acoef["CLB"]))
    assert float(dim["MY_MRP"]) / (q_area * float(dim["CREF"])) == pytest.approx(
        float(acoef["CmB"])
    )
    assert dim["FY_MRP"] == dim["MX_MRP"] == dim["MZ_MRP"] == "NA"
    a = math.radians(4.0)
    drag = float(acoef["CDB"]) * math.cos(a) + float(acoef["CLB"]) * math.sin(a)
    assert float(acoef["CDW"]) == pytest.approx(drag)
    assert float(acoef["CDW"]) == pytest.approx(float(acoef["CX"]))


def test_qprop_writes_rotor_coefficients_and_skips_the_empty_airframe(tmp_path):
    qw.workspace(tmp_path, condition="V:sweep,RPM:6000,BETA07:30,RHO:1.2,MU:1.8e-5,A:340")
    (tmp_path / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        "1001|1|001|001|001|V:sweep,RPM:6000,BETA07:30,RHO:1.2,MU:1.8e-5,A:340|20,0\n",
        encoding="ascii",
    )
    polars = collected(tmp_path)
    names = {p.name for p in polars.iterdir()}
    assert "FAM-RCOEF-P1001.csv" in names and "P1001-PROPELLER-RCOEF.csv" in names
    assert not any("AIRFRAME" in name for name in names)
    acoef = read(polars / "FAM-ACOEF-P1001.csv")
    rows = read(polars / "FAM-RCOEF-P1001.csv")
    moving, static = rows
    assert moving["ROTOR_ID"] == "0" and moving["FAMILY"] == "BLADES"
    assert moving["RPM0"] == "6000.0" and float(moving["D0"]) > 0
    assert float(moving["T"]) == pytest.approx(float(acoef[0]["T"]))
    assert float(moving["CT"]) == pytest.approx(float(acoef[0]["CT"]))
    assert float(moving["CQ"]) == pytest.approx(float(acoef[0]["CQ"]))
    assert float(moving["J0"]) == pytest.approx(20.0 / (100.0 * float(moving["D0"])))
    assert moving["CTR0"] == moving["CT"] and moving["CFZ"] == "NA"
    assert float(moving["ETA"]) > 0 and static["ETA"] == static["TC"] == "NA"
    assert float(static["CT"]) == pytest.approx(float(acoef[1]["CT"]))
    group = read(polars / "P1001-PROPELLER-RCOEF.csv")
    assert group[0]["GROUP_ID"] == "10" and group[0]["CT"] == moving["CT"]
    # The forward thrust is a negative drag in the body frame.
    assert float(acoef[0]["CDB"]) < 0

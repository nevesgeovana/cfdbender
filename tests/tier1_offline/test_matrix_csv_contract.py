"""The CSV tables follow pyflightstream's column contract: fixed context, NA, no quoting."""

from __future__ import annotations

import math

import pytest
from test_matrix_selection import three
from test_p0000_item3c_adapter import workspace

from cfdbender.matrix.cli import main

CONTEXT = [
    "ALPHA",
    "BETA",
    "MACH",
    "RE",
    "VINF",
    "VREF",
    "J",
    "J_CLOCK",
    "RPM_CLOCK",
    "ALT",
    "RHO",
    "TEMP",
    "MU",
    "SREF",
    "CREF",
    "BREF",
]


def header_of(path):
    return path.read_text(encoding="utf-8").splitlines()[0].split(",")


def test_every_table_states_the_fixed_context_after_pol_and_point(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    for path in (tmp_path / "post/study/polars").iterdir():
        header = header_of(path)
        assert header[:3] == ["POL", "POINT_ID", "POINT"]
        assert [name for name in header if name in CONTEXT] == CONTEXT


def test_a_missing_value_is_na_and_never_empty_or_quoted(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    text = (tmp_path / "post/study/polars/FAM-ACOEF-P1001.csv").read_text(encoding="utf-8")
    assert '"' not in text and ",," not in text and not text.splitlines()[1].endswith(",")
    names = header_of(tmp_path / "post/study/polars/FAM-ACOEF-P1001.csv")
    first = dict(zip(names, text.splitlines()[1].split(","), strict=True))
    assert first["VREF"] == first["BREF"] == first["RPM_CLOCK"] == "NA"
    assert float(first["ALPHA"]) == 0.0
    assert float(first["SREF"]) > 0 and float(first["CREF"]) > 0


def test_numpy_reads_the_table_to_the_header_count(tmp_path):
    np = pytest.importorskip("numpy")
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    path = tmp_path / "post/study/polars/FAM-ACOEF-P1001.csv"
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(table) == 2 and len(table.dtype.names) == len(header_of(path))
    assert table["ALPHA"].tolist() == [0.0, 2.0]


def test_a_stated_air_state_fills_velocity_density_and_reynolds(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        "1001|1|001|001|001|MACH:0.2, ALTFT:0, ALPHA:sweep|0\n",
        encoding="ascii",
    )
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    path = tmp_path / "post/study/polars/FAM-ACOEF-P1001.csv"
    row = dict(
        zip(
            header_of(path),
            path.read_text(encoding="utf-8").splitlines()[1].split(","),
            strict=True,
        )
    )
    assert float(row["MACH"]) == 0.2 and float(row["ALT"]) == 0.0
    assert math.isclose(float(row["RHO"]), 1.225, abs_tol=1e-3)
    assert math.isclose(float(row["VINF"]), 0.2 * 340.294, abs_tol=0.1)
    assert math.isclose(float(row["TEMP"]), 288.15, abs_tol=1e-6)
    assert row["RE"] != "NA" and float(row["MU"]) > 1e-5


def test_a_condition_key_spelled_as_a_context_column_fills_it_once(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        "1001|1|001|001|001|mach:0.2, alpha:sweep|2\n",
        encoding="ascii",
    )
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    path = tmp_path / "post/study/polars/FAM-ACOEF-P1001.csv"
    header = header_of(path)
    assert len(header) == len(set(header)), "a column is stated twice"
    first = path.read_text(encoding="utf-8").splitlines()[1].split(",")
    row = dict(zip(header, first, strict=True))
    assert float(row["MACH"]) == 0.2 and float(row["ALPHA"]) == 2.0


def test_text_cells_hold_no_comma_quote_or_line_break():
    from cfdbender.matrix.exports import plain_cell

    assert plain_cell('a,b "c"\nd') == "a;b 'c' d"

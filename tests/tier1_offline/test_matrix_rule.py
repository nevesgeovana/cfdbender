"""The dashed separator rule under the matrix header: written, shipped and read."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cfdbender.core.matrix import MatrixError, parse_matrix, read_matrix

ROOT = Path(__file__).resolve().parents[2]
HEADER = "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION"
ROW_A = "1001|1|001|001|001|MACH:0.3,ALPHA:2.0"
ROW_B = "1002|1|001|001|001|MACH:0.4,ALPHA:2.0"


def _pols(text: str) -> list[str]:
    rows = parse_matrix(text.encode(), Path("m.fs"), active_only=False)
    return [row.pol for row in rows]


@pytest.mark.parametrize(
    "rule",
    [
        "---|---|---|---|-----|----------------",
        "-|-|-",
        "---:|:---|:---:|---|---|---",
        "  :---  |  ---:  ",
        "--- | --- | ---",
        "+---+---+",
        "\t---\t|\t---",
        "|---|---|",
    ],
)
def test_any_dashed_rule_is_a_rule_never_a_data_row(rule):
    assert _pols("\n".join([HEADER, rule, ROW_A, ""])) == ["1001"]


def test_a_dashed_line_in_the_middle_of_data_is_dropped_like_pyflightstream():
    text = "\n".join([HEADER, "---|---", ROW_A, ":--|--:|-", ROW_B, "----", ""])
    assert _pols(text) == ["1001", "1002"]


def test_data_row_numbers_ignore_the_rule():
    text = "\n".join([HEADER, "---|---|---|---|---|---", ROW_A, ROW_A.replace("1001", "1002"), ""])
    bad = text.replace("1002|1|001|001|001|MACH:0.3,ALPHA:2.0", "1002|1|001|001")
    with pytest.raises(MatrixError, match="data row 2 needs 6 columns"):
        parse_matrix(bad.encode(), Path("m.fs"))


def test_a_matrix_without_a_rule_still_reads():
    assert _pols("\n".join([HEADER, ROW_A, ""])) == ["1001"]


def _documented_matrices() -> list[tuple[str, str]]:
    found = []
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for block in re.findall(r"```text\r?\n(.*?)```", text, re.S):
            lines = block.splitlines()
            if lines and lines[0].lstrip().startswith("POL") and "|" in lines[0]:
                found.append((path.name, block))
    return found


def test_every_documented_matrix_is_found():
    names = {name for name, _ in _documented_matrices()}
    assert {
        "matrix-columns.md",
        "openfoam-workspaces.md",
        "pyfr-workspaces.md",
        "qprop-workspaces.md",
        "su2-workspaces.md",
        "xfoil-workspaces.md",
    } <= names


@pytest.mark.parametrize(("name", "block"), _documented_matrices())
def test_every_documented_matrix_carries_a_per_column_rule_and_reads(name, block):
    lines = block.splitlines()
    header, rule = lines[0], lines[1]
    assert set(rule) <= set("-|"), f"{name}: no dashed rule under the header"
    assert rule.count("|") == header.count("|"), f"{name}: one dash run per column"
    assert len(rule) == len(header), f"{name}: rule aligned to the header"
    rows = parse_matrix(block.encode(), Path(name), active_only=False)
    assert rows


def test_shipped_smoke_matrix_carries_the_rule_and_reads():
    path = ROOT / "tools/kit/smoke/inputs/matrices/smoke.fs"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert set(lines[1]) <= set("-|")
    assert len(lines[1]) == len(lines[0])
    assert [row.pol for row in read_matrix(path)] == ["1001"]


def test_su2_reference_matrix_is_written_with_the_rule_and_plans(tmp_path):
    from cfdbender.solvers.su2._reference import prepare

    mesh = ROOT / "tests/fixtures/su2/naca0012_small.su2"
    plan = prepare(tmp_path / "ws", mesh, "pysu2")
    lines = (tmp_path / "ws/inputs/matrices/reference.fs").read_text().splitlines()
    assert set(lines[1]) <= set("-|")
    assert len(lines[1]) == len(lines[0])
    assert plan.points

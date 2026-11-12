"""P0000 item 3B: cfd-matrix plan study.fs --workspace campaign.

The absent-feature fallback only supplies the proposed call shape on the base:
it produces no rows. Assertions measure rows and refusals, not import failures.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def matrix_api():
    if importlib.util.find_spec("cfdbender.core.matrix") is None:
        return SimpleNamespace(read_matrix=lambda *a, **kw: [], MatrixError=ValueError)
    return importlib.import_module("cfdbender.core.matrix")


def matrix_file(tmp_path: Path, rows: list[str] | None = None) -> Path:
    path = tmp_path / "study.fs"
    header = "POL | RUN | SET | REF | PPROC | FLIGHT_CONDITION | SWEEP_VALUES | WALLTIME"
    body = rows or ["1001 | 1 | 002 | r003 | p001 | MACH:0.2, ALPHA:sweep | -2,0,2 | 4h"]
    path.write_text(chr(10).join(["# synthetic study", header, "-" * 40, *body, ""]))
    return path


def test_p0000_item3b_pipe_matrix_preserves_codes_conditions_and_walltime(tmp_path):
    rows = matrix_api().read_matrix(matrix_file(tmp_path))
    assert len(rows) == 1, "the matrix must produce its active row"
    row = rows[0]
    assert (row.pol, row.setup_id, row.reference_id, row.post_id) == (
        "1001",
        "s002",
        "r003",
        "p001",
    )
    assert row.condition == {"MACH": 0.2}
    assert (row.sweep_key, row.sweep_values, row.walltime) == ("ALPHA", (-2.0, 0.0, 2.0), "4h")


def test_p0000_item3b_run_filter_and_data_row_numbers(tmp_path):
    path = matrix_file(
        tmp_path,
        [
            "0001 | 0 | 002 | 003 | 001 | MACH:0.2 | NA | NA",
            "0002 | 1 | 002 | 003 | 001 | MACH:0.3 | NA | 30m",
        ],
    )
    rows = matrix_api().read_matrix(path)
    assert [(r.pol, r.row_number) for r in rows] == [("0002", 2)]
    assert len(matrix_api().read_matrix(path, active_only=False)) == 2


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("POL | RUN", "POL | POL"), "duplicate"),
        (("PPROC", "POST"), "PPROC"),
        (("-2,0,2", "0,0.0"), "SWEEP_VALUES"),
        (("-2,0,2", "nan"), "finite"),
        (("ALPHA:sweep", "ALPHA:sweep, BETA:sweep"), "one"),
        (("MACH:0.2", "MACH:0.2, mach:0.3"), "MACH"),
        (("002 | r003", "../s002 | r003"), "SET"),
        (("1001 | 1", "../1001 | 1"), "POL"),
        (("1001 | 1", "1001 | yes"), "RUN"),
        (("4h", "forever"), "WALLTIME"),
        (("-2,0,2", "NA"), "SWEEP_VALUES"),
        (("ALPHA:sweep", "ALPHA:0"), "SWEEP_VALUES"),
    ],
)
def test_p0000_item3b_malformed_matrix_is_a_named_refusal(tmp_path, replacement, message):
    api = matrix_api()
    path = matrix_file(tmp_path)
    path.write_text(path.read_text().replace(*replacement))
    with pytest.raises(api.MatrixError, match=message):
        api.read_matrix(path)


def test_p0000_item3b_duplicate_pol_refused_even_if_inactive(tmp_path):
    api = matrix_api()
    path = matrix_file(tmp_path)
    with path.open("a") as stream:
        stream.write("1001 | 0 | 002 | 003 | 001 | MACH:0.1 | NA | NA")
    with pytest.raises(api.MatrixError, match="1001"):
        api.read_matrix(path)


def test_p0000_item3b_unreadable_matrix_is_named(tmp_path):
    api = matrix_api()
    with pytest.raises(api.MatrixError, match="missing.fs"):
        api.read_matrix(tmp_path / "missing.fs")

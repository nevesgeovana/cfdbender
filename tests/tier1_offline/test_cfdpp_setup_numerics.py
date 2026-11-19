"""CFD++ rows and the SETUP and NUMERICS columns.

The adapter does not declare ``numerics_file``: a filled NUMERICS cell is warned once and
ignored, as for every solver without the capability. The retired SET header is still read,
with a warning that names the migration.
"""

from __future__ import annotations

import warnings

import pytest
from cfdpp_workspace import GRID, HEADER, matrix, use_toy, workspace

import cfdbender.solvers.cfdpp as cfdpp
from cfdbender.core.adapter import capabilities
from cfdbender.workspace.planning import plan_workspace

NUMERICS_HEADER = HEADER.replace("| SETUP |", "| SETUP | NUMERICS |")


@pytest.fixture(autouse=True)
def toy(monkeypatch):
    use_toy(monkeypatch)


def test_the_adapter_does_not_declare_the_numerics_capability():
    assert "numerics_file" not in capabilities(cfdpp)


def numerics_matrix() -> str:
    rule = "|".join("-" * 5 for _ in NUMERICS_HEADER.split("|"))
    row = (
        f"1001 | 1 | toy | MACH:0.15, REmi:6, TK:300, ALPHA:sweep | 0,10 | {GRID} | 001 | 001 | "
        "001 | 001 | 1 | 1h | 1.0 | steady"
    )
    return f"{NUMERICS_HEADER}\n{rule}\n{row}\n"


def test_a_filled_numerics_cell_is_warned_once_and_ignored(tmp_path):
    root = workspace(tmp_path / "ws", table=numerics_matrix()).resolve()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        planned = plan_workspace(root, "study")
    ignored = [
        str(w.message) for w in caught if "NUMERICS is ignored for solver cfdpp" in str(w.message)
    ]
    assert len(ignored) == 1
    assert all(point.numerics_id == "" for point in planned.points)
    assert not any("numerics" in name for p in planned.points for name in p.inputs_sha256)


def test_the_retired_set_header_is_still_read_with_a_warning(tmp_path):
    table = matrix().replace("| SETUP |", "| SET |")
    root = workspace(tmp_path / "ws", table=table).resolve()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        planned = plan_workspace(root, "study")
    assert any("column SET is now SETUP" in str(w.message) for w in caught)
    assert planned.points[0].setup_id == "s001"


def test_the_retired_geometry_header_is_still_read_with_a_warning(tmp_path):
    table = matrix().replace("| MESH |", "| GEOMETRY |")
    root = workspace(tmp_path / "ws", table=table).resolve()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        planned = plan_workspace(root, "study")
    assert any("GEOMETRY" in str(w.message) and "MESH" in str(w.message) for w in caught)
    assert planned.points[0].geometry == f"inputs/mesh/{GRID}"

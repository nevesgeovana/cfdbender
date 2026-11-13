"""Review questions: a FLIGHT_CONDITION key outside the grammar is refused as
pyflightstream refuses it; a solver's own keys are still accepted."""

from __future__ import annotations

import pytest
from test_conformance_workspace import build

from cfdbender.core.plan import PlanError
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("solver", ["mock", "su2", "openfoam", "pyfr", "xfoil", "qprop"])
def test_a_raw_re_key_is_refused_naming_the_key_and_the_accepted_set(tmp_path, solver):
    root = build(tmp_path, solver)
    matrix = root / "inputs/matrices/full.fs"
    lines = matrix.read_text(encoding="ascii").splitlines()
    header = lines[0].split("|")
    column = [name.strip() for name in header].index("FLIGHT_CONDITION")
    cells = lines[2].split("|")
    cells[column] = " RE:1000000, " + cells[column].strip() + " "
    lines[2] = "|".join(cells)
    matrix.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(PlanError) as caught:
        plan_workspace(root, "full")
    message = str(caught.value)
    assert "'RE'" in message and "not one this package knows" in message, message
    assert "REmi" in message and "closed" in message


def test_a_mistyped_key_names_its_closest_known_key(tmp_path):
    root = build(tmp_path, "mock")
    matrix = root / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii").replace("MACH", "MACHH"), encoding="ascii")
    with pytest.raises(PlanError, match="closest known key: MACH"):
        plan_workspace(root, "full")

"""An entry that does not apply to a row is warned and skipped, never silently ignored.

The warning names the row, the entry and why; planning goes on without the entry.
"""

from __future__ import annotations

import warnings

import pytest
import test_p0000_xfoil_plan as xfoil_case
from qprop_workspace import workspace as qprop_workspace

from cfdbender.core.plan import RowRequirements, plan_matrix
from cfdbender.workspace.planning import plan_workspace


def caught(call):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = call()
    return result, [str(w.message) for w in seen if "skipped" in str(w.message)]


def test_a_geometry_cell_for_a_solver_that_does_not_read_it_is_warned_and_skipped(tmp_path):
    files = {
        "init.toml": 'schema_version = 1\nsolver = "mock"\n',
        "inputs/setups/s001.toml": "",
        "inputs/references/r001.toml": "",
        "inputs/pproc/p001.toml": "",
        "inputs/geometries/wing.dat": "mesh",
        "inputs/matrices/study.fs": "POL|RUN|MESH|SETUP|REF|PPROC\n7|1|wing.dat|001|001|001\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    def ignores(row, inputs):
        return RowRequirements(("solve",), ("sims/sim_{pol}/datapoints/{point}/out.txt",))

    plan, messages = caught(lambda: plan_matrix(tmp_path, "study", validators={"mock": ignores}))
    assert plan.points[0].geometry is None
    assert len(messages) == 1
    text = messages[0]
    assert "POL" in text and "MESH" in text and "wing.dat" in text
    assert "does not read" in text


def test_unread_reference_and_post_keys_of_xfoil_are_warned_and_skipped(tmp_path):
    xfoil_case.workspace(tmp_path)
    (tmp_path / "inputs/references/r001.toml").write_text("length = 1.0\narea = 2.0\n")
    (tmp_path / "inputs/pproc/p001.toml").write_text("history = true\n")
    plan, messages = caught(lambda: plan_workspace(tmp_path, "study"))
    assert plan.points
    joined = "\n".join(messages)
    assert "area" in joined and "r001" in joined and "POL 1001" in joined
    assert "history" in joined and "p001" in joined
    assert "'length'" not in joined, "length is read by XFOIL"


def test_xfoil_with_empty_reference_and_post_warns_nothing(tmp_path):
    xfoil_case.workspace(tmp_path)
    _, messages = caught(lambda: plan_workspace(tmp_path, "study"))
    assert messages == []


@pytest.mark.parametrize(
    "name,text", [("references/r001", "area = 2.0\n"), ("pproc/p001", "x = 1\n")]
)
def test_qprop_reads_no_reference_or_post_key_and_says_so(tmp_path, name, text):
    qprop_workspace(tmp_path)
    (tmp_path / f"inputs/{name}.toml").write_text(text)
    plan, messages = caught(lambda: plan_workspace(tmp_path, "study"))
    assert plan.points
    assert len(messages) == 1 and name.split("/")[1] in messages[0]
    assert "QPROP" in messages[0]

"""P0000: shared adapter conformance through the actual child worker."""

from dataclasses import replace

import pytest
from test_p0000_item3d_config import api, plan
from test_p0000_item3g_worker import case
from test_p0000_item3g_worker import wrapper as _wrapper

from cfdbender import testing
from cfdbender.core.matrix import read_matrix
from cfdbender.core.plan import InputFiles, RowInputs

wrapper = _wrapper


@pytest.mark.parametrize(
    "rule",
    [
        "reads_init",
        "deterministic",
        "exit_evidence",
        "results",
        "no_overwrite",
        "execution_isolation",
    ],
)
def test_p0000_item3g_shared_point_conformance(tmp_path, wrapper, rule):
    getattr(testing, "assert_" + rule)(api(), case(tmp_path / "workspace"))


def test_p0000_item3g_shared_classifications(tmp_path, wrapper):
    cases = {
        outcome: case(tmp_path / outcome, outcome)
        for outcome in ("converged", "iteration_limit", "failed")
    }
    testing.assert_classifications(api(), cases)


def test_p0000_item3g_shared_validation(tmp_path, wrapper):
    root = tmp_path / "workspace"
    case(root)
    row = read_matrix(root / "inputs/matrices/study.fs")[0]
    testing.assert_validation(
        api(), replace(row, workflow="invalid"), RowInputs(root, row, InputFiles(root)), "WORKFLOW"
    )


@pytest.mark.parametrize("rule", ["run_record", "finished_no_overwrite", "collected_results"])
def test_p0000_item3g_shared_registry_conformance(tmp_path, wrapper, rule):
    root = tmp_path / "workspace"
    case(root)
    getattr(testing, "assert_" + rule)(api(), plan(root))

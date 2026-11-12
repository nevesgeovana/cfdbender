"""P0000: SU2 passes the same shared assertions as the offline mock."""

from dataclasses import replace

import pytest
from test_p0000_item3d_config import plan
from test_p0000_item3d_execute import case, execution_api

from cfdbender import testing
from cfdbender.core.matrix import read_matrix
from cfdbender.core.plan import InputFiles, RowInputs


@pytest.mark.parametrize(
    "rule",
    [
        "reads_init",
        "deterministic",
        "exit_evidence",
        "results",
        "no_overwrite",
    ],
)
def test_p0000_item3d_shared_point_conformance(tmp_path, rule):
    getattr(testing, "assert_" + rule)(execution_api(), case(tmp_path))


def test_p0000_item3d_shared_classifications(tmp_path):
    cases = {
        outcome: case(tmp_path / outcome, outcome)
        for outcome in ("converged", "iteration_limit", "failed")
    }
    testing.assert_classifications(execution_api(), cases)


def test_p0000_item3d_shared_validation(tmp_path):
    case(tmp_path)
    row = read_matrix(tmp_path / "inputs/matrices/study.fs")[0]
    inputs = RowInputs(tmp_path, row, InputFiles(tmp_path))
    testing.assert_validation(execution_api(), replace(row, workflow="invalid"), inputs, "WORKFLOW")


@pytest.mark.parametrize("rule", ["run_record", "finished_no_overwrite", "collected_results"])
def test_p0000_item3d_shared_registry_conformance(tmp_path, rule):
    case(tmp_path)
    getattr(testing, "assert_" + rule)(execution_api(), plan(tmp_path))

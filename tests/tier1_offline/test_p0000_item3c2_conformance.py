"""P0000 item 3C2: mock conformance includes registry and collected-result rules."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_p0000_item3c2_run import run_api as execute_plan
from test_p0000_item3c_adapter import workspace

from cfdbender import testing
from cfdbender.solvers import mock
from cfdbender.workspace.planning import plan_workspace

RULES = ("run_record", "finished_no_overwrite", "collected_results")


def invoke(rule, root, mutant=False):
    workspace(root)
    plan = plan_workspace(root, "study")
    helper = getattr(testing, f"assert_{rule}", lambda *a, **kw: None)
    adapter = mock
    runner = execute_plan
    if mutant and rule == "run_record":

        def runner(plan, adapter, **kwargs):
            records = execute_plan(plan, adapter, **kwargs)
            return tuple(replace(record, status="PLANNED") for record in records)
    elif mutant and rule == "finished_no_overwrite":
        saved = []

        def runner(plan, adapter, **kwargs):
            if not saved:
                saved.extend(execute_plan(plan, adapter, **kwargs))
            return tuple(saved)
    elif mutant:

        def collect(point):
            result = mock.collect(point)
            return replace(
                result,
                references=SimpleNamespace(
                    area=2.0,
                    length=3.0,
                    moment_origin=(1, 2, 3),
                    axis_system="",
                    axis_directions=("", "", ""),
                ),
            )

        adapter = SimpleNamespace(execute=mock.execute, classify=mock.classify, collect=collect)
    helper(adapter, plan, runner=runner)


@pytest.mark.parametrize("rule", RULES)
def test_p0000_item3c2_mock_conformance(tmp_path, rule):
    invoke(rule, tmp_path)


@pytest.mark.parametrize("rule", RULES)
def test_p0000_item3c2_conformance_rejects_each_mutant(tmp_path, rule):
    with pytest.raises(AssertionError, match=rule):
        invoke(rule, tmp_path, mutant=True)

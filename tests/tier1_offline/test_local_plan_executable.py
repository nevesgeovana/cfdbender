"""A local plan looks for the solver executable as a run does; a plan for the cluster does not.

The rule: ``cfd-matrix plan --local`` fails with the one missing-executable sentence
when the executable is absent on this machine, and a plan without ``--local`` never looks.
Every library solver is covered, and plan and run say the same sentence.
"""

from __future__ import annotations

import pytest
from test_conformance_workspace import case, local, state_no_build  # noqa: F401
from test_fix87_run import _break_executable

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.matrix.receipt import write_receipt
from cfdbender.workspace.planning import plan_workspace


def plan(root, *options):
    return main(["plan", "full", "--workspace", str(root), *options])


def sentence(err):
    line = next(x for x in err.splitlines() if "executable not found" in x)
    return line[line.index("executable not found") - 8 :].strip()


def test_a_local_plan_with_the_executable_present_passes(case, capsys):  # noqa: F811
    _, root = case
    assert plan(root, "--local") == 0


def test_a_local_plan_without_the_executable_fails_with_the_one_sentence(case, capsys):  # noqa: F811
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock runs no process")
    _break_executable(root, solver)
    assert plan(root, "--local") == 2
    err = capsys.readouterr().err
    assert "executable not found" in err or "launcher" in err, err
    assert "absent-" in err, err
    assert RunRegistry(root).read() == ()


def test_a_plan_for_the_cluster_does_not_look_for_the_executable(case, capsys):  # noqa: F811
    solver, root = case
    _break_executable(root, solver)
    assert plan(root) == 0
    err = capsys.readouterr().err
    assert "not found" not in err and "absent-" not in err, err


def test_plan_and_run_say_the_same_sentence(case, capsys):  # noqa: F811
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock runs no process")
    state_no_build(root)
    _break_executable(root, solver)
    assert plan(root, "--local") == 2
    from_plan = capsys.readouterr().err
    # The plan did not stop at the receipt: a run needs it, so write it as a cluster plan does.
    write_receipt(plan_workspace(root, "full"))
    assert local(root) == 2
    from_run = capsys.readouterr().err
    wanted = from_run.split("matrix not run: ", 1)[1].splitlines()[0]
    assert wanted in from_plan, (from_plan, from_run)

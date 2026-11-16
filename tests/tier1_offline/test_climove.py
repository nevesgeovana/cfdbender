"""CLIM: the verbs that act on what the workspace holds live in cfd-workspace.

cfd-matrix acts on a matrix (plan, run, collect, post, status); cfd-workspace acts on the
records, storage and logs of the workspace. The old cfd-matrix forms are refused by name
with the command to type instead, and --warning-refusals turns every warn-and-skip into a
refusal.
"""

from __future__ import annotations

import json

import pytest
import test_p0000_xfoil_plan as xfoil_case

from cfdbender.matrix.cli import build_parser as matrix_parser
from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import build_parser as workspace_parser
from cfdbender.workspace.cli import main as workspace_main

MOVED = (
    "space-in-use",
    "free-space",
    "delete-sims",
    "sync",
    "restore",
    "rebuild",
    "mark-failed",
    "mark-converged",
    "show",
    "log",
    "trace",
    "history",
    "diff",
)
KEPT = ("plan", "run", "collect", "post", "status")


def _verbs(parser):
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            return set(action.choices)
    return set()


def test_clim1_the_moved_verbs_are_workspace_verbs_and_matrix_keeps_its_own():
    matrix = _verbs(matrix_parser()[0])
    workspace = _verbs(workspace_parser())
    assert set(KEPT) <= matrix
    assert not matrix & set(MOVED)
    assert set(MOVED) <= workspace
    assert {"init", "archive"} <= workspace


@pytest.mark.parametrize("verb", MOVED)
def test_clim2_the_old_matrix_form_is_refused_by_name_with_the_new_command(verb, capsys):
    code = matrix_main([verb, "--workspace", "somewhere", "--apply"])
    err = capsys.readouterr().err
    assert code == 2
    assert f"cfd-workspace {verb}" in err
    assert f"cfd-matrix {verb}" in err


def test_clim2_a_root_option_before_the_old_verb_is_still_refused_by_name(capsys):
    assert matrix_main(["--warning-off", "show", "1001"]) == 2
    assert "cfd-workspace --warning-off show" in capsys.readouterr().err


def test_clim2_the_new_form_answers_on_a_workspace(tmp_path, capsys):
    xfoil_case.workspace(tmp_path)
    capsys.readouterr()
    assert workspace_main(["log", "--json", "--workspace", str(tmp_path)]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)

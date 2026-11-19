"""Adversarial raw-keyword cases on the real input layout."""

import pytest
from cfdpp_workspace import matrix, workspace

from cfdbender.core import setup_templates
from cfdbender.core.plan import PlanError
from cfdbender.solvers.cfdpp._schema import schema_for
from cfdbender.solvers.cfdpp._schema._model import Block, InputModel, Pair, render
from cfdbender.workspace.planning import plan_workspace


def test_raw_comment_name_is_refused_at_plan(tmp_path):
    setup = setup_templates.read("cfdpp", "subsonic").replace("\r\n", "\n")
    setup += '\n"#ignored" = { value = 7, section = "options" }\n'
    root = workspace(tmp_path / "ws", setup=setup, table=matrix(build="22.5.U3"), executables=False)
    with pytest.raises(PlanError, match="keyword.*comment"):
        plan_workspace(root, "study")


def test_direct_comment_pair_is_refused():
    model = InputModel((Block("options", (Pair("#ignored", (7,)),)),))
    with pytest.raises(ValueError, match="keyword.*comment"):
        render(model, schema_for("22.5.U3"), free={"#ignored"})

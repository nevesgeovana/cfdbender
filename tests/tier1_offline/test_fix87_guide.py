"""Q3: init writes inputs/INPUTS.md, and the example it shows plans without editing."""

from __future__ import annotations

import re
import warnings

import pytest
from test_matrix_columns_all_solvers import CASES

from cfdbender.core import init_workspace
from cfdbender.solvers import available_solvers
from cfdbender.workspace.planning import plan_workspace


def blocks(text):
    return re.findall(r"```(\w+)\n(.*?)```", text, re.S)


@pytest.mark.parametrize("solver", sorted(CASES))
def test_the_example_of_the_guide_plans_as_written(tmp_path, solver):
    init_workspace(tmp_path, solver=solver, available_solvers=available_solvers())
    guide = (tmp_path / "inputs/INPUTS.md").read_text(encoding="utf-8")
    assert f"solver: {solver}" in guide and "inputs/mesh/" in guide
    found = blocks(guide)
    matrix = next(body for kind, body in found if kind == "text" and body.startswith("POL"))
    (tmp_path / "inputs/matrices/first.fs").write_text(matrix, encoding="ascii")
    tomls = [body for kind, body in found if kind == "toml"]
    names = ["inputs/setups/s001.toml", "inputs/references/r001.toml", "inputs/pproc/p001.toml"]
    held = dict(zip(names, tomls, strict=False))
    for name in names:
        (tmp_path / name).write_text(held.get(name, ""), encoding="ascii")
    files, _ = CASES[solver]
    for name, text in files.items():
        if "geometries/" in name:
            target = tmp_path / name.replace("geometries", "mesh")
            target.write_bytes(text.encode("ascii"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(tmp_path, "first")
    assert plan.points and plan.solver == solver


def test_init_keeps_an_existing_guide(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/INPUTS.md").write_text("mine\n", encoding="utf-8")
    init_workspace(tmp_path, solver="mock", available_solvers=available_solvers())
    assert (tmp_path / "inputs/INPUTS.md").read_text(encoding="utf-8") == "mine\n"

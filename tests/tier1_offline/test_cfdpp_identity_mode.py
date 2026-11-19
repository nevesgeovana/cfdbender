"""CFD++ grid files honour the large-input identity mode: no grid byte is read.

A threshold of 64 bytes stands in for the 8 MiB one, so every grid file of the toy workspace is
"large". With the check off (the flag, the environment variable or the init key) the planned
inputs are recorded by identity and the grid is not opened; what the grid would have told is
reported as not checked.
"""

from __future__ import annotations

import warnings

import pytest
from cfdpp_workspace import matrix, use_toy, workspace
from test_cfdpp_digest_cache import LARGE, ONE, opens  # noqa: F401  (the fixture counts opens)

import cfdbender.solvers.cfdpp as cfdpp
from cfdbender.core import plan as plan_module
from cfdbender.core.adapter import load_point
from cfdbender.core.executables import bind_executable
from cfdbender.core.plan import NO_READ_ENV, is_identity_value
from cfdbender.matrix.cli import main
from cfdbender.workspace.planning import plan_workspace


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    use_toy(monkeypatch)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 64)
    monkeypatch.setattr(plan_module, "_BLOCK", 32)


def test_the_flag_plans_without_opening_a_grid_file(tmp_path, opens):  # noqa: F811
    root = workspace(tmp_path / "ws", table=ONE).resolve()
    opens.clear()
    assert main(["plan", "study", "--workspace", str(root), "--no-read-large-inputs"]) == 0
    assert opens == []


def test_the_environment_variable_plans_without_opening_a_grid_file(
    tmp_path,
    monkeypatch,
    opens,  # noqa: F811
):
    root = workspace(tmp_path / "ws", table=ONE).resolve()
    monkeypatch.setenv(NO_READ_ENV, "1")
    opens.clear()
    with pytest.warns(UserWarning, match="not read|not checked|large-input"):
        plan_workspace(root, "study")
    assert opens == []


def test_the_planned_grid_inputs_are_identities_and_the_skip_is_warned(tmp_path, monkeypatch):
    root = workspace(tmp_path / "ws", table=ONE).resolve()
    monkeypatch.setenv(NO_READ_ENV, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        planned = plan_workspace(root, "study")
    grid = {
        k: v for k, v in planned.points[0].inputs_sha256.items() if k.rsplit("/", 1)[-1] in LARGE
    }
    assert grid and all(is_identity_value(v) for v in grid.values())
    assert any("large-input check is off" in str(w.message) for w in caught)


def test_a_small_matrix_still_plans_with_the_check_on(tmp_path, opens):  # noqa: F811
    root = workspace(tmp_path / "ws", table=matrix()).resolve()
    opens.clear()
    plan_workspace(root, "study")
    assert sorted(opens) == ["open"] * len({"nodesin.bin", "cellsin.bin", "exbcsin.bin"})


def test_the_init_key_plans_without_opening_a_grid_file(tmp_path, opens):  # noqa: F811
    root = workspace(tmp_path / "ws", table=ONE).resolve()
    init = root / "init.toml"
    init.write_text(init.read_text(encoding="utf-8") + '\nlarge_input_check = "identity"\n')
    opens.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan_workspace(root, "study")
    assert opens == []


def test_a_cold_identity_plan_cannot_write_the_input_and_says_how_to_fix_it(
    tmp_path,
    monkeypatch,
    opens,  # noqa: F811
):
    root = workspace(tmp_path / "ws", table=ONE).resolve()
    monkeypatch.setenv(NO_READ_ENV, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        planned = plan_workspace(root, "study")
    context = bind_executable(load_point(root, planned.points[0], solver="cfdpp"), cfdpp)
    opens.clear()
    with pytest.raises(ValueError, match=r"nodesin\.bin.*One plan with the check on is enough"):
        cfdpp.translate(context)
    assert opens == []

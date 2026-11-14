"""A SOLVER_BUILD that the executable's own version contradicts refuses before any point runs."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_gaps87_solver_identity import PYFR_STANDIN, pyfr_case

from cfdbender.core.adapter import load_point
from cfdbender.core.run import judge_environment
from cfdbender.solvers import load_solver


def _plan(tmp_path, standin, build):
    plan = pyfr_case(tmp_path, standin)
    point = replace(plan.points[0], solver_build=build)
    return replace(plan, points=(point,))


def test_a_wrong_build_refuses_in_preflight_before_any_run(tmp_path):
    plan = _plan(tmp_path, PYFR_STANDIN, "9.9.9")
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    with pytest.raises(ValueError, match=r"SOLVER_BUILD.*9\.9\.9.*2\.0\.3"):
        load_solver("pyfr").preflight(point)
    assert not (tmp_path / plan.points[0].directory).exists()


def test_the_matching_build_passes_preflight(tmp_path):
    plan = _plan(tmp_path, PYFR_STANDIN, "2.0.3")
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    load_solver("pyfr").preflight(point)


def test_an_executable_that_cannot_say_its_version_is_judged_after_the_run(tmp_path):
    plan = _plan(tmp_path, PYFR_STANDIN.replace("--version", "--no-such-flag"), "9.9.9")
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    load_solver("pyfr").preflight(point)


SU2_STANDIN = (
    "import sys\nif '--version' in sys.argv:\n    print('SU2 v8.6.0 Harrier'); sys.exit(0)\n"
)


@pytest.mark.parametrize("build,refused", [("9.9.9", True), ("8.6.0", False)])
def test_su2_checks_the_build_against_its_executable_in_preflight(tmp_path, build, refused):
    import json

    import test_p0000_item3d_config as su2_case

    su2_case.workspace(tmp_path)
    script = tmp_path / "standin_su2.py"
    script.write_text(SU2_STANDIN, encoding="ascii")
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(
        setup.read_text(encoding="ascii")
        + f'\n[solver]\nroute = "executable"\nexecutable = {json.dumps(script.as_posix())}\n',
        encoding="ascii",
    )
    planned = su2_case.plan(tmp_path).points[0]
    point = load_point(tmp_path, replace(planned, solver_build=build), solver="su2")
    adapter = load_solver("su2")
    if refused:
        with pytest.raises(ValueError, match="SOLVER_BUILD"):
            adapter.preflight(point)
    else:
        adapter.preflight(point)


def test_the_environment_judgement_refuses_the_whole_plan(tmp_path):
    from cfdbender.core.registry import RegistryError

    plan = _plan(tmp_path, PYFR_STANDIN, "9.9.9")
    with pytest.raises(RegistryError, match="SOLVER_BUILD"):
        judge_environment(plan, load_solver("pyfr"), check_ram=lambda: None)

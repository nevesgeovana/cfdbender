"""R2-09: a prelaunch-only RAM callback must fail the during-run contract probe."""

import pytest
from test_conformance_workspace import build, executables
from test_contract_execution import test_c23_a_refusal_during_the_run_stops_the_solver as probe

from cfdbender.solvers import load_solver


def test_the_during_run_probe_rejects_two_checks_without_a_child(tmp_path, monkeypatch):
    root = build(tmp_path, "openfoam")
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes() + executables(root, "openfoam").encode())

    def prelaunch_only(point, *, check_ram):
        check_ram()
        check_ram()

    monkeypatch.setattr(load_solver("openfoam"), "execute", prelaunch_only)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        probe(("openfoam", root))

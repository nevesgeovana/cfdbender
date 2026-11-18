"""Test isolation: the host process lease is machine-wide, tests must not share it.

The lease is a named mutex shared by every interpreter on the host, so a real
solver run, or another suite, made any test that executes a fake solver fail
with "already running". Tests get a no-op lease; the tests of the lease itself
keep the real one and skip when another process holds it.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext

import pytest

from cfdbender.solvers._host._single import process_lease

_LEASE_USERS = (
    "cfdbender.matrix.rerun",
    "cfdbender.solvers._host._exe",
    "cfdbender.solvers.su2._single",
)
_REAL_LEASE_MODULES = {"test_p0000_solver_host_lease"}
_REAL_LEASE_TESTS = {
    "test_p0000_item3d_process_lock_prevents_second_interpreter",
    "test_p0000_item4_su2_plan_without_mock_or_real_solver",
}
_UNPATCHED_TESTS = {
    "test_p0000_solver_host_helpers_are_neutral",
}


@pytest.fixture(autouse=True)
def _isolated_process_lease(request, monkeypatch):
    if (
        request.module.__name__.rsplit(".", 1)[-1] in _REAL_LEASE_MODULES
        or request.node.name in _REAL_LEASE_TESTS
    ):
        try:
            with process_lease():
                pass
        except ValueError:
            pytest.skip("the host process lease is held by another process (a real run or suite)")
        return
    if request.node.originalname in _UNPATCHED_TESTS:
        return
    for name in _LEASE_USERS:
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "process_lease", lambda solver="SU2": nullcontext())


_REAL_RECEIPT_MODULES = {
    "test_cli_plan_receipt",
    "test_plan_receipt",
    "test_p053_rowhpc_profiles",
    "test_p060_int060_rowhpc_merge",
    "test_submit_batch",
}


@pytest.fixture(autouse=True)
def _planned_before_run(request, monkeypatch):
    """Tests that exercise a run, not the receipt, plan the matrix first, as a user does.

    The tests of the receipt itself keep the real check.
    """
    if request.module.__name__.rsplit(".", 1)[-1] in _REAL_RECEIPT_MODULES:
        return
    from cfdbender.matrix import cli
    from cfdbender.matrix.receipt import require_receipt, write_receipt

    def planned_first(plan, profiles=(), grouping=None):
        write_receipt(plan, profiles, grouping)
        require_receipt(plan, profiles, grouping)

    monkeypatch.setattr(cli, "require_receipt", planned_first)

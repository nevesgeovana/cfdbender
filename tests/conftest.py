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
    "cfdbender.solvers.openfoam._execute",
    "cfdbender.solvers.pyfr._execute",
    "cfdbender.solvers.qprop._execute",
    "cfdbender.solvers.su2._execute",
    "cfdbender.solvers.su2._single",
    "cfdbender.solvers.xfoil._execute",
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

"""P0000: the base lease and migrated adapters exclude each other.

LEGACY is the frozen Python implementation of su2._single at base a7c8524.
It is frozen here so a future lease identity change cannot alter both sides.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_p0000_solver_host_boundary import prepared

ROOT = Path(__file__).resolve().parents[2]
LEGACY = r'''"""Coordinate one SU2 process across interpreters on the same host."""

from __future__ import annotations

import ctypes
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def process_lease() -> Iterator[None]:
    refusal = "SU2 is already running; only one process at a time"
    if sys.platform == "win32":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateMutexW(None, False, "Local" + chr(92) + "cfdbender-su2")
        if not handle:
            raise ValueError("cannot reserve SU2 process slot")
        owned = False
        try:
            # An abandoned mutex can be acquired after a crashed owner.
            owned = kernel.WaitForSingleObject(handle, 0) in (0, 0x80)
            if not owned:
                raise ValueError(refusal)
            yield
        finally:
            if owned:
                kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
    else:
        import fcntl

        with (Path(tempfile.gettempdir()) / "cfdbender-su2.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError(refusal) from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
'''


def child_code(legacy, hold):
    source = LEGACY if legacy else "from cfdbender.solvers.su2._single import process_lease"
    return source + (
        "\nwith process_lease():\n"
        "    print('owned', flush=True)\n" + ("    input()\n" if hold else "")
    )


@pytest.mark.parametrize("legacy_holder", [False, True])
@pytest.mark.parametrize(
    "solver,label", [("su2", "SU2"), ("xfoil", "XFOIL"), ("openfoam", "OpenFOAM"), ("pyfr", "PyFR")]
)
def test_p0000_solver_host_legacy_lease_excludes_adapter(tmp_path, solver, label, legacy_holder):
    point = prepared(tmp_path, solver)
    child = subprocess.Popen(
        [sys.executable, "-c", child_code(legacy_holder, True)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout.readline().strip() == "owned"
        with pytest.raises(ValueError, match=f"{label} is already running"):
            importlib.import_module(f"cfdbender.solvers.{solver}").execute(
                point, check_ram=lambda: None
            )
        assert not (point.root / point.point.directory).exists()
        # A second interpreter of the other generation must also be refused.
        refused = subprocess.run(
            [sys.executable, "-c", child_code(not legacy_holder, False)],
            capture_output=True,
            text=True,
            check=False,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        )
        assert refused.returncode != 0 and "only one process" in refused.stderr
    finally:
        child.communicate("\n", timeout=5)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    assert child.returncode == 0
    acquired = subprocess.run(
        [sys.executable, "-c", child_code(not legacy_holder, False)],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "owned"

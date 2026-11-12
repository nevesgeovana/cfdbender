"""Coordinate one solver process across interpreters on the same host."""

from __future__ import annotations

import ctypes
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def process_lease(solver: str = "SU2") -> Iterator[None]:
    """Reserve the host process slot with the unchanged legacy lease identity.

    Parameters
    ----------
    solver : str
        Solver name for refusals. The default preserves legacy SU2 callers.

    Yields
    ------
    None
        Exclusive ownership until the context exits.
    """
    refusal = f"{solver} is already running; only one process at a time"
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
            raise ValueError(f"cannot reserve {solver} process slot")
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

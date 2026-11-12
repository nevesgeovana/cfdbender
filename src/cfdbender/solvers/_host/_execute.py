"""Shared host launch flags, timeout bounds and child-tree termination."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping

from cfdbender.core.adapter import AdapterPoint
from cfdbender.solvers._host._config import number

_RUNNING = threading.Lock()
if sys.platform == "win32":
    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    _NO_WINDOW = 0


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the child process tree and reap the immediate child."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=_NO_WINDOW,
            check=False,
        )
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def _timeout(point: AdapterPoint, opts: Mapping[str, object]) -> float:
    """Bound the configured timeout by the point walltime, when present."""
    limit = number(opts["timeout_s"], "timeout_s")
    walltime = point.point.walltime
    if walltime:
        limit = min(
            limit, float(walltime[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[walltime[-1]]
        )
    return limit

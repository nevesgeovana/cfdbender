"""Ask a solver executable for its version, without leaving anything in a point folder."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Sequence

from cfdbender.solvers._host._execute import _NO_WINDOW

UNKNOWN = "unknown"
_VERSION = re.compile(r"(?<![\w.])v?([0-9]+(?:[.][0-9]+)+)(?![\w.])")


def probe_version(
    command: Sequence[str], *, argument: str = "--version", timeout: float = 20.0
) -> str:
    """Return the version ``command argument`` prints, or ``unknown``.

    The probe runs in a scratch folder, so a stand-in that writes files when it
    does not know the argument leaves nothing in the point. Any failure to start,
    a timeout or an answer without a dotted number gives ``unknown``; the caller
    records it and never refuses on it.

    Parameters
    ----------
    command : sequence of str
        Launcher tokens and the executable.
    argument : str
        The flag that makes the solver print its version.
    timeout : float
        Seconds to wait for the answer.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="cfdb-version-") as scratch:
            done = subprocess.run(
                [*command, argument],
                cwd=scratch,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                shell=False,
                creationflags=_NO_WINDOW,
                check=False,
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        return UNKNOWN
    text = (done.stdout + b"\n" + done.stderr).decode("utf-8", errors="replace")
    found = _VERSION.search(text)
    return found[1] if found else UNKNOWN

"""Stage inputs and run PyFR under the shared host guards."""

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.settings import read_init
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._execute import _NO_WINDOW, _RUNNING, _kill_tree, _timeout
from cfdbender.solvers._host._mesh import link_mesh
from cfdbender.solvers._host._single import process_lease
from cfdbender.solvers.pyfr._config import config, options


def commands(opts: Mapping[str, object], root: Path, backend: str) -> list[list[str]]:
    launcher = cast(list[str], opts["launcher"])
    executable = str(opts.get("executable", "pyfr"))
    if executable.endswith(".py") and not launcher:
        path = Path(executable)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ValueError("PyFR stand-in does not exist")
        prefix = [sys.executable, str(path)]
    else:
        prefix = [*launcher, executable]
    return [
        prefix + ["import", "mesh.msh", "mesh.pyfrm"],
        prefix + ["-p", "run", "-b", backend, "mesh.pyfrm", "case.ini"],
    ]


def stage_case(point: AdapterPoint, text: str, folder: Path) -> None:
    """Create the point folder, link mesh.msh and write case.ini into it.

    A local run and a cluster submission stage the same files.
    """
    case = config(point.setup, point.point.condition, point.reference, point.geometry)
    folder.mkdir(parents=True, exist_ok=False)
    link_mesh(
        workspace_path(point.root, case.mesh),
        folder / "mesh.msh",
        point.point.inputs_sha256.get(case.mesh),
    )
    (folder / "case.ini").write_bytes(text.encode("ascii"))


def run(point: AdapterPoint, text: str, check_ram: Callable[[], None]) -> None:
    point.verify()
    opts = options(point.setup, read_init(point.root, available_solvers=("pyfr",)).solver_options)
    case = config(point.setup, point.point.condition, point.reference, point.geometry)
    argv = commands(opts, point.root, case.backend)
    timeout = _timeout(point, opts)
    folder = workspace_path(point.root, point.point.directory)
    check_ram()
    _ram.require_ram(0, solver="PyFR")
    if folder.exists():
        raise ValueError("existing PyFR point directory; refusing overwrite")
    if not _RUNNING.acquire(blocking=False):
        raise ValueError("PyFR is already running; only one process at a time")
    try:
        with process_lease("PyFR"):
            stage_case(point, text, folder)
            _launch(argv, folder, timeout, check_ram)
    finally:
        _RUNNING.release()


def _launch(
    argv: list[list[str]], folder: Path, timeout: float, check_ram: Callable[[], None]
) -> None:
    start = time.monotonic()
    code = 1
    timed_out = False
    process = None
    try:
        with (
            (folder / "stdout.log").open("xb") as stdout,
            (folder / "stderr.log").open("xb") as stderr,
        ):
            for command in argv:
                check_ram()
                _ram.require_ram(0, solver="PyFR")
                process = subprocess.Popen(
                    command,
                    cwd=folder,
                    env=os.environ.copy(),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    creationflags=_NO_WINDOW,
                    start_new_session=sys.platform != "win32",
                )
                while True:
                    check_ram()
                    _ram.require_ram(0, solver="PyFR")
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        timed_out = True
                        _kill_tree(process)
                        code = 124
                        break
                    try:
                        code = process.wait(timeout=min(0.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if code != 0:
                    break
    finally:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        (folder / "execution.json").write_text(
            json.dumps(
                {"argv": argv, "exit_code": code, "timed_out": timed_out, "timeout_s": timeout},
                sort_keys=True,
            ),
            encoding="ascii",
        )

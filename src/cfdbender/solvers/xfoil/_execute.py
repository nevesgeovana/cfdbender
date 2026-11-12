"""Run XFOIL with file-backed stdin and shared host process safety."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
from cfdbender.solvers._host._config import number
from cfdbender.solvers._host._execute import _NO_WINDOW, _RUNNING, _kill_tree, _timeout
from cfdbender.solvers._host._single import process_lease
from cfdbender.solvers.xfoil._config import Config, config, options, render


def argv(opts: Mapping[str, object], root: Path) -> list[str]:
    value = str(opts.get("executable") or os.environ.get("CFDBENDER_XFOIL_EXE") or "xfoil")
    launcher = cast(list[str], opts.get("launcher", []))
    if launcher:
        # A guest executable is not a host path. WSL maps the unchanged cwd.
        return [*launcher, value]
    executable = Path(value).expanduser()
    is_path = executable.is_absolute() or any(sep in value for sep in (os.sep, os.altsep) if sep)
    if is_path:
        if not executable.is_absolute():
            executable = root / executable
        allowed = executable.suffix == ".py" or os.name == "nt" or os.access(executable, os.X_OK)
        resolved = str(executable) if executable.is_file() and allowed else None
        if resolved is None and not executable.exists():
            resolved = shutil.which(str(executable))
    else:
        resolved = shutil.which(str(executable))
    if resolved is None or not Path(resolved).is_file():
        raise ValueError("XFOIL executable not found; set solver.executable or CFDBENDER_XFOIL_EXE")
    return [sys.executable, resolved] if executable.suffix == ".py" else [resolved]


def run(point: AdapterPoint, check_ram: Callable[[], None]) -> None:
    point.verify()
    declaration = read_init(point.root, available_solvers=("xfoil",))
    opts = options(point.setup, declaration.solver_options)
    values = config(point.setup, point.point.condition, point.reference, point.geometry)
    folder = workspace_path(point.root, point.point.directory)
    _launch(
        point.root,
        folder,
        opts,
        check_ram,
        lambda: _staged(point, values),
        _timeout(point, opts),
        render(values).encode("ascii"),
    )


def run_script(
    root: Path,
    folder: Path,
    script: str,
    staged: Mapping[str, bytes],
    opts: Mapping[str, object],
    check_ram: Callable[[], None],
) -> None:
    """Run one keystroke script in a new folder with the shared host safeguards.

    Parameters
    ----------
    root : Path
        Workspace root, used to resolve a relative executable.
    folder : Path
        New working directory; it must not exist.
    script : str
        ASCII keystrokes, saved as ``xfoil.in`` and fed to standard input.
    staged : mapping of str to bytes
        Extra files written into the folder before the run, such as ``airfoil.dat``.
    opts : mapping
        ``executable``, ``launcher``, ``timeout_s`` and ``ram_estimate_gib``.
    check_ram : callable
        Coordinator guard, checked before writes and throughout the process.
    """
    full = options({"solver": dict(opts)})
    _launch(
        root,
        folder,
        full,
        check_ram,
        lambda: staged,
        float(cast(float, full["timeout_s"])),
        script.encode("ascii"),
    )


def stage(point: AdapterPoint) -> None:
    """Create the point folder and write xfoil.in and any airfoil.dat without running."""
    point.verify()
    values = config(point.setup, point.point.condition, point.reference, point.geometry)
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError("existing XFOIL point directory; refusing overwrite")
    staged = _staged(point, values)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "xfoil.in").write_bytes(render(values).encode("ascii"))
    for name, content in staged.items():
        (folder / name).write_bytes(content)


def _staged(point: AdapterPoint, values: Config) -> Mapping[str, bytes]:
    if not values.airfoil_file:
        return {}
    airfoil = workspace_path(point.root, values.airfoil_file).read_bytes()
    if hashlib.sha256(airfoil).hexdigest() != point.point.inputs_sha256.get(values.airfoil_file):
        raise ValueError("airfoil changed after planning; plan again")
    return {"airfoil.dat": airfoil}


def _launch(
    root: Path,
    folder: Path,
    opts: Mapping[str, object],
    check_ram: Callable[[], None],
    stage: Callable[[], Mapping[str, bytes]],
    timeout: float,
    script: bytes,
) -> None:
    command = argv(opts, root)
    check_ram()
    _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="XFOIL")
    if folder.exists():
        raise ValueError("existing XFOIL point directory; refusing overwrite")
    if not _RUNNING.acquire(blocking=False):
        raise ValueError("XFOIL is already running; only one process at a time")
    try:
        with process_lease("XFOIL"):
            _run_owned(check_ram, opts, command, timeout, folder, stage(), script)
    finally:
        _RUNNING.release()


def _run_owned(
    check_ram: Callable[[], None],
    opts: Mapping[str, object],
    command: list[str],
    timeout: float,
    folder: Path,
    staged: Mapping[str, bytes],
    script: bytes,
) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "xfoil.in").write_bytes(script)
    for name, content in staged.items():
        (folder / name).write_bytes(content)
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    code = 1
    start = time.monotonic()
    try:
        with (
            (folder / "xfoil.in").open("rb") as stdin,
            (folder / "stdout.log").open("xb") as stdout,
            (folder / "stderr.log").open("xb") as stderr,
        ):
            check_ram()
            _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="XFOIL")
            process = subprocess.Popen(
                command,
                cwd=folder,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                creationflags=_NO_WINDOW,
                start_new_session=sys.platform != "win32",
            )
            while True:
                check_ram()
                _ram.require_ram(0, solver="XFOIL")
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    _kill_tree(process)
                    code = 124
                    break
                try:
                    code = process.wait(timeout=min(0.1, remaining))
                    check_ram()
                    _ram.require_ram(0, solver="XFOIL")
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        log = (folder / "stdout.log").read_text(encoding="utf-8", errors="replace")
        version_line = next(
            (
                line.strip()
                for line in log.splitlines()
                if re.search(r"\bXFOIL\b.*\bVersion\s+[0-9]", line, re.IGNORECASE)
            ),
            None,
        )
        (folder / "execution.json").write_text(
            json.dumps(
                {
                    "argv": command,
                    "exit_code": code,
                    "duration_s": time.monotonic() - start,
                    "version_line": version_line,
                    "timed_out": timed_out,
                    "timeout_s": timeout,
                    "ram_estimate_gib": opts["ram_estimate_gib"],
                },
                sort_keys=True,
                ensure_ascii=True,
            ),
            encoding="ascii",
        )

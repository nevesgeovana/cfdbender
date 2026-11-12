"""Stage and run one ESI case using shared host process safety primitives."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.settings import read_init
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._execute import _NO_WINDOW, _kill_tree, _timeout
from cfdbender.solvers._host._mesh import link_mesh
from cfdbender.solvers._host._single import process_lease
from cfdbender.solvers.openfoam._config import config, options, staged_mesh

_RUNNING = threading.Lock()


def argv(opts: Mapping[str, object], root: Path) -> list[str]:
    """Prepend the launcher verbatim, or replace it with a Python stand-in."""
    executable = opts.get("executable") or os.environ.get("CFDBENDER_OPENFOAM_EXE")
    if executable:
        path = Path(str(executable)).expanduser()
        path = path if path.is_absolute() else root / path
        if path.suffix != ".py" or not path.is_file():
            raise ValueError("OpenFOAM executable must name an existing .py stand-in")
        return [sys.executable, str(path)]
    launcher = opts.get("launcher", [])
    if not isinstance(launcher, list) or not all(isinstance(v, str) for v in launcher):
        raise ValueError("solver.launcher must be a list of strings")
    return [*launcher, "bash", "Allrun"]


def stage(folder: Path, text: str) -> None:
    """Split the deterministic translation into exclusive, contained case files."""
    sections = text.split("// FILE: ")
    if sections[0] or len(sections) == 1:
        raise ValueError("invalid OpenFOAM translation")
    for section in sections[1:]:
        name, body = section.split("\n", 1)
        target = workspace_path(folder, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(body.encode("ascii"))


def stage_case(point: AdapterPoint, text: str, folder: Path) -> None:
    """Create the point folder, write the case tree and Allrun, and link the mesh.

    A local run and a cluster submission stage the same files.
    """
    case = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
    )
    folder.mkdir(parents=True, exist_ok=False)
    stage(folder, text)
    link_mesh(
        workspace_path(point.root, case.mesh),
        folder / staged_mesh(case),
        point.point.inputs_sha256.get(case.mesh),
    )


def run(point: AdapterPoint, text: str, check_ram: Callable[[], None]) -> None:
    """Stage verified inputs and poll a single child under timeout and RAM guards."""
    point.verify()
    opts = options(
        point.setup, read_init(point.root, available_solvers=("openfoam",)).solver_options
    )
    command = argv(opts, point.root)
    timeout = _timeout(point, opts)
    folder = workspace_path(point.root, point.point.directory)
    check_ram()
    _ram.require_ram(0.25, solver="OpenFOAM")
    if folder.exists():
        raise ValueError("existing OpenFOAM point directory; refusing overwrite")
    if not _RUNNING.acquire(blocking=False):
        raise ValueError("OpenFOAM is already running; only one process at a time")
    try:
        # All adapters reserve the same host process slot.
        with process_lease("OpenFOAM"):
            stage_case(point, text, folder)
            _launch(command, folder, opts, timeout, check_ram)
    finally:
        _RUNNING.release()


def _launch(
    command: list[str],
    folder: Path,
    opts: Mapping[str, object],
    timeout: float,
    check_ram: Callable[[], None],
) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    process: subprocess.Popen[bytes] | None = None
    code, timed_out = 1, False
    start = time.monotonic()
    try:
        with (
            (folder / "stdout.log").open("xb") as stdout,
            (folder / "stderr.log").open("xb") as stderr,
        ):
            check_ram()
            _ram.require_ram(0.25, solver="OpenFOAM")
            process = subprocess.Popen(
                command,
                cwd=folder,
                env=env,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                shell=False,
                creationflags=_NO_WINDOW,
                start_new_session=sys.platform != "win32",
            )
            while True:
                check_ram()
                _ram.require_ram(0, solver="OpenFOAM")
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    _kill_tree(process)
                    code = 124
                    break
                try:
                    code = process.wait(timeout=min(0.1, remaining))
                    check_ram()
                    _ram.require_ram(0, solver="OpenFOAM")
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        if process is not None and process.poll() is None:
            _kill_tree(process)
        (folder / "execution.json").write_text(
            json.dumps(
                {
                    "exit_code": code,
                    "timed_out": timed_out,
                    "timeout_s": timeout,
                    "threads": opts["threads"],
                    "processes": opts["processes"],
                    "mpi_launcher": opts["mpi_launcher"],
                    "argv": command,
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )

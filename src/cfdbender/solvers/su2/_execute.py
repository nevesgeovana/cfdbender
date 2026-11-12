"""Run either SU2 route with bounded waits and durable child channel capture."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from importlib.machinery import PathFinder
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.settings import read_init
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._execute import _NO_WINDOW, _RUNNING, _kill_tree, _timeout
from cfdbender.solvers._host._mesh import link_mesh
from cfdbender.solvers._host._single import process_lease
from cfdbender.solvers.su2._config import config, mesh_name, number, options, render, staged_mesh
from cfdbender.solvers.su2.routes import select_route


def _argv(opts: Mapping[str, object], root: Path) -> list[str]:
    raw = opts.get("executable") or os.environ.get("CFDBENDER_SU2_EXE") or "SU2_CFD"
    value = str(raw)
    executable = Path(value).expanduser()
    is_path = executable.is_absolute() or any(sep in value for sep in (os.sep, os.altsep) if sep)
    if is_path:
        if not executable.is_absolute():
            executable = root / executable
        allowed = executable.suffix == ".py" or os.name == "nt" or os.access(executable, os.X_OK)
        resolved = str(executable) if executable.is_file() and allowed else None
        if resolved is None and not executable.exists():
            # A path written without its Windows suffix (SU2_CFD for SU2_CFD.exe) still resolves.
            resolved = shutil.which(str(executable))
    else:
        resolved = shutil.which(str(executable))
    if resolved is None or not Path(resolved).is_file():
        raise ValueError("SU2 executable not found; set solver.executable or CFDBENDER_SU2_EXE")
    # A Python stand-in uses the same argv/cwd/channel path in offline conformance tests.
    command = [sys.executable, resolved] if executable.suffix == ".py" else [resolved]
    return [*cast(list[str], opts.get("mpi_launcher", [])), *command, "su2.cfg"]


def stage_files(point: AdapterPoint, values: Mapping[str, object], folder: Path) -> None:
    """Create the point folder, write su2.cfg and link the mesh into it.

    A local run and a cluster submission stage the same files.
    """
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "su2.cfg").write_bytes(render(values).encode("ascii"))
    link_mesh(
        workspace_path(point.root, mesh_name(values)),
        folder / staged_mesh(values),
        point.point.inputs_sha256.get(mesh_name(values)),
    )


def stage(point: AdapterPoint) -> None:
    """Write every input of a job into its new point folder without running SU2."""
    point.verify()
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
    )
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError("existing SU2 point directory; refusing overwrite")
    stage_files(point, values, folder)


def run(point: AdapterPoint, check_ram: Callable[[], None]) -> None:
    point.verify()
    declaration = read_init(point.root, available_solvers=("su2",))
    opts = options(point.setup, declaration.solver_options)
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
    )
    raw_path = opts.get("pysu2_path") or os.environ.get("CFDBENDER_PYSU2_PATH")
    wrapper_path = None
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        wrapper_path = str((path if path.is_absolute() else point.root / path).resolve())
    route = select_route(str(opts["route"]) if "route" in opts else None, pysu2_path=wrapper_path)
    if route == "pysu2":
        argv = [sys.executable, "-m", "cfdbender.solvers.su2._pysu2_worker", "su2.cfg"]
    else:
        argv = _argv(opts, point.root)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(opts["threads"])
    if route == "pysu2":
        # Preserve this installation when the child's cwd changes to the point.
        paths = [str(Path(__file__).resolve().parents[3])]
        spec = (wrapper_path and PathFinder.find_spec("pysu2", [wrapper_path])) or (
            importlib.util.find_spec("pysu2")
        )
        if spec and spec.origin:
            module_path = Path(spec.origin).resolve().parent
            if spec.submodule_search_locations is not None:
                module_path = module_path.parent
            paths.insert(0, str(module_path))
        env["PYTHONPATH"] = os.pathsep.join(paths)
    timeout = _timeout(point, opts)
    check_ram()
    _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="SU2")
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError("existing SU2 point directory; refusing overwrite")
    if not _RUNNING.acquire(blocking=False):
        raise ValueError("SU2 is already running; only one process at a time")
    try:
        with process_lease("SU2"):
            _run_owned(point, check_ram, opts, values, argv, timeout, folder, route, env)
    finally:
        _RUNNING.release()


def _run_owned(
    point: AdapterPoint,
    check_ram: Callable[[], None],
    opts: Mapping[str, object],
    values: Mapping[str, object],
    argv: list[str],
    timeout: float,
    folder: Path,
    route: str,
    env: Mapping[str, str],
) -> None:
    stage_files(point, values, folder)
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    code = 1
    start = time.monotonic()
    try:
        with (
            (folder / "stdout.log").open("xb") as stdout,
            (folder / "stderr.log").open("xb") as stderr,
        ):
            check_ram()
            _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="SU2")
            process = subprocess.Popen(
                argv,
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
                _ram.require_ram(0, solver="SU2")
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    _kill_tree(process)
                    code = 124
                    break
                try:
                    code = process.wait(timeout=min(0.1, remaining))
                    check_ram()
                    _ram.require_ram(0, solver="SU2")
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
                    "route": route,
                    "timeout_s": timeout,
                    "threads": opts["threads"],
                    "ram_estimate_gib": opts["ram_estimate_gib"],
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )

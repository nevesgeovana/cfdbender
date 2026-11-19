"""Run either SU2 route with bounded waits and durable child channel capture."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping
from importlib.machinery import PathFinder
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.restart import copy_solution, write_pending
from cfdbender.core.settings import read_init
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._exe import check_host_free, resolve_executable
from cfdbender.solvers._host._execute import _NO_WINDOW, _kill_tree, _timeout
from cfdbender.solvers._host._mesh import link_mesh
from cfdbender.solvers._host._version import refuse_wrong_build
from cfdbender.solvers.su2._config import (
    config,
    extra_files,
    mesh_name,
    number,
    options,
    render,
    staged_mesh,
)
from cfdbender.solvers.su2._precursor import (
    CAP_WARNING,
    EXIT_FAILED,
    PRECURSOR_CFG,
    PRECURSOR_LOG,
    PRECURSOR_SCRIPT,
    after_precursor,
    capped,
    failure,
    stages,
)
from cfdbender.solvers.su2._restart import restart_files, restart_values, staged_solution
from cfdbender.solvers.su2._run_types import SEGMENT_TWO, SEGMENTS_SCRIPT
from cfdbender.solvers.su2.routes import select_route


def _argv(opts: Mapping[str, object], root: Path) -> list[str]:
    raw = opts.get("executable") or os.environ.get("CFDBENDER_SU2_EXE") or "SU2_CFD"
    value = str(raw)
    resolved = resolve_executable(value, root, "SU2", "executable")
    # A Python stand-in uses the same argv/cwd/channel path in offline conformance tests.
    command = [sys.executable, str(resolved)] if resolved.suffix == ".py" else [str(resolved)]
    return [*cast(list[str], opts.get("mpi_launcher", [])), *command, "su2.cfg"]


def job_command(point: AdapterPoint) -> list[str]:
    """Return the command line a batch script runs in the point folder, as the job would.

    The executable is named as stated (never resolved on this machine, because the job runs
    elsewhere): the launcher, ``SU2_CFD`` or the stated executable, then ``su2.cfg``.
    For an export window with two segments, the generated script runs this command with
    each configuration in order; for a point with a frozen-rotor precursor, ``run_mrf.sh`` runs
    this command once per stage, the stage's configuration file as its last argument.
    """
    declaration = read_init(point.root, available_solvers=("su2",))
    opts = options(point.setup, declaration.solver_options)
    raw = opts.get("executable") or os.environ.get("CFDBENDER_SU2_EXE") or "SU2_CFD"
    command = [*cast(list[str], opts.get("mpi_launcher", [])), str(raw)]
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
        restarted=point.point.restart_from is not None,
        precursor=point.point.precursor,
    )
    if point.point.precursor:
        return [*job_wrapper(point), *command]
    if SEGMENT_TWO in extra_files(values):
        return ["sh", SEGMENTS_SCRIPT, *command]
    return [*command, "su2.cfg"]


def job_wrapper(point: AdapterPoint) -> list[str]:
    """Return the script a point's command runs under: ``run_mrf.sh`` for a precursor point.

    A batch script puts it in front of a row's own command (an application wrapper script
    included), without that command's last argument: the script appends the configuration
    file of each stage. A point without a precursor has none.
    """
    return ["sh", PRECURSOR_SCRIPT] if point.point.precursor else []


def stage_files(point: AdapterPoint, values: Mapping[str, object], folder: Path) -> None:
    """Create the point folder, write su2.cfg and link the mesh into it.

    A local run and a cluster submission stage the same files.
    """
    solution = restart_files(point) if point.point.restart_from is not None else {}
    values = restart_values(point, dict(values))
    folder.mkdir(parents=True, exist_ok=False)
    if point.point.restart_in_job:
        # The source has not run: the job script stages the solution after it ends.
        write_pending(folder, point, staged_solution(dict(solution), values))
    else:
        copy_solution(folder, staged_solution(dict(solution), values))
    (folder / "su2.cfg").write_bytes(render(values).encode("ascii"))
    for name, text in extra_files(values).items():
        (folder / name).write_bytes(text.encode("ascii"))
    link_mesh(
        workspace_path(point.root, mesh_name(values)),
        folder / staged_mesh(values),
        point.point.inputs_sha256.get(mesh_name(values)),
        inputs=point.inputs(),
        name=mesh_name(values),
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
        run_type=point.point.run_type,
        restarted=point.point.restart_from is not None,
        precursor=point.point.precursor,
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
        run_type=point.point.run_type,
        restarted=point.point.restart_from is not None,
        precursor=point.point.precursor,
    )
    raw_path = opts.get("pysu2_path") or os.environ.get("CFDBENDER_PYSU2_PATH")
    wrapper_path = None
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        wrapper_path = str((path if path.is_absolute() else point.root / path).resolve())
    # A stated executable is a choice of the executable route: no wrapper search, no warning.
    chosen = (
        str(opts["route"]) if "route" in opts else ("executable" if "executable" in opts else None)
    )
    route = select_route(chosen, pysu2_path=wrapper_path)
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
    _run_owned(point, check_ram, opts, values, argv, timeout, folder, route, env)


#: Executables that are launcher scripts: SU2 runs inside them with their own environment.
WRAPPER_SUFFIXES = (".cmd", ".bat", ".sh", ".ps1")


def recorded_threads(opts: Mapping[str, object], argv: list[str], route: str) -> object:
    """Return the thread count execution.json records, or "unknown" behind a launcher script.

    cfdbender sets ``OMP_NUM_THREADS`` from ``threads`` for the process it starts. When that
    process is a launcher script (a ``.cmd``, ``.bat``, ``.sh`` or ``.ps1`` executable, such as
    a wrapper that runs SU2 in WSL with its own thread count), SU2 runs in the script's
    environment and does not print its thread count, so the count is not known here.
    """
    if route == "executable" and len(argv) >= 2 and argv[-2].lower().endswith(WRAPPER_SUFFIXES):
        return "unknown"
    return opts["threads"]


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
    # A point with an export window runs its two segments in order, as run_segments.sh does
    # in a job: the second starts only when the first ended with status zero.
    segments = ["su2.cfg", *((SEGMENT_TWO,) if (folder / SEGMENT_TWO).is_file() else ())]
    # A point with a precursor runs its steady stage first and stages its solution, as
    # run_mrf.sh does in a job; stage 1 writes its own log.
    spec = stages(values)
    log = None
    if spec is not None:
        segments = [PRECURSOR_CFG, *spec.segments]
        log = (folder / PRECURSOR_LOG).open("xb")
    try:
        with (
            (folder / "stdout.log").open("xb") as stdout,
            (folder / "stderr.log").open("xb") as stderr,
        ):
            for segment in segments:
                if time.monotonic() - start >= timeout:
                    timed_out, code = True, 124
                    break
                code = 1
                check_ram()
                _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="SU2")
                first = log is not None and segment == PRECURSOR_CFG
                process = subprocess.Popen(
                    [*argv[:-1], segment],
                    cwd=folder,
                    env=env,
                    stdout=log if first else stdout,
                    stderr=log if first else stderr,
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
                        status = process.wait(timeout=min(0.1, remaining))
                        check_ram()
                        _ram.require_ram(0, solver="SU2")
                        code = status
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if first and code != 0 and not timed_out:
                    ended = f"the steady stage ended with status {code} (see {PRECURSOR_LOG})"
                    note = failure(ended)
                    stderr.write((note + chr(10)).encode("ascii"))
                    code = EXIT_FAILED
                elif first and code == 0 and spec is not None:
                    if (verdict := after_precursor(folder, spec)) is not None:
                        stderr.write((verdict[1] + chr(10)).encode("ascii"))
                        code = verdict[0]
                    elif spec.converge and capped(folder):
                        stdout.write((CAP_WARNING + chr(10)).encode("ascii"))
                if code != 0:
                    break
    finally:
        if log is not None:
            log.close()
        if process is not None and process.poll() is None:
            _kill_tree(process)
        (folder / "execution.json").write_text(
            json.dumps(
                {
                    "exit_code": code,
                    "timed_out": timed_out,
                    "route": route,
                    "timeout_s": timeout,
                    "threads": recorded_threads(opts, argv, route),
                    "ram_estimate_gib": opts["ram_estimate_gib"],
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )


def check_executable(point: AdapterPoint) -> None:
    """Resolve the executable the point would run, writing nothing and asking no host slot."""
    declaration = read_init(point.root, available_solvers=("su2",))
    opts = options(point.setup, declaration.solver_options)
    raw_path = opts.get("pysu2_path") or os.environ.get("CFDBENDER_PYSU2_PATH")
    wrapper_path = None
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        wrapper_path = str((path if path.is_absolute() else point.root / path).resolve())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        route = select_route(
            str(opts["route"]) if "route" in opts else None, pysu2_path=wrapper_path
        )
    if route != "pysu2":
        _argv(opts, point.root)


def preflight(point: AdapterPoint) -> None:
    """Resolve the executable and ask the host for a slot and RAM, writing nothing.

    A row's SOLVER_BUILD is checked here when the executable can say its version
    (``--version``), so a wrong build refuses the run before any point starts.
    """
    check_executable(point)
    opts = options(point.setup, read_init(point.root, available_solvers=("su2",)).solver_options)
    chosen = (
        str(opts["route"]) if "route" in opts else ("executable" if "executable" in opts else None)
    )
    if point.point.solver_build and chosen == "executable":
        launcher = len(cast(list[str], opts.get("mpi_launcher", [])))
        refuse_wrong_build(point.point.solver_build, _argv(opts, point.root)[launcher:-1])
    check_host_free("SU2", number(opts["ram_estimate_gib"], "ram_estimate_gib"))

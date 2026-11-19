"""Stage a CFD++ point and run its ordered steps under the shared timeout and RAM guards.

The executables come from the row's build entry in ``inputs/executables.toml``
(``EXECUTABLE_FIELDS``): the serial executable when NCPUS is 1, otherwise the MPI
launcher tokens followed by the MPI executable, with ``{ncpus}`` filled. A grid
folder without the partition file of the row's NCPUS gets the ``partitioner`` step
first, run in the point folder after the input is written, its output in
``log.partition``; a partition file still missing after it stops the steps. Each
step runs without a shell and without a window, standard input closed, the solver's
console output in ``stdout.log`` (its log) and its errors in ``stderr.log``.

On a timeout or a RAM-guard trip the database's stop tool (``role = "stop"``, from
the entry's ``tools_dir``) is run first in the point folder, asking the solver to
finish its step and write its files; after ``stop_grace_s`` the process tree is
killed. Without the tool, or without evidence for it, the process is killed at once.
The point is failed either way. A step file ending in ``.py`` is run with this
Python (the offline stand-ins of the tests).
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.restart import copy_solution
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._execute import _NO_WINDOW, _kill_tree, _timeout
from cfdbender.solvers.cfdpp import _mesh
from cfdbender.solvers.cfdpp._restart import restart_files
from cfdbender.solvers.cfdpp._schema import EMITTED, Schema

SOLVER_LABEL = "CFD++"
#: Field of a build entry in inputs/executables.toml to its kind (core.executables).
EXECUTABLE_FIELDS: Mapping[str, str] = {
    "serial": "path",
    "mpi": "path",
    "mpi_launcher": "tokens",
    "partitioner": "tokens",
    "tools_dir": "path",
}
INPUT_NAME = "mcfd.inp"
PARTITION_LOG = "log.partition"
STDOUT = "stdout.log"
STDERR = "stderr.log"
EXECUTION = "execution.json"
JOB_SCRIPT = "cfdpp_job.sh"
JOB_DONE = "job.done"
STOP_ROLE = "stop"
NCPUS_TOKEN = "{ncpus}"


@dataclass(frozen=True)
class Step:
    """One step of a point: its name, its argument vector and the file of its output."""

    name: str
    argv: tuple[str, ...]
    log: str


def entry(point: AdapterPoint) -> Mapping[str, object]:
    """Return the build entry the run bound into the point, refusing a point without one."""
    table = point.setup.get("solver")
    if not isinstance(table, Mapping) or not table:
        raise ValueError(
            f"{SOLVER_LABEL}: SOLVER_BUILD {point.point.solver_build!r} has no executables on "
            "this machine; declare the build in inputs/executables.toml with serial (and mpi, "
            "mpi_launcher for NCPUS above 1) and give this machine's paths in "
            "inputs/executables.local.toml"
        )
    return table


def ncpus(point: AdapterPoint) -> int:
    """Return the row's process count (1 when the row states none)."""
    return point.point.ncpus or 1


def _python(tokens: Sequence[str]) -> list[str]:
    tokens = list(tokens)
    return [sys.executable, *tokens] if tokens and tokens[0].endswith(".py") else tokens


def _fill(tokens: object, count: int) -> list[str]:
    assert isinstance(tokens, list)
    return [str(token).replace(NCPUS_TOKEN, str(count)) for token in tokens]


def _need(table: Mapping[str, object], name: str, build: str | None, why: str) -> object:
    value = table.get(name)
    if not value:
        raise ValueError(
            f"{SOLVER_LABEL}: {why} needs the {name} field of build {build!r} in "
            "inputs/executables.toml"
        )
    return value


def argv(
    table: Mapping[str, object], count: int, *, partitioned: bool, build: str | None = None
) -> list[Step]:
    """Return the ordered steps of a point for ``count`` processes."""
    steps = []
    if count == 1:
        solver = [str(_need(table, "serial", build, "NCPUS 1"))]
    else:
        why = f"NCPUS {count}"
        launcher = _fill(_need(table, "mpi_launcher", build, why), count)
        solver = [*launcher, str(_need(table, "mpi", build, why))]
        if not partitioned:
            partitioner = _fill(
                _need(table, "partitioner", build, f"{why} without a partition file"), count
            )
            steps.append(Step("partition", tuple(_python(partitioner)), PARTITION_LOG))
    steps.append(Step("solver", tuple(_python(solver)), STDOUT))
    return steps


def partitioned(folder: Path, count: int) -> bool:
    """Say whether a grid folder holds the partition file of ``count`` processes."""
    return count == 1 or (folder / f"{_mesh.PARTITION_PREFIX}{count}").is_file()


def job_script(steps: Sequence[Step], count: int) -> str:
    """Return the shell script of a script-driven cluster job: the same steps, then job.done."""
    lines = [
        "#!/bin/sh",
        "# Written by cfdbender: the ordered steps of one CFD++ point.",
        'cd "$(dirname "$0")" || exit 1',
        "set -e",
        "finish() {",
        "    code=$?",
        "    printf " + """'{"exit_code": %s}\\n' "$code" > execution.json.tmp""",
        "    mv execution.json.tmp execution.json",
        f"    touch {JOB_DONE}",
        '    exit "$code"',
        "}",
        "trap finish 0",
    ]
    for step in steps:
        command = shlex.join(step.argv)
        if step.name == "partition":
            lines.append(f"{command} > {PARTITION_LOG} 2>&1")
            lines.append(f"test -f {_mesh.PARTITION_PREFIX}{count} || exit 1")
        else:
            lines.append(f"{command} < /dev/null > {STDOUT} 2> {STDERR}")
    return "\n".join(lines) + "\n"


def grid_digests(point: AdapterPoint, mesh: str) -> dict[str, str]:
    """Return the planned SHA-256 of each grid file, by file name."""
    prefix = mesh.rstrip("/") + "/"
    return {
        name[len(prefix) :]: digest
        for name, digest in point.point.inputs_sha256.items()
        if name.startswith(prefix) and "/" not in name[len(prefix) :]
    }


def stage_files(point: AdapterPoint, text: str, folder: Path, mesh: str) -> None:
    """Write the input, link the grid, copy a restart solution and write the job script.

    One routine for a local run and a cluster job (C-20): the folder must not exist.
    """
    solution = restart_files(point) if point.point.restart_from is not None else {}
    count = ncpus(point)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / INPUT_NAME).open("xb") as stream:
        stream.write(text.encode("ascii"))
    _mesh.stage_grid(
        workspace_path(point.root, mesh),
        folder,
        grid_digests(point, mesh),
        ncpus=count,
        inputs=point.inputs(),
        mesh=mesh,
    )
    copy_solution(folder, solution)
    table = point.setup.get("solver")
    if isinstance(table, Mapping) and table:
        steps = argv(
            table, count, partitioned=partitioned(folder, count), build=point.point.solver_build
        )
        with (folder / JOB_SCRIPT).open("x", encoding="ascii", newline="\n") as stream:
            stream.write(job_script(steps, count))


def stop_command(schema: Schema, table: Mapping[str, object]) -> list[str] | None:
    """Return the stop tool's command from ``tools_dir``, or None when it cannot be used.

    The tool is the database entry with ``role = "stop"``; it is used only with
    harvested or verified evidence for the version.
    """
    tools_dir = table.get("tools_dir")
    if not tools_dir:
        return None
    for tool in schema.entries.get("tools", ()):
        if tool.attrs.get("role") != STOP_ROLE or tool.status not in EMITTED:
            continue
        for suffix in ("", ".exe", ".py"):
            path = Path(str(tools_dir)) / f"{tool.name}{suffix}"
            if path.is_file():
                return _python([str(path)])
    return None


def executable_digest(path: str) -> str | None:
    """Return the SHA-256 of an executable file, or None when it cannot be read."""
    found = Path(path) if Path(path).is_file() else Path(shutil.which(path) or "")
    if not found.is_file():
        return None
    with found.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(
    point: AdapterPoint,
    text: str,
    mesh: str,
    schema: Schema,
    opts: Mapping[str, float],
    check_ram: Callable[[], None],
    extra: Mapping[str, object],
) -> int:
    """Stage the point and run its steps; return the exit code that decides the point."""
    point.verify()
    table = entry(point)
    count = ncpus(point)
    folder = workspace_path(point.root, point.point.directory)
    check_ram()
    _ram.require_ram(opts["ram_estimate_gib"], solver=SOLVER_LABEL)
    if folder.exists():
        raise ValueError(f"existing {SOLVER_LABEL} point directory; refusing overwrite")
    grid = workspace_path(point.root, mesh)
    steps = argv(table, count, partitioned=partitioned(grid, count), build=point.point.solver_build)
    stage_files(point, text, folder, mesh)
    timeout = _timeout(point, {"timeout_s": opts["timeout_s"]})
    start = time.monotonic()
    record: dict[str, object] = {
        "ncpus": count,
        "timeout_s": timeout,
        "timed_out": False,
        "stop_requested": False,
        "steps": [],
        "problem": "",
        **extra,
    }
    solver_path = str(table.get("serial") if count == 1 else table.get("mpi"))
    record["executable"] = solver_path
    record["executable_sha256"] = executable_digest(solver_path)
    code = 0
    try:
        for step in steps:
            remaining = timeout - (time.monotonic() - start)
            code, timed_out, stopped = _launch(
                step, folder, remaining, check_ram, stop_command(schema, table), opts
            )
            cast_steps = record["steps"]
            assert isinstance(cast_steps, list)
            cast_steps.append({"name": step.name, "argv": list(step.argv), "exit_code": code})
            record["timed_out"] = timed_out
            record["stop_requested"] = bool(record["stop_requested"]) or stopped
            if code != 0:
                break
            if (
                step.name == "partition"
                and not (folder / f"{_mesh.PARTITION_PREFIX}{count}").is_file()
            ):
                record["problem"] = (
                    f"the partition step left no {_mesh.PARTITION_PREFIX}{count}; "
                    f"see {PARTITION_LOG}"
                )
                code = 1
                break
    except BaseException as error:
        code = 1
        record["problem"] = str(error) or type(error).__name__
        raise
    finally:
        record["exit_code"] = code
        (folder / EXECUTION).write_text(
            json.dumps(record, sort_keys=True), encoding="ascii", newline="\n"
        )
    return code


def _launch(
    step: Step,
    folder: Path,
    timeout: float,
    check_ram: Callable[[], None],
    stop: list[str] | None,
    opts: Mapping[str, float],
) -> tuple[int, bool, bool]:
    """Run one step; return its exit code, whether it timed out and whether a stop was asked."""
    process: subprocess.Popen[bytes] | None = None
    code, timed_out, stopped = 1, False, False
    start = time.monotonic()
    # The solver's errors go to their own file; a tool's go with its output.
    errors = folder / STDERR if step.log == STDOUT else None
    try:
        with (
            (folder / step.log).open("xb") as stdout,
            (
                errors.open("xb") if errors is not None else open(os.devnull, "wb")  # noqa: SIM115
            ) as stderr,
        ):
            check_ram()
            _ram.require_ram(0, solver=SOLVER_LABEL)
            process = subprocess.Popen(
                list(step.argv),
                cwd=folder,
                stdout=stdout,
                stderr=stderr if errors is not None else subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                shell=False,
                creationflags=_NO_WINDOW,
                start_new_session=sys.platform != "win32",
            )
            while True:
                try:
                    check_ram()
                    _ram.require_ram(0, solver=SOLVER_LABEL)
                except ValueError:
                    stopped = _stop(process, folder, stop, opts)
                    raise
                if time.monotonic() - start >= timeout:
                    timed_out = True
                    stopped = _stop(process, folder, stop, opts)
                    code = 124
                    break
                try:
                    code = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        if process is not None and process.poll() is None:
            _kill_tree(process)
    return code, timed_out, stopped


def _stop(
    process: subprocess.Popen[bytes],
    folder: Path,
    stop: list[str] | None,
    opts: Mapping[str, float],
) -> bool:
    """Ask the solver to stop through its tool, wait the grace period, then kill."""
    asked = False
    if stop is not None and process.poll() is None:
        with (folder / "log.stop").open("ab") as log:
            subprocess.run(
                stop,
                cwd=folder,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=_NO_WINDOW,
                check=False,
                timeout=max(opts["stop_grace_s"], 1.0),
            )
        asked = True
        try:
            process.wait(timeout=opts["stop_grace_s"])
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        _kill_tree(process)
    return asked

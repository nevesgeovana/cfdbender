"""Run the section polars through the XFOIL adapter's execution path, then QPROP."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from cfdbender.core import progress
from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.solvers import load_solver
from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._config import number
from cfdbender.solvers._host._exe import resolve_executable
from cfdbender.solvers._host._execute import _NO_WINDOW, _kill_tree, _timeout
from cfdbender.solvers.qprop._config import Operating, Setup, options
from cfdbender.solvers.qprop._polar import Fit, fit_polar, read_polar_rows
from cfdbender.solvers.qprop._sections import Blade, Planned, deck


def qprop_argv(opts: Mapping[str, object], root: Path) -> list[str]:
    """Resolve the QPROP command: launcher and verbatim guest path, or a host path."""
    value = str(opts.get("executable") or os.environ.get("CFDBENDER_QPROP_EXE") or "qprop")
    launcher = cast(list[str], opts.get("launcher", []))
    if launcher:
        return [*launcher, value]
    resolved = resolve_executable(value, root, "QPROP", "executable")
    return [sys.executable, str(resolved)] if resolved.suffix == ".py" else [str(resolved)]


def _xfoil_options(opts: Mapping[str, object]) -> dict[str, object]:
    chosen: dict[str, object] = {
        "launcher": opts["launcher"],
        "timeout_s": opts.get("section_timeout_s", opts["timeout_s"]),
        "ram_estimate_gib": opts["ram_estimate_gib"],
    }
    if opts.get("xfoil_executable"):
        chosen["executable"] = opts["xfoil_executable"]
    return chosen


def _fit_folder(folder: Path) -> Fit | None:
    polar = folder / "polar.txt"
    if not polar.is_file():
        return None
    return fit_polar(read_polar_rows(polar.read_text(encoding="ascii", errors="replace")))


def _missing_angles(folder: Path, setup: Setup, narrow: bool) -> list[float]:
    """Return the requested angles of attack that the section's polar does not hold."""
    start, end, step = (-4.0, 10.0, 0.5) if narrow else setup.alpha
    requested = [start + k * step for k in range(int(round((end - start) / step)) + 1)]
    polar = folder / "polar.txt"
    found = (
        [row[0] for row in read_polar_rows(polar.read_text(encoding="ascii", errors="replace"))]
        if polar.is_file()
        else []
    )
    return [a for a in requested if all(abs(a - f) > 1e-3 for f in found)]


def _unconverged(opts: Mapping[str, object], label: str, missing: list[float]) -> None:
    """Apply the workspace policy to a section whose polar lacks requested angles."""
    angles = ", ".join(f"{a:g}" for a in missing)
    if opts.get("on_unconverged_section") == "fail":
        raise ValueError(
            f"{label}: XFOIL did not converge at alpha {angles}; "
            "on_unconverged_section is 'fail' (set it to 'warn' in init.toml to run on the rest)"
        )
    warn_or_refuse(
        f"{label}: XFOIL did not converge at alpha {angles}; "
        "those angles are dropped from this section's polar",
        stacklevel=2,
    )


def section_fits(
    point: AdapterPoint,
    folder: Path,
    setup: Setup,
    planned: tuple[Planned, ...],
    opts: Mapping[str, object],
    check_ram: Callable[[], None],
) -> list[dict[str, object]]:
    """Fit every station: given polars directly, missing ones through XFOIL.

    A failed XFOIL section is retried with a narrower alpha range; one that still
    fails reuses the nearest successful station's section parameters. Every
    station is recorded with its source. No station succeeding refuses the point.
    """
    xfoil = load_solver("xfoil")
    results: list[dict[str, object]] = []
    for item in planned:
        entry: dict[str, object] = {"index": item.index, "source": item.source, "fit": None}
        if item.spec.polar is not None:
            fit = fit_polar(read_polar_rows(item.spec.polar))
            if fit is None:
                raise ValueError(f"station {item.index}: the given polar has too few points")
            entry["fit"] = fit
        else:
            assert item.spec.shape is not None
            staged = {"airfoil.dat": item.spec.shape.encode("ascii")}
            label = f"section {item.index}/{len(planned)} (r = {item.spec.radius:.4g})"
            for narrow in (False, True):
                where = folder / "xfoil" / (item.name + ("_narrow" if narrow else ""))
                progress.emit(f"  {label}: XFOIL {'narrow sweep ' if narrow else ''}started")
                began = time.monotonic()
                xfoil.run_script(
                    point.root,
                    where,
                    deck(setup, item, narrow=narrow),
                    staged,
                    _xfoil_options(opts),
                    check_ram,
                )
                fit = _fit_folder(where)
                missing = _missing_angles(where, setup, narrow)
                took = time.monotonic() - began
                state = "converged" if fit is not None and not missing else "did not converge"
                progress.emit(f"  {label}: XFOIL {state} in {took:.0f} s")
                if missing:
                    _unconverged(opts, label, missing)
                if fit is not None:
                    entry.update(fit=fit, source="xfoil_narrow" if narrow else "xfoil")
                    break
        results.append(entry)
    good = [i for i, e in enumerate(results) if e["fit"] is not None]
    if not good:
        raise ValueError("no section polar could be produced by XFOIL")
    for i, entry in enumerate(results):
        if entry["fit"] is None:
            near = min(good, key=lambda g: abs(planned[g].spec.radius - planned[i].spec.radius))
            entry.update(fit=results[near]["fit"], source="reused", reused_from=near + 1)
    return results


def record(
    blade: Blade,
    planned: tuple[Planned, ...],
    fits: list[dict[str, object]],
    blades: int,
    beta_add: float,
    rpm: float,
) -> str:
    """Return the sections.json text: geometry, local conditions, fits and their sources."""
    sections = []
    for item, entry in zip(planned, fits, strict=True):
        fit = cast(Fit, entry["fit"])
        sections.append(
            {
                "index": item.index,
                "radius": item.spec.radius,
                "chord": item.spec.chord,
                "beta": item.spec.beta,
                "reynolds": item.reynolds,
                "mach": item.mach,
                "source": entry["source"],
                "reused_from": entry.get("reused_from"),
                "fit": {k: getattr(fit, k) for k in fit.__dataclass_fields__},
            }
        )
    return json.dumps(
        {
            "radius": blade.radius,
            "blades": blades,
            "beta_add": beta_add,
            "rpm": rpm,
            "sections": sections,
            **({"mesh_frame": dict(blade.mesh_frame)} if blade.mesh_frame is not None else {}),
        },
        sort_keys=True,
        ensure_ascii=True,
        indent=1,
    )


def run_qprop(
    point: AdapterPoint,
    folder: Path,
    op: Operating,
    rpm: float,
    opts: Mapping[str, object],
    check_ram: Callable[[], None],
) -> None:
    """Run QPROP in the point folder with captured channels and shared host guards."""
    command = [
        *qprop_argv(opts, point.root),
        "propeller.prop",
        "none",
        f"{op.speed:.10g}",
        f"{rpm:.10g}",
    ]
    timeout = _timeout(point, opts)
    estimate = number(opts["ram_estimate_gib"], "ram_estimate_gib")
    check_ram()
    _ram.require_ram(estimate, solver="QPROP")
    _run_owned(folder, command, timeout, opts, check_ram)


def _run_owned(
    folder: Path,
    command: list[str],
    timeout: float,
    opts: Mapping[str, object],
    check_ram: Callable[[], None],
) -> None:
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
            _ram.require_ram(number(opts["ram_estimate_gib"], "ram_estimate_gib"), solver="QPROP")
            process = subprocess.Popen(
                command,
                cwd=folder,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                creationflags=_NO_WINDOW,
                start_new_session=sys.platform != "win32",
            )
            while True:
                check_ram()
                _ram.require_ram(0, solver="QPROP")
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    _kill_tree(process)
                    code = 124
                    break
                try:
                    status = process.wait(timeout=min(0.1, remaining))
                    check_ram()
                    _ram.require_ram(0, solver="QPROP")
                    code = status
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
                if re.search(r"\bQPROP\b.*\bVersion\s+[0-9]", line, re.IGNORECASE)
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


def prepare(point: AdapterPoint) -> Path:
    """Return the new point folder, refusing an existing one."""
    folder = workspace_path(point.root, point.point.directory)
    if folder.exists():
        raise ValueError("existing QPROP point directory; refusing overwrite")
    return folder


__all__ = ["options", "prepare", "qprop_argv", "record", "run_qprop", "section_fits"]

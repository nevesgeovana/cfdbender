"""Validate one XFOIL operating point and render ASCII menu keystrokes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from cfdbender.core.flight import agree, mach_reynolds
from cfdbender.solvers._host._config import number

_OPTIONS = {"executable", "launcher", "timeout_s", "ram_estimate_gib"}


@dataclass(frozen=True)
class Config:
    """One alpha and its explicit airfoil, panel and operating settings."""

    airfoil: str
    airfoil_file: str | None
    alpha: float
    reynolds: float
    mach: float
    ncrit: float
    iterations: int
    panels: int | None


def options(
    setup: Mapping[str, object], workspace: Mapping[str, object] | None = None
) -> dict[str, object]:
    table = setup.get("solver", {})
    if not isinstance(table, dict):
        raise ValueError("s file [solver] must be a table")
    for values in (workspace or {}, table):
        if unknown := set(values) - _OPTIONS:
            raise ValueError("unknown XFOIL solver options: " + ", ".join(sorted(unknown)))
    result = {**(workspace or {}), **table}
    executable = result.get("executable")
    if executable is not None and (not isinstance(executable, str) or not executable.strip()):
        raise ValueError("solver.executable must be a nonempty path")
    launcher = result.setdefault("launcher", [])
    if not isinstance(launcher, list) or any(
        not isinstance(value, str) or not value.strip() or chr(0) in value for value in launcher
    ):
        raise ValueError("solver.launcher must be a list of nonempty strings")
    for key, default in (("timeout_s", 600.0), ("ram_estimate_gib", 0.25)):
        if number(result.setdefault(key, default), key) <= 0:
            raise ValueError(f"solver.{key} must be positive")
    return result


def _integer(value: object, name: str) -> int:
    numeric = number(value, name)
    if numeric < 1 or numeric != int(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def config(
    setup: Mapping[str, object],
    condition: Mapping[str, float],
    reference: Mapping[str, object] | None = None,
    geometry: str | None = None,
) -> Config:
    """Build one operating point from the setup and the row's condition and GEOMETRY.

    A FLIGHT_CONDITION stating MACH, REmi or other air keys gives mach and
    reynolds (REmi directly, else over the REF length); GEOMETRY gives the
    airfoil_file. A setup value that disagrees with the row is refused.
    """
    condition, flow = mach_reynolds(condition, reference, native_mach=True)
    updates: dict[str, object] = dict(flow)
    if "MACH" in condition:
        # A bare MACH with angles is the setup's mach, stated on the row.
        updates["mach"] = condition.pop("MACH")
    if geometry is not None:
        updates["airfoil_file"] = geometry
    setup = agree(setup, updates, "row")
    options(setup)
    allowed = {
        "airfoil",
        "airfoil_file",
        "reynolds",
        "mach",
        "ncrit",
        "iterations",
        "panels",
        "solver",
    }
    if unknown := set(setup) - allowed:
        raise ValueError("unsupported XFOIL setup: " + ", ".join(sorted(unknown)))
    if ("airfoil" in setup) == ("airfoil_file" in setup):
        raise ValueError("specify exactly one of airfoil and airfoil_file")
    airfoil = setup.get("airfoil", "")
    if "airfoil" in setup and (
        not isinstance(airfoil, str) or re.fullmatch("NACA ([0-9]{4}|[0-9]{5})", airfoil) is None
    ):
        raise ValueError("airfoil must be a NACA 4 or 5 digit designation, e.g. NACA 0012")
    filename = setup.get("airfoil_file")
    if "airfoil_file" in setup and (
        not isinstance(filename, str)
        or re.fullmatch("inputs/(airfoils|geometries|geometry)/[^/]+[.]dat", filename) is None
    ):
        raise ValueError(
            "airfoil_file must name inputs/airfoils/<name>.dat or inputs/geometries/<name>.dat"
        )
    normalized = {key.lower(): value for key, value in condition.items()}
    if set(normalized) != {"alpha"}:
        raise ValueError("XFOIL matrix conditions require only scalar alpha or an alpha sweep")
    reynolds = number(setup.get("reynolds", 0), "reynolds")
    mach = number(setup.get("mach", 0), "mach")
    ncrit = number(setup.get("ncrit", 9), "ncrit")
    if reynolds < 0:
        raise ValueError("reynolds must be non-negative (zero means inviscid)")
    if mach < 0:
        raise ValueError("mach must be non-negative")
    if ncrit <= 0:
        raise ValueError("ncrit must be positive")
    return Config(
        cast(str, airfoil),
        cast(str | None, filename),
        number(normalized["alpha"], "alpha"),
        reynolds,
        mach,
        ncrit,
        _integer(setup.get("iterations", 200), "iterations"),
        _integer(setup["panels"], "panels") if "panels" in setup else None,
    )


def render(values: Config) -> str:
    lines = ["PLOP", "G", "", "LOAD airfoil.dat" if values.airfoil_file else values.airfoil]
    if values.panels is not None:
        lines.extend(["PPAR", f"N {values.panels}", "", ""])
    lines.extend(["PANE", "OPER"])
    if values.reynolds:
        lines.append(f"VISC {values.reynolds:.15g}")
    lines.extend([f"MACH {values.mach:.15g}", f"ITER {values.iterations}"])
    if values.ncrit != 9:
        lines.extend(["VPAR", f"N {values.ncrit:.15g}", ""])
    lines.extend(["PACC", "polar.txt", "", f"ALFA {values.alpha:.15g}", "", "", "QUIT", ""])
    return chr(10).join(lines)

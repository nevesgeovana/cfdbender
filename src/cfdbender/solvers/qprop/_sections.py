"""Resolve the blade into stations and plan each station's 2D polar and QPROP files."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable
from dataclasses import dataclass

from cfdbender.geometry.blade_slicing import radial_extent, slice_blade
from cfdbender.solvers.qprop._config import Operating, Setup
from cfdbender.solvers.qprop._polar import polar_script


@dataclass(frozen=True)
class Spec:
    """One station in metres and degrees; shape and polar are file text when given."""

    radius: float
    chord: float
    beta: float
    shape: str | None
    polar: str | None


@dataclass(frozen=True)
class Blade:
    """The resolved stations in ascending radius and the tip radius in metres."""

    specs: tuple[Spec, ...]
    radius: float


@dataclass(frozen=True)
class Planned:
    """One station with its local flight condition and where its polar comes from."""

    index: int
    spec: Spec
    reynolds: float
    mach: float

    @property
    def source(self) -> str:
        """Return ``table`` when a polar file is given, else ``xfoil``."""
        return "table" if self.spec.polar is not None else "xfoil"

    @property
    def name(self) -> str:
        """Return the section folder name."""
        return f"sec_{self.index:02d}"


def airfoil_text(index: int, shape: tuple[tuple[float, float], ...]) -> str:
    """Return an XFOIL coordinate file for a unit-chord outline."""
    lines = [f"section {index}"]
    # Adding 0.0 turns a rounded negative zero into zero, so no "-0.000000" is written.
    lines += [f"{round(x, 6) + 0.0:.6f} {round(y, 6) + 0.0:.6f}" for x, y in shape]
    return "\n".join(lines) + "\n"


def _number(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label}: {text!r} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{label}: {text!r} must be a finite number")
    return value


def _table(setup: Setup, count: int | None, read: Callable[[str], bytes]) -> Blade:
    folder = setup.blade
    rows = list(csv.DictReader(io.StringIO(read(f"{folder}/blade.csv").decode("utf-8"))))
    if not rows:
        raise ValueError(f"{folder}/blade.csv has no stations")
    missing = {"r", "chord", "beta"} - set(rows[0])
    if missing:
        raise ValueError(
            f"{folder}/blade.csv needs columns r, chord, beta: missing {sorted(missing)}"
        )
    if count is not None and count != len(rows):
        raise ValueError(f"blade table carries {len(rows)} stations, but n_sections is {count}")
    specs = []
    for number, row in enumerate(rows, 1):
        label = f"{folder}/blade.csv station {number}"
        airfoil = (row.get("airfoil") or "").strip()
        polar = (row.get("polar") or "").strip()
        if not airfoil and not polar:
            raise ValueError(f"{label}: needs an airfoil or polar file")
        specs.append(
            Spec(
                _number(row["r"], f"{label} r") * setup.scale,
                _number(row["chord"], f"{label} chord") * setup.scale,
                _number(row["beta"], f"{label} beta"),
                read(f"{folder}/{airfoil}").decode("ascii") if airfoil else None,
                read(f"{folder}/{polar}").decode("ascii") if polar else None,
            )
        )
    radii = [s.radius for s in specs]
    if any(b <= a for a, b in zip(radii, radii[1:], strict=False)):
        raise ValueError(f"{folder}/blade.csv radii must be strictly ascending")
    return Blade(tuple(specs), setup.radius or radii[-1])


def resolve(setup: Setup, count: int | None, read: Callable[[str], bytes]) -> Blade:
    """Read the blade named by the setup: slice an OBJ or load a polars-2D table.

    Parameters
    ----------
    setup : Setup
        Validated setup.
    count : int or None
        Station count from the matrix (N_SECTIONS) or the setup.
    read : callable
        Tracked workspace reader, so every byte read is in the input digests.
    """
    if setup.blade.startswith("inputs/polars-2D/"):
        return _table(setup, count, read)
    if count is None:
        raise ValueError("n_sections is required for a blade OBJ (setup or N_SECTIONS)")
    text = read(setup.blade).decode("utf-8")
    sections = slice_blade(
        text,
        n_sections=count,
        spacing=setup.spacing,
        cut=setup.cut,
        axis=setup.axis,
        hub_center=setup.hub_center,
        rotation=setup.rotation,
    )
    tip = radial_extent(text, axis=setup.axis, hub_center=setup.hub_center)[1]
    specs = tuple(
        Spec(
            s.radius * setup.scale,
            s.chord * setup.scale,
            s.beta_deg,
            airfoil_text(i, s.shape),
            None,
        )
        for i, s in enumerate(sections, 1)
    )
    return Blade(specs, setup.radius or tip * setup.scale)


def rotation_rate(op: Operating, radius: float) -> float:
    """Return rpm, from RPM directly or from J = V / (n D) with D = 2 R."""
    if op.rpm is not None:
        return op.rpm
    assert op.j is not None
    return 60.0 * op.speed / (op.j * 2.0 * radius)


def plan_sections(blade: Blade, op: Operating, rpm: float) -> tuple[Planned, ...]:
    """Return each station with its local Reynolds and Mach number."""
    omega = 2.0 * math.pi * rpm / 60.0
    planned = []
    for index, spec in enumerate(blade.specs, 1):
        speed = math.hypot(op.speed, omega * spec.radius)
        planned.append(Planned(index, spec, op.rho * speed * spec.chord / op.mu, speed / op.a))
    return tuple(planned)


def beta_offset(blade: Blade, op: Operating) -> float:
    """Return the pitch offset in degrees: BETA07 minus the blade's own pitch at 0.7 R."""
    if op.beta07 is None:
        return op.dbeta
    target = 0.7 * blade.radius
    specs = blade.specs
    if target <= specs[0].radius:
        own = specs[0].beta
    elif target >= specs[-1].radius:
        own = specs[-1].beta
    else:
        upper = next(i for i, s in enumerate(specs) if s.radius >= target)
        a, b = specs[upper - 1], specs[upper]
        own = a.beta + (b.beta - a.beta) * (target - a.radius) / (b.radius - a.radius)
    return op.beta07 - own


def deck(setup: Setup, item: Planned, *, narrow: bool = False) -> str:
    """Return the XFOIL keystrokes for one station, the wide or the narrow sweep."""
    start, end, step = (-4.0, 10.0, 0.5) if narrow else setup.alpha
    return polar_script(
        reynolds=item.reynolds,
        mach=item.mach,
        alpha_start=start,
        alpha_end=end,
        alpha_step=step,
        ncrit=setup.ncrit,
        iterations=setup.iterations,
    )

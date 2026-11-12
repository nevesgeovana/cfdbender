"""Validate the QPROP setup file and the operating-point conditions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from cfdbender.core.flight import STATE_KEYS, agree, canonical_key, row_flow
from cfdbender.geometry.blade_slicing import AXES, CUTS, ROTATIONS, SPACINGS
from cfdbender.solvers._host._config import number

_OPTIONS = {"executable", "xfoil_executable", "launcher", "timeout_s", "ram_estimate_gib"}
_SETUP = {
    "blade",
    "n_blades",
    "n_sections",
    "spacing",
    "cut",
    "solver_2d",
    "axis",
    "hub_center",
    "rotation",
    "scale",
    "radius",
    "alpha_start",
    "alpha_end",
    "alpha_step",
    "ncrit",
    "iterations",
    "re_exponent",
    "title",
    "solver",
}
SOLVERS_2D = ("xfoil",)
CONDITIONS = ("V", "RPM", "J", "BETA07", "DBETA", "RHO", "MU", "A", "N_SECTIONS")
# International standard atmosphere at sea level, used when a row omits them.
DEFAULT_RHO, DEFAULT_MU, DEFAULT_A = 1.225, 1.789e-5, 340.3


@dataclass(frozen=True)
class Setup:
    """The blade, its section count and slicing convention, and the 2D polar settings."""

    blade: str
    blades: int
    n_sections: int | None
    spacing: str
    cut: str
    solver_2d: str
    axis: str
    hub_center: tuple[float, float, float]
    rotation: str
    scale: float
    radius: float | None
    alpha: tuple[float, float, float]
    ncrit: float
    iterations: int
    re_exponent: float
    title: str


@dataclass(frozen=True)
class Operating:
    """One operating point; exactly one of rpm and j is set."""

    speed: float
    rpm: float | None
    j: float | None
    beta07: float | None
    dbeta: float
    rho: float
    mu: float
    a: float
    n_sections: int | None


def options(
    setup: Mapping[str, object], workspace: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Merge and validate the executable, launcher, timeout and RAM options."""
    table = setup.get("solver", {})
    if not isinstance(table, dict):
        raise ValueError("s file [solver] must be a table")
    for values in (workspace or {}, table):
        if unknown := set(values) - _OPTIONS:
            raise ValueError("unknown QPROP solver options: " + ", ".join(sorted(unknown)))
    result = {**(workspace or {}), **table}
    for key in ("executable", "xfoil_executable"):
        value = result.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"solver.{key} must be a nonempty path")
    launcher = result.setdefault("launcher", [])
    if not isinstance(launcher, list) or any(
        not isinstance(v, str) or not v.strip() or chr(0) in v for v in launcher
    ):
        raise ValueError("solver.launcher must be a list of nonempty strings")
    for key, default in (("timeout_s", 600.0), ("ram_estimate_gib", 0.25)):
        if number(result.setdefault(key, default), key) <= 0:
            raise ValueError(f"solver.{key} must be positive")
    return result


def _count(value: object, name: str) -> int:
    numeric = number(value, name)
    if numeric < 1 or numeric != int(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _choice(setup: Mapping[str, object], name: str, allowed: tuple[str, ...], default: str) -> str:
    value = setup.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)} (got {value!r})")
    return str(value)


def setup(values: Mapping[str, object]) -> Setup:
    """Validate the s file; raise ValueError naming the first problem."""
    options(values)
    if unknown := set(values) - _SETUP:
        raise ValueError("unsupported QPROP setup: " + ", ".join(sorted(unknown)))
    blade = values.get("blade")
    if blade is None:
        raise ValueError(
            "blade is required: name inputs/geometry/<name>.obj or inputs/polars-2D/<set>"
        )
    if (
        not isinstance(blade, str)
        or re.fullmatch(
            "inputs/(geometry/[^/]+[.]obj|geometries/[^/]+[.]obj|polars-2D/[^/]+)", blade
        )
        is None
    ):
        raise ValueError("blade must name inputs/geometry/<name>.obj or inputs/polars-2D/<set>")
    if "n_blades" not in values:
        raise ValueError("n_blades is required")
    hub = values.get("hub_center", [0.0, 0.0, 0.0])
    if not isinstance(hub, list) or len(hub) != 3:
        raise ValueError("hub_center must be a list of three numbers")
    alpha = (
        number(values.get("alpha_start", -8.0), "alpha_start"),
        number(values.get("alpha_end", 16.0), "alpha_end"),
        number(values.get("alpha_step", 1.0), "alpha_step"),
    )
    if alpha[2] <= 0 or alpha[1] <= alpha[0]:
        raise ValueError("alpha_start < alpha_end and alpha_step > 0 are required")
    scale = number(values.get("scale", 1.0), "scale")
    radius = number(values["radius"], "radius") if "radius" in values else None
    if scale <= 0 or (radius is not None and radius <= 0):
        raise ValueError("scale and radius must be positive")
    ncrit = number(values.get("ncrit", 9), "ncrit")
    if ncrit <= 0:
        raise ValueError("ncrit must be positive")
    title = values.get("title", "cfdbender QPROP")
    if not isinstance(title, str) or not title.isascii() or "\n" in title:
        raise ValueError("title must be one line of ASCII text")
    return Setup(
        blade=blade,
        blades=_count(values["n_blades"], "n_blades"),
        n_sections=_count(values["n_sections"], "n_sections") if "n_sections" in values else None,
        spacing=_choice(values, "spacing", SPACINGS, "cosine"),
        cut=_choice(values, "cut", CUTS, "cylinder"),
        solver_2d=_choice(values, "solver_2d", SOLVERS_2D, "xfoil"),
        axis=_choice(values, "axis", tuple(AXES), "x"),
        hub_center=(
            number(hub[0], "hub_center"),
            number(hub[1], "hub_center"),
            number(hub[2], "hub_center"),
        ),
        rotation=_choice(values, "rotation", ROTATIONS, "ccw_from_front"),
        scale=scale,
        radius=radius,
        alpha=alpha,
        ncrit=ncrit,
        iterations=_count(values.get("iterations", 150), "iterations"),
        re_exponent=number(values.get("re_exponent", -0.5), "re_exponent"),
        title=title,
    )


def matrix_condition(condition: Mapping[str, float]) -> dict[str, float]:
    """Translate the pyflightstream keys of a point's condition into QPROP's.

    ADVANCE_RATIO is J. MACH or TASmps with the other air keys resolve to V,
    RHO, MU and A; a V, RHO, MU or A also stated must agree. ALPHA and BETA
    of zero are dropped, since the propeller runs in axial flow.
    """
    air = {
        canonical_key(key): value
        for key, value in condition.items()
        if canonical_key(key) in STATE_KEYS
    }
    held = {
        key.upper(): value
        for key, value in condition.items()
        if canonical_key(key) not in STATE_KEYS
    }
    for angle in ("ALPHA", "BETA"):
        if held.pop(angle, 0.0) != 0.0:
            raise ValueError(f"QPROP runs in axial flow; {angle} must be 0 or absent")
    if "ADVANCE_RATIO" in held:
        if "J" in held:
            raise ValueError("give ADVANCE_RATIO or J, not both")
        held["J"] = held.pop("ADVANCE_RATIO")
    if not air:
        return held
    _, state = row_flow(air, reference_length=None, label="FLIGHT_CONDITION", native_mach=False)
    if state is None:
        return held
    flow = {
        "V": state.velocity,
        "RHO": state.density,
        "MU": state.viscosity,
        "A": state.sonic_velocity,
    }
    merged = agree(held, flow, "FLIGHT_CONDITION")
    return {key: number(value, key) for key, value in merged.items()}


def operating(condition: Mapping[str, float]) -> Operating:
    """Validate one point's FLIGHT_CONDITION keys; keys match in either case."""
    held = matrix_condition(condition)
    if unknown := set(held) - set(CONDITIONS):
        raise ValueError(
            "unsupported QPROP condition: "
            + ", ".join(sorted(unknown))
            + "; accepted: "
            + ", ".join(CONDITIONS)
        )
    if "V" not in held:
        raise ValueError("V (m/s) is required")
    if held["V"] < 0:
        raise ValueError("V must not be negative")
    if "RPM" in held and "J" in held:
        raise ValueError("give only one of RPM and J")
    if "RPM" not in held and "J" not in held:
        raise ValueError("specify RPM or J")
    for key in ("RPM", "J", "RHO", "MU", "A"):
        if key in held and held[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if "J" in held and held["V"] <= 0:
        raise ValueError("J needs V greater than zero; use RPM for a static point")
    if "BETA07" in held and "DBETA" in held:
        raise ValueError("give BETA07 or DBETA, not both")
    sections = None
    if "N_SECTIONS" in held:
        sections = _count(held["N_SECTIONS"], "N_SECTIONS")
    return Operating(
        speed=held["V"],
        rpm=held.get("RPM"),
        j=held.get("J"),
        beta07=held.get("BETA07"),
        dbeta=held.get("DBETA", 0.0),
        rho=held.get("RHO", DEFAULT_RHO),
        mu=held.get("MU", DEFAULT_MU),
        a=held.get("A", DEFAULT_A),
        n_sections=sections,
    )

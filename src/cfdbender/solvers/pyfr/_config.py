"""Validate and render a two-dimensional PyFR case."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from cfdbender.core.flight import GAMMA, agree, mach_reynolds
from cfdbender.core.mesh import MESH_PREFIXES
from cfdbender.solvers._host._config import number, unknown_keys

_MESH_FOLDERS = MESH_PREFIXES


@dataclass(frozen=True)
class Case:
    mesh: str
    mach: float
    reynolds: float
    alpha: float
    order: int
    tend: float
    window_start: float
    tolerance: float
    backend: str


def config(
    setup: Mapping[str, object],
    condition: Mapping[str, float],
    reference: Mapping[str, object] | None = None,
    geometry: str | None = None,
) -> Case:
    """Build the case from the setup and the row's condition and GEOMETRY.

    A FLIGHT_CONDITION that states the air with more than MACH gives mach and
    reynolds (REmi directly, else over the REF length); GEOMETRY gives the mesh.
    A setup value that disagrees with the row is refused.
    """
    condition, flow = mach_reynolds(condition, reference, native_mach=True)
    if twice := sorted(key for key in condition if key.lower() in flow):
        raise ValueError(
            f"FLIGHT_CONDITION states {', '.join(twice)} and the air keys that resolve it; "
            "state one"
        )
    updates: dict[str, object] = dict(flow)
    if geometry is not None:
        updates["mesh"] = geometry
    setup = agree(setup, updates, "row")
    allowed = {
        "mesh",
        "mach",
        "reynolds",
        "alpha",
        "order",
        "tend",
        "window_start",
        "tolerance",
        "backend",
        "solver",
    }
    if unknown := set(setup) - allowed:
        raise unknown_keys("setup file", unknown, allowed)
    values = dict(setup)
    known = {"mach", "reynolds", "alpha", "order", "tend", "window_start", "tolerance"}
    for key, value in condition.items():
        key = key.lower()
        if key not in known:
            raise unknown_keys("FLIGHT_CONDITION", [key], known)
        values[key] = value
    mesh = values.get("mesh")
    if not isinstance(mesh, str) or not mesh.startswith(_MESH_FOLDERS) or not mesh.endswith(".msh"):
        raise ValueError("mesh must name inputs/mesh/<name>.msh")
    if values.get("mach") is None:
        raise ValueError("FLIGHT_CONDITION needs MACH (or mach in the setup file)")
    if values.get("reynolds") is None:
        raise ValueError("FLIGHT_CONDITION needs REmi (or reynolds in the setup file)")
    mach = number(values.get("mach"), "mach", "FLIGHT_CONDITION")
    reynolds = number(values.get("reynolds"), "reynolds", "FLIGHT_CONDITION")
    alpha = number(values.get("alpha", 0), "alpha", "FLIGHT_CONDITION")
    order = number(values.get("order"), "order", "setup file")
    tend = number(values.get("tend"), "tend", "setup file")
    start = number(values.get("window_start"), "window_start", "setup file")
    tolerance = number(values.get("tolerance", 0.01), "tolerance", "setup file")
    backend = values.get("backend", "cuda")
    if (
        mach <= 0
        or reynolds <= 0
        or order < 1
        or not order.is_integer()
        or tend <= 0
        or start < 0
        or start >= tend
        or tolerance <= 0
    ):
        raise ValueError("invalid PyFR flow, order, window or tolerance")
    if backend not in ("cuda", "openmp"):
        raise ValueError("backend must be cuda or openmp")
    return Case(mesh, mach, reynolds, alpha, int(order), tend, start, tolerance, backend)


def options(setup: Mapping[str, object], defaults: Mapping[str, object]) -> dict[str, object]:
    table = setup.get("solver", {})
    if not isinstance(table, dict):
        raise ValueError("solver must be a table")
    result = dict(defaults)
    result.update(table)
    if set(result) - {"launcher", "executable", "timeout_s"}:
        raise ValueError("unknown PyFR solver option")
    launcher = result.setdefault("launcher", [])
    if not isinstance(launcher, list) or any(
        not isinstance(x, str) or not x or any(ord(c) < 32 for c in x) for x in launcher
    ):
        raise ValueError("launcher must be a list of strings")
    if "executable" in result and (
        not isinstance(result["executable"], str) or not result["executable"]
    ):
        raise ValueError("executable must be a nonempty string")
    if number(result.setdefault("timeout_s", 600), "timeout_s") <= 0:
        raise ValueError("timeout_s must be positive")
    return result


def _n(x: float) -> str:
    return format(x, ".15g")


def render(case: Case) -> str:
    """Return the case.ini text; freestream from Mach, Reynolds and alpha."""
    angle = math.radians(case.alpha)
    return f"""[backend]
precision = double
rank-allocator = linear

[backend-cuda]
device-id = round-robin

[constants]
gamma = {GAMMA!r}
mu = {_n(1 / case.reynolds)}
Pr = 0.72
cpTref = 1
cpTs = 1
uinf = {_n(math.cos(angle))}
vinf = {_n(math.sin(angle))}
pinf = {_n(1 / (GAMMA * case.mach**2))}

[solver]
system = navier-stokes
order = {case.order}

[solver-time-integrator]
formulation = std
scheme = rk45
controller = pi
tstart = 0.0
tend = {_n(case.tend)}
dt = 1e-4
atol = 1e-6
rtol = 1e-6
safety-fact = 0.9
min-fact = 0.3
max-fact = 2.5

[solver-interfaces]
riemann-solver = rusanov
ldg-beta = 0.5
ldg-tau = 0.1

[solver-interfaces-line]
flux-pts = gauss-legendre

[solver-elements-quad]
soln-pts = gauss-legendre

[solver-elements-tri]
soln-pts = williams-shunn

[soln-plugin-nancheck]
nsteps = 100

[soln-plugin-writer]
dt-out = {_n(case.tend)}
basedir = .
basename = solution-{{t}}

[soln-plugin-fluidforce-wall]
nsteps = 200
file = forces.csv
header = true
quad-deg = {2 * case.order}

[soln-bcs-wall]
type = no-slp-adia-wall

[soln-bcs-far]
type = char-riem-inv
rho = 1.0
u = uinf
v = vinf
p = pinf

[soln-ics]
rho = 1.0
u = uinf
v = vinf
p = pinf
"""

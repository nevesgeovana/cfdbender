"""Resolve a matrix FLIGHT_CONDITION into one flow state, without any solver.

The cell grammar and the state resolution follow the pyflightstream matrix:
comma-separated KEY:value pairs whose keys carry their units in their names.
Five keys constrain the air (MACH, TASmps, REmi in millions, ALTFT in feet,
dISA in kelvin as an offset), five pin it (RHOkgm3, MUPas, ASMPS, TK, PPA), and
seven state the attitude (ALPHA, BETA, ADVANCE_RATIO, RPM, roll_rate,
pitch_rate, yaw_rate). Keys match case-insensitively.

The atmosphere is ISO 2533 from -2000 m to 20000 m with a Sutherland viscosity.
Exactly one of MACH or TASmps fixes the velocity. ALTFT and dISA fix the
temperature (sea level and zero when absent); a TK pin replaces it, and the
speed of sound and viscosity follow it unless ASMPS or MUPas pin them. REmi
solves the density from Re = rho V L / mu with L the reference length of the
row's REF file; otherwise the density is the ideal-gas density of the
altitude's pressure (or a PPA pin) at the state's temperature, and the Reynolds
number is an outcome. Whenever the density is solved or pinned, the pressure
is p = rho R T unless PPA pins it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

AIR_KEYS = ("MACH", "TASmps", "REmi", "ALTFT", "dISA")
PIN_KEYS = ("RHOkgm3", "MUPas", "ASMPS", "TK", "PPA")
ATTITUDE_KEYS = ("ALPHA", "BETA", "ADVANCE_RATIO", "RPM", "roll_rate", "pitch_rate", "yaw_rate")
STATE_KEYS = (*AIR_KEYS, *PIN_KEYS)
CANONICAL_KEYS = {key.upper(): key for key in (*STATE_KEYS, *ATTITUDE_KEYS)}

GAMMA = 1.4
GAS_CONSTANT = 287.05287
SEA_LEVEL_TEMPERATURE = 288.15
SEA_LEVEL_PRESSURE = 101325.0
GRAVITY = 9.80665
LAPSE_RATE = 0.0065
TROPOPAUSE_ALTITUDE = 11000.0
TROPOPAUSE_TEMPERATURE = 216.65
FLOOR_ALTITUDE = -2000.0
CEILING_ALTITUDE = 20000.0
METRES_PER_FOOT = 0.3048
SUTHERLAND_VISCOSITY = 1.716e-5
SUTHERLAND_TEMPERATURE = 273.15
SUTHERLAND_CONSTANT = 110.4


class FlightConditionError(ValueError):
    """A flight condition is under-determined, contradictory or outside the atmosphere."""


@dataclass(frozen=True)
class FlowState:
    """One resolved free stream in SI units, with where its density came from.

    ``reynolds`` is None only when no reference length was available and the
    row stated no REmi. ``density_source`` is ``solved-from-reynolds``,
    ``pinned`` or ``atmosphere``. ``stated`` holds the state keys as written.
    """

    mach: float
    velocity: float
    temperature: float
    pressure: float
    density: float
    viscosity: float
    sonic_velocity: float
    reynolds: float | None
    reynolds_length: float | None
    density_source: str
    stated: Mapping[str, float]


def canonical_key(name: str) -> str:
    """Return the pyflightstream spelling of a known key, else the key in upper case."""
    return CANONICAL_KEYS.get(name.strip().upper(), name.strip().upper())


def sutherland_viscosity(temperature: float) -> float:
    """Dynamic viscosity in Pa s at a temperature in K."""
    ratio = temperature / SUTHERLAND_TEMPERATURE
    return float(
        SUTHERLAND_VISCOSITY
        * math.pow(ratio, 1.5)
        * (SUTHERLAND_TEMPERATURE + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )


def speed_of_sound(temperature: float) -> float:
    """Speed of sound in m/s of the ideal gas at a temperature in K."""
    return math.sqrt(GAMMA * GAS_CONSTANT * temperature)


def standard_atmosphere(altitude_m: float) -> tuple[float, float]:
    """Return the ISO 2533 (temperature K, pressure Pa) at a geopotential altitude."""
    exponent = GRAVITY / (LAPSE_RATE * GAS_CONSTANT)
    if altitude_m <= TROPOPAUSE_ALTITUDE:
        temperature = SEA_LEVEL_TEMPERATURE - LAPSE_RATE * altitude_m
        return temperature, SEA_LEVEL_PRESSURE * (temperature / SEA_LEVEL_TEMPERATURE) ** exponent
    base = SEA_LEVEL_PRESSURE * (TROPOPAUSE_TEMPERATURE / SEA_LEVEL_TEMPERATURE) ** exponent
    height = altitude_m - TROPOPAUSE_ALTITUDE
    pressure = base * math.exp(-GRAVITY * height / (GAS_CONSTANT * TROPOPAUSE_TEMPERATURE))
    return TROPOPAUSE_TEMPERATURE, pressure


def states_flow(condition: Mapping[str, float]) -> bool:
    """Whether a point states more of the air than a bare MACH number.

    A row stating only MACH (with attitude keys) keeps the flow state of its
    setup, which is how matrices written before the full cell keep working.
    """
    return any(key in condition for key in STATE_KEYS if key != "MACH")


def row_flow(
    condition: Mapping[str, float],
    *,
    reference_length: float | None,
    label: str,
    native_mach: bool = True,
) -> tuple[dict[str, float], FlowState | None]:
    """Split a point's condition into its non-state keys and its resolved flow.

    Parameters
    ----------
    condition : mapping of str to float
        One point's condition with canonical keys.
    reference_length : float or None
        The REF file's length, required when REmi is stated.
    label : str
        Names the row in every refusal.
    native_mach : bool, optional
        True for a solver that reads a bare MACH itself: a row stating only
        MACH and angles is then returned unchanged with no flow state.

    Returns
    -------
    tuple
        The condition without the state keys, and the flow state or None when
        the row states no flow to resolve.
    """
    stated = any(key in condition for key in STATE_KEYS)
    if not (states_flow(condition) or (stated and not native_mach)):
        return dict(condition), None
    kept = {key: value for key, value in condition.items() if key not in STATE_KEYS}
    return kept, resolve_flow(condition, reference_length=reference_length, label=label)


def mach_reynolds(
    condition: Mapping[str, float],
    reference: Mapping[str, object] | None,
    *,
    native_mach: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the non-state condition keys and the resolved Mach and Reynolds numbers.

    For a two-dimensional solver whose case is made dimensionless by one
    reference length: REmi gives the Reynolds number directly, and other air
    keys give it over the REF file's ``length``. A BETA of zero is dropped and
    any other BETA refused. The second mapping is empty when the row states no
    flow to resolve. Since REmi is the Reynolds number itself, no density is
    solved from it, so ``MACH:0, REmi:3`` (an incompressible polar) is accepted.
    """
    if condition.get("BETA", 0.0) != 0.0:
        raise ValueError("a two-dimensional case flies at BETA 0; remove BETA from the row")
    condition = {key: value for key, value in condition.items() if key != "BETA"}
    if "REmi" in condition:
        label = "FLIGHT_CONDITION"
        if condition["REmi"] <= 0:
            raise FlightConditionError(f"{label}: REmi must be positive")
        if "RHOkgm3" in condition:
            raise FlightConditionError(
                f"{label}: FLIGHT_CONDITION states RHOkgm3 and REmi; each fixes the density, "
                "state one"
            )
        if sum(key in condition for key in ("MACH", "TASmps")) != 1:
            # The velocity refusal of the resolver, worded once.
            resolve_flow(condition, reference_length=1.0, label=label)
        rest = {key: value for key, value in condition.items() if key != "REmi"}
        kept, state = row_flow(rest, reference_length=None, label=label, native_mach=native_mach)
        flow = {"reynolds": condition["REmi"] * 1e6}
        if state is not None:
            flow["mach"] = state.mach
        elif "MACH" in kept:
            flow["mach"] = kept.pop("MACH")
        return kept, flow
    raw = (reference or {}).get("length")
    length = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    kept, state = row_flow(
        condition, reference_length=length, label="FLIGHT_CONDITION", native_mach=native_mach
    )
    if state is None:
        return kept, {}
    if state.reynolds is None:
        raise ValueError(
            "FLIGHT_CONDITION gives a Reynolds number only with REmi or with a length "
            "in the REF file"
        )
    return kept, {"mach": state.mach, "reynolds": state.reynolds}


def agree(
    table: Mapping[str, object], updates: Mapping[str, object], source: str
) -> dict[str, object]:
    """Add matrix-derived values to an input table, refusing a disagreeing value.

    Parameters
    ----------
    table : Mapping
        Setup (or condition) values as the files state them.
    updates : Mapping
        Values the matrix row determines.
    source : str
        The matrix column the updates come from, named in a refusal.

    Returns
    -------
    dict
        The table with the updates; a value equal to the stated one is accepted.
    """
    result = dict(table)
    for key, value in updates.items():
        if key in result:
            stated = result[key]
            same = (
                math.isclose(float(stated), float(value), rel_tol=1e-9)
                if isinstance(stated, (int, float))
                and isinstance(value, (int, float))
                and not isinstance(stated, bool)
                else stated == value
            )
            if not same:
                raise ValueError(
                    f"conflict: {key} is {stated!r} in the inputs and {value!r} from the "
                    f"matrix {source}; state it in one place"
                )
        result[key] = value
    return result


def resolve_flow(
    condition: Mapping[str, float], *, reference_length: float | None, label: str
) -> FlowState:
    """Resolve the state keys of one point's condition into a flow state.

    Parameters
    ----------
    condition : mapping of str to float
        One point's condition with canonical keys; attitude and other keys are ignored.
    reference_length : float or None
        The REF file's length in metres, required when REmi is stated.
    label : str
        Names the row in every refusal.

    Returns
    -------
    FlowState
        Velocity, temperature, pressure, density, viscosity and Reynolds number.

    Raises
    ------
    FlightConditionError
        No velocity or two, a non-positive pin or Reynolds number, REmi with
        RHOkgm3 or without a reference length, or ALTFT outside the atmosphere.
    """
    state = {key: float(condition[key]) for key in STATE_KEYS if key in condition}
    velocity_keys = [key for key in ("MACH", "TASmps") if key in state]
    if len(velocity_keys) != 1:
        raise FlightConditionError(
            f"{label}: FLIGHT_CONDITION states "
            f"{' and '.join(velocity_keys) if velocity_keys else 'no velocity'}; "
            "state exactly one of MACH or TASmps"
        )
    speed_key = velocity_keys[0]
    if state[speed_key] < 0:
        raise FlightConditionError(f"{label}: {speed_key} is a speed and cannot be negative")
    if "RHOkgm3" in state and "REmi" in state:
        raise FlightConditionError(
            f"{label}: FLIGHT_CONDITION states RHOkgm3 and REmi; each fixes the density, state one"
        )
    for key in (*PIN_KEYS, "REmi"):
        if key in state and state[key] <= 0:
            raise FlightConditionError(f"{label}: {key} must be positive")
    if reference_length is not None and not (
        math.isfinite(reference_length) and reference_length > 0
    ):
        raise FlightConditionError(f"{label}: the REF length must be a positive length")
    altitude_ft = state.get("ALTFT", 0.0)
    altitude_m = altitude_ft * METRES_PER_FOOT
    if not FLOOR_ALTITUDE <= altitude_m <= CEILING_ALTITUDE:
        floor_ft = math.ceil(FLOOR_ALTITUDE / METRES_PER_FOOT)
        ceiling_ft = math.floor(CEILING_ALTITUDE / METRES_PER_FOOT)
        raise FlightConditionError(
            f"{label}: ALTFT:{altitude_ft:g} is outside the modelled atmosphere, "
            f"{floor_ft} ft to {ceiling_ft} ft"
        )
    standard_temperature, standard_pressure = standard_atmosphere(altitude_m)
    temperature = state.get("TK", standard_temperature + state.get("dISA", 0.0))
    if temperature <= 0:
        raise FlightConditionError(f"{label}: dISA drives the temperature to absolute zero")
    sonic = state.get("ASMPS", speed_of_sound(temperature))
    viscosity = state.get("MUPas", sutherland_viscosity(temperature))
    velocity = state["MACH"] * sonic if speed_key == "MACH" else state["TASmps"]
    reynolds: float | None
    if "REmi" in state:
        if reference_length is None:
            raise FlightConditionError(
                f"{label}: REmi needs the reference length of the row's REF file "
                "(its length key); a Reynolds number is measured against it"
            )
        if velocity <= 0:
            raise FlightConditionError(
                f"{label}: REmi solves the density only at a positive speed; "
                "state the density with ALTFT, dISA or RHOkgm3 instead"
            )
        reynolds = state["REmi"] * 1e6
        density = reynolds * viscosity / (velocity * reference_length)
        source = "solved-from-reynolds"
        pressure = state.get("PPA", density * GAS_CONSTANT * temperature)
    else:
        if "RHOkgm3" in state:
            density = state["RHOkgm3"]
            source = "pinned"
            pressure = state.get("PPA", density * GAS_CONSTANT * temperature)
        else:
            pressure = state.get("PPA", standard_pressure)
            density = pressure / (GAS_CONSTANT * temperature)
            source = "atmosphere"
        reynolds = (
            None if reference_length is None else density * velocity * reference_length / viscosity
        )
    return FlowState(
        mach=velocity / sonic,
        velocity=velocity,
        temperature=temperature,
        pressure=pressure,
        density=density,
        viscosity=viscosity,
        sonic_velocity=sonic,
        reynolds=reynolds,
        reynolds_length=reference_length,
        density_source=source,
        stated=state,
    )

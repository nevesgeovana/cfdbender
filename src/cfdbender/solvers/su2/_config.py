"""Validate s/r/p inputs and render deterministic SU2 configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from cfdbender.core.flight import (
    ATTITUDE_KEYS,
    GAMMA,
    GAS_CONSTANT,
    STATE_KEYS,
    resolve_flow,
    speed_of_sound,
    states_flow,
    sutherland_viscosity,
)
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.rotor import (
    PeriodicState,
    RotorBlock,
    is_row_control,
    periodic_state,
    rotation_rate,
    rotor_block,
    shaft_axis,
)
from cfdbender.solvers._host._config import number as number
from cfdbender.solvers._host._config import unknown_keys
from cfdbender.solvers._host._groups import groups
from cfdbender.solvers.su2.routes import ROUTES

NEUTRAL_KEYS = {
    "mach": "MACH_NUMBER",
    "alpha": "AOA",
    "angle_of_attack": "AOA",
    "beta": "SIDESLIP_ANGLE",
    "sideslip": "SIDESLIP_ANGLE",
    "freestream_temperature": "FREESTREAM_TEMPERATURE",
    "freestream_pressure": "FREESTREAM_PRESSURE",
    "freestream_density": "FREESTREAM_DENSITY",
    "freestream_velocity": "FREESTREAM_VELOCITY",
    "reynolds": "REYNOLDS_NUMBER",
    "reynolds_length": "REYNOLDS_LENGTH",
    "area": "REF_AREA",
    "length": "REF_LENGTH",
    "iterations": "ITER",
    "cfl": "CFL_NUMBER",
    "convergence_field": "CONV_FIELD",
    "convergence_residual": "CONV_RESIDUAL_MINVAL",
    "output_frequency": "OUTPUT_WRT_FREQ",
    "mesh": "MESH_FILENAME",
}
_REVERSE = {value: key for key, value in reversed(NEUTRAL_KEYS.items())}


def reads_key(key: str) -> bool:
    """Return whether SU2 translates a neutral key or forwards a native key."""
    return (
        key.lower() in NEUTRAL_KEYS
        or key in {"moment_origin", *RAMP_KEYS}
        or re.fullmatch("[A-Z][A-Z0-9_]*", key) is not None
    )


_FLOW_WORDS = {
    "MACH_NUMBER": "MACH",
    "FREESTREAM_TEMPERATURE": "temperature",
    "FREESTREAM_PRESSURE": "pressure",
    "FREESTREAM_DENSITY": "density",
    "REYNOLDS_NUMBER": "Reynolds number",
    "REYNOLDS_LENGTH": "Reynolds length",
}


def _named(key: str) -> str:
    """Return a key as the user wrote it: its file and its neutral name, or the SU2 key."""
    if key in _FLOW_WORDS:
        return f"FLIGHT_CONDITION {_FLOW_WORDS[key]}"
    if key in _REVERSE:
        file = "REF file" if key in {"REF_AREA", "REF_LENGTH"} else "setup file"
        return f"{file} {_REVERSE[key]}"
    return key


_OPTIONS = {
    "route",
    "executable",
    "pysu2_path",
    "threads",
    "timeout_s",
    "ram_estimate_gib",
    "mpi_launcher",
}
_NUMERIC = {
    "MACH_NUMBER",
    "AOA",
    "SIDESLIP_ANGLE",
    "FREESTREAM_TEMPERATURE",
    "FREESTREAM_PRESSURE",
    "FREESTREAM_DENSITY",
    "REYNOLDS_NUMBER",
    "REYNOLDS_LENGTH",
    "REF_AREA",
    "REF_LENGTH",
    "REF_ORIGIN_MOMENT_X",
    "REF_ORIGIN_MOMENT_Y",
    "REF_ORIGIN_MOMENT_Z",
    "ITER",
    "OUTPUT_WRT_FREQ",
    "CONV_RESIDUAL_MINVAL",
}


def options(
    setup: Mapping[str, object], workspace: Mapping[str, object] | None = None
) -> dict[str, object]:
    value = setup.get("solver", {})
    if not isinstance(value, dict):
        raise ValueError("setup file [solver] must be a table")
    if unknown := set(value) - _OPTIONS:
        raise ValueError("setup file [solver] has unknown options: " + ", ".join(sorted(unknown)))
    if workspace and (unknown := set(workspace) - _OPTIONS):
        raise ValueError("init.toml [solver] has unknown options: " + ", ".join(sorted(unknown)))
    result = dict(workspace or {})
    result.update(value)
    if "route" in result and result["route"] not in ROUTES:
        raise ValueError("solver.route must be pysu2 or executable")
    launcher = result.setdefault("mpi_launcher", [])
    if not isinstance(launcher, list) or any(
        not isinstance(item, str) or not item or any(ord(c) < 32 for c in item) for item in launcher
    ):
        raise ValueError("solver.mpi_launcher must be a list of nonempty strings")
    for path_key in ("executable", "pysu2_path"):
        if path_key in result and (not isinstance(result[path_key], str) or not result[path_key]):
            raise ValueError(f"solver.{path_key} must be a nonempty path")
    threads = result.setdefault("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("solver.threads must be a positive integer")
    for key, default in (("timeout_s", 600.0), ("ram_estimate_gib", 0.25)):
        result.setdefault(key, default)
        if number(result[key], key) <= 0:
            raise ValueError(f"solver.{key} must be positive")
    return result


def render_value(value: object, name: str) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (int, float)):
        return format(number(value, name), ".15g")
    if isinstance(value, list):
        return "( " + ", ".join(render_value(item, name) for item in value) + " )"
    if (
        isinstance(value, str)
        and value
        and value.isascii()
        and not any(ord(c) < 32 or c in "%=" for c in value)
    ):
        return value
    raise ValueError(f"{name} needs an ASCII SU2 scalar or list without control characters, % or =")


def _needs(key: str) -> str:
    """Return the refusal of a required key the user left out."""
    file, _, name = _named(key).rpartition(" ")
    return f"{file} needs {name}"


def _reference_length(*tables: Mapping[str, object]) -> float | None:
    for table in tables:
        for key in ("length", "REF_LENGTH"):
            if key in table:
                return number(table[key], key)
    return None


def _flow(
    condition: Mapping[str, float], setup: Mapping[str, object], reference: Mapping[str, object]
) -> tuple[dict[str, float], dict[str, object]]:
    """Split the matrix condition into SU2 keys and the resolved free stream."""
    kept = {key: value for key, value in condition.items() if key not in STATE_KEYS}
    for key in kept:
        if key in ATTITUDE_KEYS and key not in {"ALPHA", "BETA", "ADVANCE_RATIO", "RPM"}:
            raise ValueError(f"the SU2 adapter does not read FLIGHT_CONDITION key {key}")
    if not states_flow(condition):
        if "MACH" in condition:
            kept["MACH"] = condition["MACH"]
        return kept, {}
    state = resolve_flow(
        condition,
        reference_length=_reference_length(reference, setup),
        label="FLIGHT_CONDITION",
    )
    derived: dict[str, object] = {
        "MACH_NUMBER": state.mach,
        "FREESTREAM_TEMPERATURE": state.temperature,
        "FREESTREAM_PRESSURE": state.pressure,
        "FREESTREAM_DENSITY": state.density,
    }
    if state.reynolds is not None:
        derived.update(REYNOLDS_NUMBER=state.reynolds, REYNOLDS_LENGTH=state.reynolds_length)
    if "MUPas" in state.stated:
        derived.update(VISCOSITY_MODEL="CONSTANT_VISCOSITY", MU_CONSTANT=state.viscosity)
    if "ASMPS" in state.stated and not math.isclose(
        state.sonic_velocity, speed_of_sound(state.temperature), rel_tol=1e-3
    ):
        raise ValueError(
            "ASMPS pins a speed of sound SU2 cannot hold: SU2 derives it from the "
            "temperature; pin TK instead"
        )
    return kept, derived


#: The compressible viscous solvers, which take the free-stream density from REYNOLDS_NUMBER.
VISCOUS = {"NAVIER_STOKES", "RANS", "FEM_NAVIER_STOKES", "FEM_RANS", "FEM_LES"}


def _upper(values: Mapping[str, object], key: str) -> str:
    return str(values.get(key, "")).strip().upper()


def _ignored_freestream(values: Mapping[str, object]) -> set[str]:
    """Return the free-stream keys this deck would state and SU2 would not read.

    Under SU2's default FREESTREAM_OPTION, TEMPERATURE_FS, the density follows from the
    pressure and the temperature, so FREESTREAM_DENSITY is unread. A compressible viscous
    run with REYNOLDS_NUMBER and SU2's default INIT_OPTION, REYNOLDS, also takes the
    density from the Reynolds number and recomputes the pressure, so FREESTREAM_PRESSURE
    is unread too. The flight condition still resolves both; the deck omits them.
    """
    ignored = set()
    if _upper(values, "FREESTREAM_OPTION") != "DENSITY_FS":
        ignored.add("FREESTREAM_DENSITY")
    if (
        _upper(values, "SOLVER") in VISCOUS
        and "REYNOLDS_NUMBER" in values
        and _upper(values, "INIT_OPTION") != "TD_CONDITIONS"
    ):
        ignored.add("FREESTREAM_PRESSURE")
    return ignored


def _same(stated: object, derived: object) -> bool:
    if isinstance(derived, float) and isinstance(stated, (int, float)):
        return not isinstance(stated, bool) and math.isclose(stated, derived, rel_tol=1e-9)
    return stated == derived


def config(
    setup: Mapping[str, object],
    reference: Mapping[str, object],
    post: Mapping[str, object],
    condition: Mapping[str, float],
    *,
    geometry: str | None = None,
    run_type: str = "steady",
    record: dict[str, object] | None = None,
    restarted: bool = False,
    precursor: bool = False,
    continue_from: int | None = None,
) -> dict[str, object]:
    """Return the SU2 keys of one point: the s/r/p files, the row and its run type's keys.

    ``record``, when given, receives what the run type resolved (its clock and windows).
    ``restarted`` states that the point starts from a saved solution (the row's RESTART).
    ``precursor`` states that it starts from a steady frozen-rotor stage of its own (the row's
    RESTART is ``MRF``): the stage's files are keys written ``<file>/<KEY>``. ``continue_from``
    continues a finished or killed time-accurate point in place at that SU2 time iteration
    (``run --continue``): its stage 1, if any, is not written again.
    """
    from cfdbender.solvers.su2._products import product_outputs
    from cfdbender.solvers.su2._run_types import SETUP_KEYS, apply_run_type, turbulence

    options(setup)
    if type(setup.get("mesh_link", True)) is not bool:
        raise ValueError("mesh_link must be a boolean")
    if setup.get("mesh_link", True) is False:
        raise ValueError(
            "mesh_link = false asks for a mesh copy; meshes are always linked, "
            "never copied: remove mesh_link"
        )
    if post.get("history") is not True:
        raise ValueError("pproc history must be true")
    condition, derived = _flow(condition, setup, reference)
    advance = condition.pop("ADVANCE_RATIO", None)
    rpm = condition.pop("RPM", None)
    if rpm is not None and run_type == "steady":
        raise ValueError("the SU2 adapter does not read FLIGHT_CONDITION key RPM")
    rotor = rotor_block(reference)
    if rotor is not None:
        block = cast(Mapping[str, object], reference[rotor.name])
        shaft, _ = shaft_axis(block["axis"], rotor.name)
        if shaft != "X":
            raise ValueError(
                f"REF file {rotor.name}.axis names shaft {shaft}: the SU2 adapter runs the "
                'rotor in an axial flow along +x, so it reads the shaft "X" only'
            )
    symmetry = str(setup.get("SYMMETRY", "NONE")).strip().upper()
    if symmetry not in {"NONE", "PERIODIC"}:
        raise ValueError(
            f"SYMMETRY {symmetry} is not read by the SU2 adapter: it reads NONE or PERIODIC "
            "(a mirror plane is a marker of the setup)"
        )
    periodic = periodic_state(setup)
    if geometry is not None:
        derived["MESH_FILENAME"] = geometry
    values: dict[str, object] = {}
    ramp: dict[str, float] = {}

    def add(key: str, value: object) -> None:
        if key in values:
            raise ValueError(f"conflict: {key} is supplied more than once")
        if key != "MESH_FILENAME":
            render_value(value, key)
        values[key] = value

    for table, file in (
        (setup, "setup file"),
        (reference, "REF file"),
        (post, "PPROC file"),
        (condition, "FLIGHT_CONDITION"),
    ):
        for key, value in table.items():
            if (
                key in {"solver", "mesh_link", "groups", *SETUP_KEYS}
                and table is setup
                or is_row_control(key)
                and table is setup
                or key == "history"
                and table is post
                or table is reference
                and isinstance(value, dict)
                and value.get("kind") == "rotor"
            ):
                continue
            if key in RAMP_KEYS and table is setup:
                ramp[key] = number(value, key)
                continue
            if key == "moment_origin":
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError("moment_origin needs three finite coordinates")
                for axis, coordinate in zip("XYZ", value, strict=True):
                    add(f"REF_ORIGIN_MOMENT_{axis}", number(coordinate, key))
                continue
            target = NEUTRAL_KEYS.get(key.lower())
            if target is None:
                if re.fullmatch("[A-Z][A-Z0-9_]*", key) is None:
                    if table is reference or table is post:
                        continue
                    raise unknown_keys(
                        file, [key], [*NEUTRAL_KEYS, "moment_origin", "groups", *RAMP_KEYS]
                    ) from None
                target = key
            add(target, value)
    stated = set(values)
    for key, value in derived.items():
        if key in values and not _same(values[key], value):
            source = "MESH" if key == "MESH_FILENAME" else "FLIGHT_CONDITION"
            raise ValueError(
                f"conflict: {key} is {values[key]!r} in the s/r/p files and "
                f"{value!r} from the matrix {source}; state it in one place"
            )
        values[key] = value
    for key in _ignored_freestream(values) - stated:
        values.pop(key, None)
    if "MACH_NUMBER" not in values:
        warn_or_refuse(
            "FLIGHT_CONDITION and the setup file state no MACH_NUMBER: SU2 runs with its own "
            "default free stream",
            UserWarning,
            stacklevel=2,
        )
    for key in ("REF_AREA", "REF_LENGTH", "ITER", "MESH_FILENAME"):
        if key not in values:
            if key == "MESH_FILENAME":
                raise ValueError("MESH names no mesh and setup file states none")
            raise ValueError(_needs(key))
    for key in values.keys() & _NUMERIC:
        value = number(values[key], key)
        if key in {"REF_AREA", "REF_LENGTH", "ITER", "OUTPUT_WRT_FREQ"} and value <= 0:
            raise ValueError(f"{_named(key)} must be positive")
        if key in {"ITER", "OUTPUT_WRT_FREQ"} and value != int(value):
            raise ValueError(f"{_named(key)} must be an integer")
        if key == "MACH_NUMBER" and value < 0:
            raise ValueError("FLIGHT_CONDITION MACH must be non-negative")
        if (
            key
            in {
                "FREESTREAM_TEMPERATURE",
                "FREESTREAM_PRESSURE",
                "FREESTREAM_DENSITY",
                "REYNOLDS_NUMBER",
                "REYNOLDS_LENGTH",
            }
            and value <= 0
        ):
            raise ValueError(f"{_named(key)} must be positive (check MACH, TASmps, ALTFT and REmi)")
    if "FREESTREAM_VELOCITY" in values:
        velocity = values["FREESTREAM_VELOCITY"]
        if not isinstance(velocity, list) or len(velocity) != 3:
            raise ValueError("freestream_velocity needs three finite coordinates")
        for coordinate in velocity:
            number(coordinate, "freestream_velocity")
    if not all(f"REF_ORIGIN_MOMENT_{axis}" in values for axis in "XYZ"):
        raise ValueError("moment_origin needs three finite coordinates")
    mesh = values["MESH_FILENAME"]
    if not isinstance(mesh, str) or not mesh.endswith((".su2", ".cgns")):
        raise ValueError("mesh must name a workspace-relative .su2 or .cgns file")
    # Classification needs a fixed CSV history contract. Conflicting output choices refuse.
    for key, value in {
        "MESH_FORMAT": "CGNS" if mesh.endswith(".cgns") else "SU2",
        "CONV_FILENAME": "history",
        "TABULAR_FORMAT": "CSV",
    }.items():
        if key in values and values[key] != value:
            raise ValueError(f"conflict: {key} must be {value}")
        values.setdefault(key, value)
    _history_output(values)
    _cfl_ramp(values, ramp)
    _fluid_model(values)
    product_outputs(values, post)
    turbulence(values, setup)
    if run_type == "steady":
        if advance is not None:
            _advance_ratio(values, number(advance, "ADVANCE_RATIO"), rotor)
    else:
        if advance is not None and (not math.isfinite(advance) or advance <= 0.0):
            raise ValueError("FLIGHT_CONDITION ADVANCE_RATIO must be positive")
        motion = {
            key: number(value, key)
            for key, value in (("ADVANCE_RATIO", advance), ("RPM", rpm))
            if value is not None
        }
        resolved = apply_run_type(
            values,
            run_type,
            setup=setup,
            rotor=rotor,
            periodic=periodic,
            motion=motion,
            restarted=restarted,
            precursor=precursor,
            continue_from=continue_from,
        )
        if record is not None:
            record.update(resolved)
    _periodic_warnings(values, rotor, periodic)
    _dimensionalization(values)
    _sa_variant(values)
    _heatflux_values(values)
    _wall_outputs(values)
    if requested := groups(setup):
        monitored = values.get("MARKER_MONITORING", [])
        monitoring = (
            list(monitored)
            if isinstance(monitored, list)
            else re.split(r"[\s,()]+", str(monitored).strip())
        )
        for markers in requested.values():
            for marker in markers:
                if marker not in monitoring:
                    monitoring.append(marker)
        values["MARKER_MONITORING"] = monitoring
        history_groups = list(cast(list[str], values["HISTORY_OUTPUT"]))
        if "AERO_COEFF_SURF" not in history_groups:
            history_groups.append("AERO_COEFF_SURF")
        values["HISTORY_OUTPUT"] = history_groups
    _surface_history(values)
    if precursor and continue_from is None:
        from cfdbender.solvers.su2._precursor import stage_one

        sliding = cast(Mapping[str, object], setup["sliding"])
        values.update(stage_one(values, setup, cast(int, sliding["rotating_zone"])))
    return values


def _surface_history(values: dict[str, object]) -> None:
    """Ask for the per-surface coefficients of every monitored marker, on every point.

    The loads of a part of the model are summed at post time from the per-marker columns
    (``CFx(wing)``), so the run writes them for every wall marker whatever the groups of a post
    file are: a new grouping needs only a new post. A multizone history names the group per
    zone, ``AERO_COEFF_SURF[<zone>]``.
    """
    groups = [str(group) for group in cast(list[object], values["HISTORY_OUTPUT"])]
    zones = len(cast(list[object], values["CONFIG_LIST"])) if values.get("MULTIZONE") else 0
    wanted = [f"AERO_COEFF_SURF[{i}]" for i in range(zones)] if zones else ["AERO_COEFF_SURF"]
    values["HISTORY_OUTPUT"] = [*groups, *(name for name in wanted if name not in groups)]


#: The SU2 markers that are walls: their names are plotted and monitored by default.
WALL_MARKERS = ("MARKER_EULER", "MARKER_HEATFLUX", "MARKER_ISOTHERMAL")


def _wall_outputs(values: dict[str, object]) -> None:
    """Default MARKER_PLOTTING and MARKER_MONITORING to the wall markers.

    SU2 integrates the forces only over MARKER_MONITORING and writes surface rows only
    for MARKER_PLOTTING; without them the coefficients are zero and surface.csv holds a
    header alone. A list of the user wins.
    """
    walls: list[object] = []
    for key in WALL_MARKERS:
        given = values.get(key)
        items = given if isinstance(given, list) else [] if given is None else [given]
        walls.extend(item for item in items if not _is_value(item) and item not in walls)
    if walls:
        values.setdefault("MARKER_PLOTTING", list(walls))
        values.setdefault("MARKER_MONITORING", list(walls))


def _is_value(item: object) -> bool:
    """Return whether a marker list item is a number rather than a marker name."""
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return True
    try:
        float(str(item))
    except ValueError:
        return False
    return True


def _heatflux_values(values: dict[str, object]) -> None:
    """Pair every MARKER_HEATFLUX name with 0.0 when the setup gives names alone.

    SU2 reads MARKER_HEATFLUX as (name, flux, ...). A list of names alone is parsed as
    adiabatic only by a parser detail; the adapter writes the adiabatic value it means.
    """
    given = values.get("MARKER_HEATFLUX")
    if given is None:
        return
    items = given if isinstance(given, list) else [given]
    if items and not any(_is_value(item) for item in items):
        values["MARKER_HEATFLUX"] = [part for item in items for part in (item, "0.0")]


#: The standard SA-neg of the NASA turbulence modeling resource, as SU2 8.x names it.
#: SU2's default for SA is SA-noft2, which limits each nu-tilde update to 99 percent of
#: the local value and cancels it below 1e-10; that stalled the TMR NACA 0012 runs.
#: SU2 refuses NEGATIVE without WITHFT2 unless EXPERIMENTAL is added.
SA_NEG = ["NEGATIVE", "WITHFT2"]


def _sa_variant(values: dict[str, object]) -> None:
    """Write SA_OPTIONS for standard SA-neg when SA runs without the user's own."""
    model = str(values.get("KIND_TURB_MODEL", "")).strip().upper()
    if model == "SA" and "SA_OPTIONS" not in values:
        values["SA_OPTIONS"] = list(SA_NEG)


#: The compressible finite-volume solvers, whose equations REF_DIMENSIONALIZATION scales.
COMPRESSIBLE = {"EULER", "NAVIER_STOKES", "RANS"}

#: The convergence level written with free-stream scaling when the setup states none.
SCALED_RESIDUAL_MINVAL = -12


def _dimensionalization(values: dict[str, object]) -> None:
    """Scale a compressible deck by the free stream unless the user states the scaling.

    SU2's default, DIMENSIONAL, leaves the energy rows of the implicit system about 1e5 times
    the density rows, and the linear solver then no longer solves the other equations: on the
    TMR NACA 0012 at CFL 1000 SU2's official deck converges in 688 iterations with
    FREESTREAM_PRESS_EQ_ONE and stalls at rms[Rho] -4.4 with DIMENSIONAL. The coefficients do
    not depend on it; the residual levels in history.csv do. With this scaling and no stated
    CONV_RESIDUAL_MINVAL the deck gets -12, the level SU2's own NACA 0012 case pairs with it
    (SU2's built-in default is -8). The classifier reads SU2's own verdict, so this key is the
    classifier's threshold too.
    """
    if _upper(values, "SOLVER") in COMPRESSIBLE:
        values.setdefault("REF_DIMENSIONALIZATION", "FREESTREAM_PRESS_EQ_ONE")
    if _upper(values, "REF_DIMENSIONALIZATION") == "FREESTREAM_PRESS_EQ_ONE":
        values.setdefault("CONV_RESIDUAL_MINVAL", SCALED_RESIDUAL_MINVAL)


#: SU2's own gas constants: STANDARD_AIR forces them, and IDEAL_GAS falls back to them.
SU2_AIR = {"GAS_CONSTANT": 287.058, "GAMMA_VALUE": 1.4}


def _fluid_model(values: dict[str, object]) -> None:
    """Make SU2 run on cfdbender's gas constants, or say why it will not.

    SU2's default FLUID_MODEL, STANDARD_AIR, replaces GAS_CONSTANT and GAMMA_VALUE with
    287.058 and 1.4, so the constants of the ISO 2533 atmosphere take effect only under
    IDEAL_GAS. Without FLUID_MODEL the adapter writes IDEAL_GAS and the core constants (a
    native constant of the user wins). A native FLUID_MODEL wins: the adapter then writes
    neither constant, and warns when the gas SU2 will use differs from the core one.
    """
    solver = str(values.get("SOLVER", "")).strip().upper()
    if solver.startswith("INC_"):
        # The incompressible fluid models have their own names; IDEAL_GAS is not one.
        values.setdefault("GAS_CONSTANT", GAS_CONSTANT)
        values.setdefault("GAMMA_VALUE", GAMMA)
        return
    if "FLUID_MODEL" not in values:
        values["FLUID_MODEL"] = "IDEAL_GAS"
        values.setdefault("GAS_CONSTANT", GAS_CONSTANT)
        values.setdefault("GAMMA_VALUE", GAMMA)
        return
    model = str(values["FLUID_MODEL"]).strip().upper()
    core = {"GAS_CONSTANT": GAS_CONSTANT, "GAMMA_VALUE": GAMMA}
    if model == "STANDARD_AIR":
        used = dict(SU2_AIR)
    elif model == "IDEAL_GAS":
        used = {key: number(values.get(key, SU2_AIR[key]), key) for key in core}
    else:
        used = {}
    if not used or not all(math.isclose(used[key], core[key], rel_tol=1e-9) for key in core):
        gas = (
            ", ".join(f"{key} {format(value, '.15g')}" for key, value in used.items())
            if used
            else "a gas that is not an ideal gas"
        )
        warn_or_refuse(
            f"FLUID_MODEL {model} of the setup makes SU2 use {gas}, not the core "
            f"GAS_CONSTANT {GAS_CONSTANT} and GAMMA_VALUE {GAMMA}: the free stream SU2 "
            "derives differs from the one the other solvers receive",
            UserWarning,
            stacklevel=3,
        )


def _periodic_angle(values: Mapping[str, object]) -> float | None:
    """Return the rotation angle in degrees of the setup's MARKER_PERIODIC, or None."""
    given = values.get("MARKER_PERIODIC")
    if given is None:
        return None
    parts = (
        [str(item) for item in given]
        if isinstance(given, list)
        else [part for part in re.split(r"[\s,()]+", str(given).strip()) if part]
    )
    try:
        angles = [float(part) for part in parts[5:8]]
    except ValueError:
        return None
    return max(angles, key=abs) if angles else None


def _periodic_warnings(
    values: Mapping[str, object], rotor: RotorBlock | None, periodic: PeriodicState
) -> None:
    """Warn where the row's symmetry and the setup's periodic boundary disagree."""
    if rotor is None and periodic.symmetry != "PERIODIC":
        return
    angle = _periodic_angle(values)
    if periodic.symmetry == "PERIODIC":
        if "MARKER_PERIODIC" not in values:
            warn_or_refuse(
                "SYMMETRY is PERIODIC but the setup states no MARKER_PERIODIC: the mesh needs "
                "a periodic boundary for a sector",
                UserWarning,
                stacklevel=3,
            )
        elif angle and not math.isclose(360.0 / abs(angle), periodic.copies, rel_tol=1e-6):
            warn_or_refuse(
                f"MARKER_PERIODIC turns by {angle:g} degrees, which makes "
                f"{360.0 / abs(angle):g} copies, and PERIODIC_COPIES is {periodic.copies}",
                UserWarning,
                stacklevel=3,
            )
        elif angle and not periodic.loads:
            warn_or_refuse(
                "SYMMETRY_LOADS is false: the loads are reported for the sector only",
                UserWarning,
                stacklevel=3,
            )
    elif "MARKER_PERIODIC" in values:
        warn_or_refuse(
            "the setup states MARKER_PERIODIC but SYMMETRY is not PERIODIC: the loads are "
            "reported for the sector only",
            UserWarning,
            stacklevel=3,
        )


#: SU2's own default free-stream temperature (K) and pressure (Pa).
SU2_DEFAULT_TEMPERATURE = 288.15
SU2_DEFAULT_PRESSURE = 101325.0


@dataclass(frozen=True)
class FreeStream:
    """The free stream of a deck as SU2 reads it.

    Attributes
    ----------
    mach, sonic_velocity, velocity : float
        MACH_NUMBER, the speed of sound of the ideal gas at the free-stream temperature and
        their product V, in m/s.
    density : float or None
        The free-stream density in kg/m3; None when the deck does not determine it.
    dynamic_pressure : float or None
        ``0.5 rho V_q^2`` in Pa with ``V_q`` the speed SU2 scales its force coefficients by:
        MACH_MOTION times the speed of sound when the deck states MACH_MOTION, else V.
    """

    mach: float
    sonic_velocity: float
    velocity: float
    density: float | None
    dynamic_pressure: float | None


def free_stream(values: Mapping[str, object]) -> FreeStream | None:
    """Return the free stream of a deck, or None without a Mach number or an ideal gas.

    The speed of sound is ``sqrt(GAMMA R T)`` with the deck's ideal-gas constants
    (STANDARD_AIR: SU2's own) and ``T`` FREESTREAM_TEMPERATURE (else SU2's 288.15 K). The
    density is FREESTREAM_DENSITY under ``FREESTREAM_OPTION = DENSITY_FS``; for a viscous
    deck that SU2 initialises from REYNOLDS_NUMBER it is ``Re mu / (V L)`` with the
    Sutherland (or constant) viscosity and REYNOLDS_LENGTH; otherwise ``p / (R T)``.
    """
    if "MACH_NUMBER" not in values:
        return None
    model = _upper(values, "FLUID_MODEL")
    if model == "STANDARD_AIR":
        gas, gamma = SU2_AIR["GAS_CONSTANT"], SU2_AIR["GAMMA_VALUE"]
    elif model in {"", "IDEAL_GAS"}:
        gas = number(values.get("GAS_CONSTANT", SU2_AIR["GAS_CONSTANT"]), "GAS_CONSTANT")
        gamma = number(values.get("GAMMA_VALUE", SU2_AIR["GAMMA_VALUE"]), "GAMMA_VALUE")
    else:
        return None
    temperature = number(
        values.get("FREESTREAM_TEMPERATURE", SU2_DEFAULT_TEMPERATURE), "FREESTREAM_TEMPERATURE"
    )
    mach = number(values["MACH_NUMBER"], "MACH_NUMBER")
    sonic = math.sqrt(gamma * gas * temperature)
    velocity = mach * sonic
    density: float | None
    if _upper(values, "FREESTREAM_OPTION") == "DENSITY_FS" and "FREESTREAM_DENSITY" in values:
        density = number(values["FREESTREAM_DENSITY"], "FREESTREAM_DENSITY")
    elif (
        _upper(values, "SOLVER") in VISCOUS
        and "REYNOLDS_NUMBER" in values
        and _upper(values, "INIT_OPTION") != "TD_CONDITIONS"
    ):
        viscosity = (
            number(values.get("MU_CONSTANT", 0.0), "MU_CONSTANT")
            if _upper(values, "VISCOSITY_MODEL") == "CONSTANT_VISCOSITY"
            else sutherland_viscosity(temperature)
        )
        length = number(values.get("REYNOLDS_LENGTH", 1.0), "REYNOLDS_LENGTH")
        density = (
            number(values["REYNOLDS_NUMBER"], "REYNOLDS_NUMBER") * viscosity / (velocity * length)
            if velocity > 0.0 and length > 0.0
            else None
        )
    else:
        pressure = number(
            values.get("FREESTREAM_PRESSURE", SU2_DEFAULT_PRESSURE), "FREESTREAM_PRESSURE"
        )
        density = pressure / (gas * temperature)
    if density is not None and density <= 0.0:
        density = None
    motion = number(values.get("MACH_MOTION", mach), "MACH_MOTION") * sonic
    pressure_q = None if density is None else 0.5 * density * motion * motion
    return FreeStream(mach, sonic, velocity, density, pressure_q)


def gas_state(values: Mapping[str, object]) -> tuple[float, float] | None:
    """Return the deck's gas constant and free-stream temperature, None for a non-ideal gas.

    The same constants :func:`free_stream` reads: STANDARD_AIR is SU2's own air.
    """
    model = _upper(values, "FLUID_MODEL")
    if model == "STANDARD_AIR":
        gas = SU2_AIR["GAS_CONSTANT"]
    elif model in {"", "IDEAL_GAS"}:
        gas = number(values.get("GAS_CONSTANT", SU2_AIR["GAS_CONSTANT"]), "GAS_CONSTANT")
    else:
        return None
    temperature = number(
        values.get("FREESTREAM_TEMPERATURE", SU2_DEFAULT_TEMPERATURE), "FREESTREAM_TEMPERATURE"
    )
    return gas, temperature


def _vector_text(vector: Sequence[float]) -> str:
    """Write a vector as SU2 reads the rotation keys: three numbers beside each other."""
    return " ".join(format(component, ".15g") for component in vector)


def _stated_vector(value: object, name: str) -> tuple[float, float, float]:
    """Parse a stated three-component key, a list or the text SU2 reads."""
    parts = (
        [str(item) for item in value]
        if isinstance(value, list)
        else [part for part in re.split(r"[\s,()]+", str(value).strip()) if part]
    )
    if len(parts) != 3:
        raise ValueError(f"{name} needs three finite coordinates")
    x, y, z = (number(float(part), name) for part in parts)
    return (x, y, z)


def _advance_ratio(values: dict[str, object], advance: float, rotor: RotorBlock | None) -> None:
    """Write the steady rotating frame of an advance ratio: the rotation, its origin and Mach.

    ``J = V / (n D)`` with ``V`` the free-stream speed of the deck (MACH_NUMBER times the
    speed of sound), ``n`` the rotation speed in rev/s and ``D`` the rotor block's
    diameter_m. ROTATION_RATE is ``rpm_sign 2 pi n`` along the rotor's shaft, MOTION_ORIGIN
    the rotor's origin, and MACH_MOTION the free-stream Mach number unless the setup states
    it, in which case the setup wins.

    Raises
    ------
    ValueError
        The advance ratio is not positive; the REF file has no rotor block; the setup does
        not declare GRID_MOVEMENT = ROTATING_FRAME; ROTATION_RATE is stated as well; the deck
        has no free-stream speed above zero; or MOTION_ORIGIN disagrees with the rotor.
    """
    if not math.isfinite(advance) or advance <= 0.0:
        raise ValueError("FLIGHT_CONDITION ADVANCE_RATIO must be positive")
    if rotor is None:
        raise ValueError(
            "FLIGHT_CONDITION ADVANCE_RATIO needs the rotor declaration of the REF file: add "
            'a table with kind = "rotor", axis, rpm_sign and diameter_m (for example [ROTOR])'
        )
    frame = _upper(values, "GRID_MOVEMENT")
    if frame != "ROTATING_FRAME":
        stated = f"states GRID_MOVEMENT = {frame}" if frame else "states no GRID_MOVEMENT"
        raise ValueError(
            "FLIGHT_CONDITION ADVANCE_RATIO needs a steady rotating frame: the setup file "
            f'{stated}; state GRID_MOVEMENT = "ROTATING_FRAME"'
        )
    if "ROTATION_RATE" in values:
        raise ValueError(
            "FLIGHT_CONDITION ADVANCE_RATIO and ROTATION_RATE both state the rotation: the "
            "advance ratio sets ROTATION_RATE, so remove it from the setup file and the row"
        )
    stream = free_stream(values)
    if stream is None or stream.velocity <= 0.0:
        raise ValueError(
            "FLIGHT_CONDITION ADVANCE_RATIO needs a free-stream speed above zero: state MACH "
            "above zero (an ideal gas, with the FREESTREAM_TEMPERATURE of the deck)"
        )
    rate, _ = rotation_rate(rotor, advance, stream.velocity)
    if "MOTION_ORIGIN" in values:
        given = _stated_vector(values["MOTION_ORIGIN"], "MOTION_ORIGIN")
        if not all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            for a, b in zip(given, rotor.origin, strict=True)
        ):
            raise ValueError(
                f"conflict: MOTION_ORIGIN is {given!r} in the s/r/p files and the rotor "
                f"{rotor.name} has origin {rotor.origin!r}; state it in one place"
            )
    else:
        values["MOTION_ORIGIN"] = _vector_text(rotor.origin)
    values["ROTATION_RATE"] = _vector_text(rate)
    values.setdefault("MACH_MOTION", stream.mach)
    for key, label in (("AOA", "ALPHA"), ("SIDESLIP_ANGLE", "BETA")):
        if key in values and number(values[key], key) != 0.0:
            warn_or_refuse(
                f"FLIGHT_CONDITION {label} is not zero with a steady rotating frame: the "
                "frame assumes axial flow along the shaft, and the run continues",
                UserWarning,
                stacklevel=3,
            )


#: The CFL ramp written when the setup states no CFL key.
#: SU2 8.x ramps the CFL number itself: CFL_NUMBER is the start value and CFL_ADAPT_PARAM
#: is (factor down, factor up, CFL min, CFL max). Start 5 and max 25: a conservative ramp.
#: The CFL 50 stalls measured on the TMR NACA 0012 ran dimensional; with the free-stream
#: scaling written by default SU2's own deck takes CFL 1000, but no larger default is proven
#: both convergent and accurate on that case with cfdbender's numerics. The up factor 1.2
#: raises the CFL 20 percent per iteration while the residual falls; the down factor 0.5
#: halves it when the residual grows, so a bad start recovers. Min 5 keeps the ramp from
#: falling below the start.
CFL_KEYS = ("CFL_NUMBER", "CFL_ADAPT", "CFL_ADAPT_PARAM")
CFL_RAMP: dict[str, object] = {
    "CFL_NUMBER": 5,
    "CFL_ADAPT": True,
    "CFL_ADAPT_PARAM": [0.5, 1.2, 5, 25],
}


#: Neutral keys that set one slot of the CFL ramp: its maximum and its growth factor.
RAMP_KEYS = ("cfl_max", "cfl_growth")


def _cfl_ramp(values: dict[str, object], ramp: Mapping[str, float]) -> None:
    """Write the CFL ramp.

    Without any CFL key the default ramp is written whole. A native CFL key without a
    ramp key switches the default off (no merging). cfl_max and cfl_growth write the
    default ramp with those slots replaced: the start is cfl (else 5) and is also the
    ramp's minimum.
    """
    if not ramp:
        if not any(key in values for key in CFL_KEYS):
            values.update(CFL_RAMP)
        return
    if "CFL_ADAPT_PARAM" in values:
        raise ValueError(
            "conflict: cfl_max and cfl_growth set CFL_ADAPT_PARAM; state either, not both"
        )
    if "CFL_ADAPT" in values and values["CFL_ADAPT"] is not True:
        raise ValueError("conflict: cfl_max and cfl_growth need CFL_ADAPT; remove CFL_ADAPT")
    start = number(values.get("CFL_NUMBER", CFL_RAMP["CFL_NUMBER"]), "cfl")
    down, growth, _, top = cast(list[float], CFL_RAMP["CFL_ADAPT_PARAM"])
    growth = ramp.get("cfl_growth", growth)
    top = ramp.get("cfl_max", top)
    if growth <= 1:
        raise ValueError("cfl_growth must be above 1")
    if start <= 0 or top < start:
        raise ValueError("cfl_max must be at least the start CFL (cfl, else 5)")
    values.update(CFL_NUMBER=start, CFL_ADAPT=True, CFL_ADAPT_PARAM=[down, growth, start, top])


#: The history groups the reader needs: the iteration counter, the residuals and the
#: aerodynamic coefficients. SU2's own default history holds none of the last two.
HISTORY_GROUPS = ("ITER", "RMS_RES", "AERO_COEFF")


def _history_output(values: dict[str, object]) -> None:
    """Make history.csv hold the columns the adapter reads.

    Without HISTORY_OUTPUT the adapter writes the default groups; a HISTORY_OUTPUT that
    lacks one is extended with it, with a warning, never refused.
    """
    given = values.get("HISTORY_OUTPUT")
    if given is None:
        values["HISTORY_OUTPUT"] = list(HISTORY_GROUPS)
        return
    groups = [str(item) for item in given] if isinstance(given, list) else [str(given)]
    present = {item.strip().upper() for item in groups}
    missing = [group for group in HISTORY_GROUPS if group not in present]
    if missing:
        warn_or_refuse(
            f"HISTORY_OUTPUT lacks {', '.join(missing)}, which the SU2 adapter reads from "
            "history.csv: they are added",
            UserWarning,
            stacklevel=3,
        )
        values["HISTORY_OUTPUT"] = [*groups, *missing]


def mesh_name(values: Mapping[str, object]) -> str:
    return cast(str, values["MESH_FILENAME"])


def continued_at(point: object) -> int | None:
    """Return the SU2 time iteration a continued point restarts at, or None for a fresh run."""
    planned = getattr(point, "point", point)
    value = getattr(planned, "continuation", {}).get("restart_iter")
    return value if type(value) is int else None


def staged_mesh(values: Mapping[str, object]) -> str:
    """Return the stable point-local name for the selected mesh format."""
    return "mesh.cgns" if mesh_name(values).endswith(".cgns") else "mesh.su2"


def render(values: Mapping[str, object]) -> str:
    return "".join(
        f"{key}= {render_value(staged_mesh(values) if key == 'MESH_FILENAME' else value, key)}"
        + chr(10)
        for key, value in sorted(values.items())
        if "/" not in key
    )


def extra_files(values: Mapping[str, object]) -> dict[str, str]:
    """Return the configuration files beside su2.cfg a point needs, by name.

    A key written ``<file>/<KEY>`` belongs to that file: a zone file of a sliding-mesh run
    states its own keys only; the second segment of an export window is ``su2.cfg`` with
    its keys replaced. A point with two segments also gets the job script that runs them, and
    a point with a frozen-rotor precursor stage its files and the script that runs the stages.
    """
    from cfdbender.solvers.su2._precursor import PRECURSOR_SCRIPT, script_text, stages
    from cfdbender.solvers.su2._run_types import SEGMENT_TWO, SEGMENTS_SCRIPT

    files: dict[str, dict[str, object]] = {}
    for key, value in values.items():
        if "/" in key:
            name, _, native = key.partition("/")
            files.setdefault(name, {})[native] = value
    texts: dict[str, str] = {}
    for name, own in sorted(files.items()):
        if name == SEGMENT_TWO:
            texts[name] = render({**values, **own})
        else:
            texts[name] = "".join(
                f"{key}= {render_value(value, key)}" + chr(10) for key, value in sorted(own.items())
            )
    if spec := stages(values):
        texts[PRECURSOR_SCRIPT] = script_text(spec)
    elif SEGMENT_TWO in texts:
        texts[SEGMENTS_SCRIPT] = SEGMENTS_TEXT
    return texts


#: The job script of a two-segment point: the solver command is its arguments.
SEGMENTS_TEXT = (
    "#!/bin/sh"
    + chr(10)
    + "# Run the two segments of an SU2 point with an export window, in order."
    + chr(10)
    + "# Usage: sh run_segments.sh [launcher and solver command, default SU2_CFD]"
    + chr(10)
    + "set -e"
    + chr(10)
    + 'if [ "$#" -eq 0 ]; then set -- SU2_CFD; fi'
    + chr(10)
    + '"$@" su2.cfg'
    + chr(10)
    + '"$@" su2_segment2.cfg'
    + chr(10)
)

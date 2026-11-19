"""The SU2 configuration each run type carries, so a setup file states only what is its own.

The agnostic layer owns what a run type reads from the row
(:mod:`cfdbender.core.run_types`); this module translates a run type into SU2 keys. A key the
setup states natively wins over a carried one (every carried key is a default), except where
the run type derives it from the row: a second, different statement of the rotation is
refused, as before.

* ``steady``: nothing beyond the adapter's own rules, so an existing workspace plans the same
  bytes.
* ``qsteady_rotor``: ``GRID_MOVEMENT= ROTATING_FRAME``, ``ROTATION_RATE`` from
  ``ADVANCE_RATIO`` or ``RPM``, ``MOTION_ORIGIN`` (the hub), ``MACH_MOTION``, and
  ``MARKER_PERIODIC`` from ``SYMMETRY PERIODIC`` when the setup names its periodic pair with
  ``periodic_markers``.
* ``unsteady``: ``TIME_DOMAIN``, ``TIME_MARCHING``, ``TIME_STEP`` and ``TIME_ITER`` from the
  clock (``DELTA_TIME`` and ``TIME_ITERATIONS``, or ``DELTA_THETA`` and ``REVOLUTIONS`` against
  a rotor speed), ``INNER_ITER`` from the setup's iterations and ``WINDOW_START_ITER`` from
  the averaging window (SU2's own time average, a cross-check).
* ``unsteady_rotor``: the time keys, and the rotation as a rigid motion of the whole mesh
  (one zone) or, when the setup states a ``[sliding]`` table, a sliding-mesh multizone layout:
  ``su2.cfg`` lists one ``zone_<i>.cfg`` per zone, the turning zone in rigid motion.

SU2's time iteration ``k`` (0-based) is cfdbender's step ``k + 1``.

The export window (:mod:`cfdbender.core.export_window`) is realised in two segments of one
point (:func:`export_segments`): the first runs to the step before the window writing restart
files only, the second restarts there and writes every step to the end. That an SU2
time-accurate restart continues cleanly is to be verified on a real run.

Names of this module that pyflightstream has no equivalent for are PROVISIONAL:
``periodic_markers`` and the ``[sliding]`` table (``zones``, ``rotating_zone``,
``interface``) of the setup, and the tip Mach number written as ``MACH_MOTION`` of a hover.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping

from cfdbender.core.export_window import ExportWindow, export_window
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.rotor import PeriodicState, RotorBlock, rotation_vector, rotor_speed
from cfdbender.core.run_types import (
    QSTEADY_ROTOR,
    STEADY,
    TIME_ACCURATE,
    TURBULENCE_MODEL,
    UNSTEADY,
    UNSTEADY_ROTOR,
)
from cfdbender.core.unsteady_clock import TimeStepping, rotor_time_stepping, unsteady_time_stepping
from cfdbender.core.unsteady_window import LAST_ITERS_AVG, LAST_REVS_AVG, averaging_span, stated_key

#: The setup keys of this module that are not SU2's own (PROVISIONAL names).
SETUP_KEYS = ("periodic_markers", "sliding")
#: The neutral turbulence model names SU2 offers through cfdbender, and their keys.
TURBULENCE_MODELS: Mapping[str, Mapping[str, object]] = {
    "SA": {"KIND_TURB_MODEL": "SA"},
    "SST": {"KIND_TURB_MODEL": "SST"},
}
#: The second segment of a point with an export window.
SEGMENT_TWO = "su2_segment2.cfg"
#: The job script that runs the segments in order.
SEGMENTS_SCRIPT = "run_segments.sh"
#: The time marching written when the setup states none.
TIME_MARCHING = "DUAL_TIME_STEPPING-2ND_ORDER"
#: SU2's own output files when OUTPUT_FILES is not stated.
SU2_OUTPUT_FILES = ["RESTART", "PARAVIEW", "SURFACE_PARAVIEW"]


def turbulence(values: MutableMapping[str, object], setup: Mapping[str, object]) -> str | None:
    """Translate the row's turbulence model into SU2 keys; return the neutral name or None.

    The setup's own ``KIND_TURB_MODEL`` wins over the row's with one warning when they
    differ. A name SU2 does not offer is refused, listing those it does. No transition model is
    ever written: every model is fully turbulent.
    """
    stated = setup.get(TURBULENCE_MODEL)
    if stated is None:
        return None
    name = str(stated).strip().upper()
    if name not in TURBULENCE_MODELS:
        raise ValueError(
            f"{TURBULENCE_MODEL} {stated!r} is not a model SU2 offers through cfdbender; "
            f"known: {', '.join(TURBULENCE_MODELS)}"
        )
    keys = TURBULENCE_MODELS[name]
    own = str(values.get("KIND_TURB_MODEL", "")).strip().upper()
    if own and own != keys["KIND_TURB_MODEL"]:
        warn_or_refuse(
            f"{TURBULENCE_MODEL} {name} of the row differs from KIND_TURB_MODEL {own} of the "
            "setup file: the setup file wins",
            UserWarning,
            stacklevel=3,
        )
        return own
    values.update(keys)
    return name


def _vector_text(vector: tuple[float, float, float]) -> str:
    return " ".join(format(component, ".15g") for component in vector)


def _rotor_or_refuse(rotor: RotorBlock | None, run_type: str) -> RotorBlock:
    if rotor is None:
        raise ValueError(
            f"WORKFLOW {run_type} needs the rotor declaration of the REF file: add a table "
            'with kind = "rotor", axis, rpm_sign, diameter_m, families_blades and blade1'
        )
    return rotor


def _speed(values: Mapping[str, object], rotor: RotorBlock, motion: Mapping[str, float]) -> float:
    from cfdbender.solvers.su2._config import free_stream

    stream = free_stream(values)
    velocity = 0.0 if stream is None else stream.velocity
    if "ADVANCE_RATIO" in motion and velocity <= 0.0:
        raise ValueError(
            "FLIGHT_CONDITION ADVANCE_RATIO needs a free-stream speed above zero: state MACH "
            "above zero, or state the speed as RPM"
        )
    rps, _ = rotor_speed(rotor, motion, velocity)
    return rps


def write_rotation(
    values: MutableMapping[str, object],
    rotor: RotorBlock,
    rps: float,
    *,
    target: str = "",
) -> None:
    """Write ROTATION_RATE, MOTION_ORIGIN and MACH_MOTION of a rotor turning at ``rps``.

    ``target`` prefixes the keys of a zone file (``zone_1.cfg/``). A stated ROTATION_RATE
    is refused (the row sets it); a stated MOTION_ORIGIN must be the hub; a stated
    MACH_MOTION wins. A hover (no free stream) takes the tip Mach number (PROVISIONAL).
    """
    from cfdbender.solvers.su2._config import _stated_vector, free_stream

    if f"{target}ROTATION_RATE" in values or "ROTATION_RATE" in values:
        raise ValueError(
            "FLIGHT_CONDITION ADVANCE_RATIO and ROTATION_RATE both state the rotation: the "
            "advance ratio sets ROTATION_RATE, so remove it from the setup file and the row"
        )
    origin_key = f"{target}MOTION_ORIGIN"
    stated = values.get("MOTION_ORIGIN")
    if stated is not None:
        given = _stated_vector(stated, "MOTION_ORIGIN")
        if not all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            for a, b in zip(given, rotor.origin, strict=True)
        ):
            raise ValueError(
                f"conflict: MOTION_ORIGIN is {given!r} in the s/r/p files and the rotor "
                f"{rotor.name} has origin {rotor.origin!r}; state it in one place"
            )
        if target:
            values.pop("MOTION_ORIGIN")
    if target or stated is None:
        values[origin_key] = _vector_text(rotor.origin)
    values[f"{target}ROTATION_RATE"] = _vector_text(rotation_vector(rotor, rps))
    stream = free_stream(values)
    mach = 0.0 if stream is None else stream.mach
    if mach <= 0.0 and stream is not None and stream.sonic_velocity > 0.0:
        mach = math.pi * rps * rotor.diameter_m / stream.sonic_velocity
    motion = values.pop("MACH_MOTION", mach) if target else values.get("MACH_MOTION", mach)
    values[f"{target}MACH_MOTION"] = motion


def _periodic_marker(
    values: MutableMapping[str, object],
    setup: Mapping[str, object],
    rotor: RotorBlock,
    periodic: PeriodicState,
) -> None:
    pair = setup.get("periodic_markers")
    if pair is None or periodic.symmetry != "PERIODIC" or "MARKER_PERIODIC" in values:
        return
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(m, str) for m in pair):
        raise ValueError("periodic_markers must name two markers: [periodic, donor]")
    angle = 360.0 / periodic.copies
    values["MARKER_PERIODIC"] = [
        *pair,
        *rotor.origin,
        *(angle * c + 0.0 for c in rotor.axis),
        0.0,
        0.0,
        0.0,
    ]


def _clock(
    run_type: str,
    setup: Mapping[str, object],
    rotor: RotorBlock | None,
    rps: float | None,
) -> TimeStepping:
    rpm = None if rotor is None or rps is None else 60.0 * rotor.rpm_sign * rps
    if run_type == UNSTEADY_ROTOR:
        return rotor_time_stepping("the row", setup, rpm=rpm)
    return unsteady_time_stepping("the row", setup, rpm=rpm)


def averaging_window(
    run_type: str, setup: Mapping[str, object], clock: TimeStepping
) -> tuple[int, int]:
    """Return the averaging window ``(first, last)`` of a time-accurate row, 1-based.

    The row must state ``LAST_REVS_AVG`` or ``LAST_ITERS_AVG``; nothing is defaulted. A
    revolutions window on a row with no rotor clock is warned and the whole run averaged.
    """
    if stated_key(setup) is None:
        key, example, other, turns = (
            (LAST_REVS_AVG, "0.5' averages the last half revolution", LAST_ITERS_AVG, "a")
            if run_type == UNSTEADY_ROTOR
            else (LAST_ITERS_AVG, "100' averages the last hundred steps", LAST_REVS_AVG, "no")
        )
        raise ValueError(
            f"the row names the run type {run_type} and states no {key}. An unsteady point's "
            f"polar, its time average and its rotor table are all averaged over that window, "
            f"so the row must say it: '{key}: {example}. (A row that turns {turns} rotor "
            f"may state {other} instead.)"
        )
    per_revolution = clock.steps_per_revolution
    span = averaging_span(setup, last_step=clock.time_iterations, per_revolution=per_revolution)
    if span is None:
        warn_or_refuse(
            f"{LAST_REVS_AVG} is skipped: the row states no rotor speed, so a count of "
            "revolutions has no length in solver steps; the whole run is averaged",
            UserWarning,
            stacklevel=3,
        )
        return 1, clock.time_iterations
    return span


def apply_run_type(
    values: MutableMapping[str, object],
    run_type: str,
    *,
    setup: Mapping[str, object],
    rotor: RotorBlock | None,
    periodic: PeriodicState,
    motion: Mapping[str, float],
) -> dict[str, object]:
    """Write the SU2 keys run type ``run_type`` carries; return what the point records.

    ``motion`` holds the row's ``ADVANCE_RATIO`` or ``RPM``. The record holds the clock, the
    averaging window and the export window of a time-accurate point.
    """
    record: dict[str, object] = {"run_type": run_type}
    if run_type == STEADY:
        return record
    needs_rotor = run_type in (QSTEADY_ROTOR, UNSTEADY_ROTOR)
    rps: float | None = None
    if needs_rotor:
        rotor = _rotor_or_refuse(rotor, run_type)
        rps = _speed(values, rotor, motion)
    elif motion and rotor is not None:
        rps = _speed(values, rotor, motion)  # the power-off twin: a clock, nothing turns
    if run_type == QSTEADY_ROTOR:
        assert rotor is not None and rps is not None
        for key, label in (("AOA", "ALPHA"), ("SIDESLIP_ANGLE", "BETA")):
            if key in values and float(str(values[key])) != 0.0:
                raise ValueError(
                    f"WORKFLOW qsteady_rotor states {label} {values[key]}: an angle makes the "
                    "inflow vary with azimuth, so the steady rotating frame no longer holds; "
                    "solve the rotor as unsteady_rotor"
                )
        frame = str(values.setdefault("GRID_MOVEMENT", "ROTATING_FRAME")).strip().upper()
        if frame != "ROTATING_FRAME":
            raise ValueError(
                f"WORKFLOW qsteady_rotor solves the rotor in a rotating frame, and the setup "
                f"states GRID_MOVEMENT = {frame}; remove it"
            )
        write_rotation(values, rotor, rps)
        _periodic_marker(values, setup, rotor, periodic)
        return record
    clock = _clock(run_type, setup, rotor, rps)
    window = averaging_window(run_type, setup, clock)
    exports = export_window(
        setup,
        pol="",
        time_iterations=clock.time_iterations,
        steps_per_revolution=clock.steps_per_revolution,
        averaging_first_step=window[0],
    )
    _time_keys(values, clock, window)
    if run_type == UNSTEADY_ROTOR:
        assert rotor is not None and rps is not None
        sliding = setup.get("sliding")
        if sliding is None:
            values.setdefault("GRID_MOVEMENT", "RIGID_MOTION")
            write_rotation(values, rotor, rps)
        else:
            _zones(values, sliding, rotor, rps)
    if exports is not None:
        export_segments(values, exports)
    record.update(
        clock=clock.record(),
        averaging_window=list(window),
        export_window=None if exports is None else exports.record(),
    )
    return record


def _time_keys(
    values: MutableMapping[str, object], clock: TimeStepping, window: tuple[int, int]
) -> None:
    values["TIME_DOMAIN"] = True
    values.setdefault("TIME_MARCHING", TIME_MARCHING)
    values["TIME_STEP"] = clock.delta_time_s
    values["TIME_ITER"] = clock.time_iterations
    if "ITER" in values:
        values.setdefault("INNER_ITER", values.pop("ITER"))
    # SU2 averages from its time iteration WINDOW_START_ITER, which is step window[0].
    values.setdefault("WINDOW_START_ITER", window[0] - 1)
    groups = [str(group) for group in values.get("HISTORY_OUTPUT", [])]  # type: ignore[attr-defined]
    if "TAVG_AERO_COEFF" not in groups:
        values["HISTORY_OUTPUT"] = [*groups, "TAVG_AERO_COEFF"]


def _zones(
    values: MutableMapping[str, object], sliding: object, rotor: RotorBlock, rps: float
) -> None:
    """Write a sliding-mesh multizone layout: CONFIG_LIST and one zone file per zone."""
    if not isinstance(sliding, Mapping):
        raise ValueError("sliding must be a table: zones, rotating_zone and interface")
    unknown = sorted(set(sliding) - {"zones", "rotating_zone", "interface"})
    if unknown:
        raise ValueError(
            f"sliding has unknown keys: {', '.join(unknown)} (zones, rotating_zone, interface)"
        )
    zones = sliding.get("zones")
    turning = sliding.get("rotating_zone")
    pair = sliding.get("interface")
    if type(zones) is not int or zones < 2:
        raise ValueError("sliding.zones must be the whole number of mesh zones, two or more")
    if type(turning) is not int or not 0 <= turning < zones:
        raise ValueError(f"sliding.rotating_zone must be a zone index from 0 to {zones - 1}")
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(m, str) for m in pair):
        raise ValueError(
            "sliding.interface must name the interface pair: [stationary side, rotating side]"
        )
    if "GRID_MOVEMENT" in values:
        raise ValueError(
            "a sliding-mesh setup states GRID_MOVEMENT in its zone files, which cfdbender "
            "writes; remove GRID_MOVEMENT from the setup file"
        )
    values["MULTIZONE"] = True
    values["CONFIG_LIST"] = [f"zone_{i}.cfg" for i in range(zones)]
    values.setdefault("MARKER_ZONE_INTERFACE", list(pair))
    values.setdefault("MARKER_FLUID_INTERFACE", list(pair))
    values.setdefault("KIND_INTERPOLATION", "WEIGHTED_AVERAGE")
    # A multizone time step iterates the zones OUTER_ITER times, each zone once per pass.
    if "INNER_ITER" in values and "OUTER_ITER" not in values:
        values["OUTER_ITER"] = values.pop("INNER_ITER")
        values["INNER_ITER"] = 1
    # Each zone's residuals and coefficients are their own history group (to verify on a
    # real time-accurate run: the column names SU2 gives them).
    groups = [str(group) for group in values.get("HISTORY_OUTPUT", [])]  # type: ignore[attr-defined]
    zoned = [
        f"{group}[{index}]"
        for group in ("RMS_RES", "AERO_COEFF")
        if group in groups
        for index in range(zones)
    ]
    values["HISTORY_OUTPUT"] = [
        *(g for g in groups if g not in ("RMS_RES", "AERO_COEFF")),
        *zoned,
    ]
    for index in range(zones):
        if index != turning:
            values[f"zone_{index}.cfg/GRID_MOVEMENT"] = "NONE"
    target = f"zone_{turning}.cfg/"
    values[f"{target}GRID_MOVEMENT"] = "RIGID_MOTION"
    write_rotation(values, rotor, rps, target=target)


def export_segments(values: MutableMapping[str, object], exports: ExportWindow) -> None:
    """Realise the export window: the ONE function a different mechanism replaces.

    SU2 writes its files at a constant frequency for a whole run, so the point runs in two
    segments: ``su2.cfg`` marches to the step before the window and writes restart files
    only; ``su2_segment2.cfg`` restarts there and writes every step to the end. A window
    that begins at step one is one segment writing every step. To verify on a real run: that
    the restart of a dual-time run continues cleanly.
    """
    outputs = values.get("OUTPUT_FILES", list(SU2_OUTPUT_FILES))
    if exports.first_step <= 1:
        values["OUTPUT_WRT_FREQ"] = 1
        return
    total = values["TIME_ITER"]
    values["TIME_ITER"] = exports.first_step - 1
    values["OUTPUT_FILES"] = ["RESTART"]
    target = f"{SEGMENT_TWO}/"
    values[f"{target}TIME_ITER"] = total
    values[f"{target}RESTART_SOL"] = True
    values[f"{target}RESTART_ITER"] = exports.first_step - 1
    values[f"{target}OUTPUT_FILES"] = outputs
    values[f"{target}OUTPUT_WRT_FREQ"] = 1


def time_accurate(run_type: str) -> bool:
    """Whether a run type marches in time."""
    return run_type in TIME_ACCURATE


__all__ = [
    "SETUP_KEYS",
    "TURBULENCE_MODELS",
    "UNSTEADY",
    "apply_run_type",
    "export_segments",
    "turbulence",
    "write_rotation",
]

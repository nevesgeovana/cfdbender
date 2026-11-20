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
files only, the second restarts there and writes every step to the end; verified on a real
SU2 8.6.0 run, the restart continues to the solver's run-to-run noise.

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
from cfdbender.core.unsteady_clock import (
    TimeStepping,
    revolution_steps,
    rotor_time_stepping,
    unsteady_time_stepping,
)
from cfdbender.core.unsteady_window import LAST_ITERS_AVG, LAST_REVS_AVG, averaging_span, stated_key
from cfdbender.solvers.su2._precursor import SOLUTION_NAME

#: The row key that writes the restart file every so many revolutions (PROVISIONAL name).
RESTART_EVERY_REV = "RESTART_EVERY_REV"
#: The setup keys of this module that are not SU2's own (PROVISIONAL names).
SETUP_KEYS = ("periodic_markers", "sliding", "precursor")
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
#: SU2 8.6.0's own RESTART_FILENAME when the setup states none.
SU2_RESTART_FILENAME = "restart"
#: The keys SU2 8.6.0 refuses set to NO for a transient run (CConfig: "Appending iterations
#: to the filename ... is incompatible with transient problems").
OVERWRITE_KEYS = ("WRT_RESTART_OVERWRITE", "WRT_SURFACE_OVERWRITE", "WRT_VOLUME_OVERWRITE")


def carried_keys(run_type: str, *, sliding: bool = False) -> tuple[str, ...]:
    """Return native keys a run type supplies above workspace defaults."""
    rotation = ("GRID_MOVEMENT", "ROTATION_RATE", "MOTION_ORIGIN", "MACH_MOTION")
    clock = (
        "TIME_DOMAIN",
        "TIME_MARCHING",
        "TIME_STEP",
        "TIME_ITER",
        "INNER_ITER",
        "WINDOW_START_ITER",
    )
    if run_type == QSTEADY_ROTOR:
        return rotation
    if run_type == UNSTEADY_ROTOR:
        interface = (
            ("MARKER_ZONE_INTERFACE", "MARKER_FLUID_INTERFACE", "KIND_INTERPOLATION")
            if sliding
            else ()
        )
        return (*clock, *rotation, *interface)
    return clock if run_type == UNSTEADY else ()


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
    restarted: bool = False,
    precursor: bool = False,
) -> dict[str, object]:
    """Write the SU2 keys run type ``run_type`` carries; return what the point records.

    ``motion`` holds the row's ``ADVANCE_RATIO`` or ``RPM``. The record holds the clock, the
    averaging window and the export window of a time-accurate point, and ``step_offset`` when
    ``restarted``: a time-accurate point that starts from a saved solution (the row's
    ``RESTART``) begins at SU2's time iteration ``RESTART_ITER``, :func:`restart_order` (two
    for second-order dual time stepping), and the solution is staged as the iterations before
    it. Its step ``s`` is SU2's time iteration ``s - 1 + offset`` and ``TIME_ITER``,
    ``WINDOW_START_ITER`` and the segment hand-over are shifted by the offset, so the point
    still marches every step the row asks for. ``precursor`` (the row's ``RESTART`` is ``MRF``)
    starts the point from a steady frozen-rotor solution of its own mesh, staged as the same
    iterations before ``RESTART_ITER`` (:mod:`cfdbender.solvers.su2._precursor`): the point is
    ``restarted`` in every other respect. Stage 1's files are written by ``config`` once every
    key of the point is final, so both stages share one flow model.
    """
    record: dict[str, object] = {"run_type": run_type}
    if run_type == STEADY:
        return record
    before = set(values)
    try:
        return _carried(
            values, run_type, record, setup, rotor, periodic, motion, restarted, precursor
        )
    finally:
        # The keys the run type wrote that no file or row stated: their origin is the run type.
        record["carried"] = sorted(key for key in values if key not in before)


def _carried(
    values: MutableMapping[str, object],
    run_type: str,
    record: dict[str, object],
    setup: Mapping[str, object],
    rotor: RotorBlock | None,
    periodic: PeriodicState,
    motion: Mapping[str, float],
    restarted: bool = False,
    precursor: bool = False,
) -> dict[str, object]:
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
    if precursor and (run_type != UNSTEADY_ROTOR or setup.get("sliding") is None):
        raise ValueError(
            "RESTART MRF needs a time-accurate sliding-mesh rotor row (WORKFLOW unsteady_rotor "
            "and a [sliding] table in the setup file): there is no frozen-rotor solution of a "
            "single zone to start from, and a cold start is the RESTART cell -"
        )
    clock = _clock(run_type, setup, rotor, rps)
    window = averaging_window(run_type, setup, clock)
    exports = export_window(
        setup,
        pol="",
        time_iterations=clock.time_iterations,
        steps_per_revolution=clock.steps_per_revolution,
    )
    offset = _time_keys(values, clock, window, restarted=restarted or precursor)
    every = restart_every(values, setup, clock)
    if run_type == UNSTEADY_ROTOR:
        assert rotor is not None and rps is not None
        sliding = setup.get("sliding")
        if sliding is None:
            values.setdefault("GRID_MOVEMENT", "RIGID_MOTION")
            write_rotation(values, rotor, rps)
        else:
            _zones(values, sliding, rotor, rps)
    if exports is not None:
        export_segments(values, exports, offset, every=every)
    else:
        last_step_outputs(values, every=every)
    if precursor:
        values["SOLUTION_FILENAME"] = SOLUTION_NAME
    record.update(
        clock=clock.record(),
        averaging_window=list(window),
        export_window=None if exports is None else exports.record(),
    )
    if every is not None:
        record["restart_every"] = {"revolutions": setup[RESTART_EVERY_REV], "steps": every}
    if offset:
        record["step_offset"] = offset
    return record


def restart_every(
    values: Mapping[str, object], setup: Mapping[str, object], clock: TimeStepping
) -> int | None:
    """Return the time iterations between periodic restart files, or None when not stated.

    ``RESTART_EVERY_REV: r`` (a PROVISIONAL name) writes the restart file every ``r``
    revolutions of the rotor clock, ``ceil(r S)`` time iterations with ``S`` the steps per
    revolution, and leaves the volume and surface files at their own frequency. SU2 counts the
    multiples on its own time iteration and, under second-order dual time stepping, also
    writes the iteration before each multiple, so a killed point keeps a pair to continue from.

    Raises
    ------
    ValueError
        A value that is not a positive number, or a setup (or PPROC ``output_frequency``) that
        states ``OUTPUT_WRT_FREQ`` itself.
    """
    if RESTART_EVERY_REV not in setup:
        return None
    raw = setup[RESTART_EVERY_REV]
    try:
        value = float(str(raw))
    except ValueError:
        value = math.nan
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"the row states {RESTART_EVERY_REV} as {raw!r}; it is how many revolutions apart "
            "the restart files are written, so it is a positive number"
        )
    if not clock.steps_per_revolution:
        warn_or_refuse(
            f"{RESTART_EVERY_REV} is skipped: the row turns no rotor, so a count of revolutions "
            "has no length in solver steps; the restart file is written at the last step only",
            UserWarning,
            stacklevel=3,
        )
        return None
    if "OUTPUT_WRT_FREQ" in values:
        raise ValueError(
            f"the row states {RESTART_EVERY_REV} and the setup (or the PPROC output_frequency) "
            "states OUTPUT_WRT_FREQ: both set how often the restart file is written; state one"
        )
    return max(1, revolution_steps(value, clock.steps_per_revolution))


def _refuse_appending(values: Mapping[str, object]) -> None:
    """Refuse an overwrite key set to NO, which SU2 8.6.0 refuses for a transient run."""
    for key in OVERWRITE_KEYS:
        value = values.get(key)
        if value is False or str(value).strip().upper() in ("NO", "FALSE"):
            raise ValueError(
                f"the setup states {key}= NO, which appends the iteration to every file name, "
                "and SU2 refuses it for a time-accurate run (it numbers each time step's files "
                f"itself): remove {key} or state it YES"
            )


def restart_order(values: Mapping[str, object]) -> int:
    """Return how many previous time iterations an SU2 time-accurate restart reads.

    SU2 8.6.0 reads ``<SOLUTION_FILENAME stem>_<k>`` for ``k = RESTART_ITER - 2`` and
    ``RESTART_ITER - 1`` under second-order dual time stepping, and ``RESTART_ITER - 1``
    otherwise; it refuses a negative ``k``, so a second-order run cannot restart at 0 or 1.
    """
    marching = str(values.get("TIME_MARCHING", TIME_MARCHING)).strip().upper()
    return 2 if marching == TIME_MARCHING else 1


def _time_keys(
    values: MutableMapping[str, object],
    clock: TimeStepping,
    window: tuple[int, int],
    *,
    restarted: bool = False,
) -> int:
    """Write the clock keys; return the step offset (``RESTART_ITER`` of a restarted point)."""
    values["TIME_DOMAIN"] = True
    values.setdefault("TIME_MARCHING", TIME_MARCHING)
    _refuse_appending(values)
    offset = restart_order(values) if restarted else 0
    values["TIME_STEP"] = clock.delta_time_s
    values["TIME_ITER"] = clock.time_iterations + offset
    if "ITER" in values:
        values.setdefault("INNER_ITER", values.pop("ITER"))
    # SU2 averages from its time iteration WINDOW_START_ITER, which is step window[0].
    values.setdefault("WINDOW_START_ITER", window[0] - 1 + offset)
    if offset:
        # The saved solution is staged as SU2's time iterations 0 .. offset - 1 (one
        # steady state repeated), so the run starts at iteration `offset`, the row's step 1.
        values["RESTART_SOL"] = True
        values["RESTART_ITER"] = offset
    groups = [str(group) for group in values.get("HISTORY_OUTPUT", [])]  # type: ignore[attr-defined]
    if "TAVG_AERO_COEFF" not in groups:
        values["HISTORY_OUTPUT"] = [*groups, "TAVG_AERO_COEFF"]
    return offset


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
    values.setdefault("KIND_INTERPOLATION", "NEAREST_NEIGHBOR")
    # A multizone time step iterates the zones OUTER_ITER times, each zone once per pass.
    if "INNER_ITER" in values and "OUTER_ITER" not in values:
        values["OUTER_ITER"] = values.pop("INNER_ITER")
        values["INNER_ITER"] = 1
    # Each zone's residuals and coefficients are their own history group: SU2 8.6.0's
    # multizone output names the group <GROUP>[<zone>] and its columns <field>[<zone>]
    # (CMultizoneOutput, read in SU2's source; no multizone time-accurate run yet).
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


def export_segments(
    values: MutableMapping[str, object],
    exports: ExportWindow,
    offset: int = 0,
    *,
    every: int | None = None,
    start: int | None = None,
) -> None:
    """Realise the export window: the ONE function a different mechanism replaces.

    SU2 writes its files at a constant frequency for a whole run, so the point runs in two
    segments: ``su2.cfg`` marches to the step before the window and writes restart files
    only; ``su2_segment2.cfg`` restarts there and writes every step to the end. Verified on a
    real SU2 8.6.0 run: segment two at ``RESTART_ITER= R`` (step ``R - offset + 1`` is the
    window's first) reads ``<SOLUTION_FILENAME stem>_<R-2>`` and ``_<R-1>`` (second order),
    which segment one wrote as ``<RESTART_FILENAME stem>_<k>``, so segment two's
    ``SOLUTION_FILENAME`` is segment one's ``RESTART_FILENAME`` (SU2's default ``restart``)
    and it reads them as binary, the format ``RESTART`` writes. The hand-over continues to the
    run-to-run noise of the solver. A window whose first step leaves segment one fewer marched
    steps than the restart reads (``restart_order``) is one segment writing every step.
    ``offset`` is the step offset of a point started from a saved solution. ``every`` is the
    periodic restart frequency (:func:`restart_every`), None for the end of each segment only.
    ``start`` is SU2's first time iteration of a continuation (:func:`continue_keys`): a window
    that began before it writes every remaining step in one segment.
    """
    outputs = _outputs(values)
    total = int(str(values["TIME_ITER"]))
    first = offset if start is None else start
    handover = exports.first_step - 1 + offset
    if handover - first < restart_order(values):
        values["OUTPUT_FILES"] = outputs
        values["OUTPUT_WRT_FREQ"] = _frequencies(outputs, restart=every or total, other=1)
        return
    values["TIME_ITER"] = handover
    values["OUTPUT_FILES"] = ["RESTART"]
    # Only the restart files segment two reads. SU2 8.6.0 writes a file at every multiple of
    # its frequency and at its last time iteration, and under second-order dual time a restart
    # file also before each multiple and before the last (CFlowOutput, measured): with the
    # frequency equal to TIME_ITER that is handover - 2 and handover - 1 (handover - 1 alone
    # under first order), plus time iteration 0 of a cold start. A periodic frequency adds
    # its multiples and keeps the last two.
    values["OUTPUT_WRT_FREQ"] = every or handover
    target = f"{SEGMENT_TWO}/"
    values[f"{target}TIME_ITER"] = total
    values[f"{target}RESTART_SOL"] = True
    values[f"{target}RESTART_ITER"] = handover
    # SU2 refuses a restart whose WINDOW_START_ITER is below RESTART_ITER (measured), so its
    # own time average, a cross-check, starts at segment two; cfdbender's reads the history.
    window = values.get("WINDOW_START_ITER")
    if isinstance(window, int) and window < handover:
        values[f"{target}WINDOW_START_ITER"] = handover
    values[f"{target}SOLUTION_FILENAME"] = values.get("RESTART_FILENAME", SU2_RESTART_FILENAME)
    values[f"{target}READ_BINARY_RESTART"] = True
    values[f"{target}OUTPUT_FILES"] = outputs
    values[f"{target}OUTPUT_WRT_FREQ"] = _frequencies(outputs, restart=every or total, other=1)


def _outputs(values: Mapping[str, object]) -> list[str]:
    stated = values.get("OUTPUT_FILES", list(SU2_OUTPUT_FILES))
    if isinstance(stated, list):
        return [str(item) for item in stated]
    return [item.strip() for item in str(stated).strip("() ").split(",") if item.strip()]


def _frequencies(outputs: list[str], *, restart: int, other: int) -> list[int]:
    """Return OUTPUT_WRT_FREQ for ``outputs``, one frequency per file in their order."""
    return [restart if item.upper().startswith("RESTART") else other for item in outputs]


def last_step_outputs(values: MutableMapping[str, object], *, every: int | None = None) -> None:
    """Write a time-accurate row's files at its last step only, when it has no export window.

    pyflightstream's meaning: an unsteady point with no ``EXPORT_UNSTEADY_*`` key exports its
    end-of-run files only. SU2 8.6.0 writes every file of ``OUTPUT_FILES`` at each time
    iteration that is a multiple of its ``OUTPUT_WRT_FREQ`` (1 by default for a transient run,
    so every step) and always at its last, so each file gets the frequency ``TIME_ITER``: the
    last step (and the step before it for a second-order restart file, which SU2 writes in
    pairs), and on a cold start also time iteration 0 (step 1), which every frequency
    divides. A setup that states ``OUTPUT_WRT_FREQ`` (or the PPROC ``output_frequency``) keeps
    its own frequency. ``every`` (:func:`restart_every`) writes the restart file at its
    multiples too.
    """
    if "OUTPUT_WRT_FREQ" in values:
        return
    outputs = _outputs(values)
    total = int(str(values["TIME_ITER"]))
    values["OUTPUT_FILES"] = outputs
    values["OUTPUT_WRT_FREQ"] = _frequencies(outputs, restart=every or total, other=total)


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

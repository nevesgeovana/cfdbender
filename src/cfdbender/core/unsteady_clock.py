"""The physical clock of an unsteady row, and the azimuth of the rotor at each step.

A row states its clock in ONE of two forms, never both and never half of one:

* ``DELTA_THETA`` (degrees per step) and ``REVOLUTIONS``: resolved against the rotor speed,
  ``DELTA_TIME = DELTA_THETA / (6 abs(rpm))`` and ``TIME_ITERATIONS = REVOLUTIONS 360 /
  DELTA_THETA``, which must be a whole number of steps;
* ``DELTA_TIME`` (seconds) and ``TIME_ITERATIONS`` (steps), stated directly.

:class:`TimeStepping` carries the clock and its steps per revolution. The azimuth of blade one
at a step is ``(datum + sense * step * 360 / steps_per_revolution) mod 360``, the datum being
the ``azimuth_deg`` of the ``blade1`` table of the reference file's rotor block and ``sense``
the sign of the rotor's speed (:func:`blade_azimuth_deg`).

Everything here is pure: the row is a mapping of its variables and the rotor speed is a number
the caller resolved. The semantics, names and sentences match the time-domain
post-processing of pyflightstream.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from cfdbender.core.settings import WorkspaceError

__all__ = [
    "ADVANCE_RATIO_VARIABLE",
    "DELTA_THETA_VARIABLE",
    "DELTA_TIME_VARIABLE",
    "REVOLUTIONS_VARIABLE",
    "RPM_VARIABLE",
    "TIME_ITERATIONS_VARIABLE",
    "BladeDatum",
    "TimeStepping",
    "blade_azimuth_deg",
    "blade_datum",
    "blade_one_azimuth",
    "revolution_steps",
    "rotor_time_stepping",
    "unsteady_time_stepping",
]

DELTA_THETA_VARIABLE = "DELTA_THETA"
REVOLUTIONS_VARIABLE = "REVOLUTIONS"
DELTA_TIME_VARIABLE = "DELTA_TIME"
TIME_ITERATIONS_VARIABLE = "TIME_ITERATIONS"
#: Row keys a refusal names as the way to state a rotor speed.
ADVANCE_RATIO_VARIABLE = "ADVANCE_RATIO"
RPM_VARIABLE = "RPM"


@dataclass(frozen=True)
class TimeStepping:
    """The physical clock of an unsteady run, and which form set it.

    Attributes
    ----------
    stated_form : str
        ``angular`` (``DELTA_THETA`` and ``REVOLUTIONS``) or ``explicit``
        (``DELTA_TIME`` and ``TIME_ITERATIONS``).
    delta_time_s : float
        Physical time step in s.
    time_iterations : int
        Physical time steps of the whole run.
    delta_theta_deg, revolutions : float or None
        The stated pair, where the form was angular.
    rpm : float or None
        The signed rotor speed the conversion ran against.
    """

    stated_form: str
    delta_time_s: float
    time_iterations: int
    delta_theta_deg: float | None
    revolutions: float | None
    rpm: float | None

    @property
    def steps_per_revolution(self) -> float | None:
        """Solver steps in one revolution, or None without a turning rotor.

        From the azimuthal step where the clock was stated that way, and from the seconds
        only where it was not: a ten-degree step is thirty-six steps exactly, and deriving it
        from rounded seconds would read the rounding back as physics. A rotor at rest has no
        revolution, and a product that underflows to zero is no revolution either.

        Examples
        --------
        >>> TimeStepping("angular", 0.01, 36, 10.0, 1.0, 1000.0).steps_per_revolution
        36.0
        """
        if self.delta_theta_deg:
            return 360.0 / self.delta_theta_deg
        if not self.rpm or self.delta_time_s <= 0.0:
            return None
        revolution_s = abs(self.rpm) * self.delta_time_s
        if revolution_s <= 0.0:
            return None
        return 60.0 / revolution_s

    def record(self) -> dict[str, object]:
        """Return the stated form and every form derived from it.

        ``rpm`` is included although it is an input: ``steps_per_revolution`` is computed
        from it, so a record carrying the quotient and not the divisor would show a number
        whose working could not be recovered.
        """
        return {
            "form": self.stated_form,
            "delta_time_s": self.delta_time_s,
            "time_iterations": self.time_iterations,
            "delta_theta_deg": self.delta_theta_deg,
            "revolutions": self.revolutions,
            "rpm": self.rpm,
            "steps_per_revolution": self.steps_per_revolution,
        }


def _variable(variables: Mapping[str, object], key: str) -> str | None:
    value = variables.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_float(
    sim_id: str, variables: Mapping[str, object], key: str, *, quantity: str, unit: str
) -> float:
    """One numeric row variable, refused by row and key."""
    text = _variable(variables, key)
    if text is None:
        raise WorkspaceError(
            f"case {sim_id!r} declares no {key}, and the run type it names needs "
            f"the {quantity} in {unit}. Add it to the row's variables as "
            f"'{key}: <value>'."
        )
    try:
        value = float(text)
    except ValueError:
        raise WorkspaceError(
            f"case {sim_id!r} declares {key} as {text!r}, which is not a number. "
            f"It is the {quantity} in {unit}. Matrix variables arrive as text, so the "
            "conversion happens here rather than at the command, where the refusal "
            "would name only the command and not the cell you typed."
        ) from None
    # NaN compares false against every bound, so it would pass each check below.
    if not math.isfinite(value):
        raise WorkspaceError(
            f"case {sim_id!r} declares {key} as {text!r}, which is not a finite "
            f"number. It is the {quantity} in {unit}, and a value that is NaN or "
            "infinite compares false against every bound, so it would pass each check "
            "below and be emitted, or fail much later where the refusal could name "
            "neither this case nor this key."
        )
    return value


def _required_int(
    sim_id: str, variables: Mapping[str, object], key: str, *, quantity: str, unit: str
) -> int:
    value = _required_float(sim_id, variables, key, quantity=quantity, unit=unit)
    if value != int(value):
        raise WorkspaceError(
            f"case {sim_id!r} declares {key} as {value!r}, and the {quantity} is a "
            f"count in {unit}, which has no fractional part."
        )
    # A count written as plain digits is read as an integer, not through a float, which
    # holds only 53 bits.
    stated = _variable(variables, key)
    if stated is not None:
        try:
            return int(stated)
        except ValueError:
            pass
    return int(value)


def _whole_steps(sim_id: str, revolutions: float, theta: float) -> int:
    """Return the whole time steps of ``revolutions`` at ``theta`` degrees a step, or refuse."""
    exact_steps = revolutions * 360.0 / theta
    if not math.isfinite(exact_steps):
        raise WorkspaceError(
            f"case {sim_id!r} asks for {revolutions} revolutions at {theta} degrees per "
            "step, which is more time steps than a number can hold. State a run length "
            "the solver could march."
        )
    steps = int(round(exact_steps))
    if steps < 1:
        raise WorkspaceError(
            f"case {sim_id!r} asks for {revolutions} revolutions at {theta} degrees "
            f"per step, which is {exact_steps:g} time steps, and a run marches at least one "
            "time step. State more revolutions or a smaller azimuthal step."
        )
    if abs(exact_steps - steps) > 1e-6:
        # Whole steps or a refusal: rounding silently would end the run mid-step at an
        # azimuth nobody chose.
        low = steps * theta / 360.0
        raise WorkspaceError(
            f"case {sim_id!r} asks for {revolutions} revolutions at "
            f"{theta} degrees per step, which is {exact_steps:g} time steps and not a "
            "whole number, so the run would end part way through a step at an azimuth "
            f"nobody chose. {low:g} revolutions is {steps} whole steps at this step "
            f"size; a step size that divides {revolutions} revolutions exactly is the "
            "other way to close it."
        )
    return steps


def _angular_step_seconds(sim_id: str, theta: float, rpm: float) -> float:
    """Return the seconds one azimuthal step of ``theta`` degrees lasts at ``rpm``.

    One revolution lasts 60/rpm seconds, so one degree lasts 1/(6 rpm) seconds. The
    derived precision is kept: rounding the step would end the run at an unstated azimuth.
    """
    seconds = theta / (6.0 * abs(rpm))
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise WorkspaceError(
            f"case {sim_id!r}: a step of {theta} degrees at {rpm} rev/min lasts "
            f"{seconds:g} seconds, and a time step is a positive, finite number of seconds. "
            "State a speed or a step the solver could use."
        )
    return seconds


def _explicit_clock(sim_id: str, variables: Mapping[str, object]) -> tuple[float, int]:
    """Return the stated ``DELTA_TIME`` and ``TIME_ITERATIONS``, refusing a non-positive one."""
    delta_time_s = _required_float(
        sim_id, variables, DELTA_TIME_VARIABLE, quantity="solver physical time step", unit="s"
    )
    if delta_time_s <= 0.0:
        raise WorkspaceError(
            f"case {sim_id!r} declares {DELTA_TIME_VARIABLE} as {delta_time_s:g}, and a "
            "time step is a positive number of seconds. State a positive step."
        )
    iterations = _required_int(
        sim_id,
        variables,
        TIME_ITERATIONS_VARIABLE,
        quantity="physical time step count",
        unit="steps",
    )
    if iterations < 1:
        raise WorkspaceError(
            f"case {sim_id!r} declares {TIME_ITERATIONS_VARIABLE} as {iterations}, and "
            "a run marches at least one time step. State a positive count."
        )
    return delta_time_s, iterations


def _refuse_an_azimuthal_step_outside_one_turn(sim_id: str, theta: float) -> None:
    """Refuse a ``DELTA_THETA`` that is not a positive angle of at most one turn."""
    if theta <= 0.0 or theta > 360.0:
        raise WorkspaceError(
            f"case {sim_id!r} declares {DELTA_THETA_VARIABLE} as {theta}, and an "
            "azimuthal step is a positive angle no larger than a whole revolution. A "
            "step of 360 degrees resolves nothing inside one turn."
        )


def _turning_speed(sim_id: str, rpm: float | None) -> float:
    """Return the rotor speed, refusing a rotor that does not turn."""
    if rpm is None or not math.isfinite(rpm):
        raise WorkspaceError(
            f"case {sim_id!r} resolves no rotor speed, and a degree of rotation has no "
            "duration without one. Pass the signed speed of the rotor that sets the clock, "
            f"or state the clock as {DELTA_TIME_VARIABLE} and {TIME_ITERATIONS_VARIABLE}."
        )
    if rpm == 0.0:
        raise WorkspaceError(
            f"case {sim_id!r} resolves a rotor speed of zero, and a degree of "
            "rotation has no duration on a rotor that does not turn. State the clock "
            f"as {DELTA_TIME_VARIABLE} and {TIME_ITERATIONS_VARIABLE} if this run "
            "really is stationary."
        )
    return rpm


def rotor_time_stepping(
    sim_id: str, variables: Mapping[str, object], *, rpm: float | None = None
) -> TimeStepping:
    """Resolve the physical clock of an unsteady rotor run, in either form.

    Parameters
    ----------
    sim_id : str
        The row's identifier, named by every refusal.
    variables : mapping
        The row's variables. It states ``DELTA_THETA`` and ``REVOLUTIONS``, or
        ``DELTA_TIME`` and ``TIME_ITERATIONS``; one pair, not both and not half of one.
    rpm : float or None, keyword-only
        The signed speed in rev/min of the rotor whose clock this is, resolved by the
        caller. Needed by the angular form; the explicit form records it.

    Returns
    -------
    TimeStepping

    Raises
    ------
    WorkspaceError
        If both pairs or neither are stated; if one pair is stated half; if the speed is
        missing or zero for the angular form; or if the revolutions and the azimuthal step
        do not work out to a WHOLE number of time steps, which is refused naming the two
        numbers and the nearest pair that does.

    Examples
    --------
    >>> stepping = rotor_time_stepping(
    ...     "7001", {"DELTA_THETA": "15", "REVOLUTIONS": "1.5"}, rpm=1200.0
    ... )
    >>> stepping.time_iterations, stepping.steps_per_revolution
    (36, 24.0)
    """
    angular = {
        key: _variable(variables, key)
        for key in (DELTA_THETA_VARIABLE, REVOLUTIONS_VARIABLE)
        if _variable(variables, key) is not None
    }
    explicit = {
        key: _variable(variables, key)
        for key in (DELTA_TIME_VARIABLE, TIME_ITERATIONS_VARIABLE)
        if _variable(variables, key) is not None
    }
    if angular and explicit:
        raise WorkspaceError(
            f"case {sim_id!r} states its clock in two forms at once: "
            f"{', '.join(f'{k}={v!r}' for k, v in sorted(angular.items()))} and "
            f"{', '.join(f'{k}={v!r}' for k, v in sorted(explicit.items()))}. The "
            "seconds and the step count are what the azimuthal step and the "
            "revolutions work out to at this rotor speed, so the second form is a "
            f"second set of numbers nobody keeps in agreement with the first. Keep "
            f"{DELTA_THETA_VARIABLE} and {REVOLUTIONS_VARIABLE}, or keep "
            f"{DELTA_TIME_VARIABLE} and {TIME_ITERATIONS_VARIABLE}."
        )
    if not angular and not explicit:
        raise WorkspaceError(
            f"case {sim_id!r} states no physical clock, so an unsteady run of it "
            f"has no step and no length. State '{DELTA_THETA_VARIABLE}: <deg>' and "
            f"'{REVOLUTIONS_VARIABLE}: <turns>', which are resolved against the rotor "
            f"speed, or '{DELTA_TIME_VARIABLE}: <s>' and "
            f"'{TIME_ITERATIONS_VARIABLE}: <steps>' to state them directly."
        )

    if explicit:
        # Half a pair is named here, so a user who adds the missing key meets no second refusal.
        missing = [
            key for key in (DELTA_TIME_VARIABLE, TIME_ITERATIONS_VARIABLE) if key not in explicit
        ]
        if missing:
            raise WorkspaceError(
                f"case {sim_id!r} states {', '.join(sorted(explicit))} and not "
                f"{', '.join(missing)}. The explicit clock is a PAIR: a step with no "
                "count has no length and a count with no step has no duration. Add the "
                f"missing key, or state {DELTA_THETA_VARIABLE} and "
                f"{REVOLUTIONS_VARIABLE} instead and let both be derived."
            )
        delta_time_s, iterations = _explicit_clock(sim_id, variables)
        return TimeStepping(
            stated_form="explicit",
            delta_time_s=delta_time_s,
            time_iterations=iterations,
            delta_theta_deg=None,
            revolutions=None,
            rpm=rpm,
        )

    missing = [key for key in (DELTA_THETA_VARIABLE, REVOLUTIONS_VARIABLE) if key not in angular]
    if missing:
        raise WorkspaceError(
            f"case {sim_id!r} states {', '.join(sorted(angular))} and not "
            f"{', '.join(missing)}. The angular clock is a PAIR: an azimuthal step "
            "sets how finely one revolution is resolved and the revolutions set how "
            "many there are, and neither implies the other. Add the missing key."
        )
    theta = _required_float(
        sim_id, variables, DELTA_THETA_VARIABLE, quantity="azimuthal step", unit="degrees"
    )
    revolutions = _required_float(
        sim_id, variables, REVOLUTIONS_VARIABLE, quantity="run length", unit="revolutions"
    )
    _refuse_an_azimuthal_step_outside_one_turn(sim_id, theta)
    if revolutions <= 0.0:
        raise WorkspaceError(
            f"case {sim_id!r} declares {REVOLUTIONS_VARIABLE} as {revolutions}, "
            "and a run turns for a positive number of revolutions."
        )
    speed = _turning_speed(sim_id, rpm)
    # The magnitude sets the clock whatever the direction of rotation.
    delta_time_s = _angular_step_seconds(sim_id, theta, speed)
    steps = _whole_steps(sim_id, revolutions, theta)
    return TimeStepping(
        stated_form="angular",
        delta_time_s=delta_time_s,
        time_iterations=steps,
        delta_theta_deg=theta,
        revolutions=revolutions,
        rpm=speed,
    )


def unsteady_time_stepping(
    sim_id: str, variables: Mapping[str, object], *, rpm: float | None = None
) -> TimeStepping:
    """Resolve the physical clock of a run that turns nothing.

    The seconds and the count, or the angular pair with a speed beside it: a degree of
    rotation has a duration only against a rotor speed, so a row stating the pair needs the
    speed of the rotor whose azimuth the step measures, and is then resolved exactly as
    :func:`rotor_time_stepping` resolves it. A row stating the pair and no speed is refused,
    naming the keys that would make it resolvable.

    Parameters
    ----------
    sim_id : str
        The row's identifier, named by every refusal.
    variables : mapping
        The row's variables.
    rpm : float or None, keyword-only
        The signed speed in rev/min of the rotor whose azimuth the step measures, where the
        row has one; None otherwise.

    Returns
    -------
    TimeStepping
        The resolved clock; in the explicit form it carries no rotor speed.

    Raises
    ------
    WorkspaceError
        If the row states the angular pair with no rotor speed to divide by, states
        neither pair, or states half of the explicit one.
    """
    angular = {
        key: value
        for key in (DELTA_THETA_VARIABLE, REVOLUTIONS_VARIABLE)
        if (value := _variable(variables, key)) is not None
    }
    if angular:
        if rpm is None or rpm == 0.0:
            raise WorkspaceError(
                f"case {sim_id!r} states {', '.join(sorted(angular))} and no rotor "
                "speed, and this run type meshes nothing that turns: an azimuthal step "
                "becomes seconds by dividing by a speed, and there is none here to "
                f"divide by. State '{ADVANCE_RATIO_VARIABLE}: <J>' for the rotor "
                f"whose azimuth the step measures ('{RPM_VARIABLE}' is refused on this "
                f"run type), or state the clock directly as '{DELTA_TIME_VARIABLE}: <s>' "
                f"and '{TIME_ITERATIONS_VARIABLE}: <steps>'."
            )
        return rotor_time_stepping(sim_id, variables, rpm=rpm)
    explicit = {
        key
        for key in (DELTA_TIME_VARIABLE, TIME_ITERATIONS_VARIABLE)
        if _variable(variables, key) is not None
    }
    if not explicit:
        raise WorkspaceError(
            f"case {sim_id!r} states no physical clock, so an unsteady run of it has "
            f"no step and no length. State '{DELTA_TIME_VARIABLE}: <s>' and "
            f"'{TIME_ITERATIONS_VARIABLE}: <steps>'."
        )
    missing = [
        key for key in (DELTA_TIME_VARIABLE, TIME_ITERATIONS_VARIABLE) if key not in explicit
    ]
    if missing:
        raise WorkspaceError(
            f"case {sim_id!r} states {', '.join(sorted(explicit))} and not "
            f"{', '.join(missing)}. The clock is a PAIR: a step with no count has no "
            "length and a count with no step has no duration. Add the missing key."
        )
    delta_time_s, iterations = _explicit_clock(sim_id, variables)
    return TimeStepping(
        stated_form="explicit",
        delta_time_s=delta_time_s,
        time_iterations=iterations,
        delta_theta_deg=None,
        revolutions=None,
        rpm=None,
    )


#: Steps beyond which a product of revolutions and steps a turn is no run at all; it is
#: capped there so a length check refuses it instead of the conversion overflowing.
_STEP_CAP = 10**15


def revolution_steps(number: float, per_revolution: float) -> int:
    """Return the steps ``number`` revolutions make at ``per_revolution`` steps a turn.

    Rounded up, with a margin of 1e-9 so a product that is whole to rounding error does not
    gain a step, and capped.

    Examples
    --------
    >>> revolution_steps(1.5, 24.0)
    36
    >>> revolution_steps(0.1, 36.0)
    4
    """
    product = number * per_revolution
    if not math.isfinite(product) or product >= _STEP_CAP:
        return _STEP_CAP
    return math.ceil(product - 1e-9)


def blade_azimuth_deg(
    datum_deg: object,
    *,
    step: object,
    steps_per_revolution: object,
    rpm: object,
) -> float | None:
    """Return where a blade is at a step, in degrees in [0, 360), or None.

    ``datum_deg`` is where the blade sits at step zero, and it turns by one revolution every
    ``steps_per_revolution`` steps, in the sense of the sign of ``rpm``: the azimuth is
    ``(datum + sense * step * 360 / steps_per_revolution) mod 360``.

    None where the clock is not stated: a step, a datum, a positive number of steps per
    revolution or a non-zero ``rpm`` that is not a finite number. A caller publishes that as
    not applicable, never as zero.

    Parameters
    ----------
    datum_deg : object
        Where the blade sits at step zero, in degrees.
    step : object
        The time step.
    steps_per_revolution : object
        The rotor's clock, in time steps per revolution.
    rpm : object
        The rotor's signed speed in rev/min; only its sign is read.

    Returns
    -------
    float or None
        The azimuth in degrees in [0, 360), or None where the clock is not stated.

    Examples
    --------
    >>> blade_azimuth_deg(10.0, step=18, steps_per_revolution=72, rpm=2000.0)
    100.0
    >>> blade_azimuth_deg(10.0, step=18, steps_per_revolution=72, rpm=-2000.0)
    280.0
    >>> blade_azimuth_deg(10.0, step=18, steps_per_revolution=72, rpm=None) is None
    True
    """
    stated = (datum_deg, step, steps_per_revolution, rpm)
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in stated):
        return None
    # A NaN or infinite value states nothing: a NaN speed would read as the negative sense.
    if not all(math.isfinite(float(value)) for value in stated):  # type: ignore[arg-type]
        return None
    if not steps_per_revolution > 0 or not rpm:  # type: ignore[operator]
        return None
    sense = 1.0 if rpm > 0 else -1.0  # type: ignore[operator]
    turned = sense * float(step) * 360.0 / float(steps_per_revolution)  # type: ignore[arg-type]
    return (float(datum_deg) + turned) % 360.0 + 0.0  # type: ignore[arg-type]


def blade_one_azimuth(
    step: float, *, datum_deg: float, sense: float, steps_per_revolution: float
) -> float:
    """Return where blade one of a rotor is at ``step``, in degrees, wrapped to one turn.

    The one convention of every table that states an azimuth:
    ``(datum + sense * step * 360 / steps_per_revolution) mod 360``, the datum being blade
    one's azimuth at step zero and ``sense`` the sign of the rotor's speed, all of THAT rotor.

    Raises
    ------
    WorkspaceError
        If the step, the datum and the steps per revolution do not state a clock.

    Examples
    --------
    >>> blade_one_azimuth(3, datum_deg=10.0, sense=-1.0, steps_per_revolution=8.0)
    235.0
    """
    turned = blade_azimuth_deg(
        datum_deg,
        step=step,
        steps_per_revolution=steps_per_revolution,
        rpm=1.0 if sense >= 0 else -1.0,
    )
    if turned is None:
        raise WorkspaceError(
            f"no azimuth at step {step!r}: the datum {datum_deg!r} and the "
            f"{steps_per_revolution!r} steps per revolution do not state a clock"
        )
    return turned


@dataclass(frozen=True)
class BladeDatum:
    """Where blade one sits at step zero, and the axis its azimuth is measured from.

    Attributes
    ----------
    azimuth_deg : float
        Where blade one is, measured from ``zero`` about the rotor's axis.
    zero : str
        The axis letter the azimuth is measured from, with at most one sign.
    """

    azimuth_deg: float = 0.0
    zero: str = "X"


def blade_datum(rotor: Mapping[str, object]) -> BladeDatum:
    """Return the ``blade1`` datum of a reference file's rotor table.

    The table is ``blade1 = { azimuth_deg = 45.0, zero = "X" }``. Without a ``blade1`` table
    the datum is an azimuth of zero from ``X``. The keys are ``azimuth_deg`` and ``zero``
    only; the zero is an axis letter with an optional sign.

    Parameters
    ----------
    rotor : mapping
        One rotor table of the parsed reference file.

    Returns
    -------
    BladeDatum

    Raises
    ------
    WorkspaceError
        If ``blade1`` is not a table, holds another key, or states an azimuth that is not a
        finite number or a zero that is not an axis.

    Examples
    --------
    >>> blade_datum({"blade1": {"azimuth_deg": 45.0, "zero": "-y"}})
    BladeDatum(azimuth_deg=45.0, zero='-Y')
    >>> blade_datum({})
    BladeDatum(azimuth_deg=0.0, zero='X')
    """
    stated = rotor.get("blade1", {})
    if not isinstance(stated, Mapping):
        raise WorkspaceError("blade1 must be a table: blade1 = { azimuth_deg = 0.0, zero = 'X' }")
    unknown = sorted(str(key) for key in stated if key not in ("azimuth_deg", "zero"))
    if unknown:
        raise WorkspaceError(
            f"blade1 has unknown keys: {', '.join(unknown)} (the keys are azimuth_deg and zero)"
        )
    azimuth = stated.get("azimuth_deg", 0.0)
    if (
        isinstance(azimuth, bool)
        or not isinstance(azimuth, int | float)
        or not math.isfinite(azimuth)
    ):
        raise WorkspaceError(
            f"blade1 states azimuth_deg = {azimuth!r}, which is not a finite number"
        )
    zero = stated.get("zero", "X")
    token = zero.strip().upper() if isinstance(zero, str) else ""
    if token not in {sign + axis for sign in ("", "+", "-") for axis in "XYZ"}:
        raise WorkspaceError(
            f"blade1 states zero = {zero!r}, which is not an axis; write X, Y or Z, "
            "with at most one leading sign. The sign reverses the direction the "
            "azimuth is measured FROM, not the sense in which it increases"
        )
    return BladeDatum(float(azimuth), token)

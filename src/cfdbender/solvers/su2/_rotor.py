"""What the SU2 adapter supplies for a rotor: force and moment vectors, speed and density.

SU2 writes force and moment coefficients over ``q A`` and ``q A L``, ``q`` being the dynamic
pressure of the speed it scales by (``MACH_MOTION`` times the speed of sound when the deck
states ``MACH_MOTION``). The adapter turns them back into newtons and newton metres here, so
the agnostic layer receives vectors (:class:`cfdbender.core.rotor.RotorLoads`) and forms every
rotor coefficient itself (:func:`cfdbender.core.rotor.rotor_row`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from cfdbender.core.flight import sutherland_viscosity
from cfdbender.core.rotor import (
    PeriodicState,
    RotorLoads,
    rotor_block,
    rotor_row,
    rotor_speed,
)
from cfdbender.solvers._host._config import number
from cfdbender.solvers.su2._config import (
    SU2_DEFAULT_TEMPERATURE,
    _stated_vector,
    _upper,
    free_stream,
)

FORCE_KEYS = ("CFx", "CFy", "CFz")
MOMENT_KEYS = ("CMx", "CMy", "CMz")
#: The condition columns of a product SU2 cannot fill: it has no reference span.
ABSENT_CONDITION_COLUMNS = ("BREF",)


def rotor_loads(
    values: Mapping[str, object], force: Sequence[float], moment: Sequence[float]
) -> RotorLoads | None:
    """Return the monitored loads of a deck in N and N m, or None without a free stream.

    ``force`` and ``moment`` are SU2's coefficients ``CF`` and ``CM`` of the monitored
    surface; the loads are ``CF q A`` and ``CM q A L`` about ``REF_ORIGIN_MOMENT``.
    """
    stream = free_stream(values)
    if stream is None or stream.dynamic_pressure is None:
        return None
    area = number(values["REF_AREA"], "REF_AREA")
    length = number(values["REF_LENGTH"], "REF_LENGTH")
    q = stream.dynamic_pressure
    fx, fy, fz = (q * area * c for c in force)
    mx, my, mz = (q * area * length * c for c in moment)
    ox, oy, oz = (number(values[f"REF_ORIGIN_MOMENT_{a}"], "moment_origin") for a in "XYZ")
    return RotorLoads((fx, fy, fz), (mx, my, mz), (ox, oy, oz), stream.velocity, stream.density)


def condition_cells(values: Mapping[str, object]) -> dict[str, object]:
    """Return the condition-block cells SU2's deck determines (pyflightstream's names).

    ``MACH``, ``RE`` (millions, where the deck states a Reynolds number), ``VINF``, ``VREF``
    (the speed SU2 scales its coefficients by), ``RHO``, ``TEMP``, ``MU``, ``SREF`` and
    ``CREF``. SU2 has no reference span, so ``BREF`` is not one of them.
    """
    stream = free_stream(values)
    if stream is None:
        return {}
    temperature = number(
        values.get("FREESTREAM_TEMPERATURE", SU2_DEFAULT_TEMPERATURE), "FREESTREAM_TEMPERATURE"
    )
    viscosity = (
        number(values.get("MU_CONSTANT", 0.0), "MU_CONSTANT")
        if _upper(values, "VISCOSITY_MODEL") == "CONSTANT_VISCOSITY"
        else sutherland_viscosity(temperature)
    )
    reynolds = values.get("REYNOLDS_NUMBER")
    reference = (
        math.sqrt(2.0 * stream.dynamic_pressure / stream.density)
        if stream.dynamic_pressure is not None and stream.density
        else None
    )
    return {
        "MACH": stream.mach,
        "RE": None if reynolds is None else number(reynolds, "REYNOLDS_NUMBER") / 1e6,
        "VINF": stream.velocity,
        "VREF": reference,
        "RHO": stream.density,
        "TEMP": temperature,
        "MU": viscosity,
        "SREF": number(values["REF_AREA"], "REF_AREA"),
        "CREF": number(values["REF_LENGTH"], "REF_LENGTH"),
    }


def rotor_point(
    values: Mapping[str, object],
    reference: Mapping[str, object],
    coefficients: Mapping[str, float],
    periodic: PeriodicState | None = None,
    condition: Mapping[str, float] | None = None,
) -> dict[str, object] | None:
    """Return the rotor record of one rotating-frame point, or None without a rotor.

    Parameters
    ----------
    values : mapping
        The point's SU2 deck.
    reference : mapping
        The point's REF file.
    coefficients : mapping
        The CFx, CFy, CFz, CMx, CMy and CMz of the monitored surface.
    periodic : PeriodicState, optional
        The row's symmetry; without it the monitored surface is the whole rotor.
    condition : mapping, optional
        The point's FLIGHT_CONDITION; its ``ADVANCE_RATIO`` or ``RPM`` gives the speed, else
        the deck's ``ROTATION_RATE`` does.

    Returns
    -------
    dict or None
        None when the REF file has no rotor block, the deck turns nothing, or the history
        lacks a coefficient. Otherwise the alias, the loads the adapter supplies, the
        condition cells and the rotor row :func:`cfdbender.core.rotor.rotor_row` forms.
    """
    rotor = rotor_block(reference)
    if (
        rotor is None
        or _upper(values, "GRID_MOVEMENT") not in {"ROTATING_FRAME", "RIGID_MOTION"}
        or "ROTATION_RATE" not in values
        or not all(key in coefficients for key in (*FORCE_KEYS, *MOMENT_KEYS))
    ):
        return None
    loads = rotor_loads(
        values,
        [coefficients[key] for key in FORCE_KEYS],
        [coefficients[key] for key in MOMENT_KEYS],
    )
    if loads is None or loads.speed_m_s <= 0.0:
        return None
    stated = dict(condition or {})
    if "ADVANCE_RATIO" in stated or "RPM" in stated:
        rps, _ = rotor_speed(rotor, stated, loads.speed_m_s)
    else:
        rate = _stated_vector(values["ROTATION_RATE"], "ROTATION_RATE")
        rps = math.sqrt(sum(c * c for c in rate)) / (2.0 * math.pi)
    copies = 1 if periodic is None else periodic.multiplier
    return {
        "rotor": rotor.alias,
        "blades": rotor.blades,
        "periodic_copies": 1 if periodic is None else periodic.copies,
        "symmetry_loads": "SECTOR"
        if periodic is not None and periodic.multiplier == 1 and periodic.symmetry == "PERIODIC"
        else "WHOLE",
        "loads": loads.to_dict(),
        "condition": condition_cells(values),
        "absent": list(ABSENT_CONDITION_COLUMNS),
        "row": rotor_row(rotor, loads, rps=rps, copies=copies),
    }

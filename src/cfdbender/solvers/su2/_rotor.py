"""What the SU2 adapter supplies for a rotor: force and moment vectors, speed and density.

SU2 writes force and moment coefficients over ``q A`` and ``q A L``, ``q`` being the dynamic
pressure of the speed it scales by (``MACH_MOTION`` times the speed of sound when the deck
states ``MACH_MOTION``). The adapter turns them back into newtons and newton meters here, so
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


def rotor_markers(reference: Mapping[str, object]) -> tuple[str, ...]:
    """Return the markers of the REF file's rotor: ``families_blades``, then ``families_general``.

    The rotor of a mesh is its blades and its spinner, the families its declaration names; an
    SU2 marker is named by its family. Empty without a rotor or without those lists.
    """
    block = rotor_block(reference)
    table = reference.get(block.name) if block is not None else None
    if not isinstance(table, Mapping):
        return ()
    names = [
        str(name)
        for key in ("families_blades", "families_general")
        for name in (table.get(key) or [])
    ]
    return tuple(dict.fromkeys(names))


def rotor_surfaces(
    reference: Mapping[str, object],
    walls: Sequence[str],
    surfaces: Sequence[str],
    *,
    multizone: bool,
) -> tuple[tuple[str, ...] | None, str | None]:
    """Return the markers whose own loads are the rotor's, or why no rotor row can be formed.

    SU2's total coefficients sum every monitored marker, and every wall is monitored, so the
    totals of an isolated propeller hold its nacelle and those of a zone every wall of the
    zone. The rotor's loads are the sum of the per-surface loads of its markers
    (:func:`rotor_markers`). The totals stand for the rotor only where they are the rotor's
    alone: no monitored wall is outside the rotor's markers in the deck. On a multizone point
    the turning zone names the rotor where its declaration names none of the zone's walls (or
    the history holds no per-surface load): the rotor is then the turning zone's total, every
    wall of that zone.

    Parameters
    ----------
    reference : mapping
        The point's REF file.
    walls : sequence of str
        The wall markers the deck monitors.
    surfaces : sequence of str
        The markers the history holds per-surface loads of (the turning zone's on a multizone
        point).
    multizone : bool
        Whether the point has several zones.

    Returns
    -------
    tuple
        ``(markers, None)`` to sum those markers' own loads, ``(None, None)`` for the totals,
        ``(None, reason)`` when neither is the rotor's.
    """
    declared = rotor_markers(reference)
    if multizone and not surfaces:
        return None, None
    zone_walls = list(surfaces) if multizone else list(dict.fromkeys([*walls, *surfaces]))
    others = [name for name in zone_walls if name not in declared]
    own = [name for name in declared if name in zone_walls]
    if not others or (multizone and not own):
        return None, None
    if not declared:
        return None, (
            "the rotor declaration names no families_blades or families_general and the deck "
            f"monitors {', '.join(zone_walls)}, so the totals are not the rotor's alone"
        )
    if not own:
        return None, (
            f"no marker of the rotor ({', '.join(declared)}) is a wall the deck monitors "
            f"({', '.join(zone_walls)}), so the totals are not the rotor's alone"
        )
    if missing := [name for name in own if name not in surfaces]:
        return None, (
            f"the history holds no per-surface loads of the rotor marker(s) {', '.join(missing)} "
            f"(keep AERO_COEFF_SURF in HISTORY_OUTPUT) and its totals also hold {', '.join(others)}"
        )
    return tuple(own), None


def surface_markers(headers: Sequence[str]) -> list[str]:
    """Return the markers a history holds the six per-surface loads of, in column order.

    ``CFx(blade)`` ... ``CMz(blade)``: a single-zone history, or a multizone one whose turning
    zone's columns are named as a single zone's.
    """
    names = set(headers)
    found = [
        header[4:-1] for header in headers if header.startswith("CFx(") and header.endswith(")")
    ]
    return [
        marker
        for marker in dict.fromkeys(found)
        if all(f"{key}({marker})" in names for key in (*FORCE_KEYS, *MOMENT_KEYS))
    ]


def marker_loads(row: Mapping[str, str], markers: Sequence[str]) -> dict[str, float]:
    """Return the six load coefficients of ``markers`` summed over one history row."""
    found: dict[str, float] = {}
    for key in (*FORCE_KEYS, *MOMENT_KEYS):
        values = [float(row[f"{key}({marker})"]) for marker in markers]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"history: the rotor's {key} is not finite")
        found[key] = sum(values)
    return found


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
    if loads is None:
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

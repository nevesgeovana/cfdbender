"""The rotor of a steady rotating-frame SU2 point: its reference declaration and its loads.

The REF file declares a rotor as its FlightStream reference file does, a table of its own
with ``kind = "rotor"`` (any table name)::

    [ROTOR]
    kind = "rotor"
    x_m = 0.0                  # hub centre in the mesh frame, m (default 0)
    y_m = 0.0
    z_m = 0.0
    axis = "X"                 # the declared shaft; the SU2 adapter reads X
    rpm_sign = -1              # 1: turns right-handed about the declared axis; -1: left-handed
    diameter_m = 0.4064        # rotor diameter D, m
    families_blades = ["Blade1", "Blade2", "Blade3", "Blade4", "Blade5", "Blade6"]

``families_general``, ``alias`` and ``blade1`` to ``bladeN`` are accepted and not read.
The blade count is the length of ``families_blades``, as on its FlightStream side; SU2 has no
use for it beyond the table column and a consistency check.

The flow runs along the declared axis, so thrust is positive against it: with ``axis = "X"``
the free stream is along +x and thrust is positive toward -x. With ``n`` the rotation speed
in rev/s, ``D`` the diameter and ``V`` the free-stream speed, ``J = V / (n D)``. The rotation
vector the adapter writes is ``rpm_sign 2 pi n axis_hat``.

The whole-rotor loads of a point solved on one of ``m`` periodic sectors are the sum over the
``m`` copies of the sector's force and moment, copy ``j`` turned about the axis through the
rotor origin by ``360 j / m`` degrees. With ``F`` the sector's force and ``M_o`` its moment
about the rotor origin (the moment about the reference point plus ``(r_ref - origin) x F``)::

    F_w = sum_j R_j F        M_w = sum_j R_j M_o
    T = F_w . thrust_hat     Q = -M_w . axis_hat
    CT = T / (rho n^2 D^4)   CQ = Q / (rho n^2 D^5)   CP = 2 pi CQ sign(n)   eta = J CT / CP

``thrust_hat`` is minus the axis and ``n`` is signed by ``rpm_sign``. ``Q`` is the torque the
shaft supplies about the fixed axis, signed about it, so ``CP`` is positive for a propeller
that absorbs power whichever way it turns. The components of ``F_w`` and ``M_w`` normal to
the axis cancel for equal copies and are reported as magnitudes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from cfdbender.core.refusals import warn_or_refuse

Vector = tuple[float, float, float]

#: The keys of a rotor declaration the adapter reads, and the unread keys of a FlightStream one.
READ_KEYS = ("kind", "x_m", "y_m", "z_m", "axis", "rpm_sign", "diameter_m", "families_blades")
UNREAD_KEYS = ("alias", "families_general")


@dataclass(frozen=True)
class RotorBlock:
    """One validated rotor declaration of the REF file.

    Attributes
    ----------
    name : str
        The declaration's table name, ``ROTOR`` in ``[ROTOR]``.
    diameter_m : float
        The rotor diameter D.
    blades : int or None
        The length of ``families_blades``, None when the table has none.
    origin : tuple of float
        The hub centre in the mesh frame.
    axis : tuple of float
        The unit vector of the declared axis; the flow runs along it and thrust against it.
    rpm_sign : int
        1 when the rotor turns right-handed about the axis, -1 when left-handed.
    """

    name: str
    diameter_m: float
    blades: int | None
    origin: Vector
    axis: Vector
    rpm_sign: int

    @property
    def thrust_direction(self) -> Vector:
        """The unit vector of positive thrust, against the declared axis."""
        return (-self.axis[0] + 0.0, -self.axis[1] + 0.0, -self.axis[2] + 0.0)


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"REF file {label} must be a finite number")
    return float(value)


def rotor_block(reference: Mapping[str, object]) -> RotorBlock | None:
    """Return the REF file's rotor declaration, or None when it holds none.

    Parameters
    ----------
    reference : mapping
        The parsed REF file.

    Returns
    -------
    RotorBlock or None
        The validated declaration: the one table whose ``kind`` is ``"rotor"``.

    Raises
    ------
    ValueError
        More than one rotor is declared, a required key is missing, a key is unknown or of
        the wrong type, the diameter is not positive, or the axis is not X.
    """
    found = {
        key: value
        for key, value in reference.items()
        if isinstance(value, dict) and value.get("kind") == "rotor"
    }
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"REF file declares {len(found)} rotors ({', '.join(sorted(found))}): multi-rotor "
            "rotating-frame runs are not supported yet; declare one"
        )
    ((name, block),) = found.items()
    unknown = sorted(
        key
        for key in block
        if key not in (*READ_KEYS, *UNREAD_KEYS) and re.fullmatch(r"blade[0-9]+", key) is None
    )
    if unknown:
        raise ValueError(
            f"REF file {name} has unknown keys: {', '.join(unknown)} "
            f"(the keys are {', '.join((*READ_KEYS, *UNREAD_KEYS))} and blade1 to bladeN)"
        )
    for key in ("diameter_m", "axis"):
        if key not in block:
            raise ValueError(f"REF file {name} needs {key}")
    diameter = _real(block["diameter_m"], f"{name}.diameter_m")
    if diameter <= 0.0:
        raise ValueError(f"REF file {name}.diameter_m must be positive")
    axis = block["axis"]
    if not isinstance(axis, str) or axis.strip().upper() not in ("X", "Y", "Z"):
        raise ValueError(f'REF file {name}.axis must be "X", "Y" or "Z"')
    if axis.strip().upper() != "X":
        raise ValueError(
            f"REF file {name}.axis is {axis!r}: the SU2 adapter runs the rotor in an axial "
            'flow along +x, so it reads axis = "X" only'
        )
    sign = block.get("rpm_sign", 1)
    if type(sign) is not int or sign not in (1, -1):
        raise ValueError(f"REF file {name}.rpm_sign must be 1 or -1")
    blades: int | None = None
    if "families_blades" in block:
        listed = block["families_blades"]
        if (
            not isinstance(listed, list)
            or not listed
            or not all(isinstance(i, str) for i in listed)
        ):
            raise ValueError(f"REF file {name}.families_blades must be a list of family names")
        blades = len(listed)
    origin = (
        _real(block.get("x_m", 0.0), f"{name}.x_m"),
        _real(block.get("y_m", 0.0), f"{name}.y_m"),
        _real(block.get("z_m", 0.0), f"{name}.z_m"),
    )
    return RotorBlock(name, diameter, blades, origin, (1.0, 0.0, 0.0), sign)


def rotation_rate(block: RotorBlock, advance_ratio: float, velocity: float) -> tuple[Vector, float]:
    """Return the rotation vector in rad/s and the speed n in rev/s of an advance ratio.

    Parameters
    ----------
    block : RotorBlock
        The rotor.
    advance_ratio : float
        J = V / (n D), positive.
    velocity : float
        The free-stream speed V in m/s, positive.

    Returns
    -------
    tuple
        ``rpm_sign 2 pi n axis_hat`` and ``n = V / (J D)``.
    """
    n = velocity / (advance_ratio * block.diameter_m)
    rate = 2.0 * math.pi * n * block.rpm_sign
    x, y, z = (rate * c + 0.0 for c in block.axis)
    return (x, y, z), n


def _cross(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def turn(vector: Sequence[float], axis: Sequence[float], angle: float) -> Vector:
    """Return a vector turned by ``angle`` radians right-handed about a unit axis (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    cross = _cross(axis, vector)
    along = _dot(axis, vector) * (1.0 - c)
    x, y, z = (v * c + k * s + a * along for v, k, a in zip(vector, cross, axis, strict=True))
    return (x, y, z)


def periodic_sum(
    force: Sequence[float],
    moment: Sequence[float],
    axis: Sequence[float],
    copies: int,
) -> tuple[Vector, Vector]:
    """Return the force and moment of ``copies`` periodic copies of a sector, summed.

    Parameters
    ----------
    force, moment : sequence of float
        The sector's force and its moment about a point on the axis.
    axis : sequence of float
        The unit vector of the rotation axis through that point.
    copies : int
        The number of identical sectors that make the whole; copy ``j`` is the sector turned
        by ``360 j / copies`` degrees about the axis.

    Returns
    -------
    tuple
        The summed force and the summed moment about the same point.
    """
    f = [0.0, 0.0, 0.0]
    m = [0.0, 0.0, 0.0]
    for j in range(copies):
        angle = 2.0 * math.pi * j / copies
        for total, vector in ((f, force), (m, moment)):
            turned = turn(vector, axis, angle)
            for i in range(3):
                total[i] += turned[i]
    return (f[0], f[1], f[2]), (m[0], m[1], m[2])


NOT_APPLICABLE = "NA"


def rotor_coefficients(
    *,
    thrust_n: float,
    torque_nm: float,
    rps: float,
    diameter_m: float,
    density_kg_m3: float,
    speed_m_s: float,
) -> dict[str, float | str]:
    """Return the standard coefficients of one rotor.

    The definitions are those of ``docs/post-processing-definitions.md`` of pyflightstream.
    ``n`` is signed revolutions per second and ``D`` the diameter; its magnitude normalises
    the coefficients and its sign enters power only. ``CQ`` keeps the signed torque about the
    fixed rotor axis, and reversing the rotation alone does not change ``J``::

        J   = V / (|n| D)
        CT  = T / (rho n^2 D^4)
        CQ  = Q / (rho n^2 D^5)
        CP  = 2 pi CQ sign(n)
        ETA = J CT / CP

    ``ETA`` is NA on a static point (V = 0) and where CP is zero.

    Parameters
    ----------
    thrust_n : float
        Thrust along the shaft in N.
    torque_nm : float
        Torque about the shaft in N m, signed about the fixed rotor axis.
    rps : float
        Signed revolutions per second; the sign is the sense of rotation.
    diameter_m, density_kg_m3, speed_m_s : float
        Rotor diameter in m, air density in kg/m3 and free-stream speed in m/s.

    Returns
    -------
    dict of str to float or str
        ``J``, ``CT``, ``CQ``, ``CP`` and ``ETA``.

    Raises
    ------
    ValueError
        The rotor is not turning, or the diameter or density is not positive, or the speed or
        diameter is infinite: every coefficient divides by the square of the speed.
    """
    if rps == 0:
        raise ValueError(
            "rps is zero: the rotor is not turning, so it has no thrust or torque coefficient: "
            "every one of them divides by the square of its speed"
        )
    if diameter_m <= 0:
        raise ValueError("diameter_m must be positive to define rotor coefficients")
    if math.isinf(rps) or math.isinf(diameter_m):
        raise ValueError("rps and diameter_m must be finite to define rotor coefficients")
    if not density_kg_m3 > 0:
        raise ValueError(
            f"density_kg_m3 must be positive to define rotor coefficients, got {density_kg_m3!r}"
        )
    rate = abs(rps)
    sense = 1.0 if rps > 0 else -1.0
    advance_ratio = speed_m_s / (rate * diameter_m)
    thrust_coefficient = thrust_n / (density_kg_m3 * rate**2 * diameter_m**4)
    torque_coefficient = torque_nm / (density_kg_m3 * rate**2 * diameter_m**5)
    power_coefficient = 2.0 * math.pi * torque_coefficient * sense
    values: dict[str, float | str] = {
        "J": advance_ratio,
        "CT": thrust_coefficient,
        "CQ": torque_coefficient,
        "CP": power_coefficient,
    }
    if speed_m_s == 0 or power_coefficient == 0:
        values["ETA"] = NOT_APPLICABLE
    else:
        values["ETA"] = advance_ratio * thrust_coefficient / power_coefficient
    return values


def rotor_quantities(
    block: RotorBlock,
    *,
    rotation: Sequence[float],
    velocity: float,
    density: float | None,
    dynamic_pressure: float | None,
    area: float,
    length: float,
    moment_origin: Sequence[float],
    force_coefficients: Sequence[float],
    moment_coefficients: Sequence[float],
    copies: int = 1,
) -> dict[str, float | None]:
    """Return the quantities of the whole rotor from the sector's coefficients.

    Parameters
    ----------
    block : RotorBlock
        The rotor.
    rotation : sequence of float
        The rotation vector the solver ran with, rad/s.
    velocity : float
        The free-stream speed V of the advance ratio, m/s.
    density, dynamic_pressure : float or None
        The free-stream density and the dynamic pressure SU2 divides by; None when unknown.
    area, length : float
        The reference area and length of the coefficients.
    moment_origin : sequence of float
        The point the moments are about.
    force_coefficients, moment_coefficients : sequence of float
        CFx, CFy, CFz and CMx, CMy, CMz of the monitored surface.
    copies : int, optional
        The periodic copies summed (1: the monitored surface is the whole rotor, or the
        loads are reported for the sector as computed).

    Returns
    -------
    dict
        n_rps, rpm, advance_ratio, thrust_n, torque_nm, ct, cq, cp, efficiency, and the
        magnitudes of the summed force and moment normal to the axis, ``in_plane_force_n``
        and ``in_plane_moment_nm``. A value that needs the density or the dynamic pressure
        is None when either is unknown, and the efficiency is None when CP is zero.
    """
    omega = math.sqrt(_dot(rotation, rotation))
    n = omega / (2.0 * math.pi)
    out: dict[str, float | None] = {
        "n_rps": n,
        "rpm": 60.0 * n,
        "advance_ratio": velocity / (n * block.diameter_m) if n > 0.0 else None,
        "thrust_n": None,
        "torque_nm": None,
        "in_plane_force_n": None,
        "in_plane_moment_nm": None,
        "ct": None,
        "cq": None,
        "cp": None,
        "efficiency": None,
    }
    if dynamic_pressure is None or dynamic_pressure <= 0.0 or omega == 0.0:
        return out
    force = tuple(dynamic_pressure * area * c for c in force_coefficients)
    moment = tuple(dynamic_pressure * area * length * c for c in moment_coefficients)
    arm = tuple(r - o for r, o in zip(moment_origin, block.origin, strict=True))
    about_origin = tuple(m + c for m, c in zip(moment, _cross(arm, force), strict=True))
    whole_force, whole_moment = periodic_sum(force, about_origin, block.axis, copies)
    thrust = _dot(whole_force, block.thrust_direction)
    torque = -_dot(whole_moment, block.axis)
    out["thrust_n"], out["torque_nm"] = thrust, torque
    for key, vector in (("in_plane_force_n", whole_force), ("in_plane_moment_nm", whole_moment)):
        along = _dot(vector, block.axis)
        out[key] = math.sqrt(
            sum((v - along * a) ** 2 for v, a in zip(vector, block.axis, strict=True))
        )
    if density is None or density <= 0.0:
        return out
    got = rotor_coefficients(
        thrust_n=thrust,
        torque_nm=torque,
        rps=block.rpm_sign * n,
        diameter_m=block.diameter_m,
        density_kg_m3=density,
        speed_m_s=velocity,
    )
    efficiency = got["ETA"]
    out.update(
        ct=cast(float, got["CT"]),
        cq=cast(float, got["CQ"]),
        cp=cast(float, got["CP"]),
        efficiency=None if isinstance(efficiency, str) else efficiency,
    )
    return out
    d = block.diameter_m
    ct = thrust / (density * n * n * d**4)
    cq = torque / (density * n * n * d**5)
    cp = 2.0 * math.pi * cq
    advance = velocity / (n * d)
    out.update(ct=ct, cq=cq, cp=cp, efficiency=advance * ct / cp if cp != 0.0 else None)
    return out


#: The row controls of the symmetry and the time-accurate run types: never solver keys.
ROW_CONTROLS = (
    "SYMMETRY",
    "SYMMETRY_LOADS",
    "symmetry_loads",
    "PERIODIC_COPIES",
    "DELTA_THETA",
    "REVOLUTIONS",
)
TIME_ACCURATE = ("DELTA_THETA", "REVOLUTIONS")


@dataclass(frozen=True)
class PeriodicState:
    """What the row states about the symmetry of the meshed model.

    Attributes
    ----------
    symmetry : str
        ``NONE`` or ``PERIODIC``: what was meshed.
    copies : int
        The periodic copies the sector stands for (1 without symmetry).
    loads : bool
        Whether the reported loads are the whole rotor's (the default) or the sector's.
    """

    symmetry: str
    copies: int
    loads: bool

    @property
    def multiplier(self) -> int:
        """The copies summed into the whole-rotor loads: ``copies`` or 1."""
        return self.copies if self.symmetry == "PERIODIC" and self.loads else 1


def periodic_state(setup: Mapping[str, object]) -> PeriodicState:
    """Read SYMMETRY, SYMMETRY_LOADS and PERIODIC_COPIES as the FlightStream side reads them.

    ``SYMMETRY`` says what was meshed (``NONE``, the default, or ``PERIODIC``);
    ``PERIODIC_COPIES`` is required with ``PERIODIC`` and refused otherwise;
    ``SYMMETRY_LOADS`` (the row's, else the setup's ``symmetry_loads``, else true) says whether
    the loads are the whole rotor's or the sector's. Warns and skips ``DELTA_THETA`` and
    ``REVOLUTIONS``, which belong to the time-accurate run types.

    Raises
    ------
    ValueError
        A symmetry other than NONE or PERIODIC, PERIODIC without a positive whole number of
        copies, copies without PERIODIC, or a non-boolean SYMMETRY_LOADS.
    """
    for key in TIME_ACCURATE:
        if key in setup:
            warn_or_refuse(
                f"{key} is skipped: it belongs to a time-accurate run type that the SU2 "
                "adapter does not offer yet",
                UserWarning,
                stacklevel=3,
            )
    symmetry = str(setup.get("SYMMETRY", "NONE")).strip().upper()
    if symmetry not in {"NONE", "PERIODIC"}:
        raise ValueError(
            f"SYMMETRY {symmetry} is not read by the SU2 adapter: it reads NONE or PERIODIC "
            "(a mirror plane is a marker of the setup)"
        )
    copies = setup.get("PERIODIC_COPIES")
    if symmetry != "PERIODIC":
        if copies is not None:
            raise ValueError("PERIODIC_COPIES needs SYMMETRY: PERIODIC and is refused otherwise")
        count = 1
    else:
        if copies is None:
            raise ValueError(
                "SYMMETRY: PERIODIC needs PERIODIC_COPIES, the copies the sector stands for"
            )
        if (
            isinstance(copies, bool)
            or not isinstance(copies, int | float)
            or copies != int(copies)
            or copies < 1
        ):
            raise ValueError("PERIODIC_COPIES must be a positive whole number")
        count = int(copies)
    stated = setup.get("SYMMETRY_LOADS", setup.get("symmetry_loads", True))
    if isinstance(stated, str) and stated.strip().lower() in {"true", "false"}:
        stated = stated.strip().lower() == "true"
    if not isinstance(stated, bool):
        raise ValueError("SYMMETRY_LOADS must be true or false")
    return PeriodicState(symmetry, count, stated)

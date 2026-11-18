"""The rotor of a steady rotating-frame SU2 point: its reference block and its coefficients.

The REF file may hold one rotor block::

    [rotors.PROP]
    diameter_m = 0.4064         # rotor diameter D, m
    blades = 6                  # blades of the whole rotor
    monitored_blades = 1        # blades inside the monitored surface (default: blades)
    origin = [0.0, 0.0, 0.0]    # hub centre in the mesh frame, m
    shaft = [-1.0, 0.0, 0.0]    # direction of positive thrust
    rpm_sign = 1                # +1: the rotor turns right-handed about the shaft

With ``n`` the rotation speed in rev/s, ``D`` the diameter and ``V`` the free-stream speed,
the advance ratio is ``J = V / (n D)``. The rotation vector the adapter writes is
``rpm_sign 2 pi n shaft / |shaft|`` (rad/s), so a shaft toward minus x with ``rpm_sign`` 1
gives a negative x rotation rate.

The rotor quantities of a point come from the solver's force and moment coefficients of the
monitored surface, with ``q = 0.5 rho V_q^2`` the dynamic pressure SU2 divides by, ``A`` and
``L`` the reference area and length and ``k = blades / monitored_blades``::

    F = q A (CFx, CFy, CFz)                 M_o = q A L (CMx, CMy, CMz) + (r_ref - origin) x F
    T = k F . shaft_hat                     Q = -k M_o . omega_hat
    CT = T / (rho n^2 D^4)    CQ = Q / (rho n^2 D^5)    CP = 2 pi CQ    eta = J CT / CP

``omega_hat`` is the unit vector of the rotation, so Q is the torque the shaft supplies and is
positive for a propeller that absorbs power. Thrust is positive along the shaft.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Vector = tuple[float, float, float]

#: The keys of a rotor block.
BLOCK_KEYS = ("diameter_m", "blades", "monitored_blades", "origin", "shaft", "rpm_sign")


@dataclass(frozen=True)
class RotorBlock:
    """One validated rotor block of the REF file.

    Attributes
    ----------
    name : str
        The block's name, ``PROP`` in ``[rotors.PROP]``.
    diameter_m : float
        The rotor diameter D.
    blades, monitored_blades : int
        The blades of the whole rotor and the blades inside the monitored surface.
    origin : tuple of float
        The hub centre in the mesh frame.
    shaft : tuple of float
        The unit vector of positive thrust.
    rpm_sign : int
        1 when the rotor turns right-handed about the shaft, -1 when left-handed.
    """

    name: str
    diameter_m: float
    blades: int
    monitored_blades: int
    origin: Vector
    shaft: Vector
    rpm_sign: int


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"REF file {label} must be a finite number")
    return float(value)


def _vector(value: object, label: str) -> Vector:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"REF file {label} needs three finite coordinates")
    x, y, z = (_real(item, label) for item in value)
    return (x, y, z)


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"REF file {label} must be a positive integer")
    return value


def rotor_block(reference: Mapping[str, object]) -> RotorBlock | None:
    """Return the REF file's rotor block, or None when it holds none.

    Parameters
    ----------
    reference : mapping
        The parsed REF file.

    Returns
    -------
    RotorBlock or None
        The validated block.

    Raises
    ------
    ValueError
        The block is not a table of tables, names more than one rotor, lacks a key, holds an
        unknown key, a wrong type, a non-positive diameter or a zero shaft.
    """
    table = reference.get("rotors")
    if table is None:
        return None
    if not isinstance(table, dict) or not all(isinstance(item, dict) for item in table.values()):
        raise ValueError("REF file rotors must hold one table per rotor, [rotors.<NAME>]")
    if not table:
        return None
    if len(table) > 1:
        raise ValueError(
            f"REF file rotors names {len(table)} rotors ({', '.join(sorted(table))}): "
            "multi-rotor rotating-frame runs are not supported yet; state one rotor"
        )
    ((name, block),) = table.items()
    label = f"rotors.{name}"
    if unknown := sorted(set(block) - set(BLOCK_KEYS)):
        raise ValueError(
            f"REF file {label} has unknown keys: {', '.join(unknown)} "
            f"(the keys are {', '.join(BLOCK_KEYS)})"
        )
    for key in ("diameter_m", "blades", "origin", "shaft"):
        if key not in block:
            raise ValueError(f"REF file {label} needs {key}")
    diameter = _real(block["diameter_m"], f"{label}.diameter_m")
    if diameter <= 0.0:
        raise ValueError(f"REF file {label}.diameter_m must be positive")
    blades = _count(block["blades"], f"{label}.blades")
    monitored = _count(block.get("monitored_blades", blades), f"{label}.monitored_blades")
    if monitored > blades:
        raise ValueError(f"REF file {label}.monitored_blades cannot exceed blades")
    shaft = _vector(block["shaft"], f"{label}.shaft")
    length = math.sqrt(sum(c * c for c in shaft))
    if length == 0.0:
        raise ValueError(f"REF file {label}.shaft must not be the zero vector")
    sign = block.get("rpm_sign", 1)
    if type(sign) is not int or sign not in (1, -1):
        raise ValueError(f"REF file {label}.rpm_sign must be 1 or -1")
    unit = (shaft[0] / length, shaft[1] / length, shaft[2] / length)
    return RotorBlock(
        name, diameter, blades, monitored, _vector(block["origin"], f"{label}.origin"), unit, sign
    )


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
        ``rpm_sign 2 pi n shaft_hat`` and ``n = V / (J D)``.
    """
    n = velocity / (advance_ratio * block.diameter_m)
    rate = 2.0 * math.pi * n * block.rpm_sign
    x, y, z = (rate * c + 0.0 for c in block.shaft)
    return (x, y, z), n


def _cross(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


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
) -> dict[str, float | None]:
    """Return the quantities of the whole rotor from the monitored surface's coefficients.

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

    Returns
    -------
    dict
        n_rps, rpm, advance_ratio, thrust_n, torque_nm, ct, cq, cp, efficiency. A value
        that needs the density or the dynamic pressure is None when either is unknown, and
        the efficiency is None when CP is zero.
    """
    omega = math.sqrt(_dot(rotation, rotation))
    n = omega / (2.0 * math.pi)
    scale = block.blades / block.monitored_blades
    out: dict[str, float | None] = {
        "n_rps": n,
        "rpm": 60.0 * n,
        "advance_ratio": velocity / (n * block.diameter_m) if n > 0.0 else None,
        "thrust_n": None,
        "torque_nm": None,
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
    thrust = scale * _dot(force, block.shaft)
    torque = -scale * _dot(about_origin, tuple(w / omega for w in rotation))
    out["thrust_n"], out["torque_nm"] = thrust, torque
    if density is None or density <= 0.0:
        return out
    d = block.diameter_m
    ct = thrust / (density * n * n * d**4)
    cq = torque / (density * n * n * d**5)
    cp = 2.0 * math.pi * cq
    advance = velocity / (n * d)
    out.update(ct=ct, cq=cq, cp=cp, efficiency=advance * ct / cp if cp != 0.0 else None)
    return out

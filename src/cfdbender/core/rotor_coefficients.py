"""The rotor coefficients of a family or group inside a rotor (the RCOEF columns).

``F_R`` and ``M_R`` are the force and the moment about the rotor hub in the rotor frame
R<id> of :mod:`cfdbender.core.axes`; ``n`` is the rate in rev/s signed about ``x_R``
(``sense RPM / 60``), ``D`` the diameter, ``A = pi D^2 / 4``, ``q = 0.5 RHO VINF^2`` and
``J = VINF / (|n| D)``::

    T = F_R,x    Y = F_R,y    Z = -F_R,z    Q = M_R,x    M = M_R,y    N = M_R,z
    P  = 2 pi n Q / 1000                       (kW)
    CT = T / (RHO n^2 D^4)    CQ = Q / (RHO n^2 D^5)    CP = 2 pi CQ sign(n)
    CFZ, CFY = Z, Y / (RHO n^2 D^4)            CMZ, CMY = N, M / (RHO n^2 D^5)
    ETA = J CT / CP
    TC, YC, ZC = T, Y, Z / (q A)               QC, NC, MC = Q, N, M / (q A D)
    TCA, YCA, ZCA = T, Y, Z / (q SREF)         QCA, NCA = Q, N / (q SREF BREF)
    MCA = M / (q SREF CREF)
    CT<f> = F_f,x / (RHO n^2 D^4)   CQ<f> = M_f,x / (RHO n^2 D^5)   ETA<f> = J CT<f> / CP

Thrust is positive forward along ``x_R``; ``CT<own frame>`` repeats ``CT``. ``ETAW``, for a
shaft square to the body's x, counts the wind-axis force along the stream.

THE NA RULES. A static point (``VINF = 0``) states no ``ETA``, no ``ETA<f>`` and none of
``TC`` to ``MCA``. A rotor not turning, or of unknown speed, states no coefficient over
``n^2`` and no ``ETA``; ``P`` is 0 at ``RPM = 0`` and NA when the speed is unknown. A load
component that is not known (NaN) gives NA wherever it enters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from cfdbender.core.axes import NOT_APPLICABLE, Matrix, RotorFrame, turn

#: The RCOEF columns in the rotor's own frame, in file order: name, unit, quantity, sign.
ROTOR_FRAME_COLUMNS: tuple[tuple[str, str, str, str | None], ...] = (
    ("T", "N", "thrust", "positive forward"),
    ("Q", "N m", "torque about the hub", "right-hand about x_R"),
    ("P", "kW", "shaft power", None),
    ("CT", "-", "thrust coefficient", "positive forward"),
    ("CQ", "-", "torque coefficient", "right-hand about x_R"),
    ("CP", "-", "power coefficient", None),
    ("CFZ", "-", "normal force coefficient", "positive up"),
    ("CFY", "-", "side force coefficient", "positive right"),
    ("CMZ", "-", "moment coefficient about z_R", "right-hand"),
    ("CMY", "-", "moment coefficient about y_R", "right-hand"),
    ("ETA", "-", "propulsive efficiency", None),
    ("TC", "-", "thrust over disc loading", "positive forward"),
    ("YC", "-", "side force over disc loading", "positive right"),
    ("ZC", "-", "normal force over disc loading", "positive up"),
    ("QC", "-", "torque over disc loading", "right-hand about x_R"),
    ("NC", "-", "moment about z_R over disc loading", "right-hand"),
    ("MC", "-", "moment about y_R over disc loading", "right-hand"),
    ("TCA", "-", "thrust over airframe loading", "positive forward"),
    ("YCA", "-", "side force over airframe loading", "positive right"),
    ("ZCA", "-", "normal force over airframe loading", "positive up"),
    ("QCA", "-", "torque over airframe loading", "right-hand about x_R"),
    ("NCA", "-", "moment about z_R over airframe loading", "right-hand"),
    ("MCA", "-", "moment about y_R over airframe loading", "right-hand"),
)
#: The RCOEF columns repeated per frame, the frame letter appended.
PER_FRAME_ROTOR_COLUMNS: tuple[str, ...] = ("CT", "CQ", "ETA")


def force_scale(rate_rps: float, density_kg_m3: float, diameter_m: float) -> float:
    """``rho n^2 D^4``: what a rotor force is divided by to give its coefficient."""
    return density_kg_m3 * rate_rps * rate_rps * diameter_m**4


def moment_scale(rate_rps: float, density_kg_m3: float, diameter_m: float) -> float:
    """``rho n^2 D^5``: what a rotor moment is divided by to give its coefficient."""
    return force_scale(rate_rps, density_kg_m3, diameter_m) * diameter_m


def power_scale(rate_rps: float, density_kg_m3: float, diameter_m: float) -> float:
    """``rho n^3 D^5``: what a shaft power in W is divided by to give ``CP``."""
    return moment_scale(rate_rps, density_kg_m3, diameter_m) * rate_rps


def advance_ratio(speed_m_s: float, rate_rps: float, diameter_m: float) -> float:
    """``J = V / (|n| D)``."""
    return speed_m_s / (abs(rate_rps) * diameter_m)


def power_from_torque(torque_coefficient: float, rate_rps: float) -> float:
    """``CP = 2 pi CQ sign(n)``: the sense of rotation enters the power only."""
    return 2.0 * math.pi * torque_coefficient * (1.0 if rate_rps > 0 else -1.0) + 0.0


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
    thrust_coefficient = thrust_n / force_scale(rps, density_kg_m3, diameter_m)
    torque_coefficient = torque_nm / moment_scale(rps, density_kg_m3, diameter_m)
    power_coefficient = power_from_torque(torque_coefficient, rps)
    advance = advance_ratio(speed_m_s, rps, diameter_m)
    values: dict[str, float | str] = {
        "J": advance,
        "CT": thrust_coefficient,
        "CQ": torque_coefficient,
        "CP": power_coefficient,
    }
    if speed_m_s == 0 or power_coefficient == 0:
        values["ETA"] = NOT_APPLICABLE
    else:
        values["ETA"] = advance * thrust_coefficient / power_coefficient
    return values


def _usable(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _over(numerator: float, denominator: float | None) -> float | str:
    if denominator is None or denominator == 0.0 or not math.isfinite(numerator):
        return NOT_APPLICABLE
    value = numerator / denominator
    return value + 0.0 if math.isfinite(value) else NOT_APPLICABLE


def _efficiency(advance: float | None, thrust: float | str, cp: float | str) -> float | str:
    if advance is None or advance == 0.0 or not isinstance(thrust, float):
        return NOT_APPLICABLE
    if not isinstance(cp, float) or cp == 0.0:
        return NOT_APPLICABLE
    return advance * thrust / cp + 0.0


def _loads(
    force: Sequence[float], moment: Sequence[float], matrix: Matrix
) -> tuple[float, float, float, float, float, float]:
    """``(T, Y, Z, Q, M, N)`` of a force and hub moment turned into one frame."""
    f, m = turn(matrix, force), turn(matrix, moment)
    return f[0], f[1], -f[2] + 0.0, m[0], m[1], m[2]


def rotor_frame_coefficients(
    *,
    force_n: Sequence[float],
    moment_hub_nm: Sequence[float],
    frame: RotorFrame,
    rotations: Mapping[str, Matrix],
    rpm: float | None,
    diameter_m: float,
    density_kg_m3: float | None,
    vinf_m_s: float | None,
    sref_m2: float | None,
    cref_m: float | None,
    bref_m: float | None,
) -> dict[str, float | str]:
    """Return the RCOEF values of one family or group inside rotor ``frame``.

    Parameters
    ----------
    force_n, moment_hub_nm : sequence of float
        The force (N) and the moment about the hub (N m) in the geometry frame; an
        unknown component is NaN.
    frame : RotorFrame
        The frame of the rotor the family or group sits in.
    rotations : mapping
        The geometry-to-frame matrices of every frame, as
        :func:`cfdbender.core.axes.frame_rotations` returns them.
    rpm : float or None
        The rotor's speed signed about its declared axis; None when unknown.
    diameter_m : float
        The rotor's diameter.
    density_kg_m3, vinf_m_s : float or None
        The point's density and free-stream speed.
    sref_m2, cref_m, bref_m : float or None
        The airframe reference area, chord and span.

    Returns
    -------
    dict
        The columns of :data:`ROTOR_FRAME_COLUMNS` in order, then ``CT<f>, CQ<f>, ETA<f>``
        per frame of ``rotations``; NA where a value does not exist.
    """
    speed = _usable(rpm)
    rate = None if speed is None else frame.sense * speed / 60.0
    turning = rate is not None and rate != 0.0
    diameter = _usable(diameter_m)
    if diameter is not None and diameter <= 0.0:
        diameter = None
    rho = _usable(density_kg_m3)
    vinf = _usable(vinf_m_s)
    forces = (
        force_scale(rate, rho, diameter)
        if turning and rate is not None and rho is not None and diameter is not None
        else None
    )
    moments = None if forces is None or diameter is None else forces * diameter
    advance = (
        None
        if not turning or vinf is None or rate is None or diameter is None
        else advance_ratio(vinf, rate, diameter)
    )
    pressure = None if vinf is None or vinf == 0.0 or rho is None else 0.5 * rho * vinf * vinf
    area = None if diameter is None else math.pi * diameter**2 / 4.0
    thrust, side, normal, torque, pitching, yawing = _loads(
        force_n, moment_hub_nm, rotations[frame.name]
    )
    values: dict[str, float | str] = {
        "T": _over(thrust, 1.0),
        "Q": _over(torque, 1.0),
        "P": (
            NOT_APPLICABLE
            if rate is None
            else 0.0
            if rate == 0.0
            else _over(2.0 * math.pi * rate * torque, 1000.0)
        ),
        "CT": _over(thrust, forces),
        "CQ": _over(torque, moments),
    }
    cq = values["CQ"]
    values["CP"] = power_from_torque(cq, rate) if isinstance(cq, float) and rate else NOT_APPLICABLE
    values["CFZ"] = _over(normal, forces)
    values["CFY"] = _over(side, forces)
    values["CMZ"] = _over(yawing, moments)
    values["CMY"] = _over(pitching, moments)
    values["ETA"] = _efficiency(advance, values["CT"], values["CP"])
    disc = None if pressure is None or area is None else pressure * area
    sref, cref, bref = (_usable(v) for v in (sref_m2, cref_m, bref_m))
    airframe = None if pressure is None or sref is None else pressure * sref

    def times(base: float | None, length: float | None) -> float | None:
        return None if base is None or length is None else base * length

    values.update(
        {
            "TC": _over(thrust, disc),
            "YC": _over(side, disc),
            "ZC": _over(normal, disc),
            "QC": _over(torque, times(disc, diameter)),
            "NC": _over(yawing, times(disc, diameter)),
            "MC": _over(pitching, times(disc, diameter)),
            "TCA": _over(thrust, airframe),
            "YCA": _over(side, airframe),
            "ZCA": _over(normal, airframe),
            "QCA": _over(torque, times(airframe, bref)),
            "NCA": _over(yawing, times(airframe, bref)),
            "MCA": _over(pitching, times(airframe, cref)),
        }
    )
    for name, matrix in rotations.items():
        along, _side, _normal, about, _pitch, _yaw = _loads(force_n, moment_hub_nm, matrix)
        values[f"CT{name}"] = _over(along, forces)
        values[f"CQ{name}"] = _over(about, moments)
        values[f"ETA{name}"] = _efficiency(advance, values[f"CT{name}"], values["CP"])
    wind = values.get("CTW")
    if isinstance(wind, float) and abs(frame.body_to_rotor[0][0]) < 1e-12:
        values["ETAW"] = _efficiency(advance, -wind + 0.0, values["CP"])
    return values

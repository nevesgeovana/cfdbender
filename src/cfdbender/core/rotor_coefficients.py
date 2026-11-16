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
    force_scale = (
        rho * rate * rate * diameter**4
        if turning and rate is not None and rho is not None and diameter is not None
        else None
    )
    moment_scale = None if force_scale is None or diameter is None else force_scale * diameter
    advance = (
        None
        if not turning or vinf is None or rate is None or diameter is None
        else vinf / (abs(rate) * diameter)
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
        "CT": _over(thrust, force_scale),
        "CQ": _over(torque, moment_scale),
    }
    cq = values["CQ"]
    values["CP"] = (
        2.0 * math.pi * cq * (1.0 if rate and rate > 0 else -1.0) + 0.0
        if isinstance(cq, float)
        else NOT_APPLICABLE
    )
    values["CFZ"] = _over(normal, force_scale)
    values["CFY"] = _over(side, force_scale)
    values["CMZ"] = _over(yawing, moment_scale)
    values["CMY"] = _over(pitching, moment_scale)
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
        values[f"CT{name}"] = _over(along, force_scale)
        values[f"CQ{name}"] = _over(about, moment_scale)
        values[f"ETA{name}"] = _efficiency(advance, values[f"CT{name}"], values["CP"])
    wind = values.get("CTW")
    if isinstance(wind, float) and abs(frame.body_to_rotor[0][0]) < 1e-12:
        values["ETAW"] = _efficiency(advance, -wind + 0.0, values["CP"])
    return values

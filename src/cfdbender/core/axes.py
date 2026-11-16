"""The polar frames W, S, B and R<id>, and the airframe coefficients in each.

Every load cfdbender writes per family or group is stated in the geometry frame E of
the mesh: x aft, y right, z up, about the moment reference point. The polar files turn
it into four kinds of frame, each forward, right, down:

* ``B`` (body): ``C_B/E = C2(pi)``, half a turn about y.
* ``S`` (stability): ``C_S/B = C2(-alpha_s)``.
* ``W`` (wind): ``C_W/B = C3(beta_w) C2(-alpha_s)``.
* ``R<id>`` (rotor): ``C_R/B = C2(theta) C3(psi)``, the body turned by the pitch and toe
  of rotor ``<id>``'s shaft.

``C1``, ``C2`` and ``C3`` are the passive direction-cosine matrices of :func:`dcm` about
x, y and z. ``alpha_s = atan2(w, u)`` and ``beta_w = asin(v)`` are read off the unit
velocity of the aircraft in body axes, ``(cos a cos b, cos a sin b, sin a)``.

In every frame ``CD`` is positive along ``-x`` (aft), ``CL`` along ``-z`` (up) and
``CY`` along ``+y`` (right); moments keep the right-hand rule about the frame's axes.

THE SENSE RULE of a rotor frame makes thrust positive forward: the shaft in body axes is
turned when it points aft, when it points down while square to x (a lift rotor's thrust
is positive up), and when it points left while square to x and z. A rotor whose declared
axis points aft therefore counts thrust, torque and efficiency against its declared axis.

A load component that is not known is NaN. Turning a vector skips the terms whose matrix
entry is exactly zero, so an unknown component only spoils the components it enters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vector = tuple[float, float, float]
NOT_APPLICABLE = "NA"

_Y = (0.0, 1.0, 0.0)
_Z = (0.0, 0.0, 1.0)
IDENTITY: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _matrix(rows: Sequence[Sequence[float]]) -> Matrix:
    a, b, c = (tuple(float(v) + 0.0 for v in row) for row in rows)
    return (a, b, c)  # type: ignore[return-value]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    """Return the product ``left @ right`` of two 3 by 3 matrices."""
    return _matrix(
        [[sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    )


def transpose(matrix: Matrix) -> Matrix:
    """Return the transpose (the inverse) of a direction-cosine matrix."""
    return _matrix([[matrix[j][i] for j in range(3)] for i in range(3)])


def turn(matrix: Matrix, vector: Sequence[float]) -> Vector:
    """Return ``matrix @ vector``, skipping exact zero entries so NaN stays where it enters."""
    out = [
        sum(matrix[i][j] * float(vector[j]) for j in range(3) if matrix[i][j] != 0.0) + 0.0
        for i in range(3)
    ]
    return (out[0], out[1], out[2])


def dcm(axis: Sequence[float], angle_rad: float) -> Matrix:
    """Return the passive direction-cosine matrix of a turn ``angle_rad`` about ``axis``.

    Parameters
    ----------
    axis : sequence of float
        The direction the coordinate system turns about; not null.
    angle_rad : float
        The right-handed angle, in radians.

    Returns
    -------
    tuple
        The 3 by 3 matrix ``C`` such that ``C @ v`` holds ``v`` in the turned system.

    Raises
    ------
    ValueError
        If the axis names no direction.
    """
    n = [float(c) for c in axis]
    length = math.sqrt(sum(c * c for c in n))
    if not math.isfinite(length) or length == 0.0:
        raise ValueError(f"a rotation axis {n} names no direction to turn about")
    n = [c / length for c in n]
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    tilde = ((0.0, -n[2], n[1]), (n[2], 0.0, -n[0]), (-n[1], n[0], 0.0))
    return _matrix(
        [
            [(1.0 - c) * n[i] * n[j] + (c if i == j else 0.0) - s * tilde[i][j] for j in range(3)]
            for i in range(3)
        ]
    )


#: The geometry frame (x aft, y right, z up) to the body frame (forward, right, down).
EXPORT_TO_BODY: Matrix = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))


def wind_angles(alpha_deg: float, beta_deg: float) -> tuple[float, float]:
    """Return ``(alpha_s, beta_w)`` in radians, read off the aircraft's body-axis velocity."""
    a, b = math.radians(float(alpha_deg)), math.radians(float(beta_deg))
    u, v, w = math.cos(a) * math.cos(b), math.cos(a) * math.sin(b), math.sin(a)
    return math.atan2(w, u), math.asin(max(-1.0, min(1.0, v)))


def body_to_stability(alpha_deg: float, beta_deg: float = 0.0) -> Matrix:
    """Return ``C_S/B = C2(-alpha_s)``."""
    return dcm(_Y, -wind_angles(alpha_deg, beta_deg)[0])


def body_to_wind(alpha_deg: float, beta_deg: float = 0.0) -> Matrix:
    """Return ``C_W/B = C3(beta_w) C2(-alpha_s)``."""
    alpha_s, beta_w = wind_angles(alpha_deg, beta_deg)
    return multiply(dcm(_Z, beta_w), dcm(_Y, -alpha_s))


@dataclass(frozen=True)
class RotorFrame:
    """The frame R<id> of one rotor: the body turned by the pitch and toe of its shaft.

    Attributes
    ----------
    rotor_id : int
        The rotor's id; the frame is named ``R<id>``.
    pitch_deg, toe_deg : float
        ``theta = asin(-a_z)`` (positive nose up) and ``psi = atan2(a_y, a_x)`` (positive
        toward +y; 0 for a vertical shaft).
    sense : int
        ``+1`` when ``x_R`` has the declared axis's sense, ``-1`` when the sense rule turned it.
    declared_aft : bool
        Whether the declared axis points aft.
    body_to_rotor : tuple
        ``C_R/B``.
    """

    rotor_id: int
    pitch_deg: float
    toe_deg: float
    sense: int
    declared_aft: bool
    body_to_rotor: Matrix

    @property
    def name(self) -> str:
        """The frame's name, ``R<id>``."""
        return f"R{self.rotor_id}"


def rotor_frame(axis_vector: Sequence[float], rotor_id: int = 0) -> RotorFrame:
    """Return the frame R<id> of a rotor whose shaft is ``axis_vector`` in the geometry frame.

    Parameters
    ----------
    axis_vector : sequence of float
        The declared shaft (x aft, y right, z up); it need not be a unit vector.
    rotor_id : int, optional
        The rotor's id, zero or more.

    Returns
    -------
    RotorFrame
        The frame, its angles and the sense applied.

    Raises
    ------
    ValueError
        If the axis names no direction or the id is negative.
    """
    if isinstance(rotor_id, bool) or int(rotor_id) < 0:
        raise ValueError(f"a rotor id is an integer of zero or more, got {rotor_id!r}")
    stated = [float(c) for c in axis_vector]
    length = math.sqrt(sum(c * c for c in stated))
    if not math.isfinite(length) or length == 0.0:
        raise ValueError(f"the rotor axis {stated} names no direction")
    shaft = turn(EXPORT_TO_BODY, [c / length for c in stated])
    if shaft[0] != 0.0:
        flip = shaft[0] < 0.0
    elif shaft[2] != 0.0:
        flip = shaft[2] > 0.0
    else:
        flip = shaft[1] < 0.0
    a = tuple(-c + 0.0 for c in shaft) if flip else shaft
    toe = 0.0 if a[0] == 0.0 and a[1] == 0.0 else math.atan2(a[1], a[0])
    pitch = math.asin(max(-1.0, min(1.0, -a[2])))
    return RotorFrame(
        rotor_id=int(rotor_id),
        pitch_deg=math.degrees(pitch) + 0.0,
        toe_deg=math.degrees(toe) + 0.0,
        sense=-1 if flip else 1,
        declared_aft=shaft[0] < 0.0,
        body_to_rotor=multiply(dcm(_Y, pitch), dcm(_Z, toe)),
    )


def frame_rotations(
    alpha_deg: float, beta_deg: float, rotors: Sequence[RotorFrame] = ()
) -> dict[str, Matrix]:
    """Return ``C_f/E`` for ``f`` in W, S, B and each R<id> (id ascending), in file order.

    Raises
    ------
    ValueError
        If two rotor frames carry one id.
    """
    ids = [frame.rotor_id for frame in rotors]
    if len(set(ids)) != len(ids):
        raise ValueError(f"two rotor frames carry one id: {sorted(ids)}")
    turns: dict[str, Matrix] = {
        "W": body_to_wind(alpha_deg, beta_deg),
        "S": body_to_stability(alpha_deg, beta_deg),
        "B": IDENTITY,
    }
    for frame in sorted(rotors, key=lambda item: item.rotor_id):
        turns[frame.name] = frame.body_to_rotor
    return {name: multiply(matrix, EXPORT_TO_BODY) for name, matrix in turns.items()}


def frame_names(rotor_ids: Sequence[int] = ()) -> tuple[str, ...]:
    """Return ``W``, ``S``, ``B``, then ``R<id>`` per rotor id ascending."""
    return ("W", "S", "B", *(f"R{value}" for value in sorted(int(v) for v in rotor_ids)))


def _positive_finite(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(float(v)) and float(v) > 0.0 for v in values)


def airframe_axis_coefficients(
    force_n: Sequence[float],
    moment_nm: Sequence[float],
    rotations: Mapping[str, Matrix],
    *,
    q_ref_pa: float | None,
    sref_m2: float | None,
    cref_m: float | None,
    bref_m: float | None,
) -> dict[str, float | str]:
    """Return ``CD<f>, CY<f>, CL<f>, Cl<f>, Cm<f>, Cn<f>`` for every frame of ``rotations``.

    With ``F_f`` and ``M_f`` the force and the moment about the moment reference point in
    the forward-right-down frame f::

        CD<f> = -F_f,x / (q SREF)        Cl<f> = M_f,x / (q SREF BREF)
        CY<f> =  F_f,y / (q SREF)        Cm<f> = M_f,y / (q SREF CREF)
        CL<f> = -F_f,z / (q SREF)        Cn<f> = M_f,z / (q SREF BREF)

    Parameters
    ----------
    force_n, moment_nm : sequence of float
        Force (N) and moment about the moment reference point (N m) in the geometry frame;
        an unknown component is NaN.
    rotations : mapping
        The geometry-to-frame matrices of :func:`frame_rotations`.
    q_ref_pa, sref_m2, cref_m, bref_m : float or None
        ``0.5 RHO VREF^2`` and the reference area, chord and span.

    Returns
    -------
    dict
        The coefficients in the order of ``rotations``; ``NA`` where one cannot be formed.
    """
    row: dict[str, float | str] = {}
    lengths = (bref_m, cref_m, bref_m)
    for name, matrix in rotations.items():
        f = turn(matrix, force_n)
        m = turn(matrix, moment_nm)
        values = (-f[0], f[1], -f[2], m[0], m[1], m[2])
        divisors = (1.0, 1.0, 1.0, *lengths)
        for index, stem in enumerate(("CD", "CY", "CL", "Cl", "Cm", "Cn")):
            divisor = divisors[index]
            if not _positive_finite(q_ref_pa, sref_m2, divisor):
                row[f"{stem}{name}"] = NOT_APPLICABLE
                continue
            scale = float(q_ref_pa) * float(sref_m2) * float(divisor)  # type: ignore[arg-type]
            number = float(values[index]) / scale
            row[f"{stem}{name}"] = number + 0.0 if math.isfinite(number) else NOT_APPLICABLE
    return row

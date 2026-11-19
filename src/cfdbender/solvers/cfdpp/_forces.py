"""Forces and moments of the solver to dimensionless coefficients; pure arithmetic.

The solver writes dimensional forces and moments about the coordinate origin, per
boundary selector. cfdbender normalizes them itself, with the reference values of
the reference file and the dynamic pressure of the row:

    F   = FORCE_SIGN * sum(F_selector)       M_o = MOMENT_SIGN * sum(M_selector)
    M_c = M_o - c x F                        (moment transferred to the center c)
    CF  = F / (q A)                          CM  = M_c / (q A L)

The sign constants and the axis convention are hypotheses, each to be confirmed
on a lifting case against the vendor post-processor: FORCE_SIGN assumes the file
holds the force of the fluid on the body, MOMENT_SIGN a right-handed moment about
the positive axes. The body-axis coefficients are cfdbender's own normalization,
not a solver quantity. Wind-axis projection (``neutral``) mirrors the library's
standard frame: x aft, y right, z up in three dimensions; the flow in the x-y
plane in two, where a moment about +z is nose-down positive.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from cfdbender.core.neutral import wind_directions
from cfdbender.core.results import Coefficient, References

Vector = tuple[float, float, float]

FORCE_SIGN = 1
MOMENT_SIGN = 1

_AXES: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")


@dataclass(frozen=True)
class Forces:
    """Dimensional force and moment about the coordinate origin, in the geometry frame."""

    force: Vector
    moment: Vector


def sum_forces(items: Iterable[Forces]) -> Forces:
    """Return the component-wise sum of forces and of moments."""
    force = [0.0, 0.0, 0.0]
    moment = [0.0, 0.0, 0.0]
    for item in items:
        for axis in range(3):
            force[axis] += item.force[axis]
            moment[axis] += item.moment[axis]
    return Forces((force[0], force[1], force[2]), (moment[0], moment[1], moment[2]))


def transfer_moment(moment: Vector, force: Vector, origin: Vector) -> Vector:
    """Return the moment about ``origin``, from a force and moment about the coordinate origin."""
    cross = (
        origin[1] * force[2] - origin[2] * force[1],
        origin[2] * force[0] - origin[0] * force[2],
        origin[0] * force[1] - origin[1] * force[0],
    )
    return (moment[0] - cross[0], moment[1] - cross[1], moment[2] - cross[2])


def _signed(forces: Forces, force_sign: float, moment_sign: float) -> Forces:
    return Forces(
        (force_sign * forces.force[0], force_sign * forces.force[1], force_sign * forces.force[2]),
        (
            moment_sign * forces.moment[0],
            moment_sign * forces.moment[1],
            moment_sign * forces.moment[2],
        ),
    )


def coefficients(
    forces: Forces,
    dynamic_pressure: float,
    references: References,
    *,
    inviscid: Forces | None = None,
    viscous: Forces | None = None,
    force_sign: float = FORCE_SIGN,
    moment_sign: float = MOMENT_SIGN,
) -> dict[str, Coefficient]:
    """Return the body-axis coefficients and the raw dimensional sums.

    Names: ``CFx`` ``CFy`` ``CFz`` ``CMx`` ``CMy`` ``CMz`` (total), the same with
    ``_inv`` and ``_vis`` when those parts are given, and the raw sums ``Fx`` ``Fy``
    ``Fz`` ``Mx`` ``My`` ``Mz`` in newtons and newton meters about the coordinate
    origin. In two dimensions the forces are per unit depth and the reference
    area is the chord times one.
    """
    if not math.isfinite(dynamic_pressure) or dynamic_pressure <= 0:
        raise ValueError("dynamic pressure must be finite and positive")
    q_area = dynamic_pressure * references.area
    q_area_length = q_area * references.length
    result: dict[str, Coefficient] = {}
    parts = {"": forces}
    if inviscid is not None:
        parts["_inv"] = inviscid
    if viscous is not None:
        parts["_vis"] = viscous
    for suffix, part in parts.items():
        signed = _signed(part, force_sign, moment_sign)
        centred = transfer_moment(signed.moment, signed.force, references.moment_origin)
        for index, axis in enumerate(_AXES):
            result[f"CF{axis}{suffix}"] = Coefficient(signed.force[index] / q_area, axis, "force")
            result[f"CM{axis}{suffix}"] = Coefficient(
                centred[index] / q_area_length, axis, "moment"
            )
    signed = _signed(forces, force_sign, moment_sign)
    for index, axis in enumerate(_AXES):
        result[f"F{axis}"] = Coefficient(signed.force[index], axis, "force")
        result[f"M{axis}"] = Coefficient(signed.moment[index], axis, "moment")
    return result


def neutral(
    cf: Vector, cm: Vector, alpha_deg: float, beta_deg: float, dimensions: int
) -> dict[str, float]:
    """Project body-axis coefficients onto the neutral ``CL``, ``CD`` and ``CM``.

    Three dimensions: drag and lift along the wind directions of the library's
    standard frame, ``CM = CMy``. Two dimensions (flow in the x-y plane):
    ``CD = CFx cos a + CFy sin a``, ``CL = -CFx sin a + CFy cos a``, ``CM = -CMz``.
    """
    if dimensions == 3:
        drag, lift = wind_directions(alpha_deg, beta_deg)
        return {
            "CD": sum(c * d for c, d in zip(cf, drag, strict=True)),
            "CL": sum(c * d for c, d in zip(cf, lift, strict=True)),
            "CM": cm[1],
        }
    if dimensions == 2:
        a = math.radians(alpha_deg)
        return {
            "CD": cf[0] * math.cos(a) + cf[1] * math.sin(a),
            "CL": -cf[0] * math.sin(a) + cf[1] * math.cos(a),
            "CM": -cm[2],
        }
    raise ValueError(f"dimension must be 2 or 3, not {dimensions}")

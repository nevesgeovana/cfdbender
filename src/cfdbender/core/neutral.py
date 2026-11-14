"""Neutral coefficient columns CL, CD and CM, in one standard frame for every solver.

Every collected table carries the three columns ``CL``, ``CD`` and ``CM`` with the
same meaning on every solver, and cfdbender transforms each solver's native axes into
this frame. Each solver's own column names follow them unchanged. A solver that reports
no such coefficient (a propeller has no lift or drag; a two-dimensional flow solver
reports no moment here) writes the token NA.

The standard frame
------------------
The geometry frame is x aft (downstream), y to the right, z up, as pyflightstream's
loads export states it. A two-dimensional case lies in the x-y plane with y up.

* The freestream direction is ``V = (cos a cos b, sin b, sin a cos b)`` for the row's
  ALPHA ``a`` and BETA ``b``. In two dimensions ``V = (cos a, sin a)`` (BETA is zero).
* ``CD`` is the force along ``V``; ``CL`` is the force along the lift direction
  ``(-sin a, 0, cos a)``, normal to ``V`` and in the plane of symmetry (in two dimensions
  ``(-sin a, cos a)``). Both are nondimensionalised by the dynamic pressure and the
  row's reference area.
* ``CM`` is the pitching moment about the axis through the row's moment reference point
  perpendicular to the plane of symmetry (the y axis; in two dimensions the axis normal to
  the plane, z), positive nose-up, nondimensionalised by the dynamic pressure, the row's
  reference area and reference length. It is not turned by BETA.

This table is the only place where a solver's own name is tied to a neutral one, and the
only place that transforms a native axis into the standard one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

NEUTRAL = ("CL", "CD", "CM")

#: Neutral column to the solver's own coefficient name, per solver.
NATIVE_NAMES: Mapping[str, Mapping[str, str]] = {
    "su2": {"CL": "CL", "CD": "CD", "CM": "CMy"},
    "openfoam": {"CL": "Cl", "CD": "Cd", "CM": "CmPitch"},
    "pyfr": {"CL": "Cl", "CD": "Cd"},
    "xfoil": {"CL": "CL", "CD": "CD", "CM": "CM"},
    "mock": {"CL": "CZ", "CD": "CX", "CM": "Cm"},
    "flightstream": {"CL": "CLB", "CD": "CDB", "CM": "CMB25"},
    "qprop": {},
}

#: The names that differ when the solver states a two-dimensional problem. SU2 solves a
#: planar case in the x-y plane, so its pitching moment is CMz and CMy is identically zero.
PLANAR_NAMES: Mapping[str, Mapping[str, str]] = {
    "su2": {"CM": "CMz"},
}

#: The sign that turns a native value into the standard one, where the sign differs.
#: SU2 reports the physical moment about +z of the x-y plane, positive counterclockwise,
#: which is nose-down for a flow along +x with y up: the standard pitching moment is minus it.
PLANAR_SIGNS: Mapping[str, Mapping[str, float]] = {
    "su2": {"CM": -1.0},
}


def wind_directions(
    alpha_deg: float, beta_deg: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the freestream and lift directions of the standard frame.

    Parameters
    ----------
    alpha_deg, beta_deg : float
        The row's angle of attack and sideslip, in degrees.

    Returns
    -------
    tuple
        ``(drag direction, lift direction)`` as unit vectors in the geometry frame
        (x aft, y right, z up).
    """
    a, b = math.radians(alpha_deg), math.radians(beta_deg)
    drag = (math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b))
    lift = (-math.sin(a), 0.0, math.cos(a))
    return drag, lift


def _angle(condition: Mapping[str, object] | None, key: str) -> float:
    value = (condition or {}).get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def neutral_values(
    solver: str | None,
    native: Mapping[str, float],
    *,
    dimensions: object = None,
    condition: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Return the neutral coefficients, in the standard frame, a point's own values state.

    Parameters
    ----------
    solver : str or None
        Registered solver name; None or an unknown name states nothing.
    native : mapping
        The solver's own coefficient values by its own names.
    dimensions : object, optional
        The problem dimension the result states (2 or 3); 2 selects PLANAR_NAMES.
    condition : mapping, optional
        The point's flight condition; ALPHA and BETA (degrees) turn a solver that reports
        body-axis forces into the wind axes. Absent angles are zero.

    Returns
    -------
    dict
        CL, CD and CM, each present only when the solver reported it.
    """
    names = dict(NATIVE_NAMES.get(solver or "", {}))
    signs: dict[str, float] = {}
    if dimensions == 2:
        names.update(PLANAR_NAMES.get(solver or "", {}))
        signs = dict(PLANAR_SIGNS.get(solver or "", {}))
    values = {key: signs.get(key, 1.0) * native[own] for key, own in names.items() if own in native}
    if solver == "flightstream" and all(k in native for k in ("CDB", "CYB", "CLB")):
        # Body axes (forward, right, down): the force on the body along the geometry frame is
        # (CDB, CYB, CLB) = (aft, right, up). The wind axes are the standard frame.
        force = (native["CDB"], native["CYB"], native["CLB"])
        drag, lift = wind_directions(_angle(condition, "ALPHA"), _angle(condition, "BETA"))
        values["CD"] = sum(f * d for f, d in zip(force, drag, strict=True))
        values["CL"] = sum(f * d for f, d in zip(force, lift, strict=True))
    return values


# ---------------------------------------------------------------------------------------
# The rotor frame
#
# A rotor or propeller has its own standard frame and its own neutral columns. x runs along
# the thrust direction, positive forward; z is the normal force direction, positive up; y is
# z cross x, completing a right-handed frame (y points to the right of a forward-looking
# observer's x axis). Rotor coefficients use the rotor's rotation rate n in revolutions per
# second, its diameter D and the air density rho:
#
#   J   = V / (n D)                  advance ratio, V the speed along the thrust axis
#   CT  = T / (rho n^2 D^4)          thrust, positive when the thrust points forward (+x)
#   CQ  = Q / (rho n^2 D^5)          shaft torque about +x, signed by the right-hand rule
#   CP  = P / (rho n^3 D^5)          shaft power; CP = 2 pi CQ for rotation right-handed about +x
#   CN  = N / (rho n^2 D^4)          in-plane normal force, positive up (+z)
#   CY  = Y / (rho n^2 D^4)          in-plane side force, along +y
#   EFFICIENCY = J CT / CP           propeller efficiency, as the solver states it
#
# A positive CQ is a shaft torque vector along +x: a rotor turning right-handed about +x,
# counterclockwise to an observer behind it looking forward. Reversing the rotation reverses
# the sign of the torque and so of CQ; CT, CP and the efficiency keep their meaning.
# ---------------------------------------------------------------------------------------

ROTOR = ("CT", "CQ", "CP", "CN", "CY", "J", "EFFICIENCY")

#: Neutral rotor column to the solver's own coefficient name. A column the solver does not
#: report (QPROP's isolated propeller has no in-plane force) is absent and written NA.
ROTOR_NAMES: Mapping[str, Mapping[str, str]] = {
    "qprop": {
        "CT": "CT",
        "CQ": "CQ",
        "CP": "CP",
        "J": "J",
        "EFFICIENCY": "EFFICIENCY",
    },
}


def rotor_values(solver: str | None, native: Mapping[str, float]) -> dict[str, float]:
    """Return the neutral rotor coefficients a rotor-capable solver's own values state."""
    names = ROTOR_NAMES.get(solver or "", {})
    return {key: native[own] for key, own in names.items() if own in native}

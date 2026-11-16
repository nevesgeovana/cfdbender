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

* ``CD`` and ``CL`` are the drag and lift of the wind frame W of
  :mod:`cfdbender.core.axes`: ``CD`` is the force along ``-x_W`` and ``CL`` along
  ``-z_W``, so they equal the ``CDW`` and ``CLW`` columns of the polar files. At zero
  sideslip the drag direction is ``(cos a, 0, sin a)`` and the lift direction
  ``(-sin a, 0, cos a)`` (in two dimensions ``(cos a, sin a)`` and ``(-sin a, cos a)``).
  Both are nondimensionalised by the dynamic pressure and the row's reference area.
* ``CM`` is the pitching moment about the axis through the row's moment reference point
  perpendicular to the plane of symmetry (the y axis; in two dimensions the axis normal to
  the plane, z), positive nose-up, nondimensionalised by the dynamic pressure, the row's
  reference area and reference length. It is not turned by BETA.

This table is the only place where a solver's own name is tied to a neutral one, and the
only place that transforms a native axis into the standard one.
"""

from __future__ import annotations

from collections.abc import Mapping

from cfdbender.core.axes import frame_rotations

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
    """Return the drag and lift directions of the wind frame W.

    Parameters
    ----------
    alpha_deg, beta_deg : float
        The row's angle of attack and sideslip, in degrees.

    Returns
    -------
    tuple
        ``(drag direction, lift direction)`` as unit vectors in the geometry frame
        (x aft, y right, z up): ``-x_W`` and ``-z_W`` of
        :func:`cfdbender.core.axes.frame_rotations`.
    """
    wind = frame_rotations(alpha_deg, beta_deg)["W"]
    drag = (-wind[0][0] + 0.0, -wind[0][1] + 0.0, -wind[0][2] + 0.0)
    lift = (-wind[2][0] + 0.0, -wind[2][1] + 0.0, -wind[2][2] + 0.0)
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


#: The shaft of a rotor-capable solver's isolated rotor, in the geometry frame (x aft, y
#: right, z up). QPROP's isolated propeller flies along its shaft: the shaft points forward,
#: so its frame R0 is the body frame and its printed thrust is the R0 thrust.
ROTOR_SHAFTS: Mapping[str, tuple[float, float, float]] = {"qprop": (-1.0, 0.0, 0.0)}


def rotor_loads(
    solver: str | None, native: Mapping[str, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return the force (N) and hub moment (N m) of a rotor solver's rotor, geometry frame.

    QPROP prints the thrust T along the shaft and the shaft torque Q about it; the in-plane
    components are not printed and are NaN. A solver with no rotor, or a point whose thrust
    and torque are not stated, returns None.
    """
    shaft = ROTOR_SHAFTS.get(solver or "")
    if shaft is None or "T" not in native or "Q" not in native:
        return None
    nan = float("nan")
    thrust, torque = float(native["T"]), float(native["Q"])

    def along(value: float) -> tuple[float, float, float]:
        return tuple(value * c + 0.0 if c else nan for c in shaft)  # type: ignore[return-value]

    return along(thrust), along(torque)


def airframe_load_coefficients(
    solver: str | None,
    native: Mapping[str, float],
    *,
    dimensions: object = None,
    condition: Mapping[str, object] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return the force and moment coefficient vectors of a point in the geometry frame.

    The force vector is over ``q SREF`` and the moment vector over ``q SREF CREF``, about the
    moment reference point, in the geometry frame (x aft, y right, z up). FlightStream states
    its body-axis force directly (CDB, CYB, CLB along aft, right and up). Every other airframe
    solver states CD and CL in the wind frame W (:func:`neutral_values`): the force is
    ``-CD x_W - CL z_W`` carried back to the geometry frame, its side component unknown. The
    pitching moment CM is about +y (nose up); the rolling and yawing moments are unknown. An
    unknown component is NaN. A two-dimensional case is stated in the x-z plane. A solver
    that states no CD and CL returns None.
    """
    nan = float("nan")
    values = neutral_values(solver, native, dimensions=dimensions, condition=condition)
    moment = (nan, values.get("CM", nan), nan)
    if solver == "flightstream" and all(k in native for k in ("CDB", "CYB", "CLB")):
        return (float(native["CDB"]), float(native["CYB"]), float(native["CLB"])), moment
    if "CD" not in values or "CL" not in values:
        return None
    wind = frame_rotations(_angle(condition, "ALPHA"), _angle(condition, "BETA"))["W"]
    in_wind = (-values["CD"], nan, -values["CL"])
    force = tuple(
        sum(wind[i][j] * in_wind[i] for i in range(3) if wind[i][j] != 0.0) + 0.0 for j in range(3)
    )
    return (force[0], force[1], force[2]), moment

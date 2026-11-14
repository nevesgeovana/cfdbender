"""Neutral coefficient columns CL, CD and CM, defined once for every solver.

Every collected table carries the three columns ``CL``, ``CD`` and ``CM`` with the
same meaning on every solver: the lift, drag and pitching-moment coefficients of
the solver's own wind or body axes, as the solver reports them. Each solver's own
column names follow them unchanged. A solver that reports no such coefficient
(a propeller has no lift or drag; a two-dimensional flow solver reports no moment
here) writes the token NA.

This table is the only place where a solver's own name is tied to a neutral one.
"""

from __future__ import annotations

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


def neutral_values(
    solver: str | None, native: Mapping[str, float], *, dimensions: object = None
) -> dict[str, float]:
    """Return the neutral coefficients a point's own coefficients state.

    Parameters
    ----------
    solver : str or None
        Registered solver name; None or an unknown name states nothing.
    native : mapping
        The solver's own coefficient values by its own names.
    dimensions : object, optional
        The problem dimension the result states (2 or 3); 2 selects PLANAR_NAMES.

    Returns
    -------
    dict
        CL, CD and CM, each present only when the solver reported it.
    """
    names = dict(NATIVE_NAMES.get(solver or "", {}))
    if dimensions == 2:
        names.update(PLANAR_NAMES.get(solver or "", {}))
    return {key: native[own] for key, own in names.items() if own in native}

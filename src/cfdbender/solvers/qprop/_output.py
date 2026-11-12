"""Read QPROP's printed operating-point table by header names."""

from __future__ import annotations

import math
import re


def parse_table(text: str) -> dict[str, float]:
    """Return the first operating row keyed by its header names, or {} if unreadable.

    The header is the commented line naming ``V(m/s)`` and ``rpm``; the row is the
    next numeric line (QPROP may print it behind a ``#``). Any cell that is not a
    finite number makes the whole row unreadable.
    """
    lines = text.splitlines()
    for number, line in enumerate(lines):
        names = line.lstrip("#").split()
        if "V(m/s)" not in names or "rpm" not in names:
            continue
        for follower in lines[number + 1 :]:
            cells = follower.lstrip("#").split()
            if not cells:
                continue
            try:
                values = [float(cell) for cell in cells]
            except ValueError:
                continue
            if len(values) != len(names) or not all(map(math.isfinite, values)):
                return {}
            return dict(zip(names, values, strict=True))
        return {}
    return {}


def version(text: str) -> str:
    """Return the QPROP version printed in the header, or ``unknown``."""
    found = re.search(r"QPROP\s+Version\s+([0-9]+(?:[.][0-9]+)*)", text, re.IGNORECASE)
    return found[1] if found else "unknown"


def coefficients(row: dict[str, float], *, rho: float, radius: float) -> dict[str, float]:
    """Return J, CT and CP in QPROP's and the standard normalisation, with T, Q, P.

    QPROP normalises by the tip speed and disk area. The standard forms are
    T/(rho n^2 D^4), P/(rho n^3 D^5) and J = V/(n D) with n in revolutions per
    second and D = 2R; efficiency is T V / P.
    """
    speed, rpm = row["V(m/s)"], row["rpm"]
    thrust, torque, power = row["T(N)"], row["Q(N-m)"], row["Pshaft(W)"]
    n, diameter = rpm / 60.0, 2.0 * radius
    if n <= 0.0 or diameter <= 0.0 or rho <= 0.0:
        raise ValueError("rpm, radius and rho must be positive to normalise coefficients")
    return {
        "J": speed / (n * diameter),
        "CT": thrust / (rho * n**2 * diameter**4),
        "CP": power / (rho * n**3 * diameter**5),
        "CT_QPROP": row["CT"],
        "CP_QPROP": row["CP"],
        "EFFICIENCY": thrust * speed / power if power != 0.0 else 0.0,
        "T": thrust,
        "Q": torque,
        "P": power,
    }

"""Synthetic analytic rotor blade shared by the blade-slicing and QPROP tests.

The blade follows the slicing convention: rotation axis x, thrust along +x, hub at
the origin, rotation counter-clockwise seen from +x looking aft, blade along +y.
Each ring is a cambered lens with sharp leading and trailing edges placed in the
plane y = r, with the chord centred on the section origin. Chord and pitch follow
known laws, so the slicer can be compared against them.
"""

from __future__ import annotations

import math

R_ROOT = 0.04
R_TIP = 0.20
CAMBER = 0.02
THICKNESS = 0.06
HALF_POINTS = 16


def chord_law(r: float) -> float:
    """Return the chord in metres: linear taper from root to tip."""
    return 0.030 - 0.010 * (r - R_ROOT) / (R_TIP - R_ROOT)


def beta_law(r: float) -> float:
    """Return the pitch in degrees: linear from 55 at the root to 20 at the tip."""
    return 55.0 - 35.0 * (r - R_ROOT) / (R_TIP - R_ROOT)


def ring(r: float, *, thickness: float = THICKNESS, camber: float = CAMBER):
    """Return the section outline as (axial, tangential) pairs, leading edge first."""
    c, beta = chord_law(r), math.radians(beta_law(r))
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    le = (0.5 * c * cos_b, 0.5 * c * sin_b)
    chord_dir = (-cos_b, -sin_b)
    normal = (-sin_b, cos_b)
    upper, lower = [], []
    for k in range(HALF_POINTS + 1):
        u = 0.5 * (1.0 - math.cos(math.pi * k / HALF_POINTS))
        bump = 4.0 * u * (1.0 - u)
        for side, sink in ((1.0, upper), (-1.0, lower)):
            offset = c * (camber * bump + side * thickness * bump)
            s = le[0] + u * c * chord_dir[0] + offset * normal[0]
            z = le[1] + u * c * chord_dir[1] + offset * normal[1]
            sink.append((z, s))  # (axial, tangential)
    return upper + lower[-2:0:-1]


def blade_obj(*, rings: int = 61, mirror: bool = False, shift=(0.0, 0.0, 0.0)) -> str:
    """Return Wavefront OBJ text of the synthetic blade.

    Parameters
    ----------
    rings : int
        Number of radial rings between the root and the tip.
    mirror : bool
        Mirror the tangential coordinate, giving a clockwise rotor.
    shift : tuple of float
        Translation of every vertex, used to test an off-origin hub.
    """
    lines = ["# synthetic blade"]
    count = 0
    for i in range(rings):
        r = R_ROOT + (R_TIP - R_ROOT) * i / (rings - 1)
        points = ring(r)
        count = len(points)
        for axial, tangential in points:
            t = -tangential if mirror else tangential
            lines.append(f"v {axial + shift[0]:.9f} {r + shift[1]:.9f} {t + shift[2]:.9f}")
    for i in range(rings - 1):
        for j in range(count):
            a = i * count + j + 1
            b = i * count + (j + 1) % count + 1
            lines.append(f"f {a} {b} {b + count}")
            lines.append(f"f {a} {b + count} {a + count}")
    return "\n".join(lines) + "\n"

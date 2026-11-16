"""Exact boundary curves and ruled closures shared by control constructions.

Pipeline role: retain the skin's B-spline definitions while making closure
bases. Every ruled face has the same span basis as its boundary curves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq

from ._bspline import full_knots, split_patch
from ._patch import Patch
from ._trims import Trim, iso_trim


@dataclass(frozen=True)
class ControlPartition:
    """Shared bases, named faces and measured checks of an applied control."""

    bases: dict[str, Patch]
    pieces: dict[str, tuple[Trim, ...]]
    warnings: tuple[str, ...] = ()
    checks: tuple[dict, ...] = ()


def interval(patch, direction, low, high):
    """Restrict one parameter by knot insertion and normalize the result."""
    if high < 1:
        patch = split_patch(patch, direction, high)[0]
    if low > 0:
        patch = split_patch(patch, direction, low / high)[1]
    return patch


def transform_bases(bases, rotation, translation):
    """Place each base once while its trim parameters remain unchanged."""
    return {
        key: replace(p, controls=p.controls @ rotation.T + translation) for key, p in bases.items()
    }


def boundary(patch, u):
    """Return the poles of the exact span curve at u."""
    return BSpline(
        full_knots(patch.knots_u, patch.multiplicities_u), patch.controls, patch.degree_u
    )(u)


def point(patch, u, v):
    """Evaluate one boundary or section point without fitting."""
    return BSpline(
        full_knots(patch.knots_v, patch.multiplicities_v), boundary(patch, u), patch.degree_v
    )(v)


def chord_parameter(skin, x, upper):
    """Invert a root-section chord fraction on either monotone skin side."""
    le = point(skin, 0.5, 0)
    chord = (point(skin, 0, 0) + point(skin, 1, 0)) / 2 - le
    return brentq(
        lambda u: (point(skin, u, 0) - le) @ chord / (chord @ chord) - x,
        0.5 if upper else 0,
        1 if upper else 0.5,
    )


def at_v(patch, value):
    """Return a section curve as a degenerate ruled patch."""
    poles = BSpline(
        full_knots(patch.knots_v, patch.multiplicities_v), patch.controls, patch.degree_v, axis=1
    )(value)
    return replace(
        patch,
        controls=np.stack((poles, poles), axis=1),
        knots_v=(0.0, 1.0),
        multiplicities_v=(2, 2),
        degree_v=1,
    )


def ruled(skin, first, second, role):
    """Join two curves on the skin's span basis."""
    return replace(
        skin,
        family=role,
        role=role,
        controls=np.stack((first, second)),
        knots_u=(0.0, 1.0),
        multiplicities_u=(2, 2),
        degree_u=1,
    )


def end_faces(bases, boundaries, station, prefix):
    """Close a planar section with ruled fans sharing its exact rim curves."""
    curves = [at_v(patch, station) for patch in boundaries]
    centre = np.mean([p.controls[0, 0] for p in curves], axis=0)
    trims = []
    for index, curve in enumerate(curves):
        key = f"{prefix}_{index}"
        poles = curve.controls.copy()
        poles[:, 1] = centre
        bases[key] = replace(curve, controls=poles, family="end", role="end")
        trims.append(iso_trim(key, key, (0, 1), (0, 1), "end"))
    return trims

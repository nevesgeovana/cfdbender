"""Rigid Fowler motions and 49-station measured acceptance.

Pipeline role: rotate about the stated chord-frame pivot axis, then solve a
single rigid translation for gap and overlap at the middle station. The
other 48 stations independently test that motion against the same targets.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.transform import Rotation

from cfdbender._errors import InputValueError

from ._control_curves import point
from ._fowler_section import (
    cove_points,
    curves,
    distance,
    exterior_intervals,
    frame,
    inside_count,
    minimum_x,
    section,
)
from ._records import check_entry


def _profiles(skin, outline, v, rotation, translation):
    le, axes = frame(skin, v)
    return [section(p, v, le, axes, rotation, translation=translation) for p in outline]


def _pivot(skin, v, law):
    le, axes = frame(skin, v)
    basis = axes / np.sum(axes * axes, axis=0)
    return le + basis @ [law["pivot_x"], law["pivot_z"]]


def motion(skin, patches, nested, entry, span, *, angle):
    """Return one rigid motion, constrained by the supplied gap and overlap."""
    if angle == 0:
        return np.eye(3), np.zeros(3)
    law = next(row for row in entry["kinematics"] if row["deflection_deg"] == angle)
    a, b, gap = span
    start, end = _pivot(skin, a + gap, law), _pivot(skin, b - gap, law)
    axis = end - start
    axis /= np.linalg.norm(axis)
    rotation = Rotation.from_rotvec(np.deg2rad(angle) * axis).as_matrix()
    translation = start - start @ rotation.T
    middle = (a + b) / 2
    le, axes = frame(skin, middle)
    basis = axes / np.sum(axes * axes, axis=0)
    outline = curves(skin, nested, entry)
    profiles = _profiles(skin, outline, middle, rotation, translation)
    lip = cove_points(patches, middle, le, axes)[2]
    shift_x = lip[0] - law["overlap_pct"] / 100 - minimum_x(profiles)
    translation += shift_x * basis[:, 0]

    def separation(dz):
        profiles = _profiles(skin, outline, middle, rotation, translation + dz * basis[:, 1])
        return distance(profiles, lip)

    target = law["gap_pct"] / 100
    if target <= 0 or not np.isfinite(target):
        raise InputValueError("control.kinematics.gap_pct: require a finite positive gap")
    nearest = minimize_scalar(separation, bounds=(-2, 2), method="bounded").x
    if separation(nearest) > target:
        raise InputValueError("control.kinematics: gap and overlap have no rigid solution")
    shift_z = brentq(lambda z: separation(z) - target, nearest - 2, nearest, xtol=1e-13)
    translation += shift_z * basis[:, 1]
    return rotation, translation


def _check(name, value, limit, *, bound="max", where=None, unit="%c"):
    passed = value <= limit if bound == "max" else value >= limit
    return check_entry(
        name,
        "passed" if passed else "failed",
        float(value),
        limit,
        unit=unit,
        bound=bound,
        where=where,
    )


def _nested_clearance(skin, patches, outline, stations):
    minimum = np.inf
    for v in stations:
        le, axes = frame(skin, v)
        profile = _profiles(skin, outline, v, np.eye(3), np.zeros(3))
        rim = cove_points(patches, v, le, axes)
        minimum = min(
            minimum, *(distance(profile, a, b) for a, b in zip(rim, rim[1:], strict=False))
        )
    return 100 * minimum


def measure(skin, patches, nested, entry, span, *, applied):
    """Measure nested clearance, exterior identity and every deployed section law."""
    a, b, gap = span
    stations = np.linspace(a + gap, b - gap, 49)
    outline = curves(skin, nested, entry)
    clearance = _nested_clearance(skin, patches, outline, stations)
    checks = [
        _check(
            "fowler.nested_clearance_pct", clearance, entry.get("clearance_pct", 0.1), bound="min"
        )
    ]
    # The exterior portions are exact intervals of the same clean skin.
    # Measure the actual restricted bases at a lattice, with remapped u.
    identity = 0.0
    for patch, (u0, u1) in zip(
        outline[:2],
        exterior_intervals(skin, entry),
        strict=True,
    ):
        for u in np.linspace(0, 1, 17):
            identity = max(
                identity,
                float(
                    np.linalg.norm(
                        point(patch, u, stations) - point(skin, u0 + (u1 - u0) * u, stations),
                        axis=1,
                    ).max()
                ),
            )
    checks.append(_check("closure.retracted_identity", identity, 1e-12, unit="m"))
    for angle in entry["deflection_deg"]:
        checks.extend(_angle_checks(skin, patches, applied[angle], entry, stations, angle=angle))
    return checks


def _angle_checks(skin, patches, nested, entry, stations, *, angle):
    outline = curves(skin, nested, entry)
    error_gap, error_overlap, inside = 0.0, 0.0, 0
    laws = {row["deflection_deg"]: row for row in entry.get("kinematics", [])}
    for v in stations:
        le, axes = frame(skin, v)
        profiles = _profiles(skin, outline, v, np.eye(3), np.zeros(3))
        inside += inside_count(skin, patches, profiles, v, le, axes=axes)
        if angle:
            lip = cove_points(patches, v, le, axes)[2]
            error_gap = max(error_gap, abs(100 * distance(profiles, lip) - laws[angle]["gap_pct"]))
            error_overlap = max(
                error_overlap,
                abs(100 * (lip[0] - minimum_x(profiles)) - laws[angle]["overlap_pct"]),
            )
    where = {"control": entry["name"], "deflection_deg": angle, "stations": 49}
    checks = [_check("fowler.inside_main", inside, 0, unit="count", where=where)]
    if angle:
        checks.extend(
            [
                _check("fowler.gap_error_pct", error_gap, 0.01, where=where),
                _check("fowler.overlap_error_pct", error_overlap, 0.01, where=where),
                _check(
                    "fowler.law_threshold_hits",
                    int(error_gap >= 0.01) + int(error_overlap >= 0.01),
                    0,
                    unit="count",
                    where=where,
                ),
            ]
        )
    return checks

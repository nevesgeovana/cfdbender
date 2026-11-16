"""Square Fowler coves and closed nested flaps on a pinned lifting skin.

Pipeline role: preserve the exterior as iso trims, construct the wall, roof,
lip and nose on exact span curves, and split cove faces at every skin bay.
The single p3 station states a constant z/c law in the local chord frame.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.interpolate import BSpline

from cfdbender._errors import InputValueError

from ._bspline import full_knots, insert_knot
from ._control_curves import ControlPartition, boundary, chord_parameter, end_faces, interval, ruled
from ._fowler_motion import measure, motion
from ._records import check_entry
from ._trims import iso_trim


def with_pins(component):
    """Pin cove corners and the aft joins before the skin is constructed."""
    pins = list(component["sections"].get("pin") or [])
    for entry in component.get("control_surfaces", []):
        if entry["kind"] == "fowler":
            wall, lip, nose, aft = _fractions(entry)
            pins.extend((wall, lip, nose, aft))
    return {**component, "sections": {**component["sections"], "pin": sorted(set(pins))}}


def _fractions(entry):
    cove = entry["cove"]
    wall, lip = cove.get("wall_x", 0.745), cove.get("lip_x", 0.96)
    radius = entry.get("nose", {}).get("radius", 0.02)
    if entry.get("nose", {}).get("kind", "elliptic") != "elliptic":
        raise InputValueError("control.nose.kind: expected elliptic")
    nose = 1 - entry["chord_ratio"] + radius
    aft = lip + min(radius, (1 - lip) / 2)
    if (
        not np.isfinite([wall, lip, radius]).all()
        or not 0 < wall < nose - radius < nose < lip < aft < 1
    ):
        raise InputValueError("control.cove: require wall < flap nose < nose join < lip < aft < 1")
    if entry.get("nested_dz_pct", -0.145) >= 0:
        raise InputValueError("control.nested_dz_pct: require a negative roof offset")
    return wall, lip, nose, aft


def _u(skin, x, upper):
    value = chord_parameter(skin, x, upper)
    nearest = min(skin.knots_u, key=lambda k: abs(k - value))
    if abs(nearest - value) > 1e-9:
        raise InputValueError("control.cove: boundary must be pinned before building the skin")
    return nearest


def pin_skin(skin, entry):
    """Insert cove and nose pins into an imported base without changing it."""
    for x in _fractions(entry):
        for upper in (False, True):
            u = chord_parameter(skin, x, upper)
            if min(abs(k - u) for k in skin.knots_u) > 1e-10:
                skin = insert_knot(skin, "u", u)
    return skin


def _frame(skin):
    le = boundary(skin, 0.5)
    chord = (boundary(skin, 0) + boundary(skin, 1)) / 2 - le
    normal = np.stack((-chord[:, 2], np.zeros(len(chord)), chord[:, 0]), axis=1)
    return le, chord, normal


def _tangent(skin, u):
    spline = BSpline(full_knots(skin.knots_u, skin.multiplicities_u), skin.controls, skin.degree_u)
    return spline.derivative()(u)


def _bezier(skin, poles, role, knots=(0.0, 1.0), mult=(4, 4)):
    return replace(
        skin,
        family=role,
        role=role,
        controls=np.asarray(poles),
        knots_u=knots,
        multiplicities_u=mult,
        degree_u=3,
    )


def _shape(skin, entry):
    wall, lip, nose_x, aft = _fractions(entry)
    le, chord, normal = _frame(skin)
    low, high = _u(skin, wall, False), _u(skin, lip, True)
    nose_u, aft_u = _u(skin, nose_x, False), _u(skin, aft, True)
    p2, lip_up = boundary(skin, low), boundary(skin, high)
    p3 = le + wall * chord + entry["cove"]["p3"]["z_over_c"] * normal
    lip_down = lip_up - entry.get("clearance_pct", 0.1) / 100 * normal
    drop = -entry.get("nested_dz_pct", -0.145) / 100 * normal
    upper_nose = p3 + (nose_x - wall) / (lip - wall) * (lip_down - p3) - drop
    top_lip = lip_down - drop
    lower_nose, upper_aft = boundary(skin, nose_u), boundary(skin, aft_u)
    top_slope = (lip_down - p3) / (lip - wall)
    tangent = _tangent(skin, aft_u)
    chord2 = np.sum(chord * chord, axis=1)
    dx = np.sum(tangent * chord, axis=1) / chord2
    transition = _bezier(
        skin,
        [
            top_lip,
            top_lip + (aft - lip) / 3 * top_slope,
            upper_aft - ((aft - lip) / 3 / dx)[:, None] * tangent,
            upper_aft,
        ],
        "upper_transition",
    )
    patches = {
        "cove_wall": ruled(skin, p2, p3, "cove_wall"),
        "cove_roof": ruled(skin, p3, lip_down, "cove_roof"),
        "lip_base": ruled(skin, lip_down, lip_up, "lip_base"),
        "upper_nested": ruled(skin, upper_nose, top_lip, "upper_nested"),
        "upper_transition": transition,
        "nose": _nose(skin, entry, lower_nose, upper_nose, top_slope, nose_u=nose_u),
    }
    return patches, (low, high, nose_u, aft_u)


def _nose(skin, entry, lower, upper, top_slope, *, nose_u):
    _, chord, _ = _frame(skin)
    radius = entry.get("nose", {}).get("radius", 0.02)
    # Cubic quarter-ellipse handles, with the end handles on the actual
    # lower-skin and roof tangents. The centre join has one shared tangent.
    handle = 4 * (np.sqrt(2) - 1) / 3
    tip = (lower + upper) / 2 - radius * chord
    tangent = _tangent(skin, nose_u)
    dx = np.sum(tangent * chord, axis=1) / np.sum(chord * chord, axis=1)
    bottom_handle = lower + (-handle * radius / dx)[:, None] * tangent
    half_height = (upper - lower) / 2
    return _bezier(
        skin,
        [
            lower,
            bottom_handle,
            tip - handle * half_height,
            tip,
            tip + handle * half_height,
            upper - handle * radius * top_slope,
            upper,
        ],
        "nose",
        (0.0, 0.5, 1.0),
        (4, 3, 4),
    )


def _cove(skin, patches, limits, span, previous):
    low, high, _, _ = limits
    a, b, _ = span
    bases = {"skin": skin}
    main = []
    for u in ((0, 0.5), (0.5, 1)):
        if previous < a:
            main.append(iso_trim(f"before_{len(main)}", "skin", u, (previous, a), "skin"))
    for u in ((low, 0.5), (0.5, high)):
        main.append(iso_trim(f"fore_{len(main)}", "skin", u, (a, b), "skin"))
    if previous < a:
        main.append(iso_trim("te_before", "trailing_edge", (0, 1), (previous, a), "trailing_edge"))
    bays = sorted({a, b, *(v for v in skin.knots_v if a < v < b)})
    for name in ("cove_wall", "cove_roof", "lip_base"):
        bases[name] = patches[name]
        for index, (start, end) in enumerate(zip(bays, bays[1:], strict=False)):
            main.append(iso_trim(f"{name}_{index}", name, (0, 1), (start, end), name))
    rim = [
        interval(skin, "u", 0, low),
        interval(skin, "u", high, 1),
        ruled(skin, boundary(skin, 0), boundary(skin, 1), "trailing_edge"),
    ]
    rim.extend(patches[n] for n in ("cove_wall", "cove_roof", "lip_base"))
    for name, station in (("inboard", a), ("outboard", b)):
        main.extend(end_faces(bases, rim, station, name))
    return bases, main


def _nested(skin, patches, limits, span):
    _, _, nose_u, aft_u = limits
    a, b, gap = span
    bases = {
        "skin": skin,
        **{k: v for k, v in patches.items() if not k.startswith("cove") and k != "lip_base"},
    }
    bases["trailing_edge"] = ruled(skin, boundary(skin, 0), boundary(skin, 1), "trailing_edge")
    moving = [
        iso_trim("lower", "skin", (0, nose_u), (a + gap, b - gap), "skin"),
        iso_trim("upper_aft", "skin", (aft_u, 1), (a + gap, b - gap), "skin"),
    ]
    for name in ("nose", "upper_nested", "upper_transition", "trailing_edge"):
        moving.append(iso_trim(name, name, (0, 1), (a + gap, b - gap), name))
    rim = [interval(skin, "u", 0, nose_u), interval(skin, "u", aft_u, 1)]
    rim.extend(bases[n] for n in ("nose", "upper_nested", "upper_transition", "trailing_edge"))
    for name, station in (("inboard", a + gap), ("outboard", b - gap)):
        moving.extend(end_faces(bases, rim, station, name))
    return bases, moving


def add_fowler(model, entry, span, previous):
    """Return the cove and every closed flap angle, retaining shared exterior bases."""
    return definition(model.skin, entry, span, previous)["model"]


def definition(skin, entry, span, previous=0.0):
    """Construct the pieces and retain the native-edit motion and split boundaries."""
    patches, limits = _shape(skin, entry)
    bases, main = _cove(skin, patches, limits, span, previous)
    nested, trims = _nested(skin, patches, limits, span)
    name = entry["name"]
    pieces, motions = {}, {}
    for angle in entry["deflection_deg"]:
        label = f"{name}-d{str(float(angle)).removesuffix('.0')}"
        rotation, translation = motion(skin, patches, nested, entry, span, angle=angle)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, translation
        motions[label] = matrix.tolist()
        for key in {t.base for t in trims}:
            p = nested[key]
            bases[f"{label}/{key}"] = replace(p, controls=p.controls @ rotation.T + translation)
        pieces[label] = tuple(replace(t, base=f"{label}/{t.base}") for t in trims)
    own = {k for k in bases if k != "skin" and "/" not in k}
    bases = {f"{name}_{k}" if k in own else k: p for k, p in bases.items()}
    main = [
        replace(t, face=f"{name}_{t.face}", base=f"{name}_{t.base}" if t.base in own else t.base)
        for t in main
    ]
    model = ControlPartition(bases, {"main": tuple(main), **pieces})
    model = replace(model, checks=tuple(check_partition(model, entry, span=span)))
    return {"model": model, "limits": limits, "motions": motions}


def check_partition(model, entry, *, span):
    """Recompute the Fowler checks from actual applied bases and iso faces.

    Parameters
    ----------
    model : ControlPartition
        Applied geometry in the component's local frame.
    entry : dict
        Fowler control definition used for construction.
    span : tuple
        Inboard and outboard skin parameters and the normalized end gap.

    Returns
    -------
    list of dict
        Measured checks against the stated clearance and motion targets.
    """
    skin = model.bases["skin"]
    name = entry["name"]
    patches = {n: model.bases[f"{name}_{n}"] for n in ("cove_wall", "cove_roof", "lip_base")}
    applied = {
        angle: {
            key[len(f"{name}-d{str(float(angle)).removesuffix('.0')}/") :]: value
            for key, value in model.bases.items()
            if key.startswith(f"{name}-d{str(float(angle)).removesuffix('.0')}/")
        }
        for angle in entry["deflection_deg"]
    }
    if 0 in applied:
        nested = applied[0]
    else:
        original, limits = _shape(skin, entry)
        nested, _ = _nested(skin, original, limits, span)
    checks = measure(skin, patches, nested, entry, span, applied=applied)
    expected = sorted({span[0], span[1], *(v for v in skin.knots_v if span[0] < v < span[1])})
    wanted = set(zip(expected, expected[1:], strict=False))
    mismatch = 0
    for role in ("cove_wall", "cove_roof", "lip_base"):
        actual = {
            tuple(sorted({p[1] for edge in t.loop for p in edge}))
            for t in model.pieces["main"]
            if t.role == role
        }
        mismatch += len(wanted.symmetric_difference(actual))
    checks.insert(
        0,
        check_entry(
            "fowler.cove_bay_mismatch",
            "passed" if mismatch == 0 else "failed",
            mismatch,
            0,
            where={"control": entry["name"], "bay_limits": expected},
        ),
    )
    return checks

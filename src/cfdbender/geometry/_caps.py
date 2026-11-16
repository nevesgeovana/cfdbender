"""Tip caps and the root fillet of a lifting surface.

Pipeline role: pure numerical stage of the geometry package. The tip cap
closes the wing tip (flat, rounded, ridged or none; flat, the default, is the one
planar face bounded by the tip section outline); the root fillet blends
the skin into a host surface given as a signed distance and a unit normal.
The ridged cap is a superellipse rounding whose start, ridge height and
exponent follow a profile given per chord station.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator, make_interp_spline
from scipy.optimize import brentq

from cfdbender._errors import InputValueError

from ._airfoils import cst_airfoil
from ._bspline import full_knots, insert_knot, split_patch, union_knots
from ._distributions import blended_section, chord_spacing, planform_values
from ._lifting import (
    LiftingModel,
    cap_fraction,
    continuity_internal,
    law_and_fraction,
    max_thickness_x,
    place_points,
    section_at,
    section_models,
    section_sides,
    skin_model,
    thickness_station,
    tip_section,
    with_trim_checks,
)
from ._patch import Patch, interpolate_patch
from ._trims import iso_trim

__all__ = [
    "CAP_NOSE_LENGTH",
    "CAP_NOSE_RATIO",
    "CAP_ACROSS",
    "CAP_TE_TAPER",
    "RIM_SHARED",
    "CAP_EDGE_GAP",
    "cap_edge_gap",
    "cap_length",
    "cap_patch_grids",
    "cap_plane_grids",
    "cap_rates",
    "cap_taper",
    "cap_to_space",
    "exact_loft_grids",
    "contour_segment",
    "coons_grid",
    "fillet_arc",
    "fillet_contact",
    "junction_normal",
    "lifting_model",
    "map_cap_grid",
    "outline_cap",
    "ridge_arc",
    "ridge_cross",
    "ridge_profile",
    "ridged_cap_grids",
    "rim_halves",
    "rim_shared",
    "root_fillet",
    "rounded_cap_grids",
    "skin_caps",
    "tip_cap",
    "wing_contour_normal",
    "wing_contour_point",
]


def contour_segment(section, maximum, index, u):
    """Six oriented boundary curves, beginning at the leading edge."""
    u = np.asarray(u)
    if index in (0, 1, 4, 5):
        a, b, side = [
            (0, maximum, 0),
            (maximum, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
            (1, maximum, 1),
            (maximum, 0, 1),
        ][index]
        x = chord_spacing(a, b, u)
        return _sides(section, x)[side]
    top, bottom = _te_ends(section)
    middle = (top + bottom) / 2
    z = (1 - u) * top + u * middle if index == 2 else (1 - u) * middle + u * bottom
    return np.stack((np.ones_like(u), z), axis=-1)


def _sides(section, x):
    """Return the upper and lower (x, z) points of a section at chord fractions ``x``.

    ``section`` is a stored B-spline section or a blended CST table.
    """
    if hasattr(section, "poles"):
        upper, lower = section_sides(section, np.clip(np.asarray(x, float), 0.0, 1.0))
        return (upper[0], lower[0]) if np.ndim(x) == 0 else (upper, lower)
    return cst_airfoil(x, section["upper"], section["lower"], section["trailing_edge_thickness"])


def _te_ends(section):
    """Return the z of the upper and lower trailing-edge points of a section, in chords."""
    if hasattr(section, "poles"):
        return float(section.poles[-1][1]), float(section.poles[0][1])
    te = section["trailing_edge_thickness"]
    return te / 2, -te / 2


def coons_grid(bottom, top, left, right):
    """Return the Coons blend of four boundary curves, minus the bilinear corner term."""
    u, v = (
        np.linspace(0, 1, len(bottom))[:, None, None],
        np.linspace(0, 1, len(left))[None, :, None],
    )
    ruled = (
        (1 - v) * bottom[:, None] + v * top[:, None] + (1 - u) * left[None, :] + u * right[None, :]
    )
    bilinear = (
        (1 - u) * (1 - v) * bottom[0]
        + u * (1 - v) * bottom[-1]
        + (1 - u) * v * top[0]
        + u * v * top[-1]
    )
    return ruled - bilinear


def cap_plane_grids(section, maximum):
    """Return the planar grids of a flat tip cap: six edge strips and three interior patches."""
    u, v = np.linspace(0, 1, 49), np.linspace(0, 1, 9)
    up, low = _sides(section, maximum)
    centre = (up + low) / 2
    outer = [contour_segment(section, maximum, i, u) for i in range(6)]
    inner = [0.55 * edge + 0.45 * centre for edge in outer]
    annulus = [
        (1 - v[None, :, None] ** 2) * edge[:, None] + v[None, :, None] ** 2 * inside[:, None]
        for edge, inside in zip(outer, inner, strict=True)
    ]
    interior = []
    for i in (0, 2, 4):
        first, second = inner[i], inner[i + 1]
        bottom = (1 - u[:, None]) * centre + u[:, None] * first[0]
        left = (1 - u[:, None]) * centre + u[:, None] * second[-1]
        interior.append(coons_grid(bottom, second[::-1], left, first))
    return annulus + interior


def map_cap_grid(component, grid):
    """Place a planar section-frame cap grid (x, z in unit chord) in model space."""
    return cap_to_space(component, grid[..., 0], grid[..., 1], np.zeros_like(grid[..., 0]))


CAP_TE_TAPER = (
    0.03  # chord fraction over which the rounded-cap ridge falls to the trailing-edge base plane
)
CAP_NOSE_RATIO = 0.3  # ridge height at the leading-edge point as a fraction of the cap length
CAP_NOSE_LENGTH = (
    0.15  # chord fraction over which the ridge rises from the nose height to the cap length
)


def _tip(component):
    """Return the planform fraction where the skin ends and the tip cap starts.

    :func:`tip_cap` computes it once and passes it in ``_tip`` of a shallow copy of the table.
    """
    if "_tip" in component:
        return component["_tip"]
    return cap_fraction(component)


def cap_length(component, upper, lower):
    """Return the reach of a rounded or lofted cap in tip chords.

    ``tip_cap.extent`` gives it in model units, the printed tip station; otherwise it is
    ``tip_cap.length`` times the half thickness of the section between ``upper`` and ``lower``.
    """
    cap = component["tip_cap"]
    if "extent" in cap:
        return float(cap["extent"] / planform_values(component, _tip(component))["chord"])
    return float((upper[1] - lower[1]) / 2 * cap["length"])


def cap_to_space(component, x, z, extension, rates=(0.0, 0.0)):
    """Map section (x, z) and a spanwise extension, all in unit chord, to model space."""
    tip = _tip(component)
    p = planform_values(component, tip)
    angle, chord = np.deg2rad(p["twist"]), p["chord"]
    axis = component["planform"]["twist_axis"]
    dx, dz = (x - axis) * chord, z * chord
    return np.stack(
        (
            p["leading_edge_x"]
            + axis * chord
            + dx * np.cos(angle)
            + dz * np.sin(angle)
            + rates[0] * chord * extension,
            component["planform"]["span"] / 2 * tip + chord * extension,
            p["leading_edge_z"]
            - dx * np.sin(angle)
            + dz * np.cos(angle)
            + rates[1] * chord * extension,
        ),
        axis=-1,
    )


def rounded_cap_grids(component, section, maximum, chord_count=49, across_count=17):
    """Four ridge patches of the rounded tip cap: upper fore, upper aft, lower fore, lower aft.

    Across the local thickness the cap is a half ellipse of shape exponent e: the
    spanwise extension is r(x)*sin(phi)**e with z = middle +- half*cos(phi). The
    ridge height r is the cap length (multiple times the half thickness at the
    maximum-thickness station of the tip section), constant along the chord and
    falling smoothly to zero over the last CAP_TE_TAPER of the chord, so the
    finite trailing-edge thickness is closed on the base plane. Patches continue
    the chordwise splits of the skin and meet on the ridge. Each is a true
    quadrilateral: the rim side lies on the skin tip edge, the radial sides are
    half ellipses and the fourth side is the ridge, which stands above the
    leading-edge point so no side has zero length.
    """
    upper_max, lower_max = _sides(section, maximum)
    length = cap_length(component, upper_max, lower_max)
    exponent = component["tip_cap"]["exponent"]
    u, v = np.linspace(0, 1, chord_count), np.linspace(0, 1, across_count)
    grids = []
    for a, b, side in [(0, maximum, 1), (maximum, 1, 1), (0, maximum, -1), (maximum, 1, -1)]:
        x = chord_spacing(a, b, u)
        upper, lower = _sides(section, x)
        middle, half = (upper[:, 1] + lower[:, 1]) / 2, (upper[:, 1] - lower[:, 1]) / 2
        s = np.clip((x - (1 - CAP_TE_TAPER)) / CAP_TE_TAPER, 0, 1)
        rise = np.clip(1 - x / CAP_NOSE_LENGTH, 0, 1)
        ridge = length * (1 - (3 * s**2 - 2 * s**3)) * (1 - (1 - CAP_NOSE_RATIO) * rise**2)
        phi = np.pi / 2 * (v if side > 0 else 1 - v)
        z = middle[:, None] + side * half[:, None] * np.cos(phi)[None, :]
        extension = ridge[:, None] * np.sin(phi)[None, :] ** exponent
        grids.append(cap_to_space(component, np.repeat(upper[:, :1], len(v), 1), z, extension))
    return grids


def exact_loft_grids(component, section, maximum):
    """Return the tip cap as an exact loft of ``tip_cap.sections`` sections, two grids.

    Section k lies at the extension L k / N, where L is the half thickness at the
    maximum-thickness station times ``tip_cap.length`` and N the section count. It is the tip
    section with its thickness about the camber line scaled by sqrt(1 - (s / L)^(2 / e)), for the
    shape exponent e: the first section is the tip section itself and the last collapses on the
    camber line. Each surface is sampled at ``tip_cap.control_points`` chord stations; the upper
    and lower grids meet on the collapsed section, which also closes the finite trailing-edge
    base. The planform rates carry the sections along the extended planform.
    """
    cap = component["tip_cap"]
    upper_max, lower_max = _sides(section, maximum)
    length = cap_length(component, upper_max, lower_max)
    x = chord_spacing(0.0, 1.0, np.linspace(0, 1, cap["control_points"]))
    upper, lower = _sides(section, x)
    middle, half = (upper[:, 1] + lower[:, 1]) / 2, (upper[:, 1] - lower[:, 1]) / 2
    fraction = np.linspace(0, 1, cap["sections"] + 1)
    scale = np.sqrt(np.clip(1 - fraction ** (2 / cap["exponent"]), 0, 1))
    taper, sweep, rise = cap_rates(component)
    extension = np.broadcast_to(length * fraction, (len(x), len(fraction)))
    chord_x = upper[:, :1] * (1 - taper * extension)
    return [
        cap_to_space(
            component, chord_x, middle[:, None] + side * half[:, None] * scale[None, :],
            extension, (sweep, rise),
        )
        for side in (1, -1)
    ]  # fmt: skip


def cap_rates(component):
    """Planform rates at the tip per unit of spanwise extension.

    Returns (chord loss, leading-edge sweep, leading-edge rise).

    The cap sections follow the extended planform: the leading edge continues its sweep and rise and
    the trailing edge continues the taper.
    """
    tip = _tip(component)
    a, b = (planform_values(component, eta) for eta in (tip * (1 - 1e-3), tip))
    run = 1e-3 * tip * component["planform"]["span"] / 2
    return (
        float((a["chord"] - b["chord"]) / run),
        float((b["leading_edge_x"] - a["leading_edge_x"]) / run),
        float((b["leading_edge_z"] - a["leading_edge_z"]) / run),
    )


def cap_taper(component):
    """Chord loss per unit of spanwise extension at the tip."""
    return cap_rates(component)[0]


def ridge_profile(profile, x):
    """Return start of the rounding, ridge height and exponent at chord fractions ``x``.

    Interpolation is PCHIP in the square root of the chord fraction; heights are in chord units.
    """
    knots, s = np.sqrt(profile["stations"]), np.sqrt(np.asarray(x, float))
    return tuple(
        PchipInterpolator(knots, profile[key])(s) for key in ("start", "ridge", "exponent")
    )


def ridge_arc(start, ridge, exponent, half, count):
    """Superellipse from the end of the straight rim (start, +half) to the ridge (ridge, 0).

    Arrays over chord stations. Returns (stations, count, 2) of (spanwise, vertical offset from the
    camber point), both in chord units: |z/half|**n + |(e-start)/(ridge-start)|**n = 1.
    """
    phi = np.linspace(0, np.pi / 2, count)[None, :]
    n = np.asarray(exponent)[:, None]
    e = np.asarray(start)[:, None] + (np.asarray(ridge) - np.asarray(start))[:, None] * np.sin(
        phi
    ) ** (2 / n)
    z = np.asarray(half)[:, None] * np.cos(phi) ** (2 / n)
    return np.stack((e, z), axis=-1)


def ridge_cross(start, ridge, exponent, half, count=33, fine=400):
    """Return the rim-to-ridge cross curve, a straight rim then an arc, at equal arc length.

    The result has shape (stations, count, 2).
    """
    arc = ridge_arc(start, ridge, exponent, half, fine)
    out = np.empty((len(arc), count, 2))
    for i, a in enumerate(arc):
        line = np.stack(
            (np.linspace(0, start[i], 40, endpoint=False), np.full(40, half[i])), axis=-1
        )
        curve = np.concatenate((line, a))
        length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))))
        target = np.linspace(0, length[-1], count)
        out[i] = np.stack([np.interp(target, length, curve[:, k]) for k in range(2)], axis=-1)
    return out


def ridged_cap_grids(component, section, maximum, chord_count=49, across_count=65):
    """Return four cap patches (upper, lower, fore, aft of the thickness station) and two closures.

    The tip section is extended straight to the start of the rounding, then closed by a superellipse
    whose spanwise reach, start and exponent follow the identified profile. The ridge lies on the
    camber line. The closures are the planar slivers at the trailing edge, each a four-sided patch
    (rim half, rim line, arc, camber line).
    """
    profile = component["tip_cap"]["profile"]
    taper, sweep, rise = cap_rates(component)
    grids = []
    for a, b, side in [(0, maximum, 1), (maximum, 1, 1), (0, maximum, -1), (maximum, 1, -1)]:
        u = np.linspace(0, 1, chord_count)
        x = chord_spacing(a, b, u)
        upper, lower = _sides(section, x)
        middle, half = (upper[:, 1] + lower[:, 1]) / 2, (upper[:, 1] - lower[:, 1]) / 2
        start, ridge, exponent = ridge_profile(profile, x)
        cross = ridge_cross(start, ridge, exponent, half, across_count)
        if side < 0:
            cross = cross[:, ::-1]
        z = middle[:, None] + side * cross[..., 1]
        grids.append(
            (
                "tip_cap",
                cap_to_space(
                    component,
                    np.repeat(upper[:, :1], across_count, 1) * (1 - taper * cross[..., 0]),
                    z,
                    cross[..., 0],
                    (sweep, rise),
                ),
            )
        )
    upper, lower = _sides(section, np.array([1.0]))
    middle, half = float((upper[0, 1] + lower[0, 1]) / 2), float((upper[0, 1] - lower[0, 1]) / 2)
    start, ridge, exponent = (float(v[0]) for v in ridge_profile(profile, [1.0]))
    arc = ridge_arc([start], [ridge], [exponent], [half], 65)[0]
    for side in (1, -1):
        bottom = np.stack((np.linspace(0, ridge, 65), np.zeros(65)), axis=-1)
        top = np.stack((np.linspace(0, start, 65), np.full(65, half)), axis=-1)
        left = np.stack((np.zeros(65), np.linspace(0, half, 65)), axis=-1)
        right = arc[::-1]
        plane = coons_grid(bottom, top, left, right)
        e, z = plane[..., 0], middle + side * plane[..., 1]
        grids.append(
            ("tip_te_closure", cap_to_space(component, 1 - taper * e, z, e, (sweep, rise)))
        )
    return grids


def cap_patch_grids(component, section, maximum):
    """[(family, grid)] of the whole tip cap in model space; flat caps keep the planar layout."""
    kind = component["tip_cap"]["kind"]
    if kind == "none":
        return []
    if kind == "ridged":
        return ridged_cap_grids(component, section, maximum)
    if kind == "exact_loft":
        return [("tip_cap", grid) for grid in exact_loft_grids(component, section, maximum)]
    if kind == "rounded":
        return [("tip_cap", grid) for grid in rounded_cap_grids(component, section, maximum)]
    return [
        ("tip_cap", map_cap_grid(component, grid)) for grid in cap_plane_grids(section, maximum)
    ]


def tip_cap(component) -> list[Patch]:
    """Return the tip cap of a lifting-surface component as bicubic patches.

    The list is empty for the kind ``none``; for ``flat`` it holds one planar face bounded by
    the tip section outline (:func:`outline_cap`).

    Parameters
    ----------
    component : dict
        Resolved lifting-surface component, whose ``tip_cap`` entry names the kind.

    Returns
    -------
    list of Patch
        The cap patches; empty when the kind is ``none``.
    """
    if component["tip_cap"]["kind"] == "none":
        return []
    section = tip_section(component)
    component = {**component, "_tip": cap_fraction(component, section)}
    if component["tip_cap"]["kind"] == "flat":
        poles = np.asarray(section.poles, float)
        space = cap_to_space(component, poles[:, 0], poles[:, 1], np.zeros(len(poles)))
        return [outline_cap(space, np.asarray(section.knots, float), "tip_cap")]
    maximum = max_thickness_x(section)
    return [
        interpolate_patch(grid, family=family)
        for family, grid in cap_patch_grids(component, section, maximum)
    ]


def wing_contour_point(component, models, maximum, index, u, eta):
    """Return the model-space point of a contour segment at parameter ``u`` and span ``eta``.

    The section is the stored one the skin is built from (``models`` is no longer read), so a
    contour point at the tip lies on the cap's rim.
    """
    law, fraction = law_and_fraction(component)
    section = section_at(law, eta)
    point = contour_segment(section, maximum, index, np.asarray(u))
    return np.asarray(place_points(component, point, eta, fraction), float)


def wing_contour_normal(component, models, maximum, index, u, eta):
    """Return the unit surface normal of a contour segment at parameter ``u`` and span ``eta``."""

    def point(a, b):
        return wing_contour_point(component, models, maximum, index, a, b)

    tangent = point(min(1.0, u + 1e-5), eta) - point(max(0.0, u - 1e-5), eta)
    spanwise = point(u, min(1.0, eta + 1e-5)) - point(u, max(0.0, eta - 1e-5))
    normal = np.cross(tangent, spanwise)
    return normal / np.linalg.norm(normal)


def junction_normal(component, models, maximum, index, u, eta):
    """Return the contour normal, averaged with the neighbour segment at a segment end."""
    normal = wing_contour_normal(component, models, maximum, index, u, eta)
    if u in (0.0, 1.0):
        other = ((index - 1) % 6, 1.0) if u == 0 else ((index + 1) % 6, 0.0)
        normal += wing_contour_normal(component, models, maximum, other[0], other[1], eta)
        normal /= np.linalg.norm(normal)
    return normal


def fillet_contact(component, models, maximum, index, u, radius, host_distance):
    """Find the root contact point and normal where a ball of ``radius`` touches the host."""

    def normal(eta):
        return junction_normal(component, models, maximum, index, u, eta)

    def point(eta):
        return wing_contour_point(component, models, maximum, index, u, eta)

    def equation(eta):
        return host_distance(point(eta) + radius * normal(eta)) - radius

    eta = brentq(equation, 0, 1, xtol=1e-14)
    return point(eta), normal(eta)


def fillet_arc(point, normal, radius, v, host_normal):
    """Return the fillet arc from the contact point to the host, sampled at ``v`` in [0, 1]."""
    angle = np.arccos(np.clip(normal @ host_normal, -1, 1))
    direction = (
        np.sin((1 - v) * angle)[:, None] * normal + np.sin(v * angle)[:, None] * host_normal
    ) / np.sin(angle)
    return point + radius * normal - radius * direction


def root_fillet(
    component, host_distance=lambda p: p[1], host_normal=lambda p: np.array([0.0, 1.0, 0.0])
):
    """Fillet a supplied host surface given by distance and outward unit normal.

    Host functions must represent signed Euclidean distance and its gradient.
    Their default is the plane y=0. No host is read from an original geometry.
    """
    models = section_models(component["sections"])
    section = blended_section(component["sections"], models, 0.0)
    maximum = thickness_station(component, models, 0.0)
    u, v = np.linspace(0, 1, 49), np.linspace(0, 1, 9)
    patches = []
    for i in range(6):
        grid = []
        for station in u:
            x = contour_segment(section, maximum, i, np.asarray(station))[0]
            radius = float(np.interp(x, [0, maximum, 1], component["root_fillet"]["radii"]))
            if radius <= 0:
                raise InputValueError("root_fillet.radii: enabled fillet requires positive radii")
            point, normal = fillet_contact(
                component, models, maximum, i, station, radius, host_distance
            )
            centre = point + radius * normal
            grid.append(fillet_arc(point, normal, radius, v, host_normal(centre)))
        patches.append(interpolate_patch(grid, family="root_fillet"))
    return patches


# -- caps on the skin's own rim (S8) ------------------------------------------------------------

#: Samples across a cap face, from its rim to its ridge; the face is the cubic through them.
CAP_ACROSS = 9
#: Distance allowed between a closure's rim and the skin edge it closes, in metres (S8).
RIM_SHARED = 1e-12

#: Dense cap-edge acceptance, in metres (0.005 mm).
CAP_EDGE_GAP = 5e-6


def _edge_samples(patch, axis, index):
    controls = np.asarray(patch.controls)
    poles = controls[:, index] if axis == "u" else controls[index]
    knots = full_knots(getattr(patch, f"knots_{axis}"), getattr(patch, f"multiplicities_{axis}"))
    degree = getattr(patch, f"degree_{axis}")
    return BSpline(knots, poles, degree)(np.linspace(knots[degree], knots[-degree - 1], 4001))


def cap_edge_gap(model: LiftingModel) -> float:
    """Measure each cap surface along its shared boundary, including the corner sides.

    Compare 4001 corresponding boundary points, including endpoints, on the exact spline
    edges. This includes the side edges of the corner, which a skin-rim-only check misses.
    No interior fit samples are used as evidence of boundary closure.

    Parameters
    ----------
    model : LiftingModel
        Skin, trailing edge and caps in the same coordinate frame, in metres.

    Returns
    -------
    float
        Largest sampled boundary gap in metres, or zero when no caps are present.
    """
    gaps = [0.0]
    for end, column in (("tip", -1), ("root", 0)):
        if f"{end}_cap" in model.caps:
            gaps += [
                float(np.linalg.norm(a - b, axis=1).max())
                for a, b in _outline_pairs(model, end, column)
            ]
            continue
        if f"{end}_cap_upper" not in model.caps:
            continue
        upper, lower = (model.caps[f"{end}_cap_{side}"] for side in ("upper", "lower"))
        low, up, knots = rim_halves(
            np.asarray(model.skin.controls)[:, column], _u_knots(model.skin)
        )
        degree = len(knots) - len(low) - 1
        parameters = np.linspace(knots[degree], knots[-degree - 1], 4001)
        pairs = [
            (_edge_samples(upper, "u", 0), BSpline(knots, up, degree)(parameters)),
            (_edge_samples(lower, "u", 0), BSpline(knots, low, degree)(parameters)),
            (_edge_samples(upper, "u", -1), _edge_samples(lower, "u", -1)),
            (_edge_samples(upper, "v", 0), _edge_samples(lower, "v", 0)),
        ]
        corner = model.caps.get(f"{end}_cap_corner")
        if corner is not None:
            pairs += [
                (_edge_samples(corner, "v", 0), _edge_samples(lower, "v", -1)),
                (_edge_samples(corner, "v", -1), _edge_samples(upper, "v", -1)),
                (_edge_samples(corner, "u", 0), _edge_samples(model.trailing_edge, "u", column)),
            ]
        gaps += [float(np.linalg.norm(a - b, axis=1).max()) for a, b in pairs]
    return max(gaps)


def _curve_patch(poles, knots) -> Patch:
    """Return a curve as a patch of two equal rows, so the patch tools split and refine it."""
    poles = np.asarray(poles, float)
    distinct, mult = np.unique(np.asarray(knots, float), return_counts=True)
    return Patch(
        family="",
        controls=np.stack((poles, poles), axis=1),
        knots_u=tuple(float(k) for k in distinct),
        knots_v=(0.0, 1.0),
        multiplicities_u=tuple(int(m) for m in mult),
        multiplicities_v=(2, 2),
        degree_u=int(len(knots) - len(poles) - 1),
        degree_v=1,
    )


def _u_knots(patch: Patch) -> np.ndarray:
    return full_knots(patch.knots_u, patch.multiplicities_u)


def _refined(patch: Patch, target) -> Patch:
    """Insert knots until the u vector of ``patch`` is ``target``, a superset of it."""
    have = dict(zip(patch.knots_u, patch.multiplicities_u, strict=True))
    for knot, mult in zip(*np.unique(target, return_counts=True), strict=True):
        extra = int(mult) - int(have.get(float(knot), 0))
        if extra > 0 and 0.0 < knot < 1.0:
            patch = insert_knot(patch, "u", float(knot), extra)
    return patch


def rim_halves(poles, knots):
    """Return the lower and upper halves of a section curve on one shared knot vector.

    The curve (lower trailing edge, leading edge at u = 0.5, upper trailing edge) is split at
    0.5 by exact knot insertion; the lower half is reversed, so both halves run from the leading
    edge to their trailing edge, and both are refined onto the union of their vectors, so pole i
    of one pairs with pole i of the other. Works for chord-frame and model-space poles alike.

    Returns
    -------
    tuple
        ``(lower_poles, upper_poles, knots)``.
    """
    lower, upper = split_patch(_curve_patch(poles, knots), "u", 0.5)
    mirrored = 1.0 - _u_knots(lower)[::-1]
    known = _u_knots(upper)
    # Reflecting the lower half moves a knot by rounding (1e-16); a knot of the two halves that
    # coincide must stay one knot, not become two a hair apart, or the kernel refuses the face.
    for i, knot in enumerate(mirrored):
        near = known[np.abs(known - knot) < 1e-12]
        if len(near):
            mirrored[i] = near[0]
    lower = _curve_patch(np.asarray(lower.controls)[::-1, 0], mirrored)
    common = union_knots(_u_knots(lower), known)
    lower, upper = _refined(lower, common), _refined(upper, common)
    return np.asarray(lower.controls)[:, 0], np.asarray(upper.controls)[:, 0], common


def _across(component, kind, x, half, reach):
    """Return ``(a, e)`` of a cap kind, shape (poles, CAP_ACROSS).

    ``a`` is the share of the half thickness kept and ``e`` the reach in chords, from the rim
    (first column) to the ridge (last).
    """
    cap = component["tip_cap"]
    w = np.linspace(0.0, 1.0, CAP_ACROSS)
    phi = np.pi / 2 * w
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    ones = np.ones((len(x), 1))
    if kind == "ridged":
        start, ridge, exponent = ridge_profile(cap["profile"], x)
        safe = np.maximum(half, 1e-12)
        cross = ridge_cross(start, ridge, exponent, safe, CAP_ACROSS)
        a, e = cross[..., 1] / safe[:, None], cross[..., 0]
    elif kind == "exact_loft":
        a = ones * np.sqrt(np.clip(1 - w ** (2 / cap["exponent"]), 0, 1))[None, :]
        e = ones * (reach * w)[None, :]
    elif kind == "rounded":
        s = np.clip((x - (1 - CAP_TE_TAPER)) / CAP_TE_TAPER, 0, 1)
        rise = np.clip(1 - x / CAP_NOSE_LENGTH, 0, 1)
        ridge = reach * (1 - (3 * s**2 - 2 * s**3)) * (1 - (1 - CAP_NOSE_RATIO) * rise**2)
        a = ones * np.cos(phi)[None, :]
        e = ridge[:, None] * np.sin(phi)[None, :] ** cap["exponent"]
    else:
        a, e = ones * np.cos(phi)[None, :], np.zeros((len(x), CAP_ACROSS))
    a, e = np.array(a, float), np.array(e, float)
    a[:, 0], a[:, -1], e[:, 0] = 1.0, 0.0, 0.0
    return a, e


def _across_patch(track, knots_u, family) -> Patch:
    """Return the cubic across a cap through its sampled pole tracks, ends kept exactly."""
    w = np.linspace(0.0, 1.0, CAP_ACROSS)
    spline = make_interp_spline(w, track, k=3, axis=1)
    controls = np.array(np.asarray(spline.c).transpose(1, 0, 2))
    controls[:, 0], controls[:, -1] = track[:, 0], track[:, -1]
    distinct_u, mult_u = np.unique(knots_u, return_counts=True)
    distinct_w, mult_w = np.unique(spline.t, return_counts=True)
    return Patch(
        family=family,
        controls=controls,
        knots_u=tuple(float(k) for k in distinct_u),
        knots_v=tuple(float(k) for k in distinct_w),
        multiplicities_u=tuple(int(m) for m in mult_u),
        multiplicities_v=tuple(int(m) for m in mult_w),
        degree_u=int(len(knots_u) - len(track) - 1),
        degree_v=3,
        role="cap",
    )


def outline_cap(poles, knots, family) -> Patch:
    """Return the one planar face bounded by a closed section outline.

    ``poles`` and ``knots`` are the section curve (lower trailing edge, leading edge at
    u = 0.5, upper trailing edge) in any affine frame. The face is the ruled surface between
    the two halves of the outline, which run from the leading edge to the trailing edge on one
    knot vector: its rows are the two sides, its far end is the trailing-edge base (a point for
    a sharp trailing edge) and its near end collapses at the leading edge. It is planar when
    the outline is, and its boundary curves are the outline's own curves.
    """
    low, up, common = rim_halves(poles, knots)
    distinct, mult = np.unique(common, return_counts=True)
    return Patch(
        family=family,
        controls=np.stack((low, up), axis=1),
        knots_u=tuple(float(k) for k in distinct),
        knots_v=(0.0, 1.0),
        multiplicities_u=tuple(int(m) for m in mult),
        multiplicities_v=(2, 2),
        degree_u=int(len(common) - len(low) - 1),
        degree_v=1,
        role="cap",
    )


def _outline_pairs(model, end, column):
    """Pairs of 4001 samples that must coincide: cap rims and rung against the skin and face."""
    skin = np.asarray(model.skin.controls)
    low, up, knots = rim_halves(skin[:, column], _u_knots(model.skin))
    degree = len(knots) - len(low) - 1
    parameters = np.linspace(knots[degree], knots[-degree - 1], 4001)
    cap = model.caps[f"{end}_cap"]
    te = np.asarray(model.trailing_edge.controls)[:, column]
    rung = te[0] + np.linspace(0.0, 1.0, 4001)[:, None] * (te[1] - te[0])
    return [
        (_edge_samples(cap, "u", 0), BSpline(knots, low, degree)(parameters)),
        (_edge_samples(cap, "u", -1), BSpline(knots, up, degree)(parameters)),
        (_edge_samples(cap, "v", -1), rung),
    ]


def skin_caps(component, model: LiftingModel, section, end: str = "tip") -> dict[str, Patch]:
    """Return the cap faces that close one end of a skin, by base name.

    The rim of the upper and lower faces is the skin's own boundary curve at that end, split at
    the leading edge (u = 0.5), so each cap shares its edge with the skin exactly. Across, each
    face runs from the rim to the ridge with the shape of the cap kind (``flat``, ``rounded``,
    ``ridged`` or ``exact_loft``); the two faces meet on one ridge row. A third face, degree 1
    across the base, closes the blunt trailing edge between the two faces' last pole tracks, so
    its rim is the trailing-edge face's end edge. ``section`` is the chord-frame section at that
    end. The tip end takes ``tip_cap.kind``; the root end takes ``root`` (``open`` gives none).
    A flat end is instead the one planar face bounded by the end outline (:func:`outline_cap`),
    named ``<end>_cap``.
    """
    kind = component["tip_cap"]["kind"] if end == "tip" else component.get("root", "open")
    if kind in ("none", "open"):
        return {}
    column, sign, eta = (-1, 1.0, 1.0) if end == "tip" else (0, -1.0, 0.0)
    skin = model.skin
    if kind == "flat":
        poles = np.asarray(skin.controls)[:, column]
        return {f"{end}_cap": outline_cap(poles, _u_knots(skin), f"{end}_cap")}
    low3, up3, common = rim_halves(np.asarray(skin.controls)[:, column], _u_knots(skin))
    low2, up2, _ = rim_halves(np.asarray(section.poles, float), np.asarray(section.knots, float))
    tip = {**component, "_tip": model.fraction}
    middle2 = (low2 + up2) / 2
    up_max, low_max = _sides(section, np.array([max_thickness_x(section)]))
    reach = cap_length(tip, up_max[0], low_max[0])
    a, e = _across(component, kind, middle2[:, 0], np.linalg.norm(up2 - low2, axis=1) / 2, reach)
    chord = float(planform_values(component, eta * model.fraction)["chord"])
    span = np.array([0.0, sign * chord, 0.0])
    middle3, half3 = (low3 + up3) / 2, (up3 - low3) / 2
    ridge = middle3 + e[:, -1:] * span
    faces = {}
    for side, factor, rim in (("lower", -1.0, low3), ("upper", 1.0, up3)):
        track = middle3[:, None] + factor * half3[:, None] * a[..., None] + e[..., None] * span
        track[:, 0], track[:, -1] = rim, ridge
        faces[f"{end}_cap_{side}"] = _across_patch(track, common, f"{end}_cap")
    lower, upper = faces[f"{end}_cap_lower"], faces[f"{end}_cap_upper"]
    corner = np.stack((np.asarray(lower.controls)[-1], np.asarray(upper.controls)[-1]))
    if np.max(np.linalg.norm(corner[1] - corner[0], axis=-1)) > 0.0:
        faces[f"{end}_cap_corner"] = Patch(
            family=f"{end}_cap",
            controls=corner,
            knots_u=(0.0, 1.0),
            knots_v=lower.knots_v,
            multiplicities_u=(2, 2),
            multiplicities_v=lower.multiplicities_v,
            degree_u=1,
            degree_v=3,
            role="cap_corner",
        )
    return faces


def rim_shared(model: LiftingModel) -> dict:
    """Return the S8 record ``closure.rim_shared``: closures against the skin edges they close.

    Measured: the trailing-edge face's rows against the skin's two u ends; each cap face's rim
    against an independent split of the skin's end curve; each corner's rim against the
    trailing-edge face's end edge; and the two faces of a cap against each other on the ridge.
    """
    skin, te = np.asarray(model.skin.controls), np.asarray(model.trailing_edge.controls)
    gaps = [
        np.linalg.norm(te[0] - skin[0], axis=-1).max(),
        np.linalg.norm(te[1] - skin[-1], axis=-1).max(),
    ]
    for end, column in (("tip", -1), ("root", 0)):
        if f"{end}_cap" in model.caps:
            gaps += [
                float(np.linalg.norm(a - b, axis=1).max())
                for a, b in _outline_pairs(model, end, column)
            ]
            continue
        if f"{end}_cap_upper" not in model.caps:
            continue
        low, up, _ = rim_halves(skin[:, column], _u_knots(model.skin))
        upper = np.asarray(model.caps[f"{end}_cap_upper"].controls)
        lower = np.asarray(model.caps[f"{end}_cap_lower"].controls)
        gaps += [
            np.linalg.norm(upper[:, 0] - up, axis=-1).max(),
            np.linalg.norm(lower[:, 0] - low, axis=-1).max(),
            np.linalg.norm(upper[:, -1] - lower[:, -1], axis=-1).max(),
        ]
        corner = model.caps.get(f"{end}_cap_corner")
        if corner is not None:
            rim = np.asarray(corner.controls)[:, 0]
            gaps.append(np.linalg.norm(rim - te[:, column], axis=-1).max())
    worst = float(max(gaps))
    return {
        "id": "closure.rim_shared",
        "status": "pass" if worst <= RIM_SHARED else "fail",
        "value": worst,
        "note": f"closure rims against the skin edges they close, {RIM_SHARED} m",
    }


def _cap_trim(name):
    """Return the whole-base trim of a cap face.

    The outline cap collapses to a point at the leading edge, so its two curved rims are each
    divided at their midpoint: the face's outer wire then keeps the form the reader profile
    supports, as the propeller's planar cap does.
    """
    trim = iso_trim(name, name, (0.0, 1.0), (0.0, 1.0), role="cap")
    if name not in ("tip_cap", "root_cap"):
        return trim
    loop = []
    for i, (a, b) in enumerate(trim.loop):
        if i in (0, 2):
            middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            loop.extend(((a, middle), (middle, b)))
        else:
            loop.append((a, b))
    return replace(trim, loop=tuple(loop))


def lifting_model(component, warnings=None) -> LiftingModel:
    """Return the whole geometry of a lifting surface: skin, trailing-edge face, caps and faces.

    The records hold S4 (``law.ruled_twist``), S6 (``skin.section_exact``), S7 (``trim.*``), S8
    (``closure.rim_shared``) and S9 (``continuity.internal`` per base).

    Parameters
    ----------
    component : dict
        Resolved lifting-surface component (plain values, metres and degrees).
    warnings : list, optional
        Receives the warnings of the section conversion.

    Returns
    -------
    LiftingModel
        The bases, the faces as trims of them, the named iso lines and the stage records.
    """
    model = skin_model(component, warnings)
    caps = {}
    for end in ("tip", "root"):
        caps.update(skin_caps(component, model, model.ends[end], end))
    trims = model.trims + tuple(_cap_trim(name) for name in caps)
    model = replace(model, caps=caps, trims=trims)
    records = [*model.records, rim_shared(model)]
    records += [continuity_internal(patch, name) for name, patch in caps.items()]
    records += with_trim_checks(model)
    return replace(model, records=tuple(records))

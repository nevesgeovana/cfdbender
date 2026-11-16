"""Planar sections of one propeller blade and their builder parametrization, in mm.

Pipeline role: the pure numerical core of the propeller slicer. It takes a
triangle surface (an OBJ, or a fine triangulation of IGES faces with an optional exact
projector), isolates one blade, cuts it at radial stations and measures, with the propeller
builder's own definitions, the chord, pitch angle, thickness, camber, skew and rake that
rebuild it. The reader and the writer of files live in the workspace layer.

Each station is cut by the plane at distance ``r`` from the shaft whose normal is the
blade's radial direction rotated by the station's skew, which is exactly the plane in which
the builder places that section (skew rotates a planar section about the shaft and rake
translates it along the thrust axis). The skew is found by fixed-point iteration: the
section's area centroid, which is the builder's default stacking point, must lie on the
plane's radial line. A cylinder cut would bend the section and bias its chord and pitch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.mesh.objio import plane_cut_pieces, weld

from ._propeller import _toml_literal

#: Root and tip cuts stay this fraction of the radius inside the blade's ends; the table at
#: an end station is extrapolated from the cuts by a cubic in r.
END_MARGIN = 5e-4
#: The first, coarse cuts that find the skew law stay this fraction of the radius inside.
COARSE_MARGIN = 0.01
#: Side points closer than this in sqrt(x/c) are dropped before the cubic interpolation,
#: so a densely sampled section does not oscillate.
MIN_STEP = 1e-3
#: The trailing-edge base region: points within this many chords of the aftmost point.
BASE_BAND = 5e-4
#: Points per side of a written normalized profile, at x = s**2 for s uniform in [0, 1].
PROFILE_POINTS = 161
#: A triangle is blade surface when its normal has at least this azimuthal component; a
#: hub, a spinner and a planar cap have none.
BLADE_NORMAL = 0.2
#: Radial band, as fractions of the largest radius, in which the blades are counted.
COUNT_BAND = (0.55, 0.75)
#: A sharp trailing edge is written with this half gap, in chords, so a profile reader does
#: not take the closed contour's repeated point for its trailing edge.
SHARP_HALF_GAP = 1e-5
#: Two section ends whose widths at 5 percent of the chord differ by less than this fraction of
#: the wider are equally round; the direction of motion then names the leading edge. A real
#: airfoil's round nose is several times wider than its trailing edge, far outside the tie.
ROUND_TIE = 0.05

Refine = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _unit(vector, name):
    v = np.asarray(vector, float)
    norm = float(np.linalg.norm(v)) if v.shape == (3,) and np.all(np.isfinite(v)) else 0.0
    if norm == 0.0:
        raise InputValueError(f"{name}: expected three finite numbers, not all zero")
    return v / norm


class BladeFrame:
    """Shaft origin, thrust direction, reference blade direction and its motion partner."""

    def __init__(self, origin, thrust, reference):
        self.origin = np.asarray(origin, float)
        if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
            raise InputValueError("origin: expected three finite numbers")
        self.thrust = _unit(thrust, "axis")
        e = _unit(reference, "blade_dir")
        e = e - (e @ self.thrust) * self.thrust
        if np.linalg.norm(e) < 1e-6:
            raise InputValueError("blade_dir: must not be parallel to the axis")
        self.reference = e / np.linalg.norm(e)
        self.motion = np.cross(self.reference, self.thrust)

    def radial(self, psi):
        """Return the unit radial direction at azimuth ``psi``, from the reference to the motion."""
        return np.cos(psi) * self.reference + np.sin(psi) * self.motion

    def tangential(self, psi):
        return -np.sin(psi) * self.reference + np.cos(psi) * self.motion

    def polar(self, points):
        q = np.asarray(points) - self.origin
        x, y = q @ self.reference, q @ self.motion
        return np.hypot(x, y), np.arctan2(y, x)


def _polygon_centroid(points):
    x, y = points[:, 0], points[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2
    if abs(area) < 1e-300:
        return points.mean(axis=0), 0.0
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * area), area


def _nearest_end(chain, others, tolerance):
    """Return ``(gap, index, flip_chain, flip_other)`` of the closest joinable end, or None."""
    best = None
    for k, other in enumerate(others):
        for flip_self, flip_other in ((False, False), (False, True), (True, False), (True, True)):
            end = chain[0] if flip_self else chain[-1]
            start = other[-1] if flip_other else other[0]
            gap = float(np.linalg.norm(end - start))
            if gap <= tolerance and (best is None or gap < best[0]):
                best = (gap, k, flip_self, flip_other)
    return best


def _join(pieces, tolerance):
    """Close the open pieces whose ends meet within ``tolerance``; return the closed loops.

    A seam between two faces that do not share nodes ends the mesh's own chains there.
    """
    closed = [xy for xy, shut in pieces if shut and len(xy) >= 3]
    open_ = [xy for xy, shut in pieces if not shut]
    while open_:
        chain = open_.pop()
        while True:
            if len(chain) > 2 and np.linalg.norm(chain[0] - chain[-1]) <= tolerance:
                closed.append(chain)
                break
            best = _nearest_end(chain, open_, tolerance)
            if best is None:
                break
            _, k, flip_self, flip_other = best
            other = open_.pop(k)
            chain = np.vstack(
                (chain[::-1] if flip_self else chain, other[::-1] if flip_other else other)
            )
    loops = []
    for xy in closed:
        step = np.linalg.norm(np.diff(np.vstack((xy, xy[:1])), axis=0), axis=1)
        keep = step > tolerance * 1e-3
        if keep.sum() >= 3:
            loops.append(xy[keep])
    return loops


class BladeSurface:
    """A triangle surface in mm with its frame, and the cut of one blade at one radius.

    Coincident vertices are welded first (:func:`cfdbender.mesh.objio.weld`), and each
    plane is cut by the mesh's own connectivity (:func:`cfdbender.mesh.objio.plane_cut_pieces`)
    in the plane's frame, where the plane is ``z = r``. ``refine(points, normal, offset)``,
    when given, moves each section point onto the exact surface within the plane.
    """

    def __init__(self, xyz, triangles, frame, refine: Refine | None = None):
        points = np.asarray(xyz, float).reshape(-1, 3)
        triangles = np.asarray(triangles, np.int64).reshape(-1, 3)
        self.frame = frame
        self.refine = refine
        if len(points) and len(triangles):
            self.xyz, self.triangles = weld(points, triangles)
        else:
            self.xyz, self.triangles = points, triangles
        self.radius_v, self.psi_v = frame.polar(self.xyz)
        extent = float(self.radius_v.max()) if len(self.xyz) else 0.0
        self.tolerance = max(extent, 1.0) * 1e-6

    def _loops(self, psi, r):
        frame = self.frame
        basis = np.stack((frame.tangential(psi), frame.thrust, frame.radial(psi)))
        local = (self.xyz - frame.origin) @ basis.T
        return _join(plane_cut_pieces(local, self.triangles, r) or [], self.tolerance * 100)

    def cut(self, r, psi, *, exact=False):
        """Return the section at radius ``r`` whose centroid lies on its plane's radial line.

        Returns the 2D points ``(a, b)`` (tangential toward the motion, along the thrust),
        the plane azimuth and the 3D points; ``None`` when no closed loop is found. With
        ``exact`` and a projector, the iteration converges on the mesh first and then on
        the projected points.
        """
        frame = self.frame
        refining = False
        for _ in range(60):
            best = None
            for flat in self._loops(psi, r):
                centre, area = _polygon_centroid(flat)
                if area != 0 and (best is None or abs(centre[0]) < abs(best[0][0])):
                    best = (centre, flat)
            if best is None:
                return None
            centre, flat = best
            normal, tangential = frame.radial(psi), frame.tangential(psi)
            points = (
                frame.origin + r * normal + flat[:, :1] * tangential + flat[:, 1:] * frame.thrust
            )
            if refining:
                points = self.refine(points, normal, float(frame.origin @ normal + r))
                flat = np.column_stack(
                    ((points - frame.origin) @ tangential, (points - frame.origin) @ frame.thrust)
                )
                centre, _ = _polygon_centroid(flat)
            step = math.atan2(centre[0], r)
            if abs(centre[0]) <= 1e-9 * r:
                if refining or not exact or self.refine is None:
                    return flat, psi, points
                refining = True
                continue
            psi += step
        return flat, psi, points


def _diameter(points):
    i = 0
    for _ in range(20):
        j = int(np.argmax(np.linalg.norm(points - points[i], axis=1)))
        k = int(np.argmax(np.linalg.norm(points - points[j], axis=1)))
        if k == i:
            return j, k
        i = j
    return i, int(np.argmax(np.linalg.norm(points - points[i], axis=1)))


def _spread(points, origin, direction, length, band):
    low, high = band
    x = (points - origin) @ direction / length
    y = (points - origin) @ np.array([-direction[1], direction[0]]) / length
    inside = (x >= low) & (x <= high)
    return float(np.ptp(y[inside])) if inside.any() else 0.0


def _trailing_edge(points, le, far, chord):
    """Return the trailing-edge midpoint seen from ``le``, and the points of its base region.

    The base region holds the points within ``BASE_BAND`` chords of the section's aftmost
    abscissa along the chord. The midpoint sits at that abscissa, halfway between the
    region's two extreme ordinates: the corners of a blunt base, or the tip of a sharp one.
    """
    direction = (far - le) / np.linalg.norm(far - le)
    normal = np.array([-direction[1], direction[0]])
    x, y = (points - le) @ direction, (points - le) @ normal
    region = x >= x.max() - BASE_BAND * chord
    middle = (y[region].max() + y[region].min()) / 2
    return le + x.max() * direction + middle * normal, region


def _leading_edge(points, k, te):
    """Refine the farthest vertex from the trailing edge with a local quadratic in arc length."""
    n = len(points)
    idx = [(k + j) % n for j in range(-2, 3)]
    local = points[idx]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
    if np.any(np.diff(s) <= 0):
        return points[k], 0.0, s[2]
    fits = [np.polyfit(s, local[:, c], 2) for c in (0, 1)]
    dense = np.linspace(s[1], s[3], 401)
    curve = np.column_stack([np.polyval(f, dense) for f in fits])
    distance = np.linalg.norm(curve - te, axis=1)
    best = int(np.argmax(distance))
    if distance[best] <= np.linalg.norm(points[k] - te):
        return points[k], s[2], s[2]
    return curve[best], dense[best], s[2]


def edges_and_sides(flat, ahead=None):
    """Return leading edge, trailing-edge midpoint and the two sides of a closed section.

    The extreme pair of the section gives the chord's ends; the rounder end (wider at 5
    percent of the chord) is the leading edge. When ``ahead`` (the direction of motion in the
    section's plane) is given and the widths of the two ends differ by less than
    ``ROUND_TIE`` of the wider, the end farther along ``ahead`` is the leading edge, so a
    section with two equally sharp or round ends keeps its leading edge from station to
    station; a real round nose decides as before. The trailing-edge midpoint is the middle of
    the base region at the aftmost abscissa (:func:`_trailing_edge`), and the leading edge
    is the point farthest from it, the builder's definition; both are found twice, the
    second time along the refined chord. Each side runs from the leading edge to the base
    region.
    """
    n = len(flat)
    i, j = _diameter(flat)
    chord = float(np.linalg.norm(flat[j] - flat[i]))
    direction = (flat[j] - flat[i]) / chord
    near = _spread(flat, flat[i], direction, chord, (0.0, 0.05))
    far_end = _spread(flat, flat[i], direction, chord, (0.95, 1.0))
    if ahead is not None and abs(near - far_end) <= ROUND_TIE * max(near, far_end):
        if (flat[j] - flat[i]) @ np.asarray(ahead, float) > 0:
            i, j = j, i
    elif near < far_end:
        i, j = j, i
    le, far = flat[i], flat[j]
    for _ in range(2):
        te, region = _trailing_edge(flat, le, far, chord)
        k = int(np.argmax(np.linalg.norm(flat - te, axis=1)))
        le, at, vertex = _leading_edge(flat, k, te)
        far = te
    sides = []
    for step in (1, -1):
        chain = [k]
        while not region[chain[-1]]:
            chain.append((chain[-1] + step) % n)
            if len(chain) > n:
                raise InputArtifactError(
                    "section: no trailing edge found on the cut loop", kind="geometry"
                )
        side = flat[chain]
        ahead = (at > vertex) == (step == 1)
        sides.append(np.vstack((le, side[1:] if ahead else side)))
    return le, te, sides


def _resample(side, grid):
    x = np.clip(side[:, 0], 0.0, None)
    root = np.sqrt(x)
    keep = [0]
    for m in range(1, len(x)):
        if root[m] > root[keep[-1]] + MIN_STEP or m == len(x) - 1 and root[m] > root[keep[-1]]:
            keep.append(m)
    s, y = root[keep], side[keep, 1]
    if len(s) < 2:
        raise InputArtifactError("section: a side has fewer than two points", kind="geometry")
    return CubicSpline(s, y)(grid)


def section_parameters(flat, sense, r, radius):
    """Measure one section in the builder's frame; ``sense`` is +1 when it turns toward +motion."""
    plane = np.column_stack((sense * flat[:, 0], flat[:, 1]))
    centre, _ = _polygon_centroid(plane)
    # in this frame the section moves toward +x, so a tie of the two ends goes to the larger x
    le, te, sides = edges_and_sides(plane, ahead=(1.0, 0.0))
    chord = float(np.linalg.norm(te - le))
    axis = (te - le) / chord
    normal = np.array([axis[1], -axis[0]])
    frames = [
        np.column_stack(((side - le) @ axis / chord, (side - le) @ normal / chord))
        for side in sides
    ]
    upper, lower = sorted(frames, key=lambda side: -float(np.mean(side[:, 1])))
    dense = np.linspace(0, 1, 4001)
    yu, yl = _resample(upper, dense), _resample(lower, dense)
    mean = (yu + yl) / 2
    grid = np.linspace(0, 1, PROFILE_POINTS)
    profile = [np.column_stack((grid**2, _resample(side, grid))) for side in (upper, lower)]
    gap = float(profile[0][-1, 1] - profile[1][-1, 1])
    if gap < 2 * SHARP_HALF_GAP:
        middle = (profile[0][-1, 1] + profile[1][-1, 1]) / 2
        profile[0][-1, 1], profile[1][-1, 1] = middle + SHARP_HALF_GAP, middle - SHARP_HALF_GAP
    return {
        "r_R": r / radius,
        "c_R": chord / radius,
        "beta_deg": math.degrees(math.atan2(-axis[1], -axis[0])),
        "t_c": float(np.max(yu - yl)),
        "camber": float(mean[int(np.argmax(np.abs(mean)))]),
        "te_gap_c": max(gap, 0.0),
        "rake_x_R": -float(centre[1]) / radius,
        "profile": {"upper": profile[0].tolist(), "lower": profile[1].tolist()},
    }


def _count_blades(surface, blades):
    radius = float(surface.radius_v.max()) if len(surface.xyz) else 0.0
    band = (surface.radius_v >= COUNT_BAND[0] * radius) & (
        surface.radius_v <= COUNT_BAND[1] * radius
    )
    used = np.zeros(len(surface.xyz), bool)
    used[surface.triangles.ravel()] = True
    psi = np.sort(surface.psi_v[band & used])
    if radius <= 0 or len(psi) < 3:
        raise InputArtifactError(
            "no blade found: the surface has no points in the counting band "
            f"{COUNT_BAND[0]} to {COUNT_BAND[1]} of its largest radius about the axis. "
            "Check axis and origin. Nothing was written.",
            kind="geometry",
        )
    gaps = np.diff(np.r_[psi, psi[0] + 2 * np.pi])
    threshold = math.pi / (4 * blades)
    starts = np.flatnonzero(gaps > threshold)
    if len(starts) != blades:
        raise InputValueError(
            f"blades: {blades} given, found {len(starts)} blades between "
            f"{COUNT_BAND[0]} and {COUNT_BAND[1]} of the tip radius. Nothing was written."
        )
    centres = []
    for m, stop in enumerate(starts):
        lo, hi = starts[m - 1] + 1, stop + 1
        if hi <= lo:
            hi += len(psi)
        members = np.take(psi, np.arange(lo, hi), mode="wrap")
        centres.append(float(np.angle(np.exp(1j * members).mean())))
    return radius, min(centres, key=lambda c: abs(_wrap(c)))


def _blade_like(surface):
    p = surface.xyz[surface.triangles]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    size = np.linalg.norm(normal, axis=1)
    _, psi = surface.frame.polar(p.mean(axis=1))
    tangential = (
        -np.sin(psi)[:, None] * surface.frame.reference
        + np.cos(psi)[:, None] * surface.frame.motion
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        component = np.abs(np.einsum("ij,ij->i", normal, tangential)) / size
    return (size > 0) & (component >= BLADE_NORMAL)


def _no_section(r, radius):
    return InputArtifactError(
        f"no blade found: the plane at r/R = {r / radius:.4f} cuts no closed blade section. "
        "Check axis, origin and blades, or give "
        "stations inside the blade. Nothing was "
        "written.",
        kind="geometry",
    )


def _march(surface, radii, psi_start, start):
    """Cut at ``radii`` outward then inward from index ``start``; return azimuths and cuts."""
    cuts = [None] * len(radii)
    for order in (range(start, len(radii)), range(start - 1, -1, -1)):
        psi = psi_start if order.start == start else cuts[start][1]
        for m in order:
            found = surface.cut(radii[m], psi)
            if found is None:
                raise _no_section(radii[m], radii[-1])
            cuts[m] = found
            psi = found[1]
    return cuts


def slice_blade(surface, *, blades, stations=None):
    """Return the builder table of blade 0 of ``surface`` at ``stations`` (r/R) with its record.

    Blade 0 is the blade nearest the reference direction. The radius is the tip plane's
    distance from the shaft and the root is the blade surface's innermost plane distance; a
    hub and a spinner have no azimuthal normal and are excluded. With no stations, 21 equal
    steps from the root to the tip and 0.7 are cut.
    """
    if type(blades) is not int or blades < 1:
        raise InputValueError(f"blades: expected a positive integer, got {blades!r}")
    frame = surface.frame
    outer, centre = _count_blades(surface, blades)
    sector = np.abs(_wrap(surface.psi_v - centre)) < math.pi / blades
    blade_tri = _blade_like(surface)
    corners = surface.triangles[blade_tri]
    inner = corners[sector[corners].all(axis=1)].ravel()
    if not len(inner):
        raise _no_section(0.65 * outer, outer)
    root0 = float(surface.radius_v[inner].min())
    tip0 = float(surface.radius_v[sector].max())
    coarse = np.linspace(root0 + COARSE_MARGIN * tip0, tip0 * (1 - COARSE_MARGIN), 12)
    start = int(np.argmin(np.abs(coarse - 0.65 * tip0)))
    cuts = _march(surface, coarse, centre, start)
    law = CubicSpline(coarse, np.unwrap([c[1] for c in cuts]))
    ends = []
    for at, end in ((coarse[-1], tip0), (coarse[0], root0)):
        exact = surface.cut(at, float(law(at)), exact=True)
        shift = 0.0 if exact is None else exact[1] - float(law(at))
        ends.append(float(law(end)) + shift)
    q = surface.xyz - frame.origin
    radius = float(np.max(q[sector] @ frame.radial(ends[0])))
    root = float(np.min(q[inner] @ frame.radial(ends[1])))
    reference = cuts[start][0]
    le, te, _ = edges_and_sides(reference)
    sense = 1 if (te - le)[0] < 0 else -1
    if stations is None:
        r_R = np.unique(np.round(np.r_[np.linspace(root / radius, 1.0, 21), 0.7], 6))
    else:
        r_R = np.asarray(stations, float)
        if r_R.ndim != 1 or len(r_R) < 4 or np.any(np.diff(r_R) <= 0) or r_R[-1] > 1:
            raise InputValueError(
                "stations: expected at least four strictly increasing r/R values up to 1"
            )
        if r_R[0] < root / radius - 0.005:
            raise InputValueError(
                f"stations: r/R = {r_R[0]:g} is inboard of the blade root at r/R = "
                f"{root / radius:.4f}"
            )
    low, high = root + END_MARGIN * radius, radius * (1 - END_MARGIN)
    cut_r = np.unique(np.clip(r_R * radius, low, high))
    rows = []
    for r in cut_r:
        found = surface.cut(r, float(law(r)), exact=True)
        if found is None:
            raise _no_section(r, radius)
        flat, psi, _ = found
        row = section_parameters(flat, sense, r, radius)
        row["skew_deg"] = math.degrees(-sense * psi)
        rows.append(row)
    table = _at_stations(rows, cut_r / radius, r_R)
    rotation = "positive_x" if (-sense * frame.thrust)[0] > 0 else "negative_x"
    return {
        "table": table,
        "measured": {
            "blades_found": blades,
            "radius_mm": radius,
            "root_r_R": root / radius,
            "rotation": rotation,
            "motion_dir_at_blade": (sense * frame.motion).tolist(),
            "cut_r_R": (cut_r / radius).tolist(),
            "end_margin": END_MARGIN,
        },
    }


SCALARS = ("c_R", "beta_deg", "t_c", "camber", "te_gap_c", "skew_deg", "rake_x_R")


def _at_stations(rows, cut, wanted):
    """Return rows at the wanted r/R; an end station is extrapolated from the cuts."""
    laws = {}
    if len(cut) >= 2:
        k = min(3, len(cut) - 1)
        laws = {key: make_interp_spline(cut, [row[key] for row in rows], k=k) for key in SCALARS}
    table = []
    for r in wanted:
        nearest = int(np.argmin(np.abs(cut - r)))
        row = dict(rows[nearest])
        if abs(cut[nearest] - r) > 1e-12 and laws:
            for key in SCALARS:
                row[key] = float(laws[key](r))
            row["extrapolated_from_r_R"] = float(cut[nearest])
        row["r_R"] = float(r)
        table.append(row)
    return table


def definition_text(result, *, name, frame: BladeFrame, profile_files: Sequence[str]):
    """Return schema 1 TOML that the propeller builder reads to rebuild the sliced propeller."""
    table, measured = result["table"], result["measured"]
    r = [row["r_R"] for row in table]
    beta = [row["beta_deg"] for row in table]
    reference = 0.7 if r[0] <= 0.7 <= r[-1] else r[len(r) // 2]
    pitch = float(make_interp_spline(r, beta, k=min(3, len(r) - 1))(reference))
    radius = measured["radius_mm"]
    data = {
        "schema_version": 1,
        "propeller": {
            "name": name,
            "units": "mm",
            "radius": radius,
            "hub_radius": min(measured["root_r_R"], r[0]) * radius,
            "blades": measured["blades_found"],
            "rotation": measured["rotation"],
            "pitch_setting": pitch,
            "pitch_reference_r_R": reference,
        },
        "stations": {
            "units": {"skew": "deg", "rake": "x/R"},
            "r_R": r,
            "c_R": [row["c_R"] for row in table],
            "beta_deg": beta,
            "t_c": [row["t_c"] for row in table],
            "airfoil": [f"s{i:02d}" for i in range(len(table))],
            "skew": [row["skew_deg"] for row in table],
            "rake": [row["rake_x_R"] for row in table],
        },
        "airfoils": {f"s{i:02d}": {"file": path} for i, path in enumerate(profile_files)},
        "options": {"trailing_edge": "blunt", "root_cap": "planar", "tip": "planar"},
        "output": {"blades": "all", "units": "mm"},
    }
    default = (
        np.allclose(frame.origin, 0)
        and np.allclose(frame.thrust, [-1, 0, 0])
        and np.allclose(frame.reference, [0, 0, 1])
    )
    if not default:
        data["frame"] = {
            "pitch_axis_point_mm": frame.origin.tolist(),
            "blade_dir": frame.reference.tolist(),
            "thrust_dir": frame.thrust.tolist(),
            "rotation_dir_at_blade": measured["motion_dir_at_blade"],
        }
    return "\n".join(f"{key} = {_toml_literal(value)}" for key, value in data.items()) + "\n"


def profile_text(row, label):
    """Return the section as Selig-ordered x y rows: upper TE to LE, then lower LE to TE."""
    upper, lower = np.asarray(row["profile"]["upper"]), np.asarray(row["profile"]["lower"])
    points = np.vstack((upper[::-1], lower[1:]))
    lines = [f"{label} r/R={row['r_R']:.6f} normalized chord, leading edge at 0"]
    lines += [f"{x:.12f} {y:.12f}" for x, y in points]
    return "\n".join(lines) + "\n"

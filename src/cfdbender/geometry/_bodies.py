"""Bodies of revolution and superellipse bodies as a quadrilateral curve network.

Pipeline role: pure numerical stage of the geometry package. A resolved body component (plain
values, metres) becomes bicubic patches: a pole or flat-base cap at each closed end, built from
three true quadrilaterals per azimuth sector, and one skinned band between the first and the last
station, cut at the stations and junctions by knot insertion. An end may also be open, so a
parent and a child can share its rim; a boat-tail rear closes on a flat base by a law limited in
slope.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from cfdbender._errors import InputValueError

from ._continuity import edge_jumps
from ._distributions import spline_distribution
from ._patch import Patch, interpolate_patch
from ._schema import stations as check_stations
from ._skin import (
    POLE_TARGET_DEG,
    SAMPLE_LADDER,
    band_axial,
    cap_edge_ids,
    clustered,
    fit_patch,
    pole_parameter_grids,
    skin_band,
)


def rear_kind(component) -> str:
    """Return ``pole``, ``base``, ``open`` or ``boat_tail``, the form of the rear end."""
    rear = component["rear"]
    return "boat_tail" if isinstance(rear, dict) else rear


def rear_end(component) -> float:
    """Return the axial fraction where the rear of the body ends."""
    if rear_kind(component) == "boat_tail":
        return float(component["rear"]["boat_tail"]["x_cut"])
    return 1.0


def is_full(component) -> bool:
    """Whether the body is a whole body about its axis rather than the half at y >= 0."""
    return component.get("symmetry", "half_y0") == "full"


def body_station_table(component):
    """Return the shape values (width, height, centre, two exponents) at the control stations."""
    controls = np.asarray(component["stations"], float)
    rows = []
    for section in component["sections"]:
        position = section["at"]
        if isinstance(position, str):
            position = {
                "nose": controls[0],
                "front": controls[0],
                "middle": controls[1],
                "rear": controls[-1],
            }[position]
        if section.get("shape", "circle") == "circle":
            row = [section["radius"], section["radius"], 0.0, 2.0, 2.0]
        else:
            row = [
                section["width"],
                section["height"],
                section.get("centre_height", 0.0),
                section["n_top"],
                section["n_bottom"],
            ]
        rows.append((position, row))
    rows.sort(key=lambda row: row[0])
    locations = np.array([row[0] for row in rows])
    values = np.array([row[1] for row in rows])
    # A lone declared section covers the full reference body. Multiple sections
    # extend their endpoint values to the bounding transition stations.
    if len(rows) == 1:
        station_values = np.broadcast_to(values[0], (len(controls), 5)).copy()
    else:
        station_values = spline_distribution(
            locations, values, np.clip(controls, locations[0], locations[-1])
        )[0]
    return {
        "stations": controls,
        "values": station_values,
        "axisymmetric": bool(
            np.all(station_values[:, 0] == station_values[:, 1])
            and np.all(station_values[:, 2] == 0)
            and np.all(station_values[:, 3:] == 2)
        ),
    }


def _ring_row(number, row):
    """Return one rings-file row as numbers, or refuse it naming its line."""
    defaults = {"centre_height": 0.0, "n_top": 2.0, "n_bottom": 2.0}
    try:
        values = {key: float(row[key]) for key in ("x", "width", "height")}
        values |= {key: float(row.get(key) or default) for key, default in defaults.items()}
    except (TypeError, ValueError) as error:
        raise InputValueError(
            f"rings_file: line {number} holds a value that is no number"
        ) from error
    return values


def read_rings(path) -> tuple[list[float], list[dict]]:
    """Read a rings file: sections measured elsewhere, one row per station.

    The file is CSV with the header ``x`` (axial fraction), ``width``, ``height`` and optionally
    ``centre_height``, ``n_top`` and ``n_bottom`` (defaults 0, 2 and 2). Returns the stations and
    the superellipse sections, in the form ``sections`` takes.

    Raises
    ------
    InputValueError
        When the file is empty, lacks a column, holds a value that is not a number, has fewer
        than three rows, or its stations are not strictly increasing inside (0, 1).
    """
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise InputValueError(f"rings_file: cannot read {path}") from error
    if not rows or not {"x", "width", "height"} <= set(rows[0]):
        raise InputValueError("rings_file: a header with x, width and height is required")
    values = [_ring_row(number, row) for number, row in enumerate(rows, start=2)]
    stations = [v.pop("x") for v in values]
    if len(rows) < 3 or any(b <= a for a, b in zip(stations, stations[1:], strict=False)):
        raise InputValueError("rings_file: at least three rows with increasing x are required")
    if stations[0] <= 0 or stations[-1] >= 1:
        raise InputValueError("rings_file: x must lie inside (0, 1)")
    sections = [
        {"shape": "superellipse", "at": x, **v} for x, v in zip(stations, values, strict=True)
    ]
    return stations, sections


def apply_rings(component, base, warnings) -> None:
    """Replace the sections and stations of a body by those of its ``rings_file``.

    A missing file is left for the data-file check to refuse; a file that is not a rings table
    is warned about and skipped. Given sections are replaced, and that is warned, so a file and a
    table never disagree silently.
    """
    path = component.get("rings_file", "")
    if not path or not (Path(base) / path).is_file():
        return
    try:
        stations, sections = read_rings(Path(base) / path)
    except InputValueError as error:
        warnings.append(f"{error}; skipped, the sections of the table are used")
        return
    if component["sections"] != [{"at": "middle", "shape": "circle", "radius": 0.1}]:
        warnings.append("sections: replaced by rings_file")
    component["stations"], component["sections"] = stations, sections


def check_body_symmetry(raw, component) -> None:
    """Check ``symmetry``, ``nose`` and the azimuths of a body; a whole body spans 0 to 360."""
    if component["symmetry"] not in ("full", "half_y0"):
        raise InputValueError("symmetry: expected full or half_y0")
    if component["nose"] not in ("pole", "open"):
        raise InputValueError("nose: expected pole or open")
    top = 360 if component["symmetry"] == "full" else 180
    if "azimuths" not in raw and top == 360:
        component["azimuths"] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0]
    check_stations(component["azimuths"], "azimuths", 4, 0, top)
    if component["azimuths"][0] != 0 or component["azimuths"][-1] != top:
        what = (
            "whole body requires 0 and 360" if top == 360 else "half body requires top and bottom"
        )
        raise InputValueError(f"azimuths: {what}")


def _hermite(form, table):
    """Return the boat-tail law of width, height and centre height along the axial fraction.

    Each of the three runs on its own cubic from its value and its own slope at the last station,
    so the profile is tangent to the run in every direction, to the base: the value at the last
    station scaled by ``base_height / height``, with the slope of the straight closure there.
    The law returns the three values and their derivatives in the axial fraction, last axis.
    """
    last = float(table["stations"][-1])
    x_cut = float(form["x_cut"])
    values, slope = spline_distribution(table["stations"], table["values"], last)
    height = float(values[1])
    if not height > 0 or not x_cut > last:
        raise InputValueError("rear.boat_tail: x_cut must lie beyond the last station of a body")
    span = x_cut - last
    end = float(form["base_height"]) / height
    start, start_slope = np.asarray(values[:3], float), np.asarray(slope[:3], float)
    stop, stop_slope = start * end, start * (end - 1.0) / span

    def law(axial):
        t = np.clip((np.asarray(axial, float) - last) / span, 0.0, 1.0)[..., None]
        value = (
            (2 * t**3 - 3 * t**2 + 1) * start
            + (t**3 - 2 * t**2 + t) * span * start_slope
            + (-2 * t**3 + 3 * t**2) * stop
            + (t**3 - t**2) * span * stop_slope
        )
        derivative = (
            (6 * t**2 - 6 * t) * start / span
            + (3 * t**2 - 4 * t + 1) * start_slope
            + (-6 * t**2 + 6 * t) * stop / span
            + (3 * t**2 - 2 * t) * stop_slope
        )
        return value, derivative

    return law


def check_boat_tail(component) -> None:
    """Refuse a boat-tail closure whose steepest slope exceeds ``angle_limit_deg``.

    The slope is the angle of the profile to the axis, over the closure from the last station to
    ``x_cut``, on the side, the crown and the keel alike. Does nothing for another rear.

    Raises
    ------
    InputValueError
        When ``x_cut`` is not beyond the last station, or the law needs more than the limit.
    """
    if rear_kind(component) != "boat_tail":
        return
    form = component["rear"]["boat_tail"]
    law = _hermite(form, body_station_table(component))
    _, derivative = law(np.linspace(float(component["stations"][-1]), form["x_cut"], 401))
    # the side runs with the width; the crown and the keel with the centre plus or minus height
    width, height, centre = np.moveaxis(derivative, -1, 0)
    lines = np.stack([width, centre + height, centre - height])
    steepest = np.max(np.abs(lines)) / component["length"]
    angle = float(np.degrees(np.arctan(steepest)))
    if angle > form["angle_limit_deg"] + 1e-9:
        raise InputValueError(
            f"rear.boat_tail: the closure needs {angle:.3g} degrees, above angle_limit_deg "
            f"{form['angle_limit_deg']:.3g}; move x_cut aft or raise base_height"
        )


def _closed_end(component, table, axial, values, nose):
    """Return ``values`` with the nose or the rear closed by its exponent."""
    first, last = table["stations"][[0, -1]]
    end = first if nose else last
    mask = axial < first if nose else axial > last
    exponent = component["nose_exponent" if nose else "rear_exponent"]
    endpoint, slope = spline_distribution(table["stations"], table["values"], end)
    linear = endpoint + (axial[..., None] - end) * slope
    s = axial / first if nose else (1 - axial) / (1 - last)
    s = np.clip(s, 0, 1)
    closure = s**exponent * (1 + exponent * (1 - s))
    base = 0.0 if nose or component["rear"] == "pole" else component["base_radius"]
    linear[..., :2] = base + (linear[..., :2] - base) * closure[..., None]
    return np.where(mask[..., None], linear, values)


def _boat_tail(component, table, axial, values):
    """Return ``values`` with the rear closed by the boat-tail law from the last station."""
    shape, _ = _hermite(component["rear"]["boat_tail"], table)(axial)
    last = spline_distribution(table["stations"], table["values"], float(table["stations"][-1]))[0]
    scaled = np.array(last, float) * np.ones((*axial.shape, 1))
    scaled[..., :3] = shape
    return np.where((axial > table["stations"][-1])[..., None], scaled, values)


def body_profile(component, table, axial):
    """Return the shape values at axial fractions, closing nose and rear by their exponents."""
    axial = np.asarray(axial, float)
    first, last = table["stations"][[0, -1]]
    query = np.clip(axial, first, last)
    values, _ = spline_distribution(table["stations"], table["values"], query)
    if component.get("nose", "pole") == "pole":
        values = _closed_end(component, table, axial, values, True)
    kind = rear_kind(component)
    if kind == "boat_tail":
        values = _boat_tail(component, table, axial, values)
    elif kind != "open":
        values = _closed_end(component, table, axial, values, False)
    if np.any(values[..., :2] < -1e-12) or np.any(values[..., 3:] <= 0):
        raise InputValueError(
            "sections: profile interpolation makes width, height or exponent invalid"
        )
    if table["axisymmetric"]:
        values[..., 1] = values[..., 0]
    return values


def body_points(component, table, axial, azimuth):
    """Return the surface points at axial fractions and azimuths (radians), in the body frame."""
    axial, azimuth = np.broadcast_arrays(axial, azimuth)
    parameters = body_profile(component, table, axial)
    width, height, centre, top, bottom = np.moveaxis(parameters, -1, 0)
    cosine = np.cos(azimuth)
    exponent = np.where(cosine >= 0, top, bottom)
    sine = np.sin(azimuth)
    side = sine if is_full(component) else np.maximum(0, sine)
    y = width * np.sign(side) * np.abs(side) ** (2 / exponent)
    z = centre + height * np.sign(cosine) * np.abs(cosine) ** (2 / exponent)
    return np.stack((component["length"] * axial, y, z), axis=-1)


def pole_patch_grid(component, table, angles, barycentric, nose):
    """Return the sample grid of one nose or rear cap quadrilateral."""
    radius = np.sum(barycentric, axis=-1)
    fraction = np.divide(barycentric[..., 1], radius, out=np.zeros_like(radius), where=radius > 0)
    theta = angles[0] + (angles[1] - angles[0]) * fraction
    if nose:
        axial = table["stations"][0] * radius ** (1 / component["nose_exponent"])
    else:
        axial = 1 - (1 - table["stations"][-1]) * radius ** (1 / component["rear_exponent"])
    return body_points(component, table, axial, theta)


def base_patch_grid(component, table, angles, barycentric):
    """Return the sample grid of one flat-base quadrilateral."""
    radius = np.sum(barycentric, axis=-1)
    fraction = np.divide(barycentric[..., 1], radius, out=np.zeros_like(radius), where=radius > 0)
    theta = angles[0] + (angles[1] - angles[0]) * fraction
    end = rear_end(component)
    rim = body_points(component, table, end, theta)
    centre = np.array(
        [end * component["length"], 0.0, body_profile(component, table, np.asarray(end))[2]]
    )
    return centre + radius[..., None] * (rim - centre)


def _with_ids(patch, role, ids, family=None, bay=None):
    return replace(patch, role=role, edge_ids=tuple(ids), family=family or patch.family, bay=bay)


def _cap_patches(component, table, angles, sector, plan):
    """Return the nose and rear cap patches of one azimuth sector, quad by quad."""
    kind = rear_kind(component)
    ends = []
    if component.get("nose", "pole") == "pole":
        ends.append((True, "nose", "nose_cap", "ring0"))
    if kind != "open":
        ends.append(
            (False, "rear" if kind == "pole" else "base", "rear_cap", f"ring{plan['rings']}")
        )
    patches = []
    s = clustered(plan["samples"])
    for q, barycentric in enumerate(pole_parameter_grids(plan["samples"])):
        for is_nose, family, role, ring in ends:
            if is_nose or kind == "pole":
                grid = pole_patch_grid(component, table, angles, barycentric, is_nose)
            else:
                grid = base_patch_grid(component, table, angles, barycentric)
            name = "nose" if is_nose else "rear"
            ids = cap_edge_ids(name, sector, plan["sectors"], ring, plan["full"])[q]
            role = "base" if family == "base" else role
            patches.append(_with_ids(interpolate_patch(grid, family, None, s, s), role, ids))
    return patches


def _transitions(component, table, junctions):
    """Return the axial fractions where a band is cut: stations, junctions and a flat rear."""
    stations = table["stations"]
    inside = (float(x) for x in junctions if stations[0] < x < stations[-1])
    transitions = sorted({*(float(s) for s in stations), *inside})
    if rear_kind(component) in ("base", "boat_tail"):
        transitions.append(rear_end(component))
    return transitions


def _half_edges(half, plan):
    """Return the names of the two side edges of an azimuth half, without the bay suffix."""
    halves, full = plan["halves"], plan["full"]
    low = "sym0" if half == 0 and not full else f"hseam{half}"
    high_half = (half + 1) % halves if full else half + 1
    high = "sym1" if half == halves - 1 and not full else f"hseam{high_half}"
    return low, high


def _band_patches(component, theta, half, plan):
    """Return the patches of one azimuth half: one skin through every ring, cut at the stations."""
    transitions = plan["transitions"]
    s = clustered(plan["samples"])
    axial = band_axial(transitions, plan["samples"])
    azimuth = theta[0] + s * (theta[1] - theta[0])
    # Native u is azimuth and v is axial in a band; the mesher reads that order.
    points = body_points(component, plan["table"], axial[None, :], azimuth[:, None])
    pieces = skin_band(points, s, axial, transitions)
    low, high = _half_edges(half, plan)
    patches = []
    for index, piece in enumerate(pieces):
        x0, x1 = transitions[index], transitions[index + 1]
        family = "rear" if x1 == 1 else ("front" if index == 0 else "middle")
        ids = (
            f"{low}.b{index}",
            f"{high}.b{index}",
            f"ring{index}.h{half}",
            f"ring{index + 1}.h{half}",
        )
        patches.append(_with_ids(piece, "band", ids, family, (float(x0), float(x1))))
    return patches


def body(component, junctions=()) -> list[Patch]:
    """Return the patches of a body, with extra station breaks at the ``junctions`` fractions.

    Parameters
    ----------
    component : dict
        Resolved body component (plain values, metres and degrees).
    junctions : sequence of float, optional
        Axial fractions at which extra station breaks are inserted; default empty.

    Returns
    -------
    list of Patch
        The cap and band patches of the body, in the body frame. Each carries a ``role``
        (``nose_cap``, ``rear_cap``, ``base`` or ``band``) and the names of its four edges, so
        two patches that share an edge carry the same name for it. A closed end is a cap at the
        pole or a flat base; an open end has no patch and its rim edges are the band's.

    Examples
    --------
    >>> from cfdbender.geometry._boundary import unpack
    >>> from cfdbender.geometry.config import resolve_config
    >>> raw = {"type": "body", "length_mm": 20000.0,
    ...        "sections": [{"at": "middle", "shape": "circle", "radius_mm": 1200.0}]}
    >>> patches = body(unpack(resolve_config(raw))["components"]["COMPONENT"])
    >>> len(patches) > 0
    True
    """
    best, last = None, float("inf")
    for samples in SAMPLE_LADDER:
        patches = _build(component, junctions, samples)
        turn = _pole_turn(patches)
        if turn <= POLE_TARGET_DEG:
            return patches
        if best is not None and turn > 0.8 * last:
            return best  # no convergence: a true apex, which more samples cannot smooth
        best, last = patches, turn
    return patches


def _pole_turn(patches) -> float:
    """Return the largest turn, in degrees, across a seam between two cap patches of one end."""
    caps = [p for p in patches if p.role in ("nose_cap", "rear_cap")]
    found = edge_jumps(caps)
    return max((r["jump_deg"] for r in found), default=0.0)


def _build(component, junctions, samples):
    """Return the patches of a body sampled with ``samples`` points along every edge."""
    table = body_station_table(component)
    angles = np.deg2rad(component["azimuths"])
    transitions = _transitions(component, table, junctions)
    pairs = list(zip(angles, angles[1:], strict=False))
    plan = {
        "sectors": len(pairs),
        "halves": 2 * len(pairs),
        "full": is_full(component),
        "rings": len(transitions) - 1,
        "samples": samples,
        "table": table,
        "transitions": transitions,
    }
    patches = []
    for sector, (a, b) in enumerate(pairs):
        patches += _cap_patches(component, table, (a, b), sector, plan)
        for offset, (low, high) in enumerate([(a, (a + b) / 2), ((a + b) / 2, b)]):
            patches += _band_patches(component, (low, high), 2 * sector + offset, plan)
    return patches


def axisymmetric_radius(component, axial):
    """Return the radius at axial fractions of an axisymmetric body; refuse any other."""
    table = body_station_table(component)
    if not table["axisymmetric"]:
        raise InputValueError("sections: not axisymmetric")
    return body_profile(component, table, np.asarray(axial))[..., 0]


# ---------------------------------------------------------------------------------------------
# Region layouts: a body is cut into faces by the rule of its type (fuselage or nacelle), and each
# face is fitted with the fewest spans that hold it within a tolerance of the law.
# ---------------------------------------------------------------------------------------------

#: The layouts a body can follow: a fuselage is cut where its definition changes, a nacelle in bays.
LAYOUTS = ("fuselage", "nacelle")
#: Largest span count of a direction before the tolerance is called unreachable.
MAX_SPANS = 16
_CORNERS = {
    "a": (0.0, 0.0),
    "b": (1.0, 0.0),
    "c": (0.0, 1.0),
}


def _width(component, table, x) -> float:
    """Return the half width of the law at the axial fraction ``x``."""
    return float(body_profile(component, table, np.asarray(x, float))[0])


def _cap_end(component, table, radius, nose) -> float:
    """Return the axial fraction where the half width of a closing end equals ``radius``."""
    first, last = (float(table["stations"][0]), float(table["stations"][-1]))
    near, far = (0.0, first) if nose else (last, 1.0)
    if (nose and _width(component, table, far) <= radius) or (
        not nose and _width(component, table, near) <= radius
    ):
        return far if nose else near
    for _ in range(80):
        mid = 0.5 * (near + far)
        grows = _width(component, table, mid) < radius
        near, far = (mid, far) if grows == nose else (near, mid)
    return 0.5 * (near + far)


def _constant_runs(component, table) -> list[tuple[float, float]]:
    """Return the intervals between stations over which the section does not change at all."""
    runs = []
    for x0, x1 in zip(table["stations"][:-1], table["stations"][1:], strict=True):
        values = body_profile(component, table, np.linspace(x0, x1, 9))
        if np.ptp(values, axis=0).max() <= 1e-12 * max(1.0, float(np.abs(values).max())):
            runs.append((float(x0), float(x1)))
            if len(runs) > 1 and runs[-2][1] == runs[-1][0]:
                runs[-2:] = [(runs[-2][0], runs[-1][1])]
    return runs


def _ends(component, table, cap_fraction):
    """Return where the bands start and end, and the closing patches of the two ends."""
    stations = table["stations"]
    radius = cap_fraction * float(np.max(table["values"][:, 0]))
    nose = component.get("nose", "pole") == "pole"
    kind = rear_kind(component)
    start = _cap_end(component, table, radius, True) if nose else float(stations[0])
    if kind == "pole":
        end = _cap_end(component, table, radius, False)
    else:
        end = rear_end(component) if kind in ("base", "boat_tail") else float(stations[-1])
    return start, end, nose, kind


def region_plan(component, layout="fuselage", cap_fraction=0.004, regions=()) -> dict:
    """Return the axial regions of a body under the rule of its layout.

    A ``fuselage`` is cut only where its definition changes: the start and end of a constant run
    of the section (a region that is straight, degree 1 axially), the start of the rear closure
    or of a boat-tail, the base, and every station listed in ``regions``. A ``nacelle`` is cut in
    bays at its stations and at those listed, and every bay is a degree 5 face.

    Parameters
    ----------
    component : dict
        Resolved body component.
    layout : {"fuselage", "nacelle"}
        The layout rule.
    cap_fraction : float, optional
        Radius of the nose or pole cap as a fraction of the largest half width; default 0.004.
    regions : sequence of float, optional
        Extra axial fractions at which to cut.

    Returns
    -------
    dict
        ``regions`` (``x0``, ``x1``, ``constant`` per region), ``nose`` and ``rear`` (the kind of
        each end), ``layout`` and ``degree``.

    Raises
    ------
    InputValueError
        When the layout is unknown, the cap fraction is not positive or a listed station is not
        inside the body.
    """
    if layout not in LAYOUTS:
        raise InputValueError("layout: expected fuselage or nacelle")
    if not cap_fraction > 0:
        raise InputValueError("cap_fraction: expected a positive fraction")
    table = body_station_table(component)
    start, end, nose, kind = _ends(component, table, cap_fraction)
    extra = [float(x) for x in regions]
    if any(not start < x < end for x in extra):
        raise InputValueError("regions: every station must lie inside the body")
    marks = {start, end, *extra}
    runs = _constant_runs(component, table) if layout == "fuselage" else []
    if layout == "fuselage":
        marks |= {float(x) for run in runs for x in run if start < x < end}
        if kind != "open":
            marks.add(float(table["stations"][-1]))
    else:
        marks |= {float(x) for x in table["stations"] if start < x < end}
    cuts = sorted(x for x in marks if start <= x <= end)
    cuts = [x for i, x in enumerate(cuts) if i == 0 or x - cuts[i - 1] > 1e-9]
    found = [
        {
            "x0": a,
            "x1": b,
            "constant": any(r0 <= a + 1e-12 and b <= r1 + 1e-12 for r0, r1 in runs),
        }
        for a, b in zip(cuts, cuts[1:], strict=False)
    ]
    return {
        "regions": found,
        "nose": "pole" if nose else "open",
        "rear": kind,
        "layout": layout,
        "degree": 3 if layout == "fuselage" else 5,
    }


def _axial_map(component, table, region, plan):
    """Return the map from the axial parameter of a region to the axial fraction.

    Toward a closing end the half width goes as a power of the distance, so the region is
    parametrized by that power and the law stays a polynomial in it.
    """
    x0, x1 = region["x0"], region["x1"]
    first, last = float(table["stations"][0]), float(table["stations"][-1])
    exponent_nose, exponent_rear = component["nose_exponent"], component["rear_exponent"]
    if plan["nose"] == "pole" and x0 == plan["regions"][0]["x0"] and exponent_nose != 1.0:
        w0, w1 = (x0 / first) ** exponent_nose, (x1 / first) ** exponent_nose
        return lambda v: first * (w0 + v * (w1 - w0)) ** (1.0 / exponent_nose)
    closing = plan["rear"] in ("pole", "base") and exponent_rear != 1.0
    if closing and x1 == plan["regions"][-1]["x1"]:
        span = 1.0 - last
        w0, w1 = ((1.0 - x0) / span) ** exponent_rear, ((1.0 - x1) / span) ** exponent_rear
        return lambda v: 1.0 - span * np.maximum(w0 + v * (w1 - w0), 0.0) ** (1.0 / exponent_rear)
    return lambda v: x0 + v * (x1 - x0)


def _polar_points(component, table, axial, psi):
    """Return the surface points at axial fractions and polar angles ``psi`` of the section.

    :func:`body_points` reads its azimuth as the angle of a superellipse, whose points move as
    ``|sin|`` to the power ``2 / n``: for an exponent above 2 the derivative is infinite on the
    symmetry plane and the side line, which no polynomial face follows. The region faces are
    parametrized instead by the polar angle of the point in the section scaled to a unit square,
    ``(y / width, (z - centre) / height) = rho (sin psi, cos psi)``, whose point is smooth in
    ``psi`` for an even exponent and C2 or better for any exponent of at least 3. ``psi`` and the
    superellipse angle agree at 0, a half and a whole turn, so the face edges stay where they were.
    """
    axial, psi = np.broadcast_arrays(np.asarray(axial, float), np.asarray(psi, float))
    values = body_profile(component, table, axial)
    sine, cosine = np.sin(psi), np.cos(psi)
    n = np.where(cosine >= 0, values[..., 3], values[..., 4])
    rho = (np.abs(sine) ** n + np.abs(cosine) ** n) ** (-1.0 / n)
    theta = np.arctan2(
        np.sign(sine) * (rho * np.abs(sine)) ** (n / 2),
        np.sign(cosine) * (rho * np.abs(cosine)) ** (n / 2),
    )
    return body_points(component, table, axial, theta)


def _band_function(component, table, theta, axial):
    """Return the surface function ``(u, v)`` of a band face: ``u`` azimuth, ``v`` axial."""
    low, high = theta

    def point(u, v):
        return _polar_points(component, table, axial(v), low + u * (high - low))

    return point


def _disc_polar(w, angles):
    """Return the azimuth and radius of a point of a cap triangle, smooth through the pole.

    ``w`` are the weights of the two rim vertices. Near the pole the point is the straight blend
    of the two rim directions, which is smooth; toward the rim the blend gives way to the true
    polar point, so the rim itself is the arc, uniform in azimuth. A polar map alone has no
    limit at the pole, and a polynomial patch cannot follow it.
    """
    rim = np.sum(w, axis=-1)
    fraction = np.divide(w[..., 1], rim, out=np.zeros_like(rim), where=rim > 0)
    theta = angles[0] + (angles[1] - angles[0]) * fraction
    direction = lambda a: np.array([np.sin(a), np.cos(a)])  # noqa: E731
    straight = w[..., :1] * direction(angles[0]) + w[..., 1:] * direction(angles[1])
    polar = rim[..., None] * np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    weight = (rim**2)[..., None]
    blend = (1.0 - weight) * straight + weight * polar
    return np.arctan2(blend[..., 0], blend[..., 1]), np.hypot(blend[..., 0], blend[..., 1])


def _cap_function(component, table, angles, corners, *, ring, end):
    """Return the surface function of one quadrilateral of a cap, as the original network does."""
    p = [np.asarray(c, float) for c in corners]
    exponent = component["nose_exponent" if end == "nose" else "rear_exponent"]
    last = rear_end(component)
    centre_height = float(body_profile(component, table, np.asarray(last))[2])

    def point(u, v):
        u, v = np.asarray(u, float)[..., None], np.asarray(v, float)[..., None]
        w = (1 - u) * (1 - v) * p[0] + u * (1 - v) * p[1] + u * v * p[2] + (1 - u) * v * p[3]
        theta, radius = _disc_polar(w, angles)
        if end == "base":
            rim = _polar_points(component, table, last, theta)
            centre = np.array([last * component["length"], 0.0, centre_height])
            return centre + radius[..., None] * (rim - centre)
        if end == "nose":
            axial = ring * radius ** (1.0 / exponent)
        else:
            axial = 1.0 - (1.0 - ring) * radius ** (1.0 / exponent)
        return _polar_points(component, table, axial, theta)

    return point


def _quads():
    """Return the corner weights of the three quadrilaterals of one cap triangle."""
    a, b, c = (np.array(_CORNERS[k]) for k in "abc")
    ab, bc, ca, centre = (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3
    return [(a, ab, centre, ca), (ab, b, bc, centre), (centre, bc, c, ca)]


def _closing_ends(component, plan):
    """Return ``(end, axial fraction, ring name, role)`` of each closed end of the body."""
    regions, last = plan["regions"], len(plan["regions"])
    ends = []
    if plan["nose"] == "pole":
        ends.append(("nose", regions[0]["x0"], "ring0", "nose_cap"))
    if plan["rear"] == "pole":
        ends.append(("rear", regions[-1]["x1"], f"ring{last}", "rear_cap"))
    if plan["rear"] in ("base", "boat_tail"):
        ends.append(("base", rear_end(component), f"ring{last}", "base"))
    return ends


def _faces(component, table, plan):
    """Return every face of the body as a record: its surface function, names and degrees."""
    full = is_full(component)
    angles = [(0.0, np.pi)] + ([(np.pi, 2 * np.pi)] if full else [])
    shape = {"sectors": len(angles), "halves": 2 * len(angles), "full": full}
    regions, last = plan["regions"], len(plan["regions"])
    faces = []
    ends = _closing_ends(component, plan)
    for sector, pair in enumerate(angles):
        for end, ring_x, ring, role in ends:
            for q, corners in enumerate(_quads()):
                ids = cap_edge_ids(
                    "nose" if end == "nose" else "rear", sector, shape["sectors"], ring, full
                )[q]
                point = _cap_function(component, table, pair, corners, ring=ring_x, end=end)
                creased = (False, q != 0, False, False) if end == "base" else (False,) * 4
                faces.append(
                    {
                        "point": point,
                        "role": role,
                        "ids": ids,
                        "family": "nose" if end == "nose" else "rear",
                        "bay": None,
                        "degrees": (3, 3),
                        "kind": "cap",
                        "crease": creased,
                        "region": None,
                    }
                )
        mid = sum(pair) / 2
        for offset, theta in enumerate([(pair[0], mid), (mid, pair[1])]):
            half = 2 * sector + offset
            low, high = _half_edges(half, shape)
            for k, region in enumerate(regions):
                axial = _axial_map(component, table, region, plan)
                ids = (f"{low}.b{k}", f"{high}.b{k}", f"ring{k}.h{half}", f"ring{k + 1}.h{half}")
                family = "front" if k == 0 else ("rear" if k == last - 1 else "middle")
                faces.append(
                    {
                        "point": _band_function(component, table, theta, axial),
                        "role": "band",
                        "ids": ids,
                        "family": family,
                        "bay": (region["x0"], region["x1"]),
                        "degrees": (3, 1 if region["constant"] else plan["degree"]),
                        "kind": "band",
                        "crease": (
                            False,
                            False,
                            False,
                            k == last - 1 and plan["rear"] in ("base", "boat_tail"),
                        ),
                        "region": k,
                    }
                )
    for face in faces:
        if face["kind"] == "cap":
            face["degrees"] = (plan["degree"], plan["degree"])
        else:
            face["degrees"] = (plan["degree"], face["degrees"][1])
    return faces


def _fit_face(face, n_az, n_ax):
    """Fit one face on the given span counts; returns the patch, residual and edge residuals."""
    spans = (n_az, n_ax[face["region"]] if face["kind"] == "band" else n_az)
    smooth = tuple(not c for c in face["crease"])
    return fit_patch(face["point"], spans, face["degrees"], smooth, face["family"])


def _grow(faces, results, n_az, n_ax, tolerance):
    """Return the span counts after one step toward the tolerance; None when it is met."""
    n_az2, n_ax2, met = n_az, list(n_ax), True
    for face, (_, residual, (along_u, along_v)) in zip(faces, results, strict=True):
        if residual <= tolerance:
            continue
        met = False
        needs_u = along_u > tolerance / 2 or face["kind"] == "cap"
        needs_v = along_v > tolerance / 2 and face["kind"] == "band"
        if not (needs_u or needs_v):
            needs_u = needs_v = True
        if needs_u:
            n_az2 = n_az + 1
        if needs_v and face["degrees"][1] > 1:
            n_ax2[face["region"]] = n_ax[face["region"]] + 1
    return None if met else (n_az2, n_ax2)


def body_regions(component, layout="fuselage", tolerance=None, cap_fraction=0.004, regions=()):
    """Return the patches of a body cut by the region rule of its layout.

    Each face is fitted with the fewest uniform spans that hold it within ``tolerance`` of the law
    of the body, and faces that share an edge share its curve, so the seams are the law's own: a
    tangent-continuous join measures the law and not the fit.

    Parameters
    ----------
    component : dict
        Resolved body component.
    layout : {"fuselage", "nacelle"}
        The layout rule, see :func:`region_plan`.
    tolerance : float, optional
        Largest distance in metres from the law at the samples; default 1e-4 of the body length.
    cap_fraction : float, optional
        Radius of a closing cap as a fraction of the largest half width.
    regions : sequence of float, optional
        Extra axial fractions at which to cut.

    Returns
    -------
    list of Patch
        Cap and band patches with ``role`` and ``edge_ids``; roles ``nose_cap``, ``rear_cap``,
        ``base`` and ``band``.

    Raises
    ------
    InputValueError
        As :func:`region_plan`, or when the tolerance needs more than ``MAX_SPANS`` spans.
    """
    plan = region_plan(component, layout, cap_fraction, regions)
    tolerance = 1e-4 * component["length"] if tolerance is None else float(tolerance)
    if not tolerance > 0:
        raise InputValueError("tolerance: expected a positive distance")
    table = body_station_table(component)
    faces = _faces(component, table, plan)
    n_az, n_ax = 1, [1] * len(plan["regions"])
    while True:
        results = [_fit_face(f, n_az, n_ax) for f in faces]
        step = _grow(faces, results, n_az, n_ax, tolerance)
        if step is None:
            break
        n_az, n_ax = step
        if n_az > MAX_SPANS or max(n_ax) > MAX_SPANS:
            raise InputValueError(f"tolerance: {tolerance:g} needs more than {MAX_SPANS} spans")
    return [
        _with_ids(patch, face["role"], face["ids"], face["family"], face["bay"])
        for face, (patch, _, _) in zip(faces, results, strict=True)
    ]

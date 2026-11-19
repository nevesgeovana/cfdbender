"""Reverse identification of a lifting surface from section cuts: the pure numerical half.

Pipeline role: the numerical layer of the reverse step. Given closed section contours cut from
an original skin at a series of span stations, it classifies the stations that are clean
(topology, smooth chord and thickness trends), measures chord, incidence and thickness of each
cut, builds the reference planform between a root and the outermost clean cut, and fits a CST
section to a clean cut. There is no geometry kernel here: the cuts come from the kernel
stage that cuts the original skin.

A station is rejected, never repaired: a cut that is not one closed contour with one leading and
one trailing extremum raises :class:`StationRejectedError`, and the scan records the reason. Section
fits are measured against the cut itself on an independent grid; the result states whether its
maximum residual met the tolerance and the caller decides what to do with a section that did
not.

Examples
--------
>>> import numpy as np
>>> from cfdbender.geometry._reverse import cosine
>>> np.round(cosine(3), 12).tolist()
[0.0, 0.5, 1.0]
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

from cfdbender._errors import InputValueError
from cfdbender.geometry._airfoils import cst_airfoil, fit_cst, normalized_naca

__all__ = [
    "StationRejectedError",
    "clean_band",
    "closed_contour",
    "contour_features",
    "cosine",
    "fit_section",
    "leading_extension",
    "reference_planform",
    "refine_tip",
]

#: Closing tolerance of a cut in meters: a chain of cut segments whose end points are further
#: apart than this is an open or multiple contour, not one section.
CLOSURE_TOLERANCE_M = 5e-5


class StationRejectedError(ValueError):
    """A cut that cannot be taken as one closed airfoil contour; the message says why."""


def cosine(count):
    """Return ``count`` cosine-spaced values from 0 to 1.

    Parameters
    ----------
    count : int
        Number of samples; at least three.

    Returns
    -------
    numpy.ndarray
        The samples, shape ``(count,)``.

    Raises
    ------
    InputValueError
        If ``count`` is below three.
    """
    if count < 3:
        raise InputValueError("count: at least three samples required")
    return (1 - np.cos(np.linspace(0, np.pi, count))) / 2


def closed_contour(segments, tolerance=CLOSURE_TOLERANCE_M):
    """Chain cut segments end to end into one closed contour or reject the station.

    Parameters
    ----------
    segments : iterable of array_like
        Polyline segments of one cut, each ``(n, 3)``.
    tolerance : float, optional
        Largest endpoint or closure gap, meters; default ``CLOSURE_TOLERANCE_M``.

    Returns
    -------
    numpy.ndarray
        The closed contour, shape ``(m, 3)``, whose last point equals its first.
    """
    lines = []
    for line in segments:
        line = np.asarray(line)
        line = line[np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-10]]
        if len(line) > 1:
            lines.append(line)
    if not lines:
        raise StationRejectedError("empty cut")
    chain = lines.pop(0)
    while lines:
        candidates = [
            (np.linalg.norm(chain[-1] - p[k]), i, k) for i, p in enumerate(lines) for k in (0, -1)
        ]
        gap, i, end = min(candidates)
        if gap > tolerance:
            raise StationRejectedError(f"open or multiple contour: endpoint gap {gap:.9g} m")
        p = lines.pop(i)
        if end == -1:
            p = p[::-1]
        chain = np.vstack([chain, p[1:]])
    gap = np.linalg.norm(chain[-1] - chain[0])
    if gap > tolerance:
        raise StationRejectedError(f"open contour: closure gap {gap:.9g} m")
    chain[-1] = chain[0]
    return chain


def contour_features(contour, span):
    """Measure one closed cut: leading and trailing edge, chord, incidence, thickness ratio.

    Parameters
    ----------
    contour : array_like
        Closed contour of the cut, shape ``(n, 3)``.
    span : array_like
        Spanwise direction of the cut.

    Returns
    -------
    dict
        The ``le`` and ``te`` points, ``chord``, ``thickness_ratio``, ``incidence``, the
        ``contour`` and its ``unit_contour``, the ``extrema`` count and the
        ``trailing_edge_endpoints``.
    """
    p = np.asarray(contour)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]]
    if np.linalg.norm(p[-1] - p[0]) > 1e-9:
        p = np.vstack([p, p[0]])
    normal = np.cross([1.0, 0.0, 0.0], span)
    normal /= np.linalg.norm(normal)
    # A span axis toward -y turns the cross product down. "Up" stays +z on either side, so a
    # left half measures the incidence and the upper surface of its mirror image.
    side = -1.0 if normal[2] < 0 else 1.0
    normal = side * normal
    # TE base midpoint, then geometric nose farthest from that midpoint.
    xmax, xmin = p[:, 0].max(), p[:, 0].min()
    aft = p[p[:, 0] > xmax - max(1e-9, (xmax - xmin) * 0.000002)]
    ring = p[:-1]
    before = ring - np.roll(ring, 1, axis=0)
    after = np.roll(ring, -1, axis=0) - ring
    cosine_turn = np.sum(before * after, axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-30
    )
    corners = (cosine_turn < np.cos(np.deg2rad(30))) & (ring[:, 0] > xmax - 0.02 * (xmax - xmin))
    if corners.sum() >= 2:
        aft = ring[corners]
    te_ends = np.array([aft[np.argmax(aft @ normal)], aft[np.argmin(aft @ normal)]])
    te = te_ends.mean(axis=0)
    index = int(np.argmax(np.linalg.norm(p - te, axis=1)))
    # Resolve the smooth nose between cut samples, rather than allowing the
    # sample grid to choose incidence and geometric chord.
    unique = p[:-1]
    local = unique[(np.arange(index - 4, index + 5) % len(unique))]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-12]
    curve = CubicSpline(distance[keep], local[keep])
    optimum = minimize_scalar(
        lambda t: -float(np.sum((curve(t) - te) ** 2)),
        bounds=(distance[3], distance[5]),
        method="bounded",
        options={"xatol": 1e-13},
    )
    le = curve(optimum.x)
    chord = float(np.linalg.norm(te - le))
    if chord < 1e-6:
        raise StationRejectedError("collapsed chord")
    axis = (te - le) / chord
    up = side * np.cross(axis, span)
    up /= np.linalg.norm(up)
    q = np.column_stack(((p - le) @ axis / chord, (p - le) @ up / chord))
    # A valid airfoil has one forward and one backward chordwise traversal.
    d = np.diff(q[:, 0])
    signs = np.sign(d[np.abs(d) > 2e-5])
    turns = int(np.sum(signs != np.roll(signs, 1)))
    if turns != 2:
        raise StationRejectedError(f"not one LE and TE extremum: {turns} direction changes")
    ratio = float(np.ptp(q[:, 1]))
    if ratio < 1e-5:
        raise StationRejectedError("collapsed thickness")
    angle = float(-np.degrees(np.arctan2(axis @ normal, axis[0])))
    return {
        "le": le,
        "te": te,
        "chord": chord,
        "thickness_ratio": ratio,
        "incidence": angle,
        "contour": p,
        "unit_contour": q,
        "extrema": turns,
        "trailing_edge_endpoints": te_ends,
    }


def _longest_run(mask):
    runs = np.split(np.arange(len(mask)), np.flatnonzero(np.diff(mask.astype(int))) + 1)
    runs = [r for r in runs if mask[r[0]]]
    if not runs:
        raise InputValueError("clean band: no station is clean")
    return max(runs, key=len)


def clean_band(stations, features, trend_tolerance=0.0001):
    """Classify stations by topology, then chord and thickness against interior smooth trends.

    The interior polynomial trend is a classifier, never the output planform. A departure is
    rejected conservatively and recorded. Returns the indices of the longest clean run, the
    mask of every station that passed and the diagnostics of both trends.

    Parameters
    ----------
    stations : array_like
        Span station of each cut.
    features : sequence
        Features of each cut, or None where the cut was rejected.
    trend_tolerance : float, optional
        Relative departure from the interior trend that rejects a station; default 1e-4.

    Returns
    -------
    tuple
        ``(run, mask, diagnostics)``: indices of the longest clean run, the pass mask and the
        trend diagnostics.

    Raises
    ------
    InputValueError
        If fewer than eight interior cuts are closed, no station is clean, or the clean run
        has fewer than eight cuts.
    """
    s = np.asarray(stations)
    valid = np.array([f is not None for f in features])
    interior = valid & (s > s[0] + 0.2 * np.ptp(s)) & (s < s[0] + 0.75 * np.ptp(s))
    if interior.sum() < 8:
        raise InputValueError("clean band: insufficient closed interior cuts")
    mask = valid.copy()
    diagnostics = {}
    chords = np.array([f["chord"] if f else np.nan for f in features])
    model = np.polynomial.Polynomial.fit(s[interior], chords[interior], 3)
    predicted = model(s)
    relative = abs(chords - predicted) / np.maximum(abs(predicted), 1e-12)
    mask &= relative < trend_tolerance
    diagnostics["chord"] = {
        "observed": chords.tolist(),
        "trend": predicted.tolist(),
        "relative_error": relative.tolist(),
    }
    ratios = np.array([f["thickness_ratio"] if f else np.nan for f in features])
    # Thickness over chord is rational even on a linearly lofted skin: fit the dimensional
    # thickness slope and test each interval against it.
    thickness = ratios * chords
    slopes = np.diff(thickness) / np.diff(s)
    mid = (s[1:] + s[:-1]) / 2
    inside = interior[1:] & interior[:-1] & np.isfinite(slopes)
    trend = np.polynomial.Polynomial.fit(mid[inside], slopes[inside], 2)
    residual = slopes[inside] - trend(mid[inside])
    noise = np.median(abs(residual - np.median(residual))) * 1.4826
    allowance = max(
        8 * noise,
        0.1 * np.median(abs(slopes[inside])),
        1e-5 * np.nanmedian(thickness) / np.ptp(s),
    )
    # The incoming neighbor interval is checked, so a cap does not cause the preceding
    # intact section to be discarded.
    bad = np.isfinite(slopes) & (abs(slopes - trend(mid)) > allowance)
    mask[1:] &= ~bad
    diagnostics["thickness_ratio"] = {
        "observed": ratios.tolist(),
        "incoming_thickness_slope": slopes.tolist(),
        "slope_trend": trend(mid).tolist(),
        "slope_allowance": float(allowance),
    }
    run = _longest_run(mask)
    if len(run) < 8:
        raise InputValueError("clean band: too short (fewer than eight clean cuts)")
    return run, mask, diagnostics


def refine_tip(stations, features, band, evaluate, tolerance=1e-4):
    """Refine only the clean/rejected bracket, retaining a tested clean endpoint.

    ``evaluate(station)`` returns the features of a fresh cut or raises
    :class:`StationRejectedError`. Returns the stations, features and band (a clean endpoint
    inserted when the bisection found one) and the final bracket.

    Parameters
    ----------
    stations : array_like
        Span station of each cut.
    features : list
        Features of each cut.
    band : array_like
        Indices of the clean stations.
    evaluate : callable
        Function of a station that returns the features of a fresh cut.
    tolerance : float, optional
        Bracket width at which the bisection stops; default 1e-4.

    Returns
    -------
    tuple
        ``(stations, features, band, bracket)`` after the refinement.
    """
    s = np.asarray(stations)
    last = int(band[-1])
    if last == len(s) - 1:
        return s, features, band, [float(s[-1]), float(s[-1])]
    interior = band[(s[band] > s[0] + 0.2 * np.ptp(s)) & (s[band] < s[0] + 0.75 * np.ptp(s))]
    chords = np.array([features[i]["chord"] for i in interior])
    thickness = np.array([features[i]["thickness_ratio"] for i in interior]) * chords
    models = [
        np.polynomial.Polynomial.fit(s[interior], v, degree)
        for v, degree in ((chords, 3), (thickness, 5))
    ]
    low, high = float(s[last]), float(s[last + 1])
    best = features[last]
    for _ in range(24):
        if high - low < tolerance:
            break
        mid = (low + high) / 2
        try:
            f = evaluate(mid)
            measured = np.array([f["chord"], f["chord"] * f["thickness_ratio"]])
            predicted = np.array([m(mid) for m in models])
            good = np.max(abs(measured / predicted - 1)) < 0.0001
        except ValueError:  # a failed cut or a numerical failure inside it: not clean
            good = False
        if good:
            low, best = mid, f
        else:
            high = mid
    if low > s[last]:
        s = np.insert(s, last + 1, low)
        features = features[: last + 1] + [best] + features[last + 1 :]
        band = np.r_[band, last + 1]
    return s, features, band, [low, high]


def leading_extension(stations, features, band):
    """Return the leading-edge curve through the clean band, extended linearly beyond it.

    Parameters
    ----------
    stations : array_like
        Span station of each cut.
    features : sequence
        Features of each cut.
    band : array_like
        Indices of the clean stations.

    Returns
    -------
    callable
        A function of a span station that returns the leading-edge point.
    """
    s = np.asarray(stations)[band]
    points = np.array([features[i]["le"] for i in band])
    spline = CubicSpline(s, points, bc_type="natural")

    def curve(at):
        q = np.clip(at, s[0], s[-1])
        return spline(q) + (np.asarray(at) - q)[..., None] * spline(q, 1)

    return curve


def reference_planform(
    stations,
    features,
    band,
    span,
    *,
    root_station=None,
    tip_station=None,
    hold_incidence=False,
    boundary=None,
    root_bracket=None,
    mirrored=True,
):
    """Build the planform between a root and the outermost clean cut from the clean band only.

    Leading and trailing edges are natural cubic splines of the clean stations, extended
    linearly past them. The root is a station given, or the root of ``boundary`` (a function of
    a leading-edge point that vanishes on the root surface) found inside ``root_bracket``.

    Parameters
    ----------
    stations : array_like
        Span station of each cut.
    features : sequence
        Features of each cut.
    band : array_like
        Indices of the clean stations.
    span : array_like
        Spanwise direction.
    root_station : float, optional
        Root station; required unless ``boundary`` is given.
    tip_station : float, optional
        Tip station; default the outermost clean station.
    hold_incidence : bool, optional
        Interpolate the incidence without extending it past the clean band; default False.
    boundary : callable, optional
        Function of a leading-edge point that vanishes on the root surface.
    root_bracket : tuple of float, optional
        Station bracket in which the root of ``boundary`` is sought.
    mirrored : bool, optional
        Whether the planform is the half of a mirrored wing; default True.

    Returns
    -------
    dict
        The planform measures: semi-span, span, root and tip chord, taper, sweeps, area, aspect
        ratio, mean aerodynamic chord, incidence and tip twist.

    Raises
    ------
    InputValueError
        If neither a root station nor a boundary is given, or the root does not precede the tip.
    """
    s = np.asarray(stations)[band]
    le = np.array([features[i]["le"] for i in band])
    te = np.array([features[i]["te"] for i in band])
    angle = np.array([features[i]["incidence"] for i in band])

    # Natural end slopes avoid cubic growth outside the measured interval.
    def extended_spline(x, values):
        spline = CubicSpline(s, values, bc_type="natural")
        q = np.asarray(x)
        clipped = np.clip(q, s[0], s[-1])
        result = spline(clipped)
        delta = q - clipped
        if values.ndim == 2:
            return result + delta[..., None] * spline(clipped, 1)
        return result + delta * spline(clipped, 1)

    def leading(x):
        return extended_spline(x, le)

    def trailing(x):
        return extended_spline(x, te)

    if boundary is not None:
        ends = [boundary(leading(x)) for x in root_bracket]
        if ends[0] * ends[1] > 0:
            raise InputValueError(
                f"root: the leading edge does not cross the root surface between stations "
                f"{root_bracket[0]:.6g} and {root_bracket[1]:.6g} m; a surface off the symmetry "
                "plane takes a host or its own boundary curves as root"
            )
        root_station = brentq(lambda x: boundary(leading(x)), *root_bracket, xtol=1e-10)
    if root_station is None:
        raise InputValueError("root: requires a symmetry station or a measured host boundary")
    tip = float(s[-1]) if tip_station is None else float(tip_station)
    if root_station >= tip:
        raise InputValueError("root: must precede the tip")
    q = np.linspace(root_station, tip, 1001)
    a, b = leading(q), trailing(q)
    chord = np.linalg.norm(b - a, axis=1)
    incidence = np.interp(q, s, angle) if hold_incidence else extended_spline(q, angle)
    half_area = simpson(chord, x=q)
    length = tip - root_station
    mult = 2 if mirrored else 1
    quarter = a + 0.25 * (b - a)
    normal = np.cross(b[0] - a[0], a[-1] - a[0])
    normal /= np.linalg.norm(normal)
    return {
        "semi_span_m": float(length),
        "root_chord_m": float(chord[0]),
        "tip_chord_m": float(chord[-1]),
        "taper": float(chord[-1] / chord[0]),
        "span_m": float(length * mult),
        "sweep_le_deg": float(np.degrees(np.arctan2(a[-1, 0] - a[0, 0], length))),
        "sweep_quarter_chord_deg": float(
            np.degrees(np.arctan2(quarter[-1, 0] - quarter[0, 0], length))
        ),
        "area_m2": float(mult * half_area),
        "aspect_ratio": float((mult * length) ** 2 / (mult * half_area)),
        "mac_m": float(simpson(chord**2, x=q) / half_area),
        "mac_span_position_m": float(simpson((q - root_station) * chord, x=q) / half_area),
        "mac_le_position_m": (simpson(a * chord[:, None], x=q, axis=0) / half_area).tolist(),
        "incidence_deg": float(incidence[0]),
        "tip_twist_deg": float(incidence[-1] - incidence[0]),
        "dihedral_deg": float(np.degrees(np.arctan2(span[2], abs(span[1])))),
        "mirrored": mirrored,
        "reference_plane": {
            "origin_m": a[0].tolist(),
            "normal": normal.tolist(),
            "span_direction": [float(v) for v in span],
        },
        "root_station_m": float(root_station),
        "tip_station_m": tip,
        "clean_band_m": [float(s[0]), float(s[-1])],
        "clean_tip_station_m": float(s[-1]),
        "junction": {"interval_m": [float(root_station), float(s[0])]},
        "distributions": {
            "station_m": q[::25].tolist(),
            "eta": ((q[::25] - root_station) / length).tolist(),
            "le_x_m": a[::25, 0].tolist(),
            "le_y_m": a[::25, 1].tolist(),
            "le_z_m": a[::25, 2].tolist(),
            "chord_m": chord[::25].tolist(),
            "incidence_deg": incidence[::25].tolist(),
            "twist_deg": (incidence[::25] - incidence[0]).tolist(),
        },
        "classifier_limit": (
            "Interior cubic chord trend and neighboring dimensional-thickness slope; "
            "complex distributions require review."
        ),
    }


def _section_graphs(feature):
    p = feature["contour"][:-1]
    q = feature["unit_contour"][:-1]
    nose = int(np.argmin(q[:, 0]))
    ends = [
        int(np.argmin(np.linalg.norm(p - end, axis=1)))
        for end in feature["trailing_edge_endpoints"]
    ]
    axis = (feature["te"] - feature["le"]) / feature["chord"]
    paths = []
    for direction in (1, -1):
        length = min((direction * (end - nose)) % len(p) for end in ends)
        branch = q[(nose + direction * np.arange(length + 1)) % len(p)].copy()
        endpoint = branch[-1].copy()
        endpoint[0] = 1.0
        branch = branch[(branch[:, 0] > 1e-12) & (branch[:, 0] < 1.0 - 1e-10)]
        branch = np.vstack(([0.0, 0.0], branch, endpoint))
        branch = branch[np.argsort(branch[:, 0])]
        branch = branch[np.r_[True, np.diff(branch[:, 0]) > 1e-12]]
        paths.append(branch)
    paths.sort(key=lambda b: float(np.mean(b[:, 1])), reverse=True)
    interpolators = [PchipInterpolator(np.sqrt(np.maximum(0.0, b[:, 0])), b[:, 1]) for b in paths]
    graphs = [lambda x, graph=graph: graph(np.sqrt(np.maximum(0.0, x))) for graph in interpolators]
    thickness = float(paths[0][-1, 1] - paths[1][-1, 1])
    projection = max(
        abs(float((end - feature["le"]) @ axis / feature["chord"]) - 1.0)
        for end in feature["trailing_edge_endpoints"]
    )
    if thickness < 0:
        raise StationRejectedError("negative trailing-edge thickness")
    return graphs, thickness, projection


def fit_section(contour, span, seed, *, degree=5, max_degree=20, tolerance_m=None):
    """Fit a CST section to one clean cut, raising the degree until the tolerance is met.

    The seed (a NACA designation) supplies the starting shape; the correction is fitted to the
    difference between the cut and the seed. ``tolerance_m`` defaults to the larger of 0.5 mm
    and 0.1 percent of the chord. The residual is the maximum ordinate difference against the
    cut on an independent cosine grid; ``accepted`` says whether it met the tolerance.

    Parameters
    ----------
    contour : array_like
        Closed contour of the cut, shape ``(n, 3)``.
    span : array_like
        Spanwise direction of the cut.
    seed : str
        NACA designation that supplies the starting shape.
    degree : int, optional
        First CST degree tried; default 5.
    max_degree : int, optional
        Highest CST degree tried; default 20.
    tolerance_m : float, optional
        Residual tolerance, meters; default as described above.

    Returns
    -------
    dict
        The fit: ``accepted``, ``seed``, ``degree``, the CST coefficients and the residual.

    Raises
    ------
    InputValueError
        If ``max_degree`` is below ``degree``.
    """
    if max_degree < degree:
        raise InputValueError(f"max_degree: {max_degree} is below the first degree {degree}")
    feature = contour_features(contour, span)
    chord = feature["chord"]
    tolerance_m = max(0.0005, 0.001 * chord) if tolerance_m is None else tolerance_m
    graphs, thickness, projection = _section_graphs(feature)
    x = cosine(601)
    check = cosine(2402)
    measured = [graph(x) for graph in graphs]
    target = [graph(check) for graph in graphs]
    su, sl, transform = normalized_naca(seed)
    seed_graphs = [PchipInterpolator(p[:, 0], p[:, 1]) for p in (su, sl)]
    seed_training = [graph(x) for graph in seed_graphs]
    seed_check = [graph(check) for graph in seed_graphs]
    for current in range(degree, max_degree + 1):
        initial = fit_cst(x, *seed_training, thickness, current)
        base = cst_airfoil(x, *initial, thickness)
        correction = fit_cst(
            x, measured[0] - base[0][:, 1], measured[1] - base[1][:, 1], 0.0, current
        )
        upper, lower = (initial[i] + correction[i] for i in (0, 1))
        predicted = cst_airfoil(check, upper, lower, thickness)
        residuals = np.concatenate([predicted[i][:, 1] - target[i] for i in (0, 1)])
        residual = float(np.max(abs(residuals))) * chord
        if residual <= tolerance_m:
            break
    delta = np.concatenate([target[i] - seed_check[i] for i in (0, 1)])
    return {
        "accepted": residual <= tolerance_m,
        "seed": seed,
        "degree": current,
        "n1": 0.5,
        "n2": 1.0,
        "upper": upper.tolist(),
        "lower": lower.tolist(),
        "trailing_edge_thickness": thickness,
        "chord_m": chord,
        "incidence_deg": feature["incidence"],
        "leading_edge_m": feature["le"].tolist(),
        "trailing_edge_m": feature["te"].tolist(),
        "max_residual_m": residual,
        "rms_residual_m": float(np.sqrt(np.mean(residuals**2))) * chord,
        "tolerance_m": tolerance_m,
        "max_residual_unit_chord": residual / chord,
        "seed_max_difference_m": float(np.max(abs(delta))) * chord,
        "seed_rms_difference_m": float(np.sqrt(np.mean(delta**2))) * chord,
        "measured_thickness_ratio": float(np.max(target[0] - target[1])),
        "seed_thickness_ratio": float(np.max(seed_check[0] - seed_check[1])),
        "thickness_ratio_difference": float(
            np.max(target[0] - target[1]) - np.max(seed_check[0] - seed_check[1])
        ),
        "seed_chord_transform": transform,
        "te_projection_unit_chord": projection,
        "seed_coefficient_correction_l2": float(
            np.linalg.norm(np.r_[correction[0], correction[1]])
        ),
        "sampling": {
            "training_count": len(x),
            "check_count": len(check),
            "spacing": "physical x/c cosine",
        },
        "residual_definition": (
            "Maximum ordinate difference against square-root-abscissa PCHIP of the normalized "
            "original cut on an independent physical cosine grid."
        ),
    }

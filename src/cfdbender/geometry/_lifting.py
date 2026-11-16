"""The lifting surface: one B-spline skin from its defining sections, its trims and its layout.

Pipeline role: pure numerical stage of the geometry package (S4 to S9 for a lifting surface).
A resolved lifting-surface component (plain values, metres and degrees) becomes one skin: u runs
chordwise along the stored section curve (lower trailing edge, leading edge at u = 0.5, upper
trailing edge), v spanwise and equal to the station ``eta``. Pole row k of a ruled skin is the
defining section k placed by the planform. The upper and lower faces are trims of that skin in
(u, v), the blunt trailing edge is a face of its own that shares both edges exactly, and the
four-sided blocks a mesher reads are a layout split from the skin by exact knot insertion, never
written as geometry.

The ``eta`` of a station is a fraction of the span the skin covers, which ends where the tip cap
starts: the planform span is the published span and includes the cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline, make_interp_spline

from cfdbender._errors import InputValueError

from ._bspline import full_knots, split_patch
from ._distributions import (
    chord_spacing,
    elevate_coefficients,
    planform_values,
    spline_distribution,
)
from ._patch import Patch, interpolate_patch
from ._sections import BSplineSection, bspline_sections, fit_or_read
from ._skin import split_along
from ._trims import Trim, iso_trim, trim_checks

__all__ = [
    "LiftingModel",
    "continuity_internal",
    "SECTION_EXACT",
    "bound_stations",
    "cap_fraction",
    "hinge_fractions",
    "law_and_fraction",
    "layout_blocks",
    "layout_record",
    "lifting_surface",
    "max_thickness_x",
    "planform_metrics",
    "place_points",
    "section_at",
    "section_law",
    "section_models",
    "section_points",
    "section_sides",
    "side_parameters",
    "skin_model",
    "thickness_station",
    "tip_section",
    "trailing_edge_face",
    "with_trim_checks",
]

#: Distance allowed between a skin's iso-v curve at a defining station and the placed section,
#: in metres (design D, ``skin.section_exact``).
SECTION_EXACT = 1e-12
#: Largest number of halving passes of ``law.ruled_twist``.
_TWIST_PASSES = 12


# -- CST helpers kept for the stages that still blend CST sections (fairing, fillet, regions) --


def section_models(sections):
    """Fit a CST model to each section airfoil of a lifting surface at one common degree."""
    models = [form.to_stored() for form in fit_or_read(sections)]
    degree = max(model["degree"] for model in models)
    return [
        {
            **model,
            "upper": elevate_coefficients(model["upper"], degree).tolist(),
            "lower": elevate_coefficients(model["lower"], degree).tolist(),
            "degree": degree,
        }
        for model in models
    ]


#: Section laws and skin fractions already built, by the text of the tables they read.
_LAWS: dict[str, tuple] = {}


def law_and_fraction(component):
    """Return the section law (chord-weighted) and the skin's span fraction of a surface."""
    keys = ("planform", "sections", "tip_cap", "spanwise", "control_surfaces", "distributions")
    key = repr([(k, component.get(k)) for k in keys])
    if key not in _LAWS:
        fraction = cap_fraction(component, section_at(section_law(component), 1.0))
        if len(_LAWS) >= _FITS_KEPT:
            _LAWS.pop(next(iter(_LAWS)))
        _LAWS[key] = (section_law(component, None, fraction), fraction)
    return _LAWS[key]


def section_points(component, models, eta, x, side):
    """Return model-space points of the stored section at ``eta`` for chord fractions ``x``.

    The section is the one the skin is built from (``models`` is no longer read and is kept
    for the callers' signature); ``eta`` is a station of the skin, so the points at a defining
    section lie on the skin.
    """
    law, fraction = law_and_fraction(component)
    poles = law.at(eta)
    curve = BSpline(law.knots, poles, len(law.knots) - len(poles) - 1)
    local = curve(side_parameters(poles, law.knots, x, side == "upper"))
    if np.ndim(x) == 0:
        local = local[0]
    return place_points(component, local, eta, fraction)


def thickness_station(component, models, eta):
    """Return the chord fraction of maximum thickness of the stored section at ``eta``."""
    law, _ = law_and_fraction(component)
    return max_thickness_x(section_at(law, eta))


# -- placement and the section law ------------------------------------------------------------


def place_points(component, local, eta, fraction=1.0):
    """Return model-space points of chord-frame ``(x, z)`` points at skin station ``eta``.

    ``fraction`` is the share of the half span the skin covers (:func:`cap_fraction`): the
    station is placed at the planform fraction ``eta * fraction``. The map is affine, so the
    poles of a placed curve are its poles placed.
    """
    local = np.asarray(local, float)
    s = float(eta) * float(fraction)
    p = planform_values(component, s)
    angle = np.deg2rad(float(p["twist"]))
    chord = float(p["chord"])
    axis = component["planform"]["twist_axis"]
    dx, dz = (local[..., 0] - axis) * chord, local[..., 1] * chord
    xx = float(p["leading_edge_x"]) + axis * chord + dx * np.cos(angle) + dz * np.sin(angle)
    zz = float(p["leading_edge_z"]) - dx * np.sin(angle) + dz * np.cos(angle)
    yy = np.full_like(xx, component["planform"]["span"] / 2 * s)
    return np.stack((xx, yy, zz), axis=-1)


def hinge_fractions(component) -> dict[str, float]:
    """Return the hinge chord fraction of every control surface, by name (pinned as iso-u)."""
    out = {}
    for i, surface in enumerate(component.get("control_surfaces") or []):
        hinge = surface.get("hinge")
        x = hinge if isinstance(hinge, (int, float)) and not isinstance(hinge, bool) else None
        if x is None:
            x = 1.0 - float(surface.get("chord_ratio", 0.25))
        out[str(surface.get("name") or f"surface_{i}")] = float(x)
    return out


def _sections_with_pins(component):
    """Return the ``sections`` table with every stated pin and hinge pinned."""
    sections = dict(component["sections"])
    pins = list(sections.get("pin") or [])
    pins += list(hinge_fractions(component).values())
    sections["pin"] = sorted({round(float(p), 15) for p in pins})
    return sections


@dataclass(frozen=True)
class _Law:
    """The chordwise section law of one surface: pole rows given at stations, on one vector."""

    stations: np.ndarray
    poles: np.ndarray  # (stations, poles, 2)
    knots: np.ndarray
    smooth: bool
    given: tuple = ()
    chords: np.ndarray | None = None

    def at(self, eta) -> np.ndarray:
        """Return the chord-frame poles of the section at ``eta``.

        Ruled: between two given sections the shape is their blend weighted by chord, so the
        section placed at ``eta`` is the linear loft of the two placed sections whenever the
        planform is linear there. Smooth: the not-a-knot cubic of the poles.
        """
        if len(self.stations) == 1:
            return self.poles[0].copy()
        if self.smooth:
            return spline_distribution(self.stations, self.poles, float(eta))[0]
        k = int(np.clip(np.searchsorted(self.stations, eta, side="right") - 1, 0,
                        len(self.stations) - 2))  # fmt: skip
        a, b = self.stations[k], self.stations[k + 1]
        t = (float(eta) - a) / (b - a)
        wa, wb = 1 - t, t
        if self.chords is not None:
            wa, wb = wa * self.chords[k], wb * self.chords[k + 1]
            wa, wb = wa / (wa + wb), wb / (wa + wb)
        return wa * self.poles[k] + wb * self.poles[k + 1]


#: Section conversions already made, by the text of their table: a fit takes seconds and every
#: stage of one surface reads the same sections.
_FITS: dict[str, tuple[list, list]] = {}
_FITS_KEPT = 8


def _file_stamps(sections):
    """Return (path, size, mtime) of every file a section row names, so an edit refits."""
    stamps = []
    for item in sections.get("items") or []:
        if not isinstance(item, dict):
            continue
        table = item.get("section") if isinstance(item.get("section"), dict) else {}
        for value in (item.get("file"), item.get("bspline"), table.get("file")):
            if isinstance(value, str):
                try:
                    stat = Path(value).stat()
                    stamps.append((value, stat.st_size, stat.st_mtime_ns))
                except OSError:
                    stamps.append((value, None, None))
    return stamps


def _fitted(sections, warnings):
    key = repr((sorted(sections.items(), key=lambda item: item[0]), _file_stamps(sections)))
    if key not in _FITS:
        notes: list[str] = []
        result = bspline_sections(sections, notes)
        if len(_FITS) >= _FITS_KEPT:
            _FITS.pop(next(iter(_FITS)))
        _FITS[key] = (result, notes)
    result, notes = _FITS[key]
    if warnings is not None:
        warnings.extend(notes)
    return list(result)


def section_law(component, warnings=None, fraction=None) -> _Law:
    """Return the section law of a resolved lifting surface: its stored B-spline sections.

    ``fraction`` (:func:`cap_fraction`) places the given stations for the chord weights of the
    ruled blend; without it the stations are blended by ``eta`` alone.
    """
    sections = _fitted(_sections_with_pins(component), warnings)
    stations = np.asarray(component["sections"]["eta"], float)
    if len(stations) != len(sections):
        raise InputValueError("sections.eta: one station per section required")
    chords = None
    if fraction is not None:
        chords = np.array([float(planform_values(component, e * fraction)["chord"])
                           for e in stations])  # fmt: skip
    return _Law(
        stations=stations,
        poles=np.array([np.asarray(s.poles, float) for s in sections]),
        knots=np.asarray(sections[0].knots, float),
        smooth=component.get("spanwise", {}).get("law", "ruled") == "smooth",
        given=tuple(sections),
        chords=chords,
    )


def section_at(law, eta) -> BSplineSection:
    poles = law.at(eta)
    return BSplineSection(
        poles=tuple((float(p[0]), float(p[1])) for p in poles),
        knots=tuple(float(k) for k in law.knots),
    )


def tip_section(component, warnings=None) -> BSplineSection:
    """Return the stored section at the skin's tip station, in the chord frame."""
    return section_at(section_law(component, warnings), 1.0)


def max_thickness_x(section) -> float:
    x = (1 - np.cos(np.linspace(0, np.pi, 801))) / 2
    upper, lower = section.evaluate(x)
    return float(x[int(np.argmax(upper[:, 1] - lower[:, 1]))])


def _half_thickness_at_max(section) -> float:
    upper, lower = section.evaluate(np.array([max_thickness_x(section)]))
    return float(upper[0, 1] - lower[0, 1]) / 2


def cap_fraction(component, section=None) -> float:
    """Return the share of the half span the skin covers; the tip cap takes the rest.

    The planform span is the published span, so it includes the tip cap: the skin ends where
    the cap starts, at ``span / 2 - reach``. The reach is ``tip_cap.extent`` in model units, or
    the cap's reach in tip chords times the chord there (``tip_cap.length`` half thicknesses at
    the maximum-thickness station for ``rounded`` and ``exact_loft``, the largest profile ridge
    for ``ridged``); a ``flat`` cap or none has no reach.
    """
    cap = component["tip_cap"]
    kind = cap["kind"]
    if kind in ("flat", "none"):
        return 1.0
    half = component["planform"]["span"] / 2
    if "extent" in cap:
        reach_chords = None
    elif kind == "ridged":
        reach_chords = float(max(cap["profile"]["ridge"]))
    else:
        section = section or tip_section(component)
        reach_chords = float(cap["length"]) * _half_thickness_at_max(section)
    fraction = 1.0
    for _ in range(60):
        chord = float(planform_values(component, fraction)["chord"])
        reach = float(cap["extent"]) if reach_chords is None else reach_chords * chord
        new = 1.0 - reach / half
        if not 0.0 < new <= 1.0:
            raise InputValueError("tip_cap: the cap reaches past the root of the surface")
        if abs(new - fraction) < 1e-15:
            break
        fraction = new
    return fraction


# -- the skin ---------------------------------------------------------------------------------


def _place_rows(component, law, stations, fraction):
    return np.stack([place_points(component, law.at(e), e, fraction) for e in stations], axis=1)


def _ruled_twist(component, law, stations, fraction):
    """Insert defining sections until the ruled skin follows the planform law (``law.ruled_twist``).

    Between two defining sections a ruled skin is their linear interpolation; where the twist or
    a non-linear chord or offset law makes the placed section differ from it by more than the
    analytic section tolerance times the local chord, the station halfway is made a defining
    section. The planform alone is measured: one shape, the mean of the two sections, is placed
    at both ends and between them, so a change of section shape (which the ruled blend defines)
    inserts nothing. The distance is measured on the poles, which bounds that of the curves.
    """
    tolerance = float(component["sections"].get("tolerance", {}).get("analytic", 2e-5))
    stations = sorted(stations)
    inserted, worst = 0, 0.0
    for _ in range(_TWIST_PASSES):
        added, worst = [], 0.0
        for a, b in zip(stations[:-1], stations[1:], strict=True):
            shape = (law.at(a) + law.at(b)) / 2
            ends = [place_points(component, shape, e, fraction) for e in (a, b)]
            excess = 0.0
            for t in (0.25, 0.5, 0.75):
                m = a + t * (b - a)
                exact = place_points(component, shape, m, fraction)
                chord = float(planform_values(component, m * fraction)["chord"])
                ruled = (1 - t) * ends[0] + t * ends[1]
                dev = float(np.max(np.linalg.norm(exact - ruled, axis=-1)))
                worst = max(worst, dev / chord)
                excess = max(excess, dev - tolerance * chord)
            if excess > 0:
                added.append((a + b) / 2)
        if not added:
            break
        stations = sorted({*stations, *added})
        inserted += len(added)
    record = {
        "id": "law.ruled_twist",
        "status": "pass" if worst <= tolerance else "report",
        "value": {"largest_chords": worst, "inserted": inserted},
        "note": f"ruled skin against the planform law, limit {tolerance} chords",
    }
    return stations, record


def _defining_stations(component, law, fraction):
    plan = component["planform"]
    fixed = {0.0, 1.0, *law.stations.tolist(), *plan.get("breaks", []),
             *plan.get("slope_breaks", [])}  # fmt: skip
    stations = sorted(float(e) for e in fixed if 0.0 <= float(e) <= 1.0)
    if law.smooth:
        record = {"id": "law.ruled_twist", "status": "report", "value": None,
                  "note": "spanwise law smooth: the skin interpolates its sections"}  # fmt: skip
        return stations, record
    return _ruled_twist(component, law, stations, fraction)


def _skin_patch(rows, stations, law) -> Patch:
    knots_u, mult_u = np.unique(law.knots, return_counts=True)
    if not law.smooth or len(stations) == 2:
        controls, degree = rows, 1
        knots_v = tuple(float(s) for s in stations)
        mult_v = (2, *([1] * (len(stations) - 2)), 2)
    else:
        degree = min(3, len(stations) - 1)
        spline = make_interp_spline(np.asarray(stations), rows.transpose(1, 0, 2), k=degree, axis=0)
        distinct, counts = np.unique(spline.t, return_counts=True)
        controls = np.asarray(spline.c).transpose(1, 0, 2)
        knots_v = tuple(float(k) for k in distinct)
        mult_v = tuple(int(m) for m in counts)
    return Patch(
        family="skin",
        controls=controls,
        knots_u=tuple(float(k) for k in knots_u),
        knots_v=knots_v,
        multiplicities_u=tuple(int(m) for m in mult_u),
        multiplicities_v=tuple(mult_v),
        degree_u=int(len(law.knots) - rows.shape[0] - 1),
        degree_v=degree,
        bay=(0.0, 1.0),
        role="skin",
    )


def _iso_v_poles(patch: Patch, v) -> np.ndarray:
    """Return the poles of the iso-v curve of a patch: its u poles at ``v``."""
    tv = full_knots(patch.knots_v, patch.multiplicities_v)
    return BSpline(tv, np.asarray(patch.controls).transpose(1, 0, 2), patch.degree_v)(v)


def trailing_edge_face(skin: Patch) -> Patch:
    """Return the blunt trailing-edge face: ruled across the base, the skin's own v.

    Row 0 is the skin's lower trailing-edge edge and row 1 its upper one, so the face shares
    both edge curves with the skin exactly.
    """
    controls = np.asarray(skin.controls)
    return Patch(
        family="trailing_edge",
        controls=np.stack((controls[0], controls[-1])),
        knots_u=(0.0, 1.0),
        knots_v=skin.knots_v,
        multiplicities_u=(2, 2),
        multiplicities_v=skin.multiplicities_v,
        degree_u=1,
        degree_v=skin.degree_v,
        bay=skin.bay,
        role="trailing_edge",
    )


def side_parameters(poles, knots, x, upper: bool) -> np.ndarray:
    """Return the section parameters where a chord-frame curve reaches chord fractions ``x``.

    The leading edge is the parameter 0.5 itself (``x <= 0``) and the trailing edge the end of
    the side (``x >= 1``), so faces sampled on either side meet on the skin's own edge curves.
    """
    x = np.atleast_1d(np.asarray(x, float))
    poles = np.asarray(poles, float)
    curve = BSpline(np.asarray(knots, float), poles, len(knots) - len(poles) - 1)
    lo = np.full(x.shape, 0.5 if upper else 0.0)
    hi = np.full(x.shape, 1.0 if upper else 0.5)
    for _ in range(60):
        mid = (lo + hi) / 2
        ahead = (curve(mid)[..., 0] < x) == upper
        lo, hi = np.where(ahead, mid, lo), np.where(ahead, hi, mid)
    u = (lo + hi) / 2
    u[x <= 0.0] = 0.5
    u[x >= 1.0] = 1.0 if upper else 0.0
    return u


def section_sides(section, x):
    """Return the upper and lower (x, z) points of a stored section at chord fractions ``x``.

    Unlike ``BSplineSection.evaluate``, which finds where each side crosses ``x``, the point at
    ``x = 0`` is the curve's own leading edge (parameter 0.5) on both sides.
    """
    poles, knots = np.asarray(section.poles, float), np.asarray(section.knots, float)
    curve = BSpline(knots, poles, len(knots) - len(poles) - 1)
    return tuple(curve(side_parameters(poles, knots, x, upper)) for upper in (True, False))


def _u_on_side(law_poles, knots, x, upper: bool) -> float:
    """Return the section parameter where the chord-frame curve reaches ``x`` on one side."""
    return float(side_parameters(law_poles, knots, [x], upper)[0])


def _lines(component, law, stations):
    root = section_at(law, 0.0)
    x_max = max_thickness_x(root)
    u_lines = {
        "lower_te": 0.0,
        "lower_max": _u_on_side(law.at(0.0), law.knots, x_max, upper=False),
        "le": 0.5,
        "upper_max": _u_on_side(law.at(0.0), law.knots, x_max, upper=True),
        "upper_te": 1.0,
    }
    knots = np.unique(law.knots)
    for name, h in hinge_fractions(component).items():
        for side, sign in (("lower", -1.0), ("upper", 1.0)):
            u = 0.5 + sign * 0.5 * np.sqrt(h)
            near = knots[int(np.argmin(np.abs(knots - u)))]
            u_lines[f"hinge_{name}_{side}"] = float(near if abs(near - u) < 1e-9 else u)
    v_lines = {"root": 0.0, "tip": 1.0}
    for i, eta in enumerate(stations):
        if 0.0 < eta < 1.0:
            v_lines[f"section_{i}"] = float(eta)
    for i, eta in enumerate(component.get("spanwise", {}).get("lines") or []):
        v_lines[f"line_{i}"] = float(eta)
    for bound in component["sections"].get("nacelle_bounds") or []:
        v_lines[f"nacelle_{bound['component']}_in"] = float(bound["eta_in"])
        v_lines[f"nacelle_{bound['component']}_out"] = float(bound["eta_out"])
    for i, surface in enumerate(component.get("control_surfaces") or []):
        name = str(surface.get("name") or f"surface_{i}")
        for key in ("eta_in", "eta_out"):
            value = surface.get(key)
            if isinstance(value, (int, float)) and 0.0 < value < 1.0:
                v_lines[f"{name}_{key[4:]}"] = float(value)
    return {"u": u_lines, "v": v_lines}


def bound_stations(component):
    """Return the span stations of the control sections that bound a nacelle, sorted.

    Each kept ``sections.nacelle_bounds`` entry adds its two stations as iso-v lines of the
    skin; the upper and lower faces are cut there, so a nacelle trim lies between two section
    lines. The section shape does not change.
    """
    bounds = component["sections"].get("nacelle_bounds") or []
    return sorted({float(eta) for b in bounds for eta in (b["eta_in"], b["eta_out"])})


def _skin_trims(component):
    cuts = [0.0, *[e for e in bound_stations(component) if 0.0 < e < 1.0], 1.0]
    trims = []
    for side, u in (("lower", (0.0, 0.5)), ("upper", (0.5, 1.0))):
        for k, v in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
            face = side if len(cuts) == 2 else f"{side}_{k}"
            trims.append(iso_trim(face, "skin", u, v, role=side))
    trims.append(iso_trim("trailing_edge", "trailing_edge", (0.0, 1.0), (0.0, 1.0),
                          role="trailing_edge"))  # fmt: skip
    return trims


@dataclass(frozen=True, eq=False)
class LiftingModel:
    """The geometry of one lifting surface: base surfaces, faces as trims, lines and records.

    Attributes
    ----------
    skin : Patch
        The one skin, u chordwise (leading edge at 0.5), v spanwise equal to ``eta``.
    trailing_edge : Patch
        The blunt trailing-edge face, sharing both edges with the skin.
    caps : dict
        Cap faces by base name (tip and, for a free root, root); filled by the cap stage.
    trims : tuple of Trim
        Every face, as a loop on its base.
    lines : dict
        Named iso lines: ``{"u": {name: u}, "v": {name: v}}``; the layout cuts at them.
    stations : tuple of float
        The defining stations, the skin's pole rows (ruled) or interpolated rows (smooth).
    fraction : float
        Share of the half span the skin covers (:func:`cap_fraction`).
    records : tuple of dict
        The stage records S4 to S9 of the surface.
    ends : dict
        The chord-frame section at the root and at the tip, by name.
    """

    skin: Patch
    trailing_edge: Patch
    trims: tuple[Trim, ...]
    lines: dict
    stations: tuple[float, ...]
    fraction: float
    caps: dict = field(default_factory=dict)
    records: tuple[dict, ...] = ()
    ends: dict = field(default_factory=dict)

    def bases(self) -> dict[str, Patch]:
        """Return every base surface by name."""
        return {"skin": self.skin, "trailing_edge": self.trailing_edge, **self.caps}


def _section_exact(component, law, skin, stations, fraction) -> dict:
    worst = 0.0
    for eta in stations:
        placed = place_points(component, law.at(eta), eta, fraction)
        worst = max(worst, float(np.max(np.linalg.norm(_iso_v_poles(skin, eta) - placed, axis=-1))))
    return {
        "id": "skin.section_exact",
        "status": "pass" if worst <= SECTION_EXACT else "fail",
        "value": worst,
        "note": f"iso-v poles against the placed sections, {SECTION_EXACT} m",
    }


def continuity_internal(patch: Patch, name: str) -> dict:
    """Return each interior knot's continuity order of a base, read from its multiplicity."""
    levels = {}
    for direction in ("u", "v"):
        knots = patch.knots_u if direction == "u" else patch.knots_v
        mult = patch.multiplicities_u if direction == "u" else patch.multiplicities_v
        degree = patch.degree_u if direction == "u" else patch.degree_v
        levels[direction] = [[float(k), degree - int(m)] for k, m in zip(knots[1:-1], mult[1:-1],
                                                                          strict=True)]  # fmt: skip
    return {"id": "continuity.internal", "status": "report", "base": name, "value": levels,
            "note": "C order per interior knot: degree minus multiplicity"}  # fmt: skip


def skin_model(component, warnings=None) -> LiftingModel:
    """Return the skin, trailing-edge face, trims, lines and records of a lifting surface.

    The caps are added by :func:`cfdbender.geometry._caps.lifting_model`, which is the
    entry point that builds the whole surface.
    """
    fraction = cap_fraction(component, section_at(section_law(component, warnings), 1.0))
    law = section_law(component, None, fraction)
    stations, twist = _defining_stations(component, law, fraction)
    rows = _place_rows(component, law, stations, fraction)
    skin = _skin_patch(rows, stations, law)
    te = trailing_edge_face(skin)
    trims = tuple(_skin_trims(component))
    records = [twist, _section_exact(component, law, skin, stations, fraction)]
    records += [continuity_internal(skin, "skin"), continuity_internal(te, "trailing_edge")]
    return LiftingModel(
        skin=skin,
        trailing_edge=te,
        trims=trims,
        lines=_lines(component, law, stations),
        stations=tuple(float(s) for s in stations),
        fraction=fraction,
        records=tuple(records),
        ends={"root": section_at(law, 0.0), "tip": section_at(law, 1.0)},
    )


def with_trim_checks(model: LiftingModel) -> list[dict]:
    """Return the S7 records of a model's faces."""
    return trim_checks(model.bases(), model.trims)


# -- the layout of the mesh routes (F.1) ------------------------------------------------------


def _family(lines, low, high):
    middle = (low + high) / 2
    if middle < 0.5:
        return "lower_aft" if middle < lines["lower_max"] else "lower_fore"
    return "upper_fore" if middle < lines["upper_max"] else "upper_aft"


def layout_blocks(model: LiftingModel) -> list[dict]:
    """Return the four-sided blocks of the layout: the skin split at every named iso line.

    Each entry is ``{"patch", "base", "family", "u", "v"}`` with the block's (u, v) interval on
    its base; the patch is cut by exact knot insertion, so it evaluates to the base. The family
    names are the ones the mesh rules read (``upper_fore`` ... ``trailing_lower``). The layout is
    for the mesh routes only and never changes the geometry written.
    """
    u_lines = model.lines["u"]
    us = sorted({v for v in u_lines.values() if 0.0 < v < 1.0})
    vs = sorted({v for v in model.lines["v"].values() if 0.0 < v < 1.0})
    u_edges, v_edges = [0.0, *us, 1.0], [0.0, *vs, 1.0]
    blocks = []
    strips = split_along(model.skin, "u", us) if us else [model.skin]
    for i, strip in enumerate(strips):
        family = _family(u_lines, u_edges[i], u_edges[i + 1])
        for j, piece in enumerate(split_along(strip, "v", vs) if vs else [strip]):
            blocks.append({"patch": _named(piece, family, v_edges, j), "base": "skin",
                           "family": family, "u": [u_edges[i], u_edges[i + 1]],
                           "v": [v_edges[j], v_edges[j + 1]]})  # fmt: skip
    halves = split_patch(model.trailing_edge, "u", 0.5)
    for (u, family), half in zip(((0.0, "trailing_lower"), (0.5, "trailing_upper")), halves,
                                 strict=True):  # fmt: skip
        for j, piece in enumerate(split_along(half, "v", vs) if vs else [half]):
            blocks.append({"patch": _named(piece, family, v_edges, j), "base": "trailing_edge",
                           "family": family, "u": [u, u + 0.5],
                           "v": [v_edges[j], v_edges[j + 1]]})  # fmt: skip
    return blocks


def _named(piece, family, v_edges, j):
    return Patch(
        family=family,
        controls=piece.controls,
        knots_u=piece.knots_u,
        knots_v=piece.knots_v,
        multiplicities_u=piece.multiplicities_u,
        multiplicities_v=piece.multiplicities_v,
        degree_u=piece.degree_u,
        degree_v=piece.degree_v,
        bay=(v_edges[j], v_edges[j + 1]),
        role="layout",
    )


def layout_record(model: LiftingModel) -> dict:
    """Return the layout sidecar of a surface: its named lines and blocks, plain data only."""
    return {
        "lines": {"u": dict(model.lines["u"]), "v": dict(model.lines["v"])},
        "blocks": [
            {"base": b["base"], "family": b["family"], "u": b["u"], "v": b["v"]}
            for b in layout_blocks(model)
        ],
    }


# -- the faces the generator writes until the trimmed-skin export ------------------------------


def _skin_point(skin, u, v):
    tu = full_knots(skin.knots_u, skin.multiplicities_u)
    tv = full_knots(skin.knots_v, skin.multiplicities_v)
    along_v = BSpline(tv, np.asarray(skin.controls).transpose(1, 0, 2), skin.degree_v)(v)
    return BSpline(tu, along_v, skin.degree_u)(np.asarray(u, float))


def _side_grid(model, law, eta, x, upper):
    return _skin_point(model.skin, side_parameters(law.at(eta), law.knots, x, upper), eta)


def _interim_grid(model, law, entry, chord_count=49, span_count=9):
    """Sample the skin over one family and bay, shape (chord, span, 3).

    The chord fractions are the nose-clustered ones the mesh spacing rules expect.
    """
    s = np.linspace(0, 1, chord_count)
    grid = []
    for eta in np.linspace(*entry["bay"], span_count):
        family = entry["family"]
        if family.startswith("trailing_"):
            upper = _skin_point(model.skin, [1.0], eta)[0]
            lower = _skin_point(model.skin, [0.0], eta)[0]
            fraction = s / 2 if family == "trailing_upper" else (1 + s) / 2
            points = (1 - fraction[:, None]) * upper + fraction[:, None] * lower
        else:
            side, region = family.split("_")
            mid = max_thickness_x(section_at(law, eta))
            a, b = (0, mid) if region == "fore" else (mid, 1)
            points = _side_grid(model, law, eta, chord_spacing(a, b, s), side == "upper")
        grid.append(points)
    return np.stack(grid, axis=1)


def lifting_surface(component) -> list[Patch]:
    """Return the skin of a lifting surface as the four-sided faces the generator writes today.

    The shape is the one skin of :func:`skin_model`; it is sampled per layout family and bay at
    the nose-clustered chord fractions the mesh spacing rules read and written as bicubic faces,
    six families per bay. This is the generator's face set until the export writes the skin with
    its trims; the trims and the exact layout are in :func:`skin_model` and :func:`layout_blocks`.
    A bay ends only at a defining section: section lines that do not define shape (nacelle
    bounds, control-surface ends, ``spanwise.lines``) are iso lines of the skin, not face edges.

    Parameters
    ----------
    component : dict
        Resolved lifting-surface component (plain values, metres and degrees).

    Returns
    -------
    list of Patch
        The skin faces, six families per spanwise bay.

    Examples
    --------
    >>> from cfdbender.geometry import resolve_config, stages
    >>> config = resolve_config({"type": "lifting_surface"})
    >>> component = stages.unpack(config)["components"]["COMPONENT"]
    >>> faces = stages.lifting_surface(component)
    >>> len(faces) > 0
    True
    """
    model = skin_model(component)
    law = section_law(component, None, model.fraction)
    vs = sorted({0.0, 1.0, *model.stations})
    families =("upper_fore", "upper_aft", "lower_fore", "lower_aft", "trailing_upper",
                "trailing_lower")  # fmt: skip
    return [
        interpolate_patch(
            _interim_grid(model, law, {"bay": (a, b), "family": f}), family=f, bay=(a, b)
        )  # fmt: skip
        for a, b in zip(vs[:-1], vs[1:], strict=True)
        for f in families
    ]


def planform_metrics(component):
    """Return the planform area and mean aerodynamic chord by quadrature of the chord.

    The area is that of the whole span, the mirrored pair. Under ``planform.height`` the resolved
    ``planform.area`` is that of one surface, so there it is half of the area returned here.
    """

    def chord(eta):
        return float(planform_values(component, eta)["chord"])

    integral = quad(chord, 0, 1, epsabs=1e-11)[0]
    return {
        "area": component["planform"]["span"] * integral,
        "mac": quad(lambda eta: chord(eta) ** 2, 0, 1, epsabs=1e-11)[0] / integral,
    }

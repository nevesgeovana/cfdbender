"""Private input vocabulary of a lifting surface: the names a user writes.

Pipeline role: private helper of :mod:`cfdbender.geometry.config`. A fresh configuration is
written with the key names of the published geometry input form (``[surface]``, ``[planform]``
with ``sref_m2``, ``aspect_ratio`` and ``taper`` or ``semispan_mm`` and the two chords,
``[[stations]]``, ``[cap]``, ``[[control]]``, ``[[engine]]``). This module reads those names and
hands the resolver the quantities it works with, in metres and degrees. The names the first
drafts used were never released: each is refused with a message naming its replacement, never
read as an alias. A resolved configuration is not translated; it already holds resolved names.

A station, a control surface span and an engine axis are fractions of the span station where the
tip cap starts (``eta``): the resolver's own eta runs over the span the skin covers, the same
quantity. Only the half width of an engine is a length, so :func:`finish_lifting` divides it by
the skin's share of the semi-span.
"""

from __future__ import annotations

import json
from pathlib import Path

from cfdbender._errors import InputValueError
from cfdbender.geometry._schema import known_keys, number

__all__ = ["adopt_surface_file", "finish_lifting", "refuse_renamed", "translate_lifting"]

#: Old planform names, with the replacement the refusal names.
_RENAMED_PLANFORM = {
    "span": "semispan_mm (half the span, tip cap included) or sref_m2 with aspect_ratio",
    "height": "semispan_mm",
    "area": "sref_m2",
    "root_chord": "root_chord_mm",
    "tip_chord": "tip_chord_mm",
    "sweep": "le_sweep_deg",
    "dihedral": "dihedral_deg",
    "orientation": "surface.span_axis",
    "twist_axis": "surface.twist_axis",
}
_RENAMED_TOP = {
    "tip_cap": "cap (with type in place of kind)",
    "control_surfaces": "control (with eta = [inner, outer] in place of eta_in and eta_out)",
}
_RENAMED_SECTIONS = {
    "items": "stations (each with eta and section = {kind = ...})",
    "airfoils": 'stations (each with section = {kind = "naca", code = ...})',
    "eta": "stations (each with eta)",
    "nacelle_bounds": "engine (eta_axis and half_width_mm)",
}
_SET_A = ("semispan_mm", "root_chord_mm", "tip_chord_mm")
_SURFACE_KEYS = ("name", "span_axis", "section_type", "model", "twist_axis")
_STATION_KEYS = ("eta", "twist_deg", "te_thickness_mm", "section", "note")
_CONTROL_KEYS = ("name", "kind", "eta", "chord_ratio", "cove", "hinge_line")
_ENGINE_KEYS = ("name", "eta_axis", "half_width_mm")
_PLANFORM_TO_INTERNAL = {
    "sref_m2": ("area", 1.0),
    "root_chord_mm": ("root_chord", 1e-3),
    "tip_chord_mm": ("tip_chord", 1e-3),
    "le_sweep_deg": ("sweep", 1.0),
    "dihedral_deg": ("dihedral", 1.0),
}
#: Keys of the proposal no verb implements yet: each is warned about and skipped.
_NOT_IMPLEMENTED = {
    "bay": "kinked planforms by bays are not implemented, the single trapezoid is built",
}
_FILE_LEVEL = ("units", "aircraft", "acceptance", "reader", "dominance")


def adopt_surface_file(source):
    """Return ``source`` with a ``[surface]`` or ``[body]`` file read as one component.

    A file of the published form holds no ``type``: ``[surface]`` makes it a lifting surface and
    ``[body]`` a body. The file-level tables (``units``, ``aircraft``, ``acceptance``,
    ``reader``) stay at the file level; every other table belongs to the component.

    Parameters
    ----------
    source : Mapping
        A parsed configuration file.

    Returns
    -------
    Mapping
        The file with a ``components`` table, or ``source`` itself when it already has one or
        holds neither table.

    Examples
    --------
    >>> from cfdbender.geometry.stages import adopt_surface_file
    >>> adopted = adopt_surface_file({"surface": {"name": "W"}})
    >>> adopted["components"]["W"]["type"]
    'lifting_surface'
    """
    if "components" in source or "type" in source:
        return source
    kind = "lifting_surface" if "surface" in source else "body" if "body" in source else None
    if kind is None:
        return source
    component = {k: v for k, v in source.items() if k not in _FILE_LEVEL}
    head = component.get("surface" if kind == "lifting_surface" else "body")
    name = head.get("name", "COMPONENT") if isinstance(head, dict) else "COMPONENT"
    if kind == "body" and isinstance(head, dict):
        body = component.pop("body")
        component.update({k: v for k, v in body.items() if k != "name"})
        if "type" in body:
            component["body_type"] = component.pop("type")
    component["type"] = kind
    adopted = {k: v for k, v in source.items() if k in _FILE_LEVEL}
    adopted["components"] = {str(name): component}
    return adopted


def refuse_renamed(label, table, renamed):
    """Refuse each old key of ``table`` naming its replacement."""
    if not isinstance(table, dict):
        return
    for old, new in renamed.items():
        if old in table:
            raise InputValueError(f"{label}.{old}: renamed to {new}; the old name is refused")


def _planform_set(name, planform):
    """Refuse a planform that mixes the two complete sets, naming the redundant keys."""
    a = [k for k in _SET_A if k in planform]
    b = [k for k in ("sref_m2", "aspect_ratio") if k in planform]
    if a and b:
        raise InputValueError(
            f"{name}.planform: one set only, {', '.join(a)} and {', '.join(b)} are redundant "
            "together; give sref_m2, aspect_ratio and taper, or semispan_mm, root_chord_mm and "
            "tip_chord_mm"
        )


def _translate_planform(name, raw, vertical):
    """Replace the planform names by the quantities the resolver reads, in metres and degrees."""
    planform = raw.get("planform")
    if not isinstance(planform, dict):
        return
    _planform_set(name, planform)
    single_fin = vertical and _on_the_plane(planform.get("root_le_mm"))
    if "root_le_mm" in planform:
        _root_le(name, raw, planform.pop("root_le_mm"))
    for key in (*_PLANFORM_TO_INTERNAL, "semispan_mm"):
        if key in planform:
            positive = None if key.endswith("_deg") else 0
            value = number(planform.pop(key), f"{name}.planform.{key}", positive, strict=True)
            planform[_internal_name(key, vertical)] = value * _factor(key, vertical)
    if single_fin and "area" in planform and "aspect_ratio" in planform:
        # Sref of a single fin on the plane is that one surface's area; the resolver reads a
        # height, and the span of the mirrored pair that the aspect ratio refers to is twice it.
        number(planform["aspect_ratio"], f"{name}.planform.aspect_ratio", 0, strict=True)
        planform["height"] = 0.5 * (2 * planform["area"] * planform["aspect_ratio"]) ** 0.5


def _on_the_plane(point):
    """Whether a root leading edge lies on the symmetry plane (absent counts as on it)."""
    return not isinstance(point, list) or len(point) != 3 or point[1] == 0


def _internal_name(key, vertical):
    """Return the resolver's name of a planform key; a fin keeps its one-sided height."""
    if key == "semispan_mm":
        return "height" if vertical else "span"
    return _PLANFORM_TO_INTERNAL[key][0]


def _factor(key, vertical):
    """Return the factor from the input unit to the resolver's: a semispan becomes a span."""
    if key == "semispan_mm":
        return 1e-3 if vertical else 2e-3
    return _PLANFORM_TO_INTERNAL[key][1]


def _root_le(name, raw, point):
    """Place the component at the root leading edge ``root_le_mm``."""
    if not isinstance(point, list) or len(point) != 3:
        raise InputValueError(f"{name}.planform.root_le_mm: requires three values")
    for value in point:
        number(value, f"{name}.planform.root_le_mm")
    placement = raw.setdefault("placement", {})
    if "translation" in placement:
        raise InputValueError(
            f"{name}.planform.root_le_mm: given together with placement.translation"
        )
    placement["translation"] = [v * 1e-3 for v in point]


def _translate_surface(name, raw, warnings):
    """Read ``[surface]``; return whether the span axis is vertical."""
    surface = raw.pop("surface", None)
    if surface is None:
        return False
    known_keys(surface, _SURFACE_KEYS, f"{name}.surface", warnings)
    if "name" in surface and str(surface["name"]) != name:
        # a [surface] file is named by surface.name (adopt_surface_file); inside a components
        # table the table key names the component and a different surface.name is not used
        warnings.append(
            f"{name}.surface.name: the component is named by its table key {name!r}, skipped"
        )
    for key in ("section_type", "model"):
        _accept_or_skip(name, surface, key, warnings)
    axis = surface.get("span_axis", "y")
    if axis not in ("y", "z"):
        raise InputValueError(f"{name}.surface.span_axis: expected y or z")
    if "twist_axis" in surface:
        raw.setdefault("planform", {})["twist_axis"] = surface["twist_axis"]
    if axis == "z":
        raw.setdefault("planform", {})["orientation"] = "vertical"
    return axis == "z"


def _accept_or_skip(name, surface, key, warnings):
    """Accept ``full`` for a ``surface`` key and skip ``half`` by name."""
    if key not in surface:
        return
    value = surface[key]
    if value not in ("full", "half"):
        raise InputValueError(f"{name}.surface.{key}: expected full or half")
    if value != "full":
        warnings.append(f"{name}.surface.{key}: {value!r} is not implemented, skipped")


def _translate_cap(name, raw, warnings):
    """Read ``[cap]`` as the tip cap; ``type`` is the kind, a template has no reader yet."""
    cap = raw.pop("cap", None)
    if cap is None:
        return
    if not isinstance(cap, dict):
        raise InputValueError(f"{name}.cap: expected a table")
    cap = dict(cap)
    if "template" in cap:
        cap.pop("template")
        warnings.append(f"{name}.cap.template: cap templates are not implemented, skipped")
    kind = cap.pop("type", None)
    if kind == "original":
        warnings.append(f"{name}.cap.type: 'original' needs a template, not implemented, skipped")
    elif kind is not None:
        cap["kind"] = kind
    raw["tip_cap"] = cap


def _read_bspline_file(path, base, file):
    """Read a section file: a JSON object with ``degree``, ``knots`` and ``poles``."""
    target = Path(base) / file
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InputValueError(f"{path}.section.file: file not found") from None
    except (OSError, ValueError) as error:
        raise InputValueError(f"{path}.section.file: not a JSON file ({error})") from None
    if not isinstance(data, dict) or not {"degree", "knots", "poles"} <= set(data):
        raise InputValueError(f"{path}.section.file: expected degree, knots and poles")
    return {k: data[k] for k in ("degree", "knots", "poles")}


def _station_section(path, section, base):
    """Return the section item of a station (the resolver's item form)."""
    if not isinstance(section, dict) or "kind" not in section:
        raise InputValueError(f"{path}.section.kind: required")
    kind = section["kind"]
    body = {k: v for k, v in section.items() if k != "kind"}
    if kind == "naca":
        return {"naca": body.get("code")}
    if kind == "coords":
        return {"file": body.get("file")}
    if kind == "cst":
        return {k: v for k, v in body.items() if k in ("cst_upper", "cst_lower", "te")}
    if kind != "bspline":
        raise InputValueError(f"{path}.section.kind: expected bspline, naca, cst or coords")
    if "file" in body:
        return {"bspline": _read_bspline_file(path, base, body["file"])}
    return {"bspline": body}


def _station_twist(name, raw, stations, warnings):
    """Map the stations' ``twist_deg`` to the linear twist law; other cases skip by name."""
    given = [s for s in stations if "twist_deg" in s]
    if not given:
        return
    planform = raw.setdefault("planform", {})
    if "incidence" in planform or "tip_twist" in planform:
        raise InputValueError(f"{name}.stations.twist_deg: given together with planform.incidence")
    for s in given:
        number(s["twist_deg"], f"{name}.stations.twist_deg")
    first, last = given[0], given[-1]
    if first["eta"] != 0:
        warnings.append(f"{name}.stations.twist_deg: needs a station at eta 0, skipped")
    elif len(given) == 1:
        planform["incidence"] = first["twist_deg"]
    elif len(given) == 2 and last["eta"] == 1:
        planform["incidence"] = first["twist_deg"]
        planform["tip_twist"] = last["twist_deg"] - first["twist_deg"]
    else:
        warnings.append(
            f"{name}.stations.twist_deg: only a twist at eta 0, or at eta 0 and 1, is "
            "implemented, skipped"
        )


def _translate_stations(name, raw, notes, base):
    """Read ``[[stations]]`` as the section items, the twist law and the knot-only breaks."""
    stations = raw.pop("stations", None)
    if stations is None:
        return
    warnings = notes["warnings"]
    if not isinstance(stations, list) or not all(isinstance(s, dict) for s in stations):
        raise InputValueError(f"{name}.stations: expected a list of tables")
    for i, s in enumerate(stations):
        known_keys(s, _STATION_KEYS, f"{name}.stations[{i}]", warnings)
        number(s.get("eta"), f"{name}.stations[{i}].eta", 0, 1)
        s.pop("note", None)
        if "te_thickness_mm" in s:
            s.pop("te_thickness_mm")
            warnings.append(
                f"{name}.stations[{i}].te_thickness_mm: per-station trailing edge thickness "
                "is not implemented, skipped"
            )
    _station_twist(name, raw, stations, warnings)
    items, breaks, index = [], [], []
    for i, s in enumerate(stations):
        if "section" not in s:
            breaks.append(s["eta"])
            continue
        item = _station_section(f"{name}.stations[{i}]", s["section"], base)
        items.append({"eta": s["eta"], **item})
        index.append(i)
    notes.setdefault("station_index", {})[name] = index
    if items:
        raw.setdefault("sections", {})["items"] = items
    if breaks:
        raw.setdefault("planform", {})["breaks"] = breaks


def _translate_control(name, raw, warnings):
    """Read ``[[control]]`` as the control surfaces; ``eta`` is the span interval."""
    table = raw.pop("control", None)
    if table is None:
        return
    if not isinstance(table, list) or not all(isinstance(c, dict) for c in table):
        raise InputValueError(f"{name}.control: expected a list of tables")
    surfaces = []
    for i, source in enumerate(table):
        path = f"{name}.control[{i}]"
        refuse_renamed(path, source, {"eta_in": "eta = [inner, outer]", "eta_out": "eta"})
        surface = _control_surface(path, source, warnings)
        if surface is not None:
            surfaces.append(surface)
    raw["control_surfaces"] = surfaces


def _control_surface(path, source, warnings):
    """Return one control surface in the resolver's form, or ``None`` when it must be skipped."""
    surface = {k: v for k, v in source.items() if k not in ("eta", "hinge_line")}
    eta = source.get("eta")
    if not isinstance(eta, list) or len(eta) != 2:
        raise InputValueError(f"{path}.eta: expected [inner, outer]")
    surface["eta_in"], surface["eta_out"] = eta
    if source.get("kind") == "plain":
        surface["kind"] = "plain_flap"
    if "hinge_line" in source:
        warnings.append(f"{path}.hinge_line: not implemented, skipped")
    cove = source.get("cove")
    tabulated = isinstance(cove, dict) and "from" in cove
    if source.get("kind") == "fowler" and (cove is None or tabulated):
        warnings.append(
            f"{path}.cove: a fowler flap without a stated cove (a tabulated cove is not "
            f"implemented); the flap {source.get('name', '')!r} is skipped"
        )
        return None
    return surface


def translate_lifting(name, raw, notes, base):
    """Translate a fresh lifting-surface input to the resolver's names, in place.

    A resolved input is left alone. Every old name is refused with its replacement.
    """
    if notes["resolved"]:
        return
    warnings = notes["warnings"]
    refuse_renamed(name, raw, _RENAMED_TOP)
    refuse_renamed(f"{name}.planform", raw.get("planform"), _RENAMED_PLANFORM)
    refuse_renamed(f"{name}.sections", raw.get("sections"), _RENAMED_SECTIONS)
    vertical = _translate_surface(name, raw, warnings)
    _translate_planform(name, raw, vertical)
    _translate_cap(name, raw, warnings)
    _translate_stations(name, raw, notes, base)
    _translate_control(name, raw, warnings)
    for key, note in _NOT_IMPLEMENTED.items():
        if raw.pop(key, None) is not None:
            warnings.append(f"{name}.{key}: {note}, skipped")


def finish_lifting(name, component, notes):
    """Turn the engines of a fresh input into nacelle bounds, in skin fractions of the span."""
    if notes["resolved"]:
        return
    engines = component.pop("engine", None)
    if engines is None:
        return
    from cfdbender.geometry._lifting import cap_fraction  # deferred: one call, no cycle

    share = cap_fraction(component)
    _apply_engines(name, component, engines, share, notes["warnings"])


def _apply_engines(name, component, engines, share, warnings):
    """Append the nacelle bounds of ``[[engine]]``: the axis plus and minus the half width."""
    if engines is None:
        return
    if not isinstance(engines, list) or not all(isinstance(e, dict) for e in engines):
        raise InputValueError(f"{name}.engine: expected a list of tables")
    semispan = component["planform"]["span"] / 2
    bounds = component["sections"].setdefault("nacelle_bounds", [])
    for i, engine in enumerate(engines):
        path = f"{name}.engine[{i}]"
        known_keys(engine, _ENGINE_KEYS, path, warnings)
        axis = number(engine.get("eta_axis"), f"{path}.eta_axis", 0, 1)
        if "half_width_mm" not in engine:
            warnings.append(
                f"{path}.half_width_mm: absent, the bounding sections are not derived, the "
                "engine is skipped"
            )
            continue
        half = number(engine["half_width_mm"], f"{path}.half_width_mm", 0, strict=True)
        reach = half * 1e-3 / semispan
        reach /= share  # the half width is a length; eta runs over the skin, not the cap
        low, high = axis - reach, axis + reach
        if not 0 <= low < high <= 1:
            raise InputValueError(f"{path}.half_width_mm: the bounds leave the semi-span")
        bounds.append({"component": str(engine.get("name", "")), "eta_in": low, "eta_out": high})

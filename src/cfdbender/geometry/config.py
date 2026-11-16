"""One TOML input boundary for the parametric geometry.

Pipeline role: pure configuration stage of the geometry package. Values use
metres, degrees and unitless fractions. A configuration is read from TOML,
validated, completed with documented defaults and returned with a unit and a
source (``given`` or ``default``) on every value. The resolved form can be
written back and read again without change.

Generation defaults that the supplied values do not fix are explicit in
``default_component``; they are generic examples, never inferred from any
particular aircraft, and no component name is known to the library. A key a
table does not know is skipped and reported in ``warnings``; the refusals are
listed in :mod:`cfdbender.geometry._schema`.
"""

from __future__ import annotations

import math
import re
import tomllib
from copy import deepcopy
from pathlib import Path

from cfdbender._digest import optional_file_sha256
from cfdbender._errors import InputValueError
from cfdbender.geometry._airfoils import naca_camber
from cfdbender.geometry._blister import blister_notes
from cfdbender.geometry._bodies import apply_rings, check_boat_tail, check_body_symmetry
from cfdbender.geometry._boundary import definition_units, unit_for, unpack
from cfdbender.geometry._component_keys import (
    body_sections,
    check_blister,
    check_body_extras,
    check_common,
    check_fillet,
    check_lifting_extras,
)
from cfdbender.geometry._fillet import fillet_notes
from cfdbender.geometry._identity import SCHEMA_MAJOR
from cfdbender.geometry._propeller_input import adopt_propeller, validate_definition
from cfdbender.geometry._schema import (
    TYPES,
    number,
    read_acceptance,
    read_dominance,
    read_reader,
    read_units,
    read_variants_and_registry,
    scale_units,
    stations,
)
from cfdbender.geometry._sections import trailing_edge_notes
from cfdbender.geometry._toml_body_keys import translate_body, user_names
from cfdbender.geometry._toml_keys import adopt_surface_file, finish_lifting, translate_lifting

__all__ = ["load_config", "resolve_config"]

RESOLVED_FORMAT = "cfdbender-geometry-resolved"

#: The optional tables and keys with no default, by component type and dotted place. Anywhere else
#: the same name is an unknown key.
FREE_BY_TYPE = {
    "propeller": frozenset({"definition"}),
    "lifting_surface": frozenset(
        {"distributions", "planform.root_chord", "planform.tip_chord", "planform.area"}
        | {"planform.height", "tip_cap.extent"}
        | {"sections.physical", "sections.nacelle_bounds", "engine"}
    ),
    "fillet": frozenset({"laws", "mesh_zone", "spine"}),
    "blister": frozenset({"stations", "profile", "sections"}),
}


def default_component(kind):
    """Return fresh plain data, with no shared mutable defaults."""
    common = {
        "type": kind,
        "version": "1.0.0",
        "placement": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]},
        "attach": {"parent": "", "station": 0.0},
        "original": "",
        "creases": [],
        "open_findings": [],
    }
    if kind == "lifting_surface":
        common.update(
            planform={
                "span": 1.0,
                "aspect_ratio": 8.0,
                "taper": 1.0,
                "sweep": 0.0,
                "orientation": "horizontal",
                "sweep_reference": "leading_edge",
                "dihedral": 0.0,
                "incidence": 0.0,
                "tip_twist": 0.0,
                "twist_axis": 0.25,
                "control_stations": [0.0, 1 / 3, 2 / 3, 1.0],
                "breaks": [],
                "slope_breaks": [],
            },
            sections={
                "airfoils": ["NACA0012", "NACA0012"],
                "eta": [0.0, 1.0],
                "degree": 5,
                "n1": 0.5,
                "n2": 1.0,
                "trailing_edge_thickness": 0.00252,
                "blend": "spline",
                "items": [],
                "form": "bspline",
                "continuity": "C2",
                "tolerance": {"analytic": 2e-5, "file": 1e-4},
                "pin": [],
            },
            spanwise={"law": "ruled", "lines": []},
            root="open",
            trailing_edge="blunt",
            control_surfaces=[],
            tip_cap={
                "kind": "flat",
                "length": 1.0,
                "exponent": 1.0,
                "sections": 30,
                "control_points": 31,
                "profile": {
                    "stations": [0.0, 0.25, 1.0],
                    "start": [0.0, 0.01, 0.015],
                    "ridge": [0.01, 0.07, 0.078],
                    "exponent": [2.3, 2.3, 2.6],
                },
            },
            root_fillet={"enabled": False, "radii": [0.0, 0.0, 0.0]},
            continuity="G1",
        )
    elif kind == "body":
        common.update(
            length=1.0,
            stations=[0.15, 0.35, 0.75],
            sections=[{"at": "middle", "shape": "circle", "radius": 0.1}],
            azimuths=[0.0, 60.0, 120.0, 180.0],
            rear="pole",
            rings_file="",
            nose="pole",
            nose_cap_fraction=0.004,
            regions=[],
            tolerance=1e-4,
            degree=3,
            symmetry="half_y0",
            base_radius=0.05,
            nose_exponent=0.5,
            rear_exponent=1.0,
            continuity="G1",
        )
    elif kind == "fillet":
        common.update(
            hosts=[],
            extend_lower_to_te=True,
            patches=20,
            wing_band=0.02,
            host_band=0.02,
        )
    elif kind == "blister":
        common.update(host="", sections_file="", tips="pole")
    elif kind == "wing_fuselage_fairing":
        common.update(
            hosts=[],
            start=-0.25,
            end=1.25,
            stations=[-0.25, 0.0, 0.3, 1.0, 1.25],
            width=[0.0, 0.1, 0.15, 0.1, 0.0],
            footprint_upper=45.0,
            footprint_lower=135.0,
            body_blend_radius=0.05,
            wing_blend_radius=0.05,
            start_exponent=0.5,
            end_exponent=0.5,
        )
    elif kind == "propeller":
        common.update(
            radius=1.0,
            hub_radius=0.2,
            blade_count=2,
            rotation_sense=1,
            stations=[0.2, 0.6, 1.0],
            chord=[0.12, 0.10, 0.06],
            twist=[0.0, 0.0, 0.0],
            xbar=[0.0, 0.0, 0.0],
            ybar=[0.0, 0.0, 0.0],
            setting_angle=0.0,
            pitch_reference=0.7,
            sections={
                "airfoils": ["NACA0012"],
                "eta": [0.0],
                "degree": 5,
                "n1": 0.5,
                "n2": 1.0,
                "trailing_edge_thickness": 0.00252,
                "blend": "spline",
            },
        )
    else:
        raise InputValueError(f"type: unsupported {kind!r}")
    return common


#: Keys of the one-skin parametrization that resolve and are checked but that the generated
#: surface of this version does not depend on, by component type. ``spanwise.law`` and
#: ``sections.pin`` shape the skin the generator writes, so they are not listed;
#: ``spanwise.lines`` and ``root`` reach only the layout model (iso lines and root caps),
#: which no verb writes yet.
_NOT_GENERATED = {
    "lifting_surface": (
        ("spanwise", "lines"),
        ("root",),
        ("sections", "continuity"),
        ("sections", "tolerance"),
    ),
    "body": (("regions",), ("nose_cap_fraction",), ("tolerance",), ("degree",)),
}


def warn_not_generated(name, component, warnings):
    """Warn for each key of ``_NOT_GENERATED`` given a value other than its default."""
    defaults = default_component(component["type"])
    for path in _NOT_GENERATED.get(component["type"], ()):
        value, default = component, defaults
        for key in path:
            value, default = value[key], default[key]
        if value != default:
            warnings.append(
                f"{name}.{'.'.join(path)}: resolved, but the generator of this version does not "
                "apply it yet; the surface is built as with the default"
            )


def merge_values(defaults, supplied, path="", warnings=None, free=frozenset(), *, place=""):
    """Overlay supplied values on defaults; a key the defaults do not know warns and is skipped."""
    result = deepcopy(defaults)
    for key, value in supplied.items():
        label = f"{path}.{key}".strip(".")
        inner = f"{place}.{key}".strip(".")
        if key not in defaults and inner not in free:
            if warnings is None:
                raise InputValueError(f"{label}: unknown parameter")
            warnings.append(f"{label}: unknown key, skipped")
            continue
        if isinstance(result.get(key), dict):
            if not isinstance(value, dict):
                raise InputValueError(f"{label}: expected a table")
            result[key] = merge_values(result[key], value, label, warnings, free, place=inner)
        else:
            result[key] = deepcopy(value)
    return result


def validate_ridge_profile(profile):
    """Validate a ridged cap profile.

    Per chord station: spanwise start of the rounding, ridge height and superellipse exponent.
    """
    stations(profile["stations"], "cap.profile.stations")
    if profile["stations"][0] != 0 or profile["stations"][-1] != 1:
        raise InputValueError("cap.profile.stations: must run from 0 to 1")
    for key in ("start", "ridge", "exponent"):
        values = profile[key]
        if not isinstance(values, list) or len(values) != len(profile["stations"]):
            raise InputValueError(f"cap.profile.{key}: count must match the stations")
        for value in values:
            number(value, f"cap.profile.{key}", 0, strict=key == "exponent")
    if any(r <= s for r, s in zip(profile["ridge"], profile["start"], strict=True)):
        raise InputValueError("cap.profile.ridge: must exceed start at every station")
    if any(n < 1.2 for n in profile["exponent"]):
        raise InputValueError("cap.profile.exponent: at least 1.2 required")


def resolve_planform(supplied, defaults):
    """Complete an independent trapezoidal set, rejecting redundant disagreement.

    With ``height`` (a single-sided surface, a fin) ``area`` is the area of that one surface;
    otherwise it is the area of the mirrored pair.
    """
    p = deepcopy(defaults)
    sides = 2.0 if "height" in supplied else 1.0
    if "height" in supplied:
        if "span" in supplied:
            raise InputValueError("planform.height: given together with planform.span")
        p["span"] = 2 * number(supplied["height"], "planform.height", 0, strict=True)
    span = number(p["span"], "planform.span", 0, strict=True)
    if "root_chord" in supplied:
        root = number(supplied["root_chord"], "planform.root_chord", 0, strict=True)
        tip = supplied.get("tip_chord", root * p["taper"])
        taper = tip / root
        area = span * (root + tip) / 2
        ar = span**2 / area
    elif "area" in supplied:
        area = sides * number(supplied["area"], "planform.area", 0, strict=True)
        ar = supplied.get("aspect_ratio", span**2 / area)
        if "span" not in supplied and "height" not in supplied and "aspect_ratio" in supplied:
            span = math.sqrt(area * ar)
        taper = p["taper"]
        root = 2 * area / (span * (1 + taper))
        tip = root * taper
    else:
        ar, taper = p["aspect_ratio"], p["taper"]
        number(ar, "planform.aspect_ratio", 0, strict=True)
        number(taper, "planform.taper", 0, strict=True)
        area = span**2 / ar
        root = 2 * area / (span * (1 + taper))
        tip = root * taper
    expected = {
        "span": span,
        "root_chord": root,
        "tip_chord": tip,
        "area": area / sides,
        "aspect_ratio": span**2 / area,
        "taper": taper,
    }
    for key, value in expected.items():
        number(value, f"planform.{key}", 0, strict=True)
        if key in supplied and not math.isclose(supplied[key], value, rel_tol=1e-10, abs_tol=1e-12):
            given = [k for k in (*expected, "height") if k in supplied]
            raise InputValueError(
                f"planform.{key}: inconsistent independent set; given {', '.join(given)}, and "
                "a trapezoidal planform is fixed by sref_m2, aspect_ratio and taper, or by "
                "semispan_mm with root_chord_mm and tip_chord_mm"
            )
    p.update(expected)
    return p


def check_slope_breaks(component):
    """Refuse slope breaks outside (0, 1) or missing from a supplied distribution's stations."""
    breaks = component["planform"]["slope_breaks"]
    if not breaks:
        return
    stations(breaks, "planform.slope_breaks", 1)
    if breaks[0] <= 0 or breaks[-1] >= 1:
        raise InputValueError("planform.slope_breaks: must lie strictly inside (0, 1)")
    for key, law in component.get("distributions", {}).items():
        missing = [b for b in breaks if b not in law["stations"]]
        if missing:
            raise InputValueError(f"distributions.{key}.stations: slope breaks {missing} missing")


def apply_orientation(raw, component):
    """Check ``planform.orientation``; a vertical surface is rolled 90 degrees unless placed."""
    orientation = component["planform"]["orientation"]
    if orientation not in ("horizontal", "vertical"):
        raise InputValueError("planform.orientation: expected horizontal or vertical")
    if orientation == "vertical" and "rotation" not in raw.get("placement", {}):
        component["placement"]["rotation"] = [90.0, 0.0, 0.0]


def resolve_lifting(raw, component, notes):
    """Complete and validate a lifting-surface component; return its derived area and mean chord."""
    component["planform"] = resolve_planform(raw.get("planform", {}), component["planform"])
    p, section = component["planform"], component["sections"]
    apply_orientation(raw, component)
    if "eta" not in raw.get("sections", {}) and not section["items"]:
        count = len(section["airfoils"])
        section["eta"] = [i / max(1, count - 1) for i in range(count)]
    stations(p["control_stations"], "planform.control_stations")
    if p["control_stations"][0] != 0 or p["control_stations"][-1] != 1:
        raise InputValueError("planform.control_stations: must cover [0, 1]")
    if p["breaks"]:
        stations(p["breaks"], "planform.breaks", 1)
    if p["sweep_reference"] not in ("leading_edge", "quarter_chord"):
        raise InputValueError("planform.sweep_reference: unsupported reference")
    number(p["twist_axis"], "planform.twist_axis", 0, 1)
    for key in ("sweep", "dihedral", "incidence", "tip_twist"):
        number(p[key], f"planform.{key}", -89, 89)
    for key, distribution in component.get("distributions", {}).items():
        if key not in ("leading_edge_x", "leading_edge_z", "chord", "twist"):
            raise InputValueError(f"distributions.{key}: unsupported distribution")
        stations(distribution["stations"], f"distributions.{key}.stations")
        if len(distribution["values"]) != len(distribution["stations"]):
            raise InputValueError(f"distributions.{key}.values: count mismatch")
        for value in distribution["values"]:
            number(
                value,
                f"distributions.{key}.values",
                0 if key == "chord" else None,
                strict=key == "chord",
            )
    check_slope_breaks(component)
    check_lifting_extras(notes["name"], component, raw, notes["warnings"], notes["findings"])
    warn_not_generated(notes["name"], component, notes["warnings"])
    if component["tip_cap"]["kind"] == "ridged":
        validate_ridge_profile(component["tip_cap"]["profile"])
    number(component["tip_cap"]["length"], "cap.length", 0, strict=True)
    number(component["tip_cap"]["exponent"], "cap.exponent", 0, strict=True)
    if len(component["root_fillet"]["radii"]) != 3:
        raise InputValueError("root_fillet.radii: requires LE, maximum thickness and TE values")
    for radius in component["root_fillet"]["radii"]:
        number(radius, "root_fillet.radii", 0)
    validate_sections(section)
    taper, root = p["taper"], p["root_chord"]
    return {
        "area": p["area"],
        "mac": 2 * root * (1 + taper + taper**2) / (3 * (1 + taper)),
        "mac_eta": (1 + 2 * taper) / (3 * (1 + taper)),
    }


def validate_sections(section):
    """Validate the section table of a lifting surface or propeller."""
    stations(section["eta"], "sections.eta", 1)
    if len(section["airfoils"]) != len(section["eta"]):
        raise InputValueError("sections.airfoils: count differs from sections.eta")
    for name in section["airfoils"]:
        if section.get("items") and name == "":
            continue
        if not isinstance(name, str) or not re.fullmatch(r"NACA(?:\d{4}|\d{5})", name):
            raise InputValueError("sections.airfoils: expected NACA designation")
        try:
            naca_camber(name, [0.5])
        except InputValueError as error:
            raise InputValueError(f"sections.airfoils: {name}: {error}") from error
    if not isinstance(section["degree"], int) or section["degree"] < 1:
        raise InputValueError("sections.degree: expected positive integer")
    for key in ("n1", "n2"):
        number(section[key], f"sections.{key}", 0, strict=True)
    number(section["trailing_edge_thickness"], "sections.trailing_edge_thickness", 0, 0.1)
    if section["blend"] != "spline":
        raise InputValueError("sections.blend: must be spline")


def resolve_body(raw, component, notes):
    """Validate a body component and return its (empty) derived quantities."""
    body_sections(notes["name"], component["sections"], notes["warnings"])
    body_sections(notes["name"], raw.get("sections", []), [])
    number(component["length"], "length", 0, strict=True)
    stations(component["stations"], "stations", 3)
    if component["stations"][0] <= 0 or component["stations"][-1] >= 1:
        raise InputValueError("stations: body transitions must be inside (0, 1)")
    check_body_symmetry(raw, component)
    check_body_extras(notes["name"], component, notes["warnings"], notes["findings"])
    warn_not_generated(notes["name"], component, notes["warnings"])
    if not component["sections"]:
        raise InputValueError("sections: requires a section")
    for i, section in enumerate(component["sections"]):
        name = f"sections[{i}]"
        if section.get("at") not in ("nose", "front", "middle", "rear"):
            number(section.get("at"), name + ".at", 0, 1)
        if section.get("shape", "circle") == "circle":
            number(section.get("radius"), name + ".radius", 0, strict=True)
        elif section["shape"] == "superellipse":
            for key in ("width", "height", "n_top", "n_bottom"):
                number(section.get(key), name + "." + key, 0, strict=True)
            number(section.get("centre_height", 0), name + ".centre_height")
        else:
            raise InputValueError(name + ".shape: unsupported shape")
    for key in ("base_radius", "nose_exponent", "rear_exponent"):
        number(component[key], key, 0, strict=True)
    check_boat_tail(component)
    return {}


def resolve_fairing(raw, component, notes):
    """Validate a wing-fuselage fairing component and return its (empty) derived quantities."""
    if len(component["hosts"]) != 2 or len(set(component["hosts"])) != 2:
        raise InputValueError("hosts: requires distinct body and lifting surface")
    check_common(notes["name"], component, notes["warnings"])
    stations(component["stations"], "stations", 5, component["start"], component["end"])
    if len(component["width"]) != len(component["stations"]):
        raise InputValueError("width: count differs from stations")
    for value in component["width"]:
        number(value, "width", 0)
    for key in ("body_blend_radius", "wing_blend_radius", "start_exponent", "end_exponent"):
        number(component[key], key, 0, strict=True)
    for key in ("footprint_upper", "footprint_lower"):
        number(component[key], key, 0, 180)
    return {}


def resolve_fillet(raw, component, notes):
    """Validate a fillet component and return its (empty) derived quantities."""
    check_fillet(notes["name"], component, notes["warnings"], notes["findings"])
    fillet_notes(notes["name"], component, notes["warnings"])
    return {}


def resolve_blister(raw, component, notes):
    """Validate a blister component and return its (empty) derived quantities."""
    check_blister(notes["name"], component, notes["warnings"], notes["findings"])
    blister_notes(notes["name"], component, notes["warnings"])
    return {}


def resolve_propeller(raw, component, notes):
    """Validate a propeller component and return its (empty) derived quantities."""
    if "definition" in component:
        component["definition"] = validate_definition(component["definition"])
        check_common(notes["name"], component, notes["warnings"])
        return {}
    number(component["radius"], "radius", 0, strict=True)
    number(component["hub_radius"], "hub_radius", 0, component["radius"], strict=True)
    if component["hub_radius"] >= component["radius"]:
        raise InputValueError("hub_radius: must be below radius")
    stations(component["stations"], "stations", 2, component["hub_radius"] / component["radius"], 1)
    if type(component["blade_count"]) is not int or component["blade_count"] < 1:
        raise InputValueError("blade_count: expected positive integer")
    if component["rotation_sense"] not in (-1, 1):
        raise InputValueError("rotation_sense: expected -1 or 1")
    for key in ("chord", "twist", "xbar", "ybar"):
        if len(component[key]) != len(component["stations"]):
            raise InputValueError(f"{key}: count differs from stations")
        for value in component[key]:
            number(value, key, 0 if key == "chord" else None, strict=key == "chord")
    number(component["setting_angle"], "setting_angle", -180, 180)
    number(component["pitch_reference"], "pitch_reference", 0, 1)
    validate_sections(component["sections"])
    check_common(notes["name"], component, notes["warnings"])
    notes["warnings"].append("propeller: generation is deferred to 0.58.0, the values are kept")
    return {}


def provenance(value, supplied, path="", *, propeller_units=None):
    """Wrap each leaf of a component as ``{value, unit, source}``; source is given or default."""
    if isinstance(value, dict):
        provided = supplied if isinstance(supplied, dict) else {}
        if path.endswith("definition"):
            propeller_units = definition_units(value)
        return {
            k: provenance(
                v, provided.get(k), f"{path}.{k}".strip("."), propeller_units=propeller_units
            )
            for k, v in value.items()
        }
    if isinstance(value, list) and value and isinstance(value[0], dict):
        given = supplied if isinstance(supplied, list) else []
        return [
            provenance(
                v, given[i] if i < len(given) else None, path, propeller_units=propeller_units
            )
            for i, v in enumerate(value)
        ]
    return {
        "value": value,
        "unit": unit_for(path, propeller_units),
        "source": "default" if supplied is None else "given",
    }


def validate_relationships(components):
    """Refuse attachment cycles, missing parents and missing or wrong hosts."""
    for name, component in components.items():
        parent = component["attach"]["parent"]
        seen = {name}
        while parent:
            if parent not in components or parent in seen:
                raise InputValueError(f"{name}.attach.parent: missing parent or cycle")
            seen.add(parent)
            parent = components[parent]["attach"]["parent"]
        kind = component["type"]
        if kind == "wing_fuselage_fairing" and len(components) > 1:
            for host, host_kind in zip(
                component["hosts"], ("body", "lifting_surface"), strict=True
            ):
                if host not in components or components[host]["type"] != host_kind:
                    raise InputValueError(f"{name}.hosts: expected {host_kind} host {host}")
        hosts = (
            component["hosts"]
            if kind == "fillet"
            else [component["host"]]
            if kind == "blister"
            else []
        )
        for host in hosts:
            if host not in components or host == name:
                raise InputValueError(f"{name}.hosts: missing host {host!r}")


def _skip_absent_references(components, warnings):
    """Warn about and skip a nacelle bound or a split that names no component of the file."""
    for name, component in components.items():
        section = component.get("sections")
        if isinstance(section, dict) and section.get("nacelle_bounds"):
            kept = [b for b in section["nacelle_bounds"] if b["component"] in components]
            for bound in section["nacelle_bounds"]:
                if bound not in kept:
                    warnings.append(
                        f"{name}.engine.name: {bound['component']!r} is no component of the "
                        "file, its bounding sections are skipped"
                    )
            section["nacelle_bounds"] = kept
        for surface in component.get("control_surfaces", []):
            for other in surface["split_around"]:
                if other not in components:
                    warnings.append(
                        f"{name}.control.{surface['name']}.split_around: {other!r} is no "
                        "component, the piece stays continuous"
                    )
            surface["split_around"] = [o for o in surface["split_around"] if o in components]


def assembly_defaults(components, supplied):
    """Seed the generic blend radius of a fairing, 0.4 of its wing host's root chord.

    Explicit inputs always win. No component is known by name: the hosts are the ones the
    fairing itself lists.
    """
    for name, component in components.items():
        raw = supplied[name]
        if component["type"] == "wing_fuselage_fairing":
            body_name, wing_name = component["hosts"]
            kinds = [components[n]["type"] if n in components else "" for n in component["hosts"]]
            if kinds == ["body", "lifting_surface"]:
                chord = components[wing_name]["planform"]["root_chord"]
                for key in ("body_blend_radius", "wing_blend_radius"):
                    if key not in raw:
                        component[key] = 0.4 * chord
                if "parent" not in raw.get("attach", {}):
                    component["attach"]["parent"] = body_name


def _resolve_component(name, raw, notes, base):
    """Resolve one component; every message names the key as the user wrote it."""
    first = len(notes["warnings"])
    kind = raw.get("type") if isinstance(raw, dict) else None
    try:
        result = _resolve_component_keys(name, raw, notes, base)
    except InputValueError as error:
        if notes["resolved"]:
            raise
        index = notes.get("station_index", {}).get(name)
        raise InputValueError(user_names(str(error), kind, index)) from error
    if not notes["resolved"]:
        index = notes.get("station_index", {}).get(name)
        notes["warnings"][first:] = [user_names(w, kind, index) for w in notes["warnings"][first:]]
    return result


def _translate_inputs(name, raw, notes, base):
    """Read the published key names of a fresh component and skip a cap shape."""
    if raw["type"] == "propeller" and "definition" in raw:
        notes.setdefault("propeller_given", {})[name] = deepcopy(raw["definition"])
        raw["definition"] = validate_definition(raw["definition"], base)
    elif raw["type"] == "lifting_surface":
        translate_lifting(name, raw, notes, base)
    elif raw["type"] == "body":
        translate_body(name, raw, notes)
    cap = raw.get("tip_cap")
    if isinstance(cap, dict) and "shape" in cap:
        cap.pop("shape")
        notes["warnings"].append(f"{name}.cap.shape: parameter not yet defined, skipped")
        notes["findings"].append(
            f"{name}.cap.shape: given and skipped until the parameter is defined"
        )


def _resolve_component_keys(name, raw, notes, base):
    """Merge, validate and return one component and its derived quantities."""
    if "type" not in raw:
        raise InputValueError(f"components.{name}.type: required")
    if raw["type"] not in TYPES:
        raise InputValueError(f"components.{name}.type: unsupported {raw['type']!r}")
    raw = deepcopy(raw)
    _translate_inputs(name, raw, notes, base)
    free = FREE_BY_TYPE.get(raw["type"], frozenset())
    component = merge_values(
        default_component(raw["type"]), raw, f"components.{name}", notes["warnings"], free
    )
    notes["name"] = name
    if raw["type"] == "body":
        apply_rings(component, base, notes["warnings"])
    resolvers = {
        "lifting_surface": resolve_lifting,
        "body": resolve_body,
        "fillet": resolve_fillet,
        "blister": resolve_blister,
        "wing_fuselage_fairing": resolve_fairing,
        "propeller": resolve_propeller,
    }
    derived = resolvers[raw["type"]](raw, component, notes)
    if not re.fullmatch(r"\d+\.\d+\.\d+", component["version"]):
        raise InputValueError(f"{name}.version: expected X.Y.Z")
    for key in ("translation", "rotation"):
        values = component["placement"][key]
        if len(values) != 3:
            raise InputValueError(f"{name}.placement.{key}: requires three values")
        for value in values:
            number(value, f"{name}.placement.{key}")
    number(component["attach"]["station"], f"{name}.attach.station", 0, 1)
    _check_data_files(name, component, base, notes["warnings"])
    if raw["type"] == "lifting_surface":
        finish_lifting(name, component, notes)
    if name in notes.get("propeller_given", {}):
        raw["definition"] = notes["propeller_given"][name]
    return component, derived, raw


def _check_data_files(name, component, base, warnings):
    """Refuse a data file that holds geometry and is missing; warn about a missing original."""
    needed = {key: component.get(key, "") for key in ("rings_file", "sections_file")}
    sections = component.get("sections")
    items = sections.get("items", []) if isinstance(sections, dict) else []
    needed |= {f"sections.items[{i}].file": item.get("file", "") for i, item in enumerate(items)}
    for key, path in needed.items():
        if path and not (base / path).is_file():
            raise InputValueError(f"{name}.{key}: file not found")
    for item in items:
        if item.get("file"):  # the generator reads it from wherever the input was read
            item["file"] = str((base / item["file"]).resolve())
    if component.get("sections_file"):  # a blister's sections, read when it is built
        component["sections_file"] = str((base / component["sections_file"]).resolve())
    if items:
        trailing_edge_notes(name, sections, warnings)
    original = component.get("original", "").split("#")[0]
    if original and not (base / original).is_file():
        warnings.append(f"{name}.original: file not found; the distance checks are skipped")


def _warn_metre_files(components, unit, warnings):
    """Warn that a blister's sections file is read in metres whatever ``[units]`` declares.

    The keys of the file are converted from the declared unit; the file's own ``y`` and ``z``
    are not, so a file written in that unit would build a blister that many times too large.
    """
    for name, component in components.items():
        if component.get("sections_file"):
            warnings.append(
                f"{name}.sections_file: its y and z are read in metres; units.length = "
                f"{unit!r} applies to the keys of this file, not to the sections file"
            )


def _file_digests(components, base):
    """Return the identity of every data file a component names: its role and its SHA-256."""
    rows = []
    for name, component in sorted(components.items()):
        paths = {
            "rings_file": component.get("rings_file", ""),
            "sections_file": component.get("sections_file", ""),
            "original": component.get("original", "").split("#")[0],
        }
        for i, item in enumerate(
            component.get("sections", {}).get("items", [])
            if isinstance(component.get("sections"), dict)
            else []
        ):
            if "file" in item:
                paths[f"sections.items[{i}].file"] = item["file"]
        for role, path in sorted(paths.items()):
            if path:
                digest = optional_file_sha256(base / path)
                rows.append({"role": f"{name}.{role}", "sha256": digest or ""})
        for aid, profile in sorted(component.get("definition", {}).get("airfoils", {}).items()):
            if "file" in profile:
                digest = optional_file_sha256(base / profile["file"])
                rows.append({"role": f"{name}.airfoils.{aid}.file", "sha256": digest or ""})
    return rows


def _guarded(label, work, *args):
    """Run ``work``; a value of the wrong form is a refusal naming ``label``, not a crash."""
    try:
        return work(*args)
    except (TypeError, AttributeError, KeyError) as error:
        raise InputValueError(f"{label}: a value has the wrong form ({error!r})") from error


def resolve_config(data, stem=None, base=None):
    """Validate a geometry configuration and resolve it with the source of every value.

    Parameters
    ----------
    data : dict
        A configuration as read from TOML: fresh input (a ``components``
        table, or a single component carrying ``type``) or a previously
        resolved configuration (``format = "cfdbender-geometry-resolved"``),
        whose derived quantities are recomputed and compared.
    stem : str, optional
        File stem, the default of ``aircraft.name``.
    base : str or os.PathLike, optional
        Directory that the data files named in the input (``rings_file``,
        ``sections_file``, ``original``) are read from; the working directory when absent.

    Returns
    -------
    dict
        The resolved configuration: ``format``, ``units`` (m and deg),
        ``aircraft``, ``acceptance``, ``reader``, ``components`` with each leaf wrapped as
        ``{value, unit, source}`` (source ``given`` or ``default``), ``derived`` quantities,
        ``dominance``, ``variants``, ``configurations`` (the effective member names) and
        ``registry`` (members with their versions, ``extends`` and variants), ``files`` (the
        SHA-256 of each data file), ``warnings`` and ``findings``. A resolved input that
        checks out is returned as a copy of itself.

    Raises
    ------
    InputValueError
        When the units are unknown, no component is given, a component lacks its type or an
        X.Y.Z version, a value or relationship is out of range, a dominance or extends cycle
        exists, a value has the wrong form, or a resolved input's derived quantity is
        inconsistent. Unknown keys do not raise: they are skipped and listed in ``warnings``.

    Examples
    --------
    >>> from cfdbender.geometry import resolve_config
    >>> resolved = resolve_config({"type": "body", "length_mm": 2000.0})
    >>> resolved["components"]["COMPONENT"]["length"]["value"]
    2.0
    """
    return _guarded("configuration", _resolve, data, stem, base)


def _resolve(data, stem, base):
    """Resolve a configuration; :func:`resolve_config` is the documented entry."""
    resolved = data.get("format") == RESOLVED_FORMAT
    source = adopt_propeller(adopt_surface_file(unpack(data)))
    if "type" in source:
        source = {"components": {"COMPONENT": source}}
    notes = {"warnings": [], "findings": [], "name": "", "resolved": resolved}
    known = {"aircraft", "units", "acceptance", "reader", "components", "dominance"}
    known |= {"configurations", "variants"}
    if resolved:
        known |= {"format", "derived", "warnings", "findings", "files", "registry", "schema"}
    for key in sorted(set(source) - known):
        notes["warnings"].append(f"{key}: unknown key, skipped")
    length_scale, angle_scale = read_units(source.get("units", {}), notes["warnings"])
    if not resolved:
        source = scale_units(source, "", length_scale, angle_scale)
    if not source.get("components"):
        raise InputValueError("components: requires at least one component")
    base = Path(base) if base is not None else Path()
    components, derived, raws = {}, {}, {}
    for name, raw in source["components"].items():
        components[name], derived[name], raws[name] = _guarded(
            f"components.{name}", _resolve_component, name, raw, notes, base
        )
    if not resolved and length_scale != 1.0:
        _warn_metre_files(components, source["units"]["length"], notes["warnings"])
    assembly_defaults(components, raws)
    validate_relationships(components)
    _skip_absent_references(components, notes["warnings"])
    aircraft = fill_aircraft(source.get("aircraft", {}), stem, notes["warnings"])
    registry_input = _registry_input(source, resolved)
    variants, registry = read_variants_and_registry(
        source.get("variants", {}), registry_input, components, notes["warnings"]
    )
    result = {
        "format": RESOLVED_FORMAT,
        "schema": SCHEMA_MAJOR,
        "units": {"length": "m", "angle": "deg"},
        "aircraft": provenance(aircraft, source.get("aircraft", {}), "aircraft"),
        "acceptance": provenance(
            read_acceptance(source.get("acceptance", {}), notes["warnings"]),
            source.get("acceptance", {}),
            "acceptance",
        ),
        "reader": provenance(
            read_reader(source.get("reader", {}), notes["warnings"]),
            source.get("reader", {}),
            "reader",
        ),
        "components": {
            name: provenance(component, raws[name], name) for name, component in components.items()
        },
        "derived": derived,
        "dominance": provenance(
            read_dominance(source.get("dominance", []), components, notes["warnings"]),
            source.get("dominance", []),
            "dominance",
        ),
        "variants": provenance(variants, source.get("variants", {}), "variants"),
        "configurations": {
            name: [m["component"] for m in row["members"]] for name, row in registry.items()
        },
        "registry": provenance(registry, registry_input, "registry"),
        "files": _file_digests(components, base),
        "warnings": notes["warnings"],
        "findings": notes["findings"],
    }
    return _checked_resolved(data, source, result) if resolved else result


def _checked_resolved(data, source, result):
    """Return a copy of a resolved input whose derived quantities check out.

    The digests of its data files are measured again, so a file that changed after the input was
    resolved moves the identity and is reported, instead of keeping the identity it had.
    """
    for name, quantities in result["derived"].items():
        for key, expected in quantities.items():
            supplied = source.get("derived", {}).get(name, {}).get(key)
            if supplied is None or not math.isclose(
                supplied, expected, rel_tol=1e-12, abs_tol=1e-14
            ):
                raise InputValueError(f"derived.{name}.{key}: inconsistent quantity")
    checked = deepcopy(data)
    if source.get("files", []) != result["files"]:
        checked["files"] = result["files"]
        checked["warnings"] = [
            *source.get("warnings", []),
            "files: a data file differs from the one this input was resolved with; the "
            "identity follows the file on disk",
        ]
    return checked


def _registry_input(source, resolved):
    """Return the configuration table of an input: the resolved registry, or the configurations."""
    table = source.get("registry" if resolved else "configurations", {})
    return {k: ({"members": v} if isinstance(v, list) else v) for k, v in table.items()}


def fill_aircraft(supplied, stem, warnings):
    """Resolve ``[aircraft]``; a missing name warns and the file stem is used."""
    table = {"name": stem or "unnamed"}
    if "name" not in supplied:
        warnings.append("aircraft.name: absent, the file stem is used")
    for key, value in supplied.items():
        if key == "name":
            table["name"] = value
        else:
            warnings.append(f"aircraft.{key}: unknown key, skipped")
    if not isinstance(table["name"], str) or not table["name"]:
        raise InputValueError("aircraft.name: expected text")
    return table


def _read_config(path):
    """Read a geometry TOML file, naming the correction when it is missing."""
    try:
        with Path(path).open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError as error:
        raise InputValueError(
            f"geometry configuration not found: {path}; "
            "provide an existing geometry configuration file"
        ) from error


def load_config(path):
    """Read a TOML geometry configuration file and return it validated and resolved.

    Parameters
    ----------
    path : str or os.PathLike
        The TOML file to read; the data files it names are read relative to its folder.

    Returns
    -------
    dict
        The resolved configuration, as returned by :func:`resolve_config`.
    """
    path = Path(path)
    return resolve_config(_read_config(path), stem=path.stem, base=path.parent)

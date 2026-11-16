"""Checks of the component keys that the first geometry release did not have.

Pipeline role: pure configuration stage, called by
:mod:`cfdbender.geometry.config` after a component's defaults and supplied
values are merged. It checks the keys common to every component, the section
items, the physical section form, the tip cap and control-surface tables of a
lifting surface, the rear forms of a body, and the fillet and blister
components. A key it does not know was skipped with a warning before this point;
what it refuses is named in the message with the key's dotted path.
"""

from __future__ import annotations

import math
import re

from cfdbender._errors import InputValueError
from cfdbender.geometry._propeller_input import (
    DEFINITION_KEYS as PROPELLER_DEFINITION_KEYS,
)
from cfdbender.geometry._propeller_input import TABLE_KEYS as PROPELLER_TABLE_KEYS
from cfdbender.geometry._schema import (
    check_body_law,
    check_bspline_item,
    check_lifting_layout,
    check_section_law,
    known_keys,
    number,
    stations,
)

__all__ = [
    "PROPELLER_DEFINITION_KEYS",
    "PROPELLER_TABLE_KEYS",
    "body_sections",
    "check_blister",
    "check_body_extras",
    "check_common",
    "check_fillet",
    "check_lifting_extras",
]

SURFACE_KINDS = ("aileron", "elevator", "rudder", "plain_flap", "fowler")
_CAP_KINDS = ("flat", "rounded", "ridged", "exact_loft", "none")
_NACA = re.compile(r"NACA(?:\d{4}|\d{5})")
_ITEM_KEYS = ("eta", "naca", "cst_upper", "cst_lower", "te", "file", "bspline")
_ITEM_KEYS_BY_FORM = {
    "naca": ("eta", "naca"),
    "cst_upper": ("eta", "cst_upper", "cst_lower", "te"),
    "file": ("eta", "file"),
    "bspline": ("eta", "bspline"),
}
_PHYSICAL_KEYS = ("nose_radius", "boat_tail_deg", "te_thickness", "thickness_at")
_BOUND_KEYS = ("component", "eta_in", "eta_out")
_BODY_SECTION_KEYS = (
    "at",
    "shape",
    "radius",
    "width",
    "height",
    "centre_height",
    "z_c",
    "n_top",
    "n_bottom",
)
_BOAT_TAIL_KEYS = ("x_cut", "base_height", "angle_limit_deg")
_MESH_ZONE_KEYS = ("across", "across_min", "ratio", "le_cell", "end_cell", "growth", "max_cell")
_KINEMATICS_KEYS = ("deflection_deg", "pivot_x", "pivot_z", "gap_pct", "overlap_pct")


def check_common(name, component, warnings):
    """Check ``original``, ``creases`` and ``open_findings`` of any component."""
    if not isinstance(component["original"], str):
        raise InputValueError(f"{name}.original: expected file#component text")
    if not isinstance(component["open_findings"], list) or not all(
        isinstance(f, str) for f in component["open_findings"]
    ):
        raise InputValueError(f"{name}.open_findings: expected a list of text")
    for i, crease in enumerate(component["creases"]):
        if not isinstance(crease, dict) or not isinstance(crease.get("at"), str):
            raise InputValueError(f"{name}.creases[{i}]: expected at and deg")
        number(crease.get("deg"), f"{name}.creases[{i}].deg", 0, 180)
        for key in set(crease) - {"at", "deg"}:
            warnings.append(f"{name}.creases[{i}].{key}: unknown key, skipped")


def _item_form(name, i, item):
    """Return which form a section item takes: ``naca``, ``cst`` or ``file``."""
    path = f"{name}.sections.items[{i}]"
    forms = [k for k in ("naca", "cst_upper", "file", "bspline") if k in item]
    if len(forms) != 1:
        raise InputValueError(
            f"{path}: expected one of naca, cst_upper with cst_lower, file, bspline"
        )
    number(item.get("eta"), f"{path}.eta", 0, 1)
    if forms[0] == "naca":
        if not isinstance(item["naca"], str) or not _NACA.fullmatch(f"NACA{item['naca']}"):
            raise InputValueError(f"{path}.naca: expected a NACA designation")
    elif forms[0] == "cst_upper":
        for key in ("cst_upper", "cst_lower"):
            values = item.get(key)
            if not isinstance(values, list) or len(values) < 2:
                raise InputValueError(f"{path}.{key}: expected at least two coefficients")
            for v in values:
                number(v, f"{path}.{key}")
        number(item.get("te", 0.0), f"{path}.te", 0, 0.1)
    elif forms[0] == "bspline":
        check_bspline_item(f"{path}.bspline", item["bspline"])
    elif not isinstance(item["file"], str) or not item["file"]:
        raise InputValueError(f"{path}.file: expected a path")
    return forms[0]


def _check_physical(name, physical, raw, warnings):
    """Check the physical form of a section, the alternative to a designation."""
    if physical is None:
        return
    if "airfoils" in raw or "items" in raw:
        raise InputValueError(f"{name}.sections.physical: given together with naca sections")
    known_keys(physical, _PHYSICAL_KEYS, f"{name}.sections.physical", warnings)
    for key in _PHYSICAL_KEYS:
        if key in physical:
            number(physical[key], f"{name}.sections.physical.{key}", 0)
    for key, high in (("nose_radius", 0.5), ("thickness_at", 0.5), ("boat_tail_deg", 89)):
        if key in physical:
            number(physical[key], f"{name}.sections.physical.{key}", 0, high, strict=True)
    missing = [k for k in ("nose_radius", "boat_tail_deg", "thickness_at") if k not in physical]
    if missing:  # refused with the input, not when the skin is built
        raise InputValueError(f"{name}.sections.physical: {', '.join(missing)} required")


def _check_item_degree(path, item, degree):
    """Refuse a ``bspline`` item of a degree above the surface's: it cannot be lowered exactly."""
    spec = item.get("bspline")
    if isinstance(spec, dict) and isinstance(degree, int) and spec["degree"] > degree:
        raise InputValueError(
            f"{path}.bspline.degree: {spec['degree']} is above sections.degree {degree}; a "
            "section is read exactly and its degree can be raised to the surface's, not lowered"
        )


def _check_sections(name, component, raw, warnings):
    """Resolve ``sections.items`` into the legacy airfoil and station lists, check the rest."""
    section = component["sections"]
    check_section_law(name, section, warnings)
    _check_physical(name, section.get("physical"), raw, warnings)
    items = section["items"]
    if not items:
        return
    if not isinstance(items, list):
        raise InputValueError(f"{name}.sections.items: expected a list of tables")
    for key in ("airfoils", "eta"):
        if key in raw:
            warnings.append(f"{name}.sections.{key}: sections.items takes its place, skipped")
    for i, item in enumerate(items):
        known_keys(item, _ITEM_KEYS, f"{name}.sections.items[{i}]", warnings)
    forms = [_item_form(name, i, item) for i, item in enumerate(items)]
    for i, (item, form) in enumerate(zip(items, forms, strict=True)):
        known_keys(item, _ITEM_KEYS_BY_FORM[form], f"{name}.sections.items[{i}]", warnings)
        _check_item_degree(f"{name}.sections.items[{i}]", item, section["degree"])
    stations([item["eta"] for item in items], f"{name}.sections.items", 1)
    section["eta"] = [item["eta"] for item in items]
    section["airfoils"] = [
        f"NACA{item['naca']}" if f == "naca" else "" for item, f in zip(items, forms, strict=True)
    ]


_SURFACE_DEFAULTS = {
    "kind": "plain_flap",
    "span_basis": "semi_span",
    "exposed_root_m": "host",
    "chord_ratio": 0.25,
    "gap_span": 0.005,
    "gap_chord": 0.0,
    "gap_closure": "end_faces",
    "deflection_deg": [0],
    "split_around": [],
}
_SURFACE_FREE = frozenset(
    {"name", "eta_in", "eta_out", "hinge", "hinge_axis", "nose", "cove"}
    | {"nested_dz_pct", "clearance_pct", "kinematics"}
)


def _fill_surface(name, i, raw, warnings):
    """Overlay one control surface table on its defaults; an unknown key warns and is skipped."""
    if not isinstance(raw, dict):
        raise InputValueError(f"{name}.control[{i}]: expected a table")
    surface = {k: (list(v) if isinstance(v, list) else v) for k, v in _SURFACE_DEFAULTS.items()}
    for key, value in raw.items():
        if key in _SURFACE_DEFAULTS or key in _SURFACE_FREE:
            surface[key] = value
        else:
            warnings.append(f"{name}.control[{i}].{key}: unknown key, skipped")
    return surface


def _check_interval(path, surface):
    """Check the span interval of a control surface; ``host`` is computed later."""
    for key in ("eta_in", "eta_out"):
        if surface.get(key) != "host":
            number(surface.get(key), f"{path}.eta", 0, 1)
    if "host" not in (surface["eta_in"], surface["eta_out"]):
        if surface["eta_in"] >= surface["eta_out"]:
            raise InputValueError(f"{path}.eta: overlaps or inverts the span interval")


def _check_choices(path, surface):
    """Check the named choices and the bounded numbers of a control surface."""
    if surface["kind"] not in SURFACE_KINDS:
        raise InputValueError(f"{path}.kind: expected one of {SURFACE_KINDS}")
    if surface["span_basis"] not in ("semi_span", "exposed"):
        raise InputValueError(f"{path}.span_basis: expected semi_span or exposed")
    if surface["gap_closure"] not in ("end_faces", "strip"):
        raise InputValueError(
            f"{path}.gap_closure: expected end_faces or strip, open leaves free edges"
        )
    number(surface["chord_ratio"], f"{path}.chord_ratio", 0.05, 0.5)
    number(surface["gap_span"], f"{path}.gap_span", 0, 0.05)
    number(surface["gap_chord"], f"{path}.gap_chord", 0)


def _check_surface(name, i, surface):
    """Check one control surface table of a lifting surface (C.7)."""
    path = f"{name}.control[{i}]"
    if not isinstance(surface.get("name"), str) or not surface["name"]:
        raise InputValueError(f"{path}.name: required")
    _check_choices(path, surface)
    _check_interval(path, surface)
    deflections = surface["deflection_deg"]
    if not isinstance(deflections, list) or not deflections:
        raise InputValueError(f"{path}.deflection_deg: expected a list")
    for d in deflections:
        number(d, f"{path}.deflection_deg", -89, 89)
    if surface["kind"] == "fowler":
        _check_fowler(path, surface, deflections)
        return
    for key in ("cove", "nested_dz_pct", "clearance_pct", "kinematics", "nose"):
        if key in surface:
            surface.pop(key)
            surface.setdefault("_skipped", []).append(f"{path}.{key}")


def _check_fowler(path, surface, deflections):
    """Check the extra keys of a Fowler flap; kinematics are required for a nonzero deflection."""
    cove = surface.get("cove", {})
    if "p3" not in cove:
        raise InputValueError(f"{path}.cove.p3: required for a fowler flap")
    number(cove["p3"].get("eta"), f"{path}.cove.p3.eta", 0, 1)
    number(cove["p3"].get("z_over_c"), f"{path}.cove.p3.z_over_c")
    number(cove.get("wall_x", 0.745), f"{path}.cove.wall_x", 0, 1, strict=True)
    number(cove.get("lip_x", 0.96), f"{path}.cove.lip_x", 0, 1, strict=True)
    number(surface.get("clearance_pct", 0.10), f"{path}.clearance_pct", 0)
    number(surface.get("nested_dz_pct", -0.145), f"{path}.nested_dz_pct")
    kinematics = surface.get("kinematics", [])
    needed = {d for d in deflections if d != 0}
    given = {row.get("deflection_deg") for row in kinematics}
    for row in kinematics:
        for key in ("pivot_x", "pivot_z", "gap_pct", "overlap_pct"):
            number(row.get(key), f"{path}.kinematics.{key}")
    if needed - given:
        raise InputValueError(
            f"{path}.kinematics: required for deflection {sorted(needed - given)}"
        )


def _check_no_overlap(name, surfaces):
    """Refuse two control surfaces of one host whose stated span intervals overlap."""
    stated = sorted(
        (s["eta_in"], s["eta_out"], s["name"])
        for s in surfaces
        if "host" not in (s["eta_in"], s["eta_out"])
    )
    for (_, end, left), (start, _, right) in zip(stated, stated[1:], strict=False):
        if start < end:
            raise InputValueError(f"{name}.control: {left} and {right} overlap in span")


def _check_tip_cap(cap):
    """Check the kind of a tip cap and the counts the exact loft reads."""
    if cap["kind"] not in _CAP_KINDS:
        raise InputValueError("cap.type: expected flat, rounded, ridged, exact_loft or none")
    for key in ("sections", "control_points"):
        if not isinstance(cap[key], int) or isinstance(cap[key], bool) or cap[key] < 2:
            raise InputValueError(f"cap.{key}: expected an integer of at least 2")


def _warn_loft_counts(name, cap, raw, warnings):
    """Warn that the exact-loft counts given to another kind of cap are not read."""
    if cap["kind"] == "exact_loft":
        return
    for key in ("sections", "control_points"):
        if key in raw:
            warnings.append(
                f"{name}.cap.{key}: read by an exact_loft cap only, a {cap['kind']} cap skips it"
            )


def _check_tip_extent(name, cap, raw, warnings):
    """Check ``tip_cap.extent``, the reach of a rounded or lofted cap in model units."""
    _warn_loft_counts(name, cap, raw, warnings)
    if "extent" not in cap:
        return
    number(cap["extent"], f"{name}.cap.extent", 0, strict=True)
    if "length" in raw:
        raise InputValueError(f"{name}.cap.extent: given together with cap.length")
    if cap["kind"] not in ("rounded", "exact_loft"):
        warnings.append(f"{name}.cap.extent: a {cap['kind']} cap has no reach, skipped")
        cap.pop("extent")


def _surface_tables(path, surface, warnings):
    """Warn about and skip the unknown keys of the tables inside a control surface."""
    if "cove" in surface:
        known_keys(surface["cove"], ("wall_x", "lip_x", "p3"), f"{path}.cove", warnings)
        if "p3" in surface["cove"]:
            known_keys(surface["cove"]["p3"], ("eta", "z_over_c"), f"{path}.cove.p3", warnings)
    if "nose" in surface:
        known_keys(surface["nose"], ("kind", "radius"), f"{path}.nose", warnings)
    if "kinematics" in surface:
        if not isinstance(surface["kinematics"], list):
            raise InputValueError(f"{path}.kinematics: expected a list of tables")
        for row in surface["kinematics"]:
            known_keys(row, _KINEMATICS_KEYS, f"{path}.kinematics", warnings)


def _check_nacelle_bounds(name, component, warnings):
    """Check the two control sections that bound each nacelle trim (requirement 4)."""
    bounds = component["sections"].get("nacelle_bounds", [])
    if not isinstance(bounds, list):
        raise InputValueError(f"{name}.sections.nacelle_bounds: expected a list of tables")
    for i, bound in enumerate(bounds):
        path = f"{name}.sections.nacelle_bounds[{i}]"
        known_keys(bound, _BOUND_KEYS, path, warnings)
        number(bound.get("eta_in"), f"{path}.eta_in", 0, 1)
        number(bound.get("eta_out"), f"{path}.eta_out", 0, 1)
        if bound["eta_in"] >= bound["eta_out"]:
            raise InputValueError(f"{path}.eta_in: expected below eta_out")
        if not isinstance(bound.get("component"), str):
            raise InputValueError(f"{path}.component: required")


def check_lifting_extras(name, component, raw, warnings, findings):
    """Check the C.2 and C.3 keys added to a lifting surface; fills sections from ``items``."""
    check_common(name, component, warnings)
    _check_sections(name, component, raw.get("sections", {}), warnings)
    for key, law in component.get("distributions", {}).items():
        known_keys(law, ("stations", "values"), f"{name}.distributions.{key}", warnings)
    if component["trailing_edge"] == "collapsed":
        warnings.append(f"{name}.trailing_edge: collapse is a mesh key, blunt is kept")
        component["trailing_edge"] = "blunt"
    if component["trailing_edge"] != "blunt":
        raise InputValueError(f"{name}.trailing_edge: expected blunt")
    check_lifting_layout(name, component)
    _check_tip_cap(component["tip_cap"])
    _check_tip_extent(name, component["tip_cap"], raw.get("tip_cap", {}), warnings)
    _check_nacelle_bounds(name, component, warnings)
    component["control_surfaces"] = [
        _fill_surface(name, i, raw, warnings) for i, raw in enumerate(component["control_surfaces"])
    ]
    names = [s.get("name") for s in component["control_surfaces"]]
    if len(set(names)) != len(names):
        raise InputValueError(f"{name}.control.name: duplicate")
    for i, surface in enumerate(component["control_surfaces"]):
        _surface_tables(f"{name}.control[{i}]", surface, warnings)
        _check_surface(name, i, surface)
        for skipped in surface.pop("_skipped", []):
            warnings.append(f"{skipped}: used by fowler only, skipped")
    _check_no_overlap(name, component["control_surfaces"])
    findings.extend(f"{name}: {text}" for text in component["open_findings"])


def body_sections(name, sections, warnings):
    """Warn about and skip the unknown keys of a body's sections; ``z_c`` is ``centre_height``."""
    if not isinstance(sections, list):
        raise InputValueError(f"{name}.sections: expected a list of tables")
    for i, section in enumerate(sections):
        path = f"{name}.sections[{i}]"
        known_keys(section, _BODY_SECTION_KEYS, path, warnings)
        if "z_c" in section:
            height = section.pop("z_c")
            if section.setdefault("centre_height", height) != height:
                raise InputValueError(f"{path}.z_c: differs from centre_height, the same key")


def check_body_extras(name, component, warnings, findings):
    """Check ``rings_file``, the rear forms and ``continuity`` of a body."""
    check_common(name, component, warnings)
    check_body_law(name, component)
    findings.extend(f"{name}: {text}" for text in component["open_findings"])
    if component["continuity"] != "G1":
        raise InputValueError(
            f"{name}.continuity: expected G1; the one exception is a declared crease"
        )
    rear = component["rear"]
    if isinstance(rear, dict):
        form = rear.get("boat_tail")
        if set(rear) != {"boat_tail"} or not isinstance(form, dict):
            raise InputValueError(f"{name}.rear: expected pole, base, open or a boat_tail table")
        known_keys(form, _BOAT_TAIL_KEYS, f"{name}.rear.boat_tail", warnings)
        number(form.get("x_cut"), f"{name}.rear.boat_tail.x_cut", 0, 1, strict=True)
        number(form.get("base_height"), f"{name}.rear.boat_tail.base_height", 0)
        number(
            form.get("angle_limit_deg"),
            f"{name}.rear.boat_tail.angle_limit_deg",
            0,
            90,
            strict=True,
        )
    elif rear not in ("base", "pole", "open"):
        raise InputValueError(f"{name}.rear: expected pole, base, open or a boat_tail table")
    if not isinstance(component["rings_file"], str):
        raise InputValueError(f"{name}.rings_file: expected a path")


def check_fillet(name, component, warnings, findings):
    """Check a fillet component: two hosts, a patch count and the bands."""
    check_common(name, component, warnings)
    findings.extend(f"{name}: {text}" for text in component["open_findings"])
    hosts = component["hosts"]
    if len(hosts) != 2 or len(set(hosts)) != 2 or not all(isinstance(h, str) and h for h in hosts):
        raise InputValueError(f"{name}.hosts: requires two distinct host names")
    if not isinstance(component["patches"], int) or component["patches"] < 4:
        raise InputValueError(f"{name}.patches: at least 4 required")
    for key in ("wing_band", "host_band"):
        number(component[key], f"{name}.{key}", 0, strict=True)
    if not isinstance(component["extend_lower_to_te"], bool):
        raise InputValueError(f"{name}.extend_lower_to_te: expected true or false")
    if "mesh_zone" in component:
        known_keys(component["mesh_zone"], _MESH_ZONE_KEYS, f"{name}.mesh_zone", warnings)
    if "spine" in component:
        known_keys(component["spine"], ("from", "to"), f"{name}.spine", warnings)
    if "laws" in component:
        known_keys(component["laws"], ("radius", "handle", "setback"), f"{name}.laws", warnings)


def _check_blister_law(name, component):
    """Refuse a blister without increasing stations, or without a depth and a sections file."""
    x = component.get("stations", {}).get("x")
    if x is None:
        raise InputValueError(f"{name}.stations.x: required")
    stations(x, f"{name}.stations.x", 2, -math.inf, math.inf)
    depth = component.get("profile", {}).get("depth")
    if depth is None and not component["sections_file"]:  # a sections file gives the depth
        raise InputValueError(f"{name}.profile.depth: required")
    if depth is None:
        return
    if isinstance(depth, list):  # one depth per station; the tips are poles
        for value in depth:
            number(value, f"{name}.profile.depth", 0)
    else:
        number(depth, f"{name}.profile.depth", 0, strict=True)


def check_blister(name, component, warnings, findings):
    """Check a blister component: a host, increasing stations and a depth."""
    check_common(name, component, warnings)
    findings.extend(f"{name}: {text}" for text in component["open_findings"])
    if not component["host"]:
        raise InputValueError(f"{name}.host: required")
    if "stations" in component:
        known_keys(component["stations"], ("x", "of"), f"{name}.stations", warnings)
    if "profile" in component:
        known_keys(
            component["profile"],
            ("depth", "plateau", "mirror_x", "of"),
            f"{name}.profile",
            warnings,
        )
    _check_blister_law(name, component)
    if component["sections_file"] == "" and "sections" not in component:
        component["sections"] = {"kind": "superellipse"}
    if component["tips"] != "pole":
        warnings.append(f"{name}.tips: only pole exists, pole is used")
        component["tips"] = "pole"

"""Private input vocabulary of a body, and the user's names in messages.

Pipeline role: private helper of :mod:`cfdbender.geometry.config`. A body is written with
``length_mm``, ``nose``, ``tail``, ``parts`` and ``sections`` whose shapes carry ``half_width_mm``,
``half_height_mm`` and ``centre_z_mm`` (the high-level body form). This module reads those names
and hands the resolver meters. The earlier names were never released and are refused with their
replacement. :func:`user_names` rewrites a message of the resolver so that it names the key as
the user wrote it, never the resolver's own table.
"""

from __future__ import annotations

import re

from cfdbender._errors import InputValueError
from cfdbender.geometry._schema import known_keys, number
from cfdbender.geometry._toml_keys import refuse_renamed

__all__ = ["translate_body", "user_names"]

_RENAMED_BODY = {
    "length": "length_mm",
    "rear": "tail (a table with kind)",
    "stations": "parts",
}
_RENAMED_SECTION = {
    "width": "half_width_mm",
    "height": "half_height_mm",
    "centre_height": "centre_z_mm",
    "z_c": "centre_z_mm",
    "radius": "radius_mm",
}
_SECTION_KEYS: tuple[str, ...] = (
    "at",
    "shape",
    "half_width_mm",
    "half_height_mm",
    "centre_z_mm",
    "radius_mm",
)
_SECTION_KEYS += ("n_top", "n_bottom")
_SECTION_TO_INTERNAL = {
    "half_width_mm": "width",
    "half_height_mm": "height",
    "centre_z_mm": "centre_height",
    "radius_mm": "radius",
}
_LOW_LEVEL = ("axis_yz", "stations_x", "creases", "patch", "rim")


def _translate_nose(name, raw, warnings):
    """Read ``nose = {kind}``; a ``cap`` nose has no builder yet and the pole is built."""
    nose = raw.get("nose")
    if nose is None:
        return
    if not isinstance(nose, dict):
        raise InputValueError(f'{name}.nose: expected a table, nose = {{kind = "pole"}}')
    known_keys(nose, ("kind", "patch"), f"{name}.nose", warnings)
    if "patch" in nose:
        warnings.append(f"{name}.nose.patch: the low-level patch form is not implemented, skipped")
    kind = nose.get("kind", "pole")
    if kind == "cap":
        warnings.append(f"{name}.nose.kind: 'cap' is not implemented, a pole nose is built")
        kind = "pole"
    raw["nose"] = kind


def _translate_tail(name, raw, warnings):
    """Read ``tail = {kind, at}``; a base cut by position needs a builder that is not there."""
    tail = raw.pop("tail", None)
    if tail is None:
        return
    if not isinstance(tail, dict):
        raise InputValueError(f'{name}.tail: expected a table, tail = {{kind = "base"}}')
    if "boat_tail" in tail:
        raw["rear"] = {"boat_tail": tail["boat_tail"]}
        return
    known_keys(tail, ("kind", "at", "x"), f"{name}.tail", warnings)
    for key in ("at", "x"):
        if key in tail:
            warnings.append(
                f"{name}.tail.{key}: a base cut by position is not implemented, skipped"
            )
    raw["rear"] = tail.get("kind", "pole")


def _translate_parts(name, raw, warnings):
    """Read ``parts``: the patch boundaries as x/L; the interior ones are the transitions."""
    parts = raw.pop("parts", None)
    if parts is None:
        return
    if not isinstance(parts, list) or not all(isinstance(p, (int, float)) for p in parts):
        raise InputValueError(f"{name}.parts: expected a list of numbers")
    for p in parts:
        number(p, f"{name}.parts", 0, 1)
    if any(a >= b for a, b in zip(parts, parts[1:], strict=False)):
        raise InputValueError(f"{name}.parts: the boundaries must increase along the body")
    interior = [p for p in parts if 0 < p < parts[-1]]
    if len(interior) < 3:
        warnings.append(
            f"{name}.parts: needs three interior boundaries; the default transitions are kept, "
            "skipped"
        )
        return
    raw["stations"] = interior


def _translate_sections(name, raw, warnings):
    """Read the section shapes: lengths are half extents in millimeters."""
    sections = raw.get("sections")
    if sections is None:
        return
    if not isinstance(sections, list):
        raise InputValueError(f"{name}.sections: expected a list of tables")
    result = []
    for i, section in enumerate(sections):
        path = f"{name}.sections[{i}]"
        if not isinstance(section, dict):
            raise InputValueError(f"{path}: expected a table")
        refuse_renamed(path, section, _RENAMED_SECTION)
        known_keys(section, _SECTION_KEYS, path, warnings)
        new = {}
        for key, value in section.items():
            if key in _SECTION_TO_INTERNAL:
                new[_SECTION_TO_INTERNAL[key]] = number(value, f"{path}.{key}") * 1e-3
            else:
                new[key] = value
        result.append(new)
    raw["sections"] = result


def translate_body(name, raw, notes):
    """Translate a fresh body input to the resolver's names, in place."""
    if notes["resolved"]:
        return
    warnings = notes["warnings"]
    refuse_renamed(name, raw, _RENAMED_BODY)
    if "length_mm" in raw:
        raw["length"] = number(raw.pop("length_mm"), f"{name}.length_mm", 0, strict=True) * 1e-3
    kind = raw.pop("body_type", "fuselage")
    if kind not in ("fuselage", "nacelle"):
        raise InputValueError(f"{name}.type: expected fuselage or nacelle")
    if kind == "nacelle":
        warnings.append(f"{name}.type: a nacelle spinner is not implemented, a body is built")
    _translate_nose(name, raw, warnings)
    _translate_tail(name, raw, warnings)
    _translate_parts(name, raw, warnings)
    _translate_sections(name, raw, warnings)
    for key in _LOW_LEVEL:
        if raw.pop(key, None) is not None:
            warnings.append(f"{name}.{key}: the low-level patch form is not implemented, skipped")


_LIFTING_RULES = [
    (r"sections\.items\[(\d+)\]\.bspline", r"stations[\1].section"),
    (r"sections\.items\[(\d+)\]\.naca", r"stations[\1].section.code"),
    (r"sections\.items\[(\d+)\]\.(file|cst_upper|cst_lower|te)", r"stations[\1].section.\2"),
    (r"sections\.items\[(\d+)\]", r"stations[\1].section"),
    (r"sections\.(?:items|airfoils|eta)\b", "stations"),
    (r"sections\.nacelle_bounds(?:\[(\d+)\])?", "engine"),
    (r"\btip_cap\.kind\b", "cap.type"),
    (r"\btip_cap\b", "cap"),
    (r"\bcontrol_surfaces\b", "control"),
]
_BODY_RULES = [
    (r"\bsections\[(\d+)\]\.width\b", r"sections[\1].half_width_mm"),
    (r"\bsections\[(\d+)\]\.height\b", r"sections[\1].half_height_mm"),
    (r"\bsections\[(\d+)\]\.centre_height\b", r"sections[\1].centre_z_mm"),
    (r"\bsections\[(\d+)\]\.radius\b", r"sections[\1].radius_mm"),
    (r"(?<![\w.])length:", "length_mm:"),
    (r"\.length:", ".length_mm:"),
    (r"(?<![\w.])stations:", "parts:"),
    (r"\.stations:", ".parts:"),
    (r"\brear\b(?=[:.])", "tail"),
]


def user_names(text, kind, index_map=None):
    """Return ``text`` with the resolver's key names replaced by the names the user wrote."""
    if "the old name is refused" in text:
        return text
    rules = _LIFTING_RULES if kind == "lifting_surface" else _BODY_RULES if kind == "body" else []
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    if index_map:
        text = re.sub(
            r"stations\[(\d+)\]", lambda m: f"stations[{_station(index_map, m.group(1))}]", text
        )
    return text


def _station(index_map, item):
    """Return the position in the user's list of the item the resolver counts as ``item``."""
    i = int(item)
    return index_map[i] if i < len(index_map) else i

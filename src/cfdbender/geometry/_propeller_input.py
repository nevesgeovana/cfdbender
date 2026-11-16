"""Strict dimensioned propeller definitions, shared by the API and TOML boundary.

Pipeline role: validate schema 1 before any fitting or kernel work. Radius is
in mm; station coordinates are fractions of radius and angles are degrees.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from cfdbender._errors import InputValueError

from ._schema import number, stations

DEFINITION_KEYS = {
    "schema_version",
    "propeller",
    "stations",
    "airfoils",
    "options",
    "output",
    "frame",
}
TABLE_KEYS = {
    "propeller": {
        "name",
        "units",
        "radius",
        "hub_radius",
        "blades",
        "rotation",
        "pitch_setting",
        "pitch_reference_r_R",
    },
    "stations": {
        "r_R",
        "c_R",
        "beta_deg",
        "t_c",
        "airfoil",
        "x_ref",
        "y_ref",
        "skew",
        "rake",
        "units",
    },
    "options": {
        "trailing_edge",
        "te_thickness",
        "root_cap",
        "tip",
        "root_extension_r_R",
        "loft",
        "fit_tol_chord",
        "thickness_from_t_c",
        "skew_tip_deg",
        "skew_tip_s_R",
        "skew_power",
        "rake_tip_x_R",
        "rake_tip_deg",
        "rake_power",
    },
    "output": {"iges", "units", "blades", "title", "note"},
    "frame": {"pitch_axis_point_mm", "blade_dir", "thrust_dir", "rotation_dir_at_blade"},
}


def adopt_propeller(data):
    """Wrap standalone schema 1 data as one geometry component, retaining its own units.

    Parameters
    ----------
    data : mapping
        Standalone propeller schema 1 data, or a configuration that is returned unchanged.

    Returns
    -------
    mapping
        The configuration holding the data as one propeller component.
    """
    if "schema_version" not in data and "propeller" not in data:
        return data
    strict_table(data, DEFINITION_KEYS, "definition")
    name = data.get("propeller", {}).get("name", "PROPELLER")
    return {
        "aircraft": {"name": name},
        "components": {"COMPONENT": {"type": "propeller", "definition": data}},
    }


def strict_table(value, keys, path):
    """Require a table whose keys all belong to the named schema node."""
    if not isinstance(value, dict):
        raise InputValueError(f"{path}: expected a table")
    for key in value:
        if key not in keys:
            raise InputValueError(
                f"{path}.{key}: unknown key; the keys are {', '.join(sorted(keys))}"
            )


def _choice(table, key, choices, path):
    if table[key] not in choices:
        raise InputValueError(
            f"{path}.{key}: got {table[key]!r}, expected one of {', '.join(map(str, choices))}"
        )


def _airfoils(table, base):
    if not isinstance(table, dict) or not table:
        raise InputValueError("airfoils: expected at least one named profile")
    for name, spec in table.items():
        path = f"airfoils.{name}"
        strict_table(spec, {"file", "naca", "cst"}, path)
        if len(spec) != 1:
            raise InputValueError(f"{path}: expected exactly one of file, naca, cst")
        if "naca" in spec:
            code = spec["naca"]
            if (
                not isinstance(code, str)
                or len(code) != 4
                or not code.isascii()
                or not code.isdigit()
                or int(code[-2:]) == 0
            ):
                raise InputValueError(
                    f"{path}.naca: expected a four digit designation with positive thickness"
                )
        elif "file" in spec:
            if not isinstance(spec["file"], str) or not spec["file"]:
                raise InputValueError(f"{path}.file: expected a path")
            file = Path(base) / spec["file"]
            if not file.is_file() or file.stat().st_size == 0:
                raise InputValueError(f"{path}.file: missing or empty file")
            spec["file"] = str(file.resolve())
        else:
            _cst(spec["cst"], f"{path}.cst")


def _cst(spec, path):
    strict_table(spec, {"upper", "lower", "trailing_edge_thickness", "n1", "n2"}, path)
    for key in ("upper", "lower"):
        values = spec.get(key)
        if not isinstance(values, list) or len(values) < 2:
            raise InputValueError(f"{path}.{key}: expected at least two coefficients")
        for value in values:
            number(value, f"{path}.{key}")
    if len(spec["upper"]) != len(spec["lower"]):
        raise InputValueError(f"{path}.lower: count differs from upper")
    for key, default in (("n1", 0.5), ("n2", 1.0), ("trailing_edge_thickness", 0.0)):
        number(spec.setdefault(key, default), f"{path}.{key}", 0)


def _columns(data):
    p, table = data["propeller"], data["stations"]
    radial = table.get("r_R")
    stations(radial, "stations.r_R", 4, max(p["hub_radius"] / p["radius"], 1e-12), 1)
    for key in ("c_R", "beta_deg", "t_c", "airfoil", "x_ref", "y_ref", "skew", "rake"):
        if key not in table and key in ("x_ref", "y_ref", "skew", "rake"):
            continue
        values = table.get(key)
        if not isinstance(values, list) or len(values) != len(radial):
            raise InputValueError(f"stations.{key}: expected one value per r_R")
        for value in values:
            if key == "airfoil":
                if not isinstance(value, str) or value not in data["airfoils"]:
                    raise InputValueError(
                        f"stations.airfoil: unknown profile id {value!r}; the [airfoils] "
                        f"table defines {', '.join(sorted(data['airfoils']))}"
                    )
            else:
                number(
                    value,
                    f"stations.{key}",
                    0 if key in ("c_R", "t_c") else None,
                    strict=key in ("c_R", "t_c"),
                )
    if not radial[0] <= p["pitch_reference_r_R"] <= radial[-1]:
        raise InputValueError("propeller.pitch_reference_r_R: outside station range")


def offsets(data):
    """Return skew angles and aft displacement fractions at every defining station."""
    table, options = data["stations"], data["options"]
    r = np.asarray(table["r_R"], float)
    units = table.get("units", {})
    strict_table(units, {"skew", "rake"}, "stations.units")
    results = []
    for key, default, permitted in (
        ("skew", "deg", ("deg", "s/R")),
        ("rake", "x/R", ("x/R", "deg")),
    ):
        unit = units.get(key, default)
        if unit not in permitted:
            raise InputValueError(f"stations.units.{key}: expected {permitted}")
        laws = (
            [
                (f"{key}_tip_{suffix}", law_unit)
                for suffix, law_unit in (("deg", "deg"), ("s_R", "s/R"))
            ]
            if key == "skew"
            else [("rake_tip_x_R", "x/R"), ("rake_tip_deg", "deg")]
        )
        present = [(name, u) for name, u in laws if name in options]
        if len(present) > 1 or (present and key in table):
            raise InputValueError(f"{key}: column and tip laws are mutually exclusive")
        power = number(options.get(f"{key}_power", 2.0), f"options.{key}_power", 0, strict=True)
        if present:
            name, unit = present[0]
            tip = number(options[name], f"options.{name}")
            value = tip * ((r - r[0]) / (1 - r[0])) ** power
        else:
            value = np.asarray(table.get(key, np.zeros(len(r))), float)
        if key == "rake" and unit == "deg":
            if np.any(np.abs(value) >= 90):
                raise InputValueError("stations.rake: angles must lie inside (-90, 90)")
            value = r * np.tan(np.deg2rad(value))
        elif key == "skew" and unit == "s/R":
            value = np.rad2deg(value / r)
        results.append(value)
    return results[0], results[1]


def axes(data):
    """Return origin, radial, thrust and motion unit vectors of a checked frame."""
    sign = 1 if data["propeller"]["rotation"] == "positive_x" else -1
    defaults = {
        "pitch_axis_point_mm": [0, 0, 0],
        "blade_dir": [0, 0, 1],
        "thrust_dir": [-1, 0, 0],
        "rotation_dir_at_blade": [0, -sign, 0],
    }
    frame = {**defaults, **data.get("frame", {})}
    vectors = []
    for key, value in frame.items():
        if not isinstance(value, list) or len(value) != 3:
            raise InputValueError(f"frame.{key}: expected three finite numbers")
        for v in value:
            number(v, f"frame.{key}")
        vector = np.asarray(value, float)
        if key != "pitch_axis_point_mm":
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise InputValueError(f"frame.{key}: zero direction")
            vector = vector / norm
        vectors.append(vector)
    origin, radial, thrust, motion = vectors
    gram = np.array([radial, thrust, motion]) @ np.array([radial, thrust, motion]).T
    if np.max(np.abs(gram - np.eye(3))) > 1e-12:
        raise InputValueError("frame: directions must be orthogonal")
    return origin, radial, thrust, motion


def _options(data):
    options = data["options"]
    defaults = {
        "trailing_edge": "blunt",
        "root_cap": "planar",
        "tip": "planar",
        "loft": "cubic",
        "fit_tol_chord": 1e-5,
        "thickness_from_t_c": True,
    }
    for key, value in defaults.items():
        options.setdefault(key, value)
    for key, choices in (
        ("trailing_edge", ("blunt", "sharp")),
        ("root_cap", ("planar", "none")),
        ("tip", ("planar", "none")),
        ("loft", ("cubic",)),
    ):
        _choice(options, key, choices, "options")
    number(options["fit_tol_chord"], "options.fit_tol_chord", 0, strict=True)
    if type(options["thickness_from_t_c"]) is not bool:
        raise InputValueError("options.thickness_from_t_c: expected boolean")
    if "te_thickness" in options:
        number(options["te_thickness"], "options.te_thickness", 0)
    if "root_extension_r_R" in options:
        extension = number(options["root_extension_r_R"], "options.root_extension_r_R", 0)
        if extension >= data["stations"]["r_R"][0]:
            raise InputValueError("options.root_extension_r_R: must be inboard of first station")
    output = data["output"]
    output.setdefault("blades", "all")
    output.setdefault("units", "mm")
    _choice(output, "blades", ("one", "all"), "output")
    _choice(output, "units", ("mm",), "output")
    for key in ("iges", "title", "note"):
        if key in output and not isinstance(output[key], str):
            raise InputValueError(f"output.{key}: expected text")


def validate_definition(supplied, base=Path()):
    """Return independent, completed schema 1 data, refusing each invalid key by name."""
    strict_table(supplied, DEFINITION_KEYS, "definition")
    data = deepcopy(supplied)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise InputValueError(f"schema_version: got {data.get('schema_version')!r}, expected 1")
    for key, keys in TABLE_KEYS.items():
        strict_table(data.setdefault(key, {}), keys, key)
    p = data["propeller"]
    number(p.get("radius"), "propeller.radius", 0, strict=True)
    if type(p.get("blades")) is not int or p["blades"] < 1:
        raise InputValueError("propeller.blades: expected a positive integer")
    defaults = {
        "units": "mm",
        "hub_radius": 0.0,
        "rotation": "positive_x",
        "pitch_setting": 30.0,
        "pitch_reference_r_R": 0.7,
        "name": "PROPELLER",
    }
    for key, value in defaults.items():
        p.setdefault(key, value)
    _choice(p, "units", ("mm",), "propeller")
    _choice(p, "rotation", ("positive_x", "negative_x"), "propeller")
    number(p["hub_radius"], "propeller.hub_radius", 0)
    if p["hub_radius"] >= p["radius"]:
        raise InputValueError("propeller.hub_radius: must be below radius")
    number(p["pitch_setting"], "propeller.pitch_setting", -89, 89)
    number(p["pitch_reference_r_R"], "propeller.pitch_reference_r_R", 0, 1)
    if not isinstance(p["name"], str) or not p["name"]:
        raise InputValueError("propeller.name: expected nonempty text")
    _airfoils(data.get("airfoils"), base)
    _columns(data)
    _options(data)
    offsets(data)
    axes(data)
    return data

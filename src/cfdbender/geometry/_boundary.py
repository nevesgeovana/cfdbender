"""Private helpers at the file and value boundary of the geometry package.

Pipeline role: private helpers of :mod:`cfdbender.geometry.config` and
:mod:`cfdbender.geometry.component`. For the configuration: the unit of a
dotted parameter path and the stripping of the ``{value, unit, source}`` wrappers
of a resolved configuration back to plain values (checking each unit). For the
component record: the named deviation profile's limit, the SHA-256 of a file, and
the draft 2020-12 JSON Schema export of the record model (the packaged copy is
compared with it in the test suite).
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from cfdbender._digest import file_sha256, text_sha256
from cfdbender._errors import InputValueError
from cfdbender._textio import write_text
from cfdbender.geometry.component import DEVIATION_PROFILES, ComponentRecord

_DEG_KEYS = frozenset(
    {
        "sweep",
        "dihedral",
        "incidence",
        "tip_twist",
        "twist",
        "rotation",
        "azimuths",
        "setting_angle",
        "footprint_upper",
        "footprint_lower",
        "boat_tail_deg",
    }
)
_M_KEYS = frozenset(
    {
        "span",
        "root_chord",
        "tip_chord",
        "radius",
        "hub_radius",
        "base_radius",
        "translation",
        "radii",
        "mac",
        "centre_height",
        "body_blend_radius",
        "wing_blend_radius",
        "base_height",
    }
)
#: Keys whose unit depends on the table that holds them: the values of a spanwise law, and the
#: station and depth of a blister. A section of a body holds lengths; a fairing's width is a ratio.
_UNIT_BY_TABLE = {
    ("chord", "values"): "m",
    ("leading_edge_x", "values"): "m",
    ("leading_edge_z", "values"): "m",
    ("twist", "values"): "deg",
    ("stations", "x"): "m",
    ("profile", "depth"): "m",
    ("profile", "plateau"): "m",
    ("profile", "mirror_x"): "m",
    ("nose", "radius"): "1",
}
_BODY_SECTION_LENGTHS = frozenset({"width", "height", "z_c"})


def _unit_from_table(parts):
    """Return the unit a key takes from its table, or ``None`` when its name alone decides."""
    if len(parts) < 2:
        return None
    if (
        parts[-1] in _BODY_SECTION_LENGTHS
        and parts[-2] == "sections"
        and len(parts) >= 3
        and parts[-3] != "components"
    ):
        return "m"
    return _UNIT_BY_TABLE.get((parts[-2], parts[-1]))


def unit_for(path, propeller_units=None):
    """Return the unit at a dotted path: ``deg``, ``mm``, ``m``, ``m2`` or ``1``."""
    parts = path.split(".")
    key = parts[-1]
    if len(parts) >= 3 and parts[-3] == "definition":
        if parts[-2] == "propeller":
            return {"radius": "mm", "hub_radius": "mm", "pitch_setting": "deg"}.get(key, "1")
        if parts[-2] == "stations" and key in ("skew", "rake"):
            default = "deg" if key == "skew" else "x/R"
            return (propeller_units or {}).get(key, default)
    by_table = _unit_from_table(parts)
    if by_table is not None:
        return by_table
    if key in _DEG_KEYS or key.endswith("_deg"):
        return "deg"
    if key == "area":
        return "m2"
    if key.endswith("_mm"):
        return "mm"
    if key in _M_KEYS or key.endswith("_m"):
        return "m"
    if key == "length" and "tip_cap" not in path:
        return "m"
    return "1"


def definition_units(value):
    """Read the declared column units of a raw or wrapped schema 1 definition."""
    table = value.get("stations", {})
    if not isinstance(table, dict) or "r_R" not in table:
        return None
    units = table.get("units", {})
    if not isinstance(units, dict):
        return None
    return {
        key: unit.get("value") if isinstance(unit, dict) else unit for key, unit in units.items()
    }


def unpack(value, path="", *, propeller_units=None):
    """Strip the ``{value, unit, source}`` wrappers, checking each unit, and return plain values.

    Parameters
    ----------
    value : object
        A resolved configuration value: a wrapper, a mapping, a list or a plain value.
    path : str, optional
        Dotted path of ``value``, which names the expected unit; default empty.
    propeller_units : mapping, optional
        Skew and rake column units while descending a schema 1 definition;
        inferred automatically at the definition boundary.

    Returns
    -------
    object
        A deep copy of ``value`` with every wrapper replaced by its plain value.

    Raises
    ------
    InputValueError
        If a wrapper carries a unit other than the one its path implies, or a source other than
        ``given`` or ``default``.

    Examples
    --------
    >>> from cfdbender.geometry.stages import unpack
    >>> unpack({"name": {"value": "wing", "unit": "1", "source": "given"}})
    {'name': 'wing'}
    """
    if isinstance(value, list):
        return [unpack(v, path, propeller_units=propeller_units) for v in value]
    if isinstance(value, dict) and set(value) == {"value", "unit", "source"}:
        expected = unit_for(path, propeller_units)
        if value["unit"] != expected:
            raise InputValueError(f"{path}.unit: expected {expected}")
        if value["source"] not in ("given", "default"):
            raise InputValueError(f"{path}.source: expected given or default")
        return deepcopy(value["value"])
    if isinstance(value, dict):
        if path.endswith("definition"):
            propeller_units = definition_units(value)
        return {
            k: unpack(v, f"{path}.{k}".strip("."), propeller_units=propeller_units)
            for k, v in value.items()
        }
    return deepcopy(value)


def deviation_limit_mm(local_size_mm: float, profile: str = "study_default") -> float:
    """Return the allowed deviation for a local chord or diameter under a named profile."""
    try:
        constants = DEVIATION_PROFILES[profile]
    except KeyError:
        raise InputValueError(
            f"deviation profile {profile!r}: unknown; available: {sorted(DEVIATION_PROFILES)}"
        ) from None
    return max(constants["floor_mm"], constants["local_fraction"] * local_size_mm)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of the exact bytes of a file as lowercase hex."""
    return file_sha256(path)


def export_schema() -> dict:
    """Return the draft 2020-12 JSON Schema of :class:`ComponentRecord`.

    Returns
    -------
    dict
        The schema, carrying its ``$schema`` and ``title`` keys.
    """
    schema = ComponentRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Shared geometry component"
    return schema


def schema_digest() -> str:
    """Return the SHA-256 of the exported schema in its canonical JSON form."""
    text = json.dumps(export_schema(), sort_keys=True, separators=(",", ":"))
    return text_sha256(text)


def _toml_value(value):
    """Render a plain value as an inline TOML value."""
    if isinstance(value, dict):
        return (
            "{ " + ", ".join(f"{json.dumps(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return repr(value)
    raise InputValueError(f"Unsupported TOML value: {value!r}")


def write_config(config, path):
    """Write a resolved configuration as TOML that :func:`load_config` reads back unchanged.

    Parameters
    ----------
    config : dict
        Resolved configuration of plain values.
    path : str or Path
        File to write.
    """
    text = "\n".join(f"{json.dumps(k)} = {_toml_value(v)}" for k, v in config.items()) + "\n"
    write_text(path, text)

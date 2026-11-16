"""Per-component surface mesh controls, validated before a kernel is opened.

Pipeline role: pure input boundary for the workspace mesh strategies. Geometry
membership and IGES names are supplied by the caller, never inferred from table
order. Realized topology and quality errors are checked by ``mesh_measurements``.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError

from ._mesh_rules import constant, geometric_cap, pin_fractions

ERRORS = {
    "E01": "follows names an absent component",
    "E02": "primary edge division mismatch",
    "E03": "follower cycle",
    "E04": "component absent from IGES names",
    "E05": "both count and step given",
    "E06": "invalid follower declaration",
    "E07": "strategy 2 forbids triangles",
    "E08": "junction ring counts differ",
    "E09": "opposite block sides differ",
    "E10": "carry row growth exceeds limit",
    "E11": "pole must have valence 3 or be open",
    "E12": "transition table needs role flexible",
    "E13": "azimuth rule conflicts with count or step",
    "E14": "body aspect outside band",
    "E15": "lifting surface needs chordwise and spanwise cells and law",
    "E16": "invalid mesh variant or control surface in clean variant",
}

_LAWS = {"constant", "geometric_cap", "cosine", "tanh"}
_COMPONENT_KEYS = {
    "type",
    "role",
    "follows",
    "fill",
    "chordwise",
    "spanwise",
    "x_cells",
    "x_step_m",
    "azimuth_cells",
    "azimuth_step_deg",
    "azimuth_rule",
    "families",
    "transition",
    "blocks",
    "trailing_edge",
    "collapse_tolerance_m",
    "mesh_zone",
    "parent",
}
_DIRECTION_KEYS = {
    "cells",
    "law",
    "ratio",
    "growth_until",
    "te_cells",
    "hinge_rows",
    "tip_cells",
    "tip_ratio",
    "first_cell",
    "root_cells",
    "root_ratio",
}
_TRANSITION_KEYS = {"target_ratio", "max_aspect", "max_skew_deg", "max_triangle_fraction"}


def _fail(code: str, where: str) -> None:
    raise InputValueError(f"{code}: {ERRORS[code]}: {where}")


def _table(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValueError(f"{where}: expected a table")
    return dict(value)


def _positive(value: Any, where: str, *, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or (integer and (not isinstance(value, int)))
    ):
        raise InputValueError(
            f"{where}: expected a finite positive {'integer' if integer else 'number'}"
        )


def _unknown(table: dict, allowed: set[str], where: str, warnings: list[str]) -> dict:
    for key in sorted(set(table) - allowed):
        warnings.append(f"{where}.{key}: unknown key ignored")
    return {key: value for key, value in table.items() if key in allowed}


def _distribution(value: Any, where: str, warnings: list[str], *, required: bool) -> dict:
    table = _unknown(_table(value, where), _DIRECTION_KEYS, where, warnings)
    if required and not {"cells", "law"} <= table.keys():
        _fail("E15", where)
    for key in ("cells", "te_cells", "root_cells", "tip_cells"):
        if key in table:
            if key in {"root_cells", "tip_cells"} and table[key] == 0 and type(table[key]) is int:
                continue
            _positive(table[key], f"{where}.{key}", integer=True)
    if "law" in table and table["law"] not in _LAWS:
        raise InputValueError(f"{where}.law: expected one of {sorted(_LAWS)}")
    for key in ("ratio", "growth_until", "tip_ratio", "first_cell", "root_ratio"):
        if key in table:
            _positive(table[key], f"{where}.{key}")
    for value in table.get("hinge_rows", []):
        _positive(value, f"{where}.hinge_rows")
        if value >= 1:
            raise InputValueError(f"{where}.hinge_rows: fractions must be inside (0, 1)")
    return table


def _transition(value: Any, where: str, warnings: list[str]) -> dict:
    table = _unknown(_table(value, where), _TRANSITION_KEYS, where, warnings)
    for key, value in table.items():
        if key == "max_triangle_fraction" and value == 0:
            continue
        _positive(value, f"{where}.{key}")
    if table.get("max_triangle_fraction", 0) > 1:
        raise InputValueError(f"{where}.max_triangle_fraction: at most 1")
    return table


def _role(c: dict, name: str) -> str:
    if c.get("parent"):
        c.setdefault("role", "follower")
        c.setdefault("follows", c["parent"])
        c.setdefault("fill", "full_quad")
    role = c.setdefault("role", "primary")
    if role not in {"primary", "flexible", "follower"}:
        raise InputValueError(f"{name}.role: primary, flexible or follower required")
    follows = c.get("follows", [])
    follows = [follows] if isinstance(follows, str) else follows
    if not isinstance(follows, list) or any(not isinstance(x, str) for x in follows):
        _fail("E06", name)
    if (role == "follower" and not follows) or (role == "primary" and follows):
        _fail("E06", name)
    c["follows"] = follows
    if c.get("fill", "full_quad") not in {"full_quad", "quad_dominant"}:
        raise InputValueError(f"{name}.fill: full_quad or quad_dominant required")
    return role


def _body_controls(c: dict, name: str) -> None:
    if "x_cells" in c and "x_step_m" in c:
        _fail("E05", name)
    if "azimuth_cells" in c and "azimuth_step_deg" in c:
        _fail("E05", name)
    if "azimuth_rule" in c and {"azimuth_cells", "azimuth_step_deg"} & c.keys():
        _fail("E13", name)
    for key in ("x_cells", "azimuth_cells", "x_step_m", "azimuth_step_deg", "collapse_tolerance_m"):
        if key in c:
            _positive(c[key], f"{name}.{key}", integer=key.endswith("cells"))


def _azimuth_rule(raw, name: str, warnings: list[str]) -> dict:
    rule = _unknown(
        _table(raw, name),
        {"mode", "target_aspect", "aspect_band", "reference"},
        f"{name}.azimuth_rule",
        warnings,
    )
    if rule.get("mode", "aspect") != "aspect":
        raise InputValueError(f"{name}.azimuth_rule.mode: aspect required")
    _positive(rule.get("target_aspect", 1.0), f"{name}.azimuth_rule.target_aspect")
    band = rule.get("aspect_band", [0.67, 1.5])
    if not isinstance(band, list) or len(band) != 2:
        raise InputValueError(f"{name}.azimuth_rule.aspect_band: two positive limits required")
    for value in band:
        _positive(value, f"{name}.azimuth_rule.aspect_band")
    if band[0] > band[1]:
        raise InputValueError(f"{name}.azimuth_rule.aspect_band: limits reversed")
    return rule


def _blocks(raw, name: str, warnings: list[str]) -> dict:
    blocks = _unknown(
        _table(raw, name), {"topology", "i", "j", "poles"}, f"{name}.blocks", warnings
    )
    poles = _table(blocks.get("poles", {}), f"{name}.blocks.poles")
    if any(value not in {"valence3", "open"} for value in poles.values()):
        _fail("E11", name)
    return blocks


def _component(name: str, raw: Any, strategy: int, warnings: list[str]) -> dict:
    c = _unknown(_table(raw, name), _COMPONENT_KEYS, name, warnings)
    role = _role(c, name)
    _body_controls(c, name)
    if "transition" in c:
        if role == "primary":
            _fail("E12", name)
        c["transition"] = _transition(c["transition"], f"{name}.transition", warnings)
    if strategy == 2 and (
        c.get("fill") == "quad_dominant" or c.get("transition", {}).get("max_triangle_fraction", 0)
    ):
        _fail("E07", name)
    required = c.get("type") == "lifting_surface" and role != "follower"
    for direction in ("chordwise", "spanwise"):
        if required or direction in c:
            c[direction] = _distribution(
                c.get(direction, {}), f"{name}.{direction}", warnings, required=required
            )
    if "azimuth_rule" in c:
        c["azimuth_rule"] = _azimuth_rule(c["azimuth_rule"], name, warnings)
    if "blocks" in c:
        c["blocks"] = _blocks(c["blocks"], name, warnings)
    if c.get("trailing_edge", "blunt") not in {"blunt", "collapsed"}:
        raise InputValueError(f"{name}.trailing_edge: blunt or collapsed required")
    return c


def _order(components: dict) -> list[str]:
    result: list[str] = []
    active: list[str] = []

    def visit(name: str) -> None:
        if name in active:
            _fail("E03", " -> ".join([*active, name]))
        if name in result:
            return
        active.append(name)
        for neighbour in components[name]["follows"]:
            visit(neighbour)
        active.pop()
        result.append(name)

    for name in components:
        visit(name)
    return result


def _components(
    tables: dict, members: set[str], names: set[str], mesh: dict, warnings: list[str]
) -> dict:
    components = {}
    for name, table in tables.items():
        if name not in members:
            warnings.append(f"W01: {name}: outside configuration; table ignored")
            continue
        if name not in names:
            _fail("E04", name)
        c = _component(name, table, mesh["strategy"], warnings)
        if mesh["variant"] == "clean" and c.get("parent"):
            _fail("E16", name)
        for neighbour in c["follows"]:
            if neighbour not in members:
                _fail("E01", f"{name}: {neighbour}")
            if neighbour not in tables:
                raise InputValueError(f"{name}.follows: {neighbour} has no mesh controls")
        components[name] = c
    return components


def mesh_controls(source: Any, *, members: set[str], names: set[str]) -> dict[str, Any]:
    """Validate a mapping or TOML using actual configuration and IGES membership.

    Parameters
    ----------
    source : mapping or path
        The controls mapping, or the TOML file holding it.
    members, names : set of str
        The component and IGES membership names the controls may refer to.

    Returns
    -------
    dict
        A fresh mapping with warnings and dependency order. Unknown keys warn and
        are skipped.

    Raises
    ------
    InputValueError
        When mesh controls are contradictory or unsafe, by error code.
    """
    raw = (
        tomllib.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, (str, Path))
        else deepcopy(_table(source, "controls"))
    )
    warnings: list[str] = []
    result = _unknown(raw, {"mesh", "defaults", "components", "junctions"}, "controls", warnings)
    mesh = _unknown(
        _table(result.get("mesh", {}), "mesh"),
        {"strategy", "configuration", "variant", "iges", "output"},
        "mesh",
        warnings,
    )
    strategy = mesh.get("strategy", 1)
    if type(strategy) is not int or strategy not in (1, 2):
        raise InputValueError("mesh.strategy: 1 or 2 required")
    mesh["strategy"] = strategy
    if mesh.get("variant", "clean") not in {"clean", "cs", "csf"}:
        _fail("E16", "mesh.variant")
    mesh.setdefault("variant", "clean")
    components = _components(
        _table(result.get("components", {}), "components"), members, names, mesh, warnings
    )
    defaults = _unknown(
        _table(result.get("defaults", {}), "defaults"),
        {"lifting_surface", "body", "quality", "transition", "junction"},
        "defaults",
        warnings,
    )
    if "transition" in defaults:
        defaults["transition"] = _transition(
            defaults["transition"], "defaults.transition", warnings
        )
        if strategy == 2 and defaults["transition"].get("max_triangle_fraction", 0):
            _fail("E07", "defaults.transition")
    return {
        **result,
        "mesh": mesh,
        "defaults": defaults,
        "components": components,
        "order": _order(components),
        "warnings": warnings,
    }


def mesh_measurements(code: str, where: str, measured: Any, threshold: Any) -> dict:
    """Refuse a realized topology or quality defect with its schema error name.

    E02 uses mismatch <= limit, E08/E09 equal counts, E10 growth <= limit,
    and E14 an inclusive aspect band.

    Parameters
    ----------
    code : str
        The schema error name the check belongs to.
    where : str
        The input the measurement describes.
    measured : float or sequence
        The measured value (a sequence of counts for E08/E09).
    threshold : float or tuple
        The limit, or the inclusive band for E14.

    Returns
    -------
    dict
        The measurement for stage records.

    Raises
    ------
    InputValueError
        When the measurement violates its threshold or the code is unsupported.
    """
    if code in {"E08", "E09"}:
        passed = len(set(measured)) == 1
    elif code == "E14":
        passed = threshold[0] <= measured <= threshold[1]
    elif code in {"E02", "E10"}:
        passed = math.isfinite(measured) and measured <= threshold
    else:
        raise InputValueError(f"measurement: unsupported error {code}")
    if not passed:
        _fail(code, f"{where}: measured {measured}, threshold {threshold}")
    return {
        "check": code,
        "input": where,
        "measured": measured,
        "threshold": threshold,
        "passed": True,
    }


def _span_nodes(values: dict) -> np.ndarray:
    count = values["cells"]
    root, tip = values.get("root_cells", 0), values.get("tip_cells", 0)
    if root + tip >= count:
        raise InputValueError("spanwise: root and tip runs must leave a constant interior")
    widths = np.ones(count)
    for n, ratio, reverse in (
        (root, values.get("root_ratio", 1.3), False),
        (tip, values.get("tip_ratio", 1.3), True),
    ):
        if ratio < 1:
            raise InputValueError("spanwise: refinement ratios must be at least 1")
        run = np.exp(-np.arange(n, 0, -1) * np.log(ratio))
        if reverse and n:
            widths[-n:] = run[::-1]
        elif n:
            widths[:n] = run
    return np.r_[0.0, np.cumsum(widths) / widths.sum()]


def _first_cell_nodes(count: int, first: float, growth_until: float) -> np.ndarray:
    if first > 1 / count or not 0 < growth_until < 1:
        raise InputValueError("first_cell: must allow nondecreasing cells summing to 1")
    if first == 1 / count:
        return np.asarray(constant(count))
    candidates = []
    for k in range(1, count):
        exponents = np.minimum(np.arange(count), k)
        low, high = 0.0, float(np.log(1 / first) / k)
        for _ in range(60):
            mid = (low + high) / 2
            if first * np.exp(exponents * mid).sum() > 1:
                high = mid
            else:
                low = mid
        widths = first * np.exp(exponents * ((low + high) / 2))
        nodes = np.r_[0.0, np.cumsum(widths)]
        candidates.append((abs(nodes[k] - growth_until), nodes))
    if not candidates:
        raise InputValueError("first_cell: at least two cells required")
    return min(candidates, key=lambda item: item[0])[1]


def spacing_nodes(table: Mapping[str, Any], *, direction: str) -> np.ndarray:
    """Build a fresh normalized spacing law, preserving the requested cell count.

    Span root and tip runs grow toward the constant interior. Coarse scenarios
    call this again with their new controls; no rows of a finer mesh are removed.
    """
    values = _distribution(table, direction, [], required=True)
    count, law = values["cells"], values["law"]
    x = np.linspace(0.0, 1.0, count + 1)
    if direction == "spanwise" and (values.get("root_cells") or values.get("tip_cells")):
        result = _span_nodes(values)
    elif law == "constant":
        result = np.asarray(constant(count))
    elif law == "geometric_cap":
        result = _geometric_nodes(values)
    elif law == "cosine":
        result = (1 - np.cos(np.pi * x)) / 2
    else:
        beta = values.get("ratio", 1.15)
        result = (1 + np.tanh(beta * (2 * x - 1)) / np.tanh(beta)) / 2
    result[0], result[-1] = 0.0, 1.0
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise InputValueError(f"{direction}: spacing is below floating point resolution")
    return result


def _geometric_nodes(values: dict) -> np.ndarray:
    if "first_cell" in values:
        return _first_cell_nodes(
            values["cells"], values["first_cell"], values.get("growth_until", 0.5)
        )
    return np.asarray(
        geometric_cap(
            values["cells"], values.get("ratio", 1.15), growth_until=values.get("growth_until", 0.5)
        )
    )


def azimuth_count(stations, rule: Mapping[str, Any], *, ring_count: int = 0) -> tuple[int, float]:
    """Derive a half-body count from measured (x, half perimeter, axial step) rows.

    A numeric reference must be a measured station. This boundary never invents
    an unmeasured perimeter or silently moves the requested reference station.

    Parameters
    ----------
    stations : array_like
        Measured rows of ``(x, half perimeter, axial step)``, all finite.
    rule : mapping
        The azimuth rule: ``target_aspect`` (default 1.0) and ``reference``
        (``"max_perimeter"`` or a measured station ``x``).
    ring_count : int, optional
        The count the result may not fall below.

    Returns
    -------
    tuple of int and float
        The half-body azimuth count and the reference station ``x``.

    Raises
    ------
    InputValueError
        When the rows are malformed or not positive, the aspect is not positive,
        ``ring_count`` is not a nonnegative integer, or the reference is not
        exactly one measured station.
    """
    rows = np.asarray(stations, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3 or not len(rows) or not np.all(np.isfinite(rows)):
        raise InputValueError(
            "azimuth_rule: finite measured station, perimeter and step rows required"
        )
    if np.any(rows[:, 1:] <= 0):
        raise InputValueError("azimuth_rule: perimeter and axial step must be positive")
    aspect = rule.get("target_aspect", 1.0)
    _positive(aspect, "azimuth_rule.target_aspect")
    if type(ring_count) is not int or ring_count < 0:
        raise InputValueError("azimuth_rule: ring_count must be a nonnegative integer")
    reference = rule.get("reference", "max_perimeter")
    if reference == "max_perimeter":
        index = int(np.argmax(rows[:, 1]))
    else:
        if (
            isinstance(reference, bool)
            or not isinstance(reference, (int, float))
            or not math.isfinite(reference)
        ):
            raise InputValueError(
                "azimuth_rule.reference: measured station or max_perimeter required"
            )
        found = np.flatnonzero(np.abs(rows[:, 0] - reference) <= 1e-12)
        if len(found) != 1:
            raise InputValueError("azimuth_rule.reference: exactly one measured station required")
        index = int(found[0])
    return max(math.ceil(rows[index, 1] / (aspect * rows[index, 2])), ring_count), float(
        rows[index, 0]
    )


def hinge_nodes(table: Mapping[str, Any]) -> np.ndarray:
    """Pin chord fractions at exact nodes without adding cells or deleting strips."""
    nodes = spacing_nodes(table, direction="chordwise")
    hinges = table.get("hinge_rows", [])
    if not hinges:
        return nodes
    if len(set(hinges)) >= len(nodes) - 1:
        raise InputValueError("hinge rows: more cells are needed to keep every interval nonempty")
    return pin_fractions(nodes, hinges)

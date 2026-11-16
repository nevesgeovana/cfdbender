"""Spacing laws and the rules a patch mesh is measured against.

Pipeline role: pure numerical stage of the geometry package, upstream of the
mesher in the workspace. It holds the normalized node laws of an edge (constant,
geometric growth to a plateau, constant with a refined tip), the seed laws that
turn a patch's role into those laws, the solver of opposite-edge cell counts, and
the rules measured on the realized edge nodes of a mesh: spanwise width constant
except at the tip, chordwise width growing from the leading edge then constant,
constant azimuth and axial step on a body band, bounded step on a pole patch.
It imports no geometry kernel.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError

from ._distributions import planform_values
from ._lifting import section_models, thickness_station

__all__ = [
    "DEFAULT_CONTROLS",
    "chord_fraction",
    "chord_nodes",
    "constant",
    "count_classes",
    "geometric_cap",
    "insert_seed",
    "load_mesh_controls",
    "check_body",
    "check_lifting",
    "tip_refinement",
]

TOLERANCE = 0.01  # relative deviation allowed on a successive-width ratio
AXIAL_TOLERANCE = 0.10  # relative deviation of a body x step from x_length/x_cells
POLE_BAND = (0.5, 3.0)  # nose and rear pole-patch step relative to the x step
POLE_SPREAD = 0.35  # largest relative spread of cell width along a pole-patch edge

#: Mesh controls of the study this mesher was written for; a caller overrides any entry.
DEFAULT_CONTROLS: dict[str, dict[str, float]] = {
    "lifting_surface": {
        "chord_cells": 40,
        "chord_growth": 1.15,
        "growth_until": 0.5,
        "span_cells": 32,
        "tip_cells": 6,
        "tip_ratio": 1.3,
        "thickness_cells": 1,
    },
    "body": {"x_cells": 80, "azimuth_cells": 24},
    "checks": {"distance_tolerance_m": 0.0005, "symmetry_tolerance_m": 0.0000001},
    "assembly": {"candidate_edge_cells": 6},
}


def load_mesh_controls(source: Any = None) -> dict[str, dict[str, float]]:
    """Return the mesh controls: the defaults, overridden by a mapping or a TOML file.

    Parameters
    ----------
    source : str, Path or mapping, optional
        A TOML file path or a mapping of tables to keys; None keeps the defaults.

    Returns
    -------
    dict
        Table name to its keys and values, a copy of the defaults with the overrides applied.

    Raises
    ------
    InputValueError
        If a table or key is unknown, a table is not a mapping, or a value is not a finite
        positive number (a positive integer for a cell count).
    """
    given: dict[str, Any] = {}
    if isinstance(source, (str, Path)):
        given = tomllib.loads(Path(source).read_text(encoding="utf-8"))
    elif source is not None:
        given = dict(source)
    unknown = sorted(set(given) - set(DEFAULT_CONTROLS))
    if unknown:
        raise InputValueError(
            f"controls: unknown table {unknown}; available {sorted(DEFAULT_CONTROLS)}"
        )
    result = {table: dict(values) for table, values in DEFAULT_CONTROLS.items()}
    for table, values in given.items():
        if not isinstance(values, Mapping):
            raise InputValueError(f"controls.{table}: must be a table of keys and numbers")
        wrong = sorted(set(values) - set(DEFAULT_CONTROLS[table]))
        if wrong:
            raise InputValueError(
                f"controls.{table}: unknown key {wrong}; "
                f"available {sorted(DEFAULT_CONTROLS[table])}"
            )
        for key, value in values.items():
            _control_value(table, key, value)
        result[table].update(values)
    return result


def _control_value(table, key, value):
    """Refuse a control that is not a finite positive number (a positive integer for a count)."""
    if key.endswith("_cells"):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise InputValueError(f"controls.{table}.{key}: must be a positive integer")
    elif (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise InputValueError(f"controls.{table}.{key}: must be a finite positive number")


def _count(cells) -> None:
    if isinstance(cells, bool) or not isinstance(cells, (int, np.integer)) or cells < 1:
        raise InputValueError("cells: must be a positive integer")


def _nodes(widths):
    result = np.r_[0.0, np.cumsum(widths / np.sum(widths))]
    result[-1] = 1.0
    return result


def constant(cells):
    """Return ``cells + 1`` equally spaced normalized nodes.

    Parameters
    ----------
    cells : int
        Number of cells; a positive integer.

    Returns
    -------
    numpy.ndarray
        The nodes from 0 to 1, shape ``(cells + 1,)``.
    """
    _count(cells)
    return np.linspace(0.0, 1.0, cells + 1)


def geometric_cap(cells, ratio=1.15, cap_index=None, growth_until=0.5):
    """Return nodes with geometric cell growth followed by a plateau.

    ``cap_index`` is the first plateau cell (zero based). When omitted, the index whose plateau
    starts nearest ``growth_until`` of the length is chosen. The count is exact; the split is
    quantized by cells.
    """
    _count(cells)
    if not np.isfinite(ratio) or ratio < 1 or not 0 < growth_until < 1:
        raise InputValueError("ratio: must be finite and >= 1; growth_until must be in (0, 1)")

    def widths(k):
        # Subtract the maximum exponent to avoid overflow at large counts.
        return np.exp((np.minimum(np.arange(cells), k) - k) * np.log(ratio))

    if cap_index is None:
        cap_index = min(range(cells), key=lambda k: abs(_nodes(widths(k))[k] - growth_until))
    if (
        isinstance(cap_index, bool)
        or not isinstance(cap_index, (int, np.integer))
        or not 0 <= cap_index < cells
    ):
        raise InputValueError("cap_index: must be an integer in [0, cells)")
    result = _nodes(widths(cap_index))
    if np.any(np.diff(result) <= 0):
        raise InputValueError("spacing: exceeds floating point resolution")
    return result


def tip_refinement(cells, refined_cells=4, ratio=1.3):
    """Return nodes of constant interior cells, then successive division by ``ratio`` at the tip.

    Parameters
    ----------
    cells : int
        Total number of cells; a positive integer.
    refined_cells : int, optional
        Number of cells refined at the tip, in ``[0, cells)``; default 4.
    ratio : float, optional
        Successive width division at the tip, finite and at least 1; default 1.3.

    Returns
    -------
    numpy.ndarray
        The normalized nodes from 0 to 1, shape ``(cells + 1,)``.

    Raises
    ------
    InputValueError
        If ``refined_cells`` or ``ratio`` is out of range, or the spacing exceeds floating point
        resolution.
    """
    _count(cells)
    if (
        isinstance(refined_cells, bool)
        or not isinstance(refined_cells, (int, np.integer))
        or not 0 <= refined_cells < cells
    ):
        raise InputValueError("refined_cells: must be an integer in [0, cells)")
    if not np.isfinite(ratio) or ratio < 1:
        raise InputValueError("tip ratio: must be finite and >= 1")
    exponents = np.maximum(0, np.arange(cells) - (cells - refined_cells - 1))
    result = _nodes(np.exp(-exponents * np.log(ratio)))
    if np.any(np.diff(result) <= 0):
        raise InputValueError("spacing: exceeds floating point resolution")
    return result


def _free_counts(members, fixed, target):
    """Return the rounded mean-length cell count of every class not already fixed."""
    return {
        r: max(1, int(round(float(np.mean(values)) / target)))
        for r, values in members.items()
        if r not in fixed
    }


def count_classes(loops, seeds, default_cells=4, lengths=None, target=None):
    """Solve the opposite-edge cell-count equalities; refuse contradictory fixed counts.

    ``loops`` maps a surface to its four signed edges, ``seeds`` maps an edge to its fixed
    normalized nodes. With ``lengths`` and ``target`` the free classes take the rounded mean
    length over ``target``; otherwise they take ``default_cells``.

    Parameters
    ----------
    loops : dict
        Surface to its four signed edge tags.
    seeds : dict
        Edge tag to its fixed normalized nodes.
    default_cells : int, optional
        Cell count of a free class when no lengths are given; default 4.
    lengths : dict, optional
        Edge tag to its length, used with ``target``.
    target : float, optional
        Target cell length for the free classes.

    Returns
    -------
    dict
        Edge tag to its cell count.

    Raises
    ------
    InputValueError
        If two seeded edges of one opposite-edge class fix different cell counts.
    """
    parent = {abs(e): abs(e) for loop in loops.values() for e in loop}

    def root(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for loop in loops.values():
        for i in (0, 1):
            a, b = root(abs(loop[i])), root(abs(loop[i + 2]))
            parent[b] = a
    fixed = {}
    for edge, nodes in seeds.items():
        r, count = root(edge), len(nodes) - 1
        if r in fixed and fixed[r] != count:
            raise InputValueError(
                f"edge {edge}: opposite-edge constraint conflict, {fixed[r]} versus {count} cells"
            )
        fixed[r] = count
    if lengths is not None:
        members = {}
        for edge in parent:
            members.setdefault(root(edge), []).append(lengths[edge])
        fixed.update(_free_counts(members, fixed, target))
    return {edge: fixed.get(root(edge), default_cells) for edge in parent}


def insert_seed(seeds, signed_edge, nodes):
    """Record the nodes of an edge walked in its own direction; refuse a different earlier seed.

    Parameters
    ----------
    seeds : dict
        Edge tag to its recorded nodes; updated in place.
    signed_edge : int
        Signed edge tag; a negative sign means the nodes were walked against the edge.
    nodes : array_like
        Normalized nodes as walked.

    Raises
    ------
    InputValueError
        If the edge already has a seed that differs.
    """
    x = np.asarray(nodes) if signed_edge > 0 else 1 - np.asarray(nodes)[::-1]
    tag = abs(signed_edge)
    if tag in seeds and (
        len(x) != len(seeds[tag]) or not np.allclose(x, seeds[tag], atol=2e-7, rtol=0)
    ):
        raise InputValueError(f"edge {tag}: shared edge has inconsistent source parameterizations")
    seeds[tag] = x


def pin_fractions(nodes, fractions):
    """Move the nearest nodes of a normalized law onto exact fractions, keeping the cell count.

    Each fraction takes its own node in order; the law between pinned nodes is mapped linearly,
    so no strip is added or removed (hinge rows of a control surface's parent).
    """
    nodes = np.asarray(nodes, dtype=float)
    pins = sorted({float(f) for f in fractions})
    count = len(nodes) - 1
    if any(not 0 < f < 1 for f in pins) or len(pins) >= count:
        raise InputValueError("hinge rows: fractions inside (0, 1) and more cells than rows needed")
    indices = [0]
    for i, value in enumerate(pins):
        nearest = int(np.argmin(np.abs(nodes - value)))
        indices.append(max(indices[-1] + 1, min(count - len(pins) + i, nearest)))
    indices.append(count)
    result = np.interp(nodes, nodes[indices], [0.0, *pins, 1.0])
    if np.any(np.diff(result) <= 0):
        raise InputValueError("hinge rows: spacing below floating point resolution")
    return result


def chord_nodes(component, controls, family, eta):
    """Return the chordwise nodes of a fore or aft skin patch at span fraction ``eta``, or None.

    Parameters
    ----------
    component : dict
        Resolved lifting-surface component.
    controls : dict
        Mesh controls for the lifting surface (``chord_cells``, ``chord_growth``, ``growth_until``).
    family : str
        Patch family, such as ``upper_fore``; a trailing-edge family has no chordwise nodes.
    eta : float
        Span fraction at which the thickness station is taken.

    Returns
    -------
    numpy.ndarray or None
        Normalized nodes from 0 to 1, or None for a trailing-edge family.
    """
    if family.startswith("trailing"):
        return None
    mid = thickness_station(component, section_models(component["sections"]), eta)
    global_x = geometric_cap(
        controls["chord_cells"], controls["chord_growth"], growth_until=controls["growth_until"]
    )
    if controls.get("hinge_rows"):
        global_x = pin_fractions(global_x, controls["hinge_rows"])
    split = int(np.argmin(abs(global_x - mid)))
    if family.endswith("fore"):
        x = global_x[: split + 1] / global_x[split]
        return np.sqrt(x)
    x = (global_x[split:] - global_x[split]) / (1 - global_x[split])
    return (np.sqrt(mid + (1 - mid) * x) - np.sqrt(mid)) / (1 - np.sqrt(mid))


def _ratios(widths):
    return widths[1:] / widths[:-1]


def chord_fraction(component, points):
    """Return the chord fraction of model-space points on a lifting surface.

    Parameters
    ----------
    component : dict
        Resolved lifting-surface component.
    points : numpy.ndarray
        Model-space points, shape ``(n, 3)``.

    Returns
    -------
    numpy.ndarray
        The chord fraction of each point, shape ``(n,)``.
    """
    half = component["planform"]["span"] / 2
    eta = np.clip(points[:, 1] / half, 0, 1)
    values = planform_values(component, eta)
    theta = np.deg2rad(values["twist"])
    chord = values["chord"]
    axis = component["planform"]["twist_axis"]
    dx = (points[:, 0] - values["leading_edge_x"] - axis * chord) * np.cos(theta) - (
        points[:, 2] - values["leading_edge_z"]
    ) * np.sin(theta)
    return dx / chord + axis


def spanwise(points, controls, half):
    y = points[np.argsort(points[:, 1]), 1]
    widths = np.diff(y)
    if len(widths) < 2:
        return True, {"cells": len(widths)}
    ratio = _ratios(widths)
    shrink = 1 / controls["tip_ratio"]
    flat = np.abs(ratio - 1) <= TOLERANCE
    step = np.abs(ratio - shrink) <= TOLERANCE * shrink
    ok = bool(np.all(flat | step))
    first = int(np.argmax(step)) if step.any() else len(ratio)
    ok = ok and bool(np.all(step[first:]))
    tip = bool(abs(y[-1] - half) < 1e-6)
    expected = min(controls["tip_cells"], len(ratio))
    if tip and step.sum() != expected:
        ok = False
    return ok, {
        "cells": len(widths),
        "shrinking_steps": int(step.sum()),
        "reaches_tip": tip,
        "max_ratio_error": float(
            np.max(np.minimum(np.abs(ratio - 1), np.abs(ratio - shrink) / shrink))
        ),
    }


def chordwise(component, points, controls):
    x = np.sort(chord_fraction(component, points))
    widths = np.diff(x)
    if len(widths) < 2:
        return True, {"cells": len(widths)}
    ratio = _ratios(widths)
    growth = controls["chord_growth"]
    flat = np.abs(ratio - 1) <= TOLERANCE
    step = np.abs(ratio - growth) <= TOLERANCE * growth
    ok = bool(np.all(flat | step))
    if flat.any():
        ok = ok and not bool(np.any(step[int(np.argmax(flat)) :]))
    return ok, {
        "cells": len(widths),
        "growth_steps": int(step.sum()),
        "ratios": [] if ok else [round(float(v), 4) for v in ratio],
        "max_ratio_error": float(
            np.max(np.minimum(np.abs(ratio - 1), np.abs(ratio - growth) / growth))
        ),
    }


def _constant_spread(widths):
    mean = float(np.mean(widths))
    return float(np.max(np.abs(widths - mean)) / mean), mean


def body_edge(points, kind, x_step, azimuth_step_deg):
    if kind == "azimuth":
        angle = np.degrees(np.unwrap(np.arctan2(points[:, 2], points[:, 1])))
        spread, mean = _constant_spread(np.abs(np.diff(angle)))
        ok = spread <= TOLERANCE and abs(mean - azimuth_step_deg) <= TOLERANCE * azimuth_step_deg
        return ok, {"cells": len(angle) - 1, "angle_step_deg": mean, "spread": spread}
    widths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    spread, mean = _constant_spread(widths)
    if kind == "axial":
        ok = spread <= TOLERANCE and abs(mean / x_step - 1) <= AXIAL_TOLERANCE
    else:
        ok = spread <= POLE_SPREAD and POLE_BAND[0] <= mean / x_step <= POLE_BAND[1]
    return bool(ok), {
        "cells": len(widths),
        "step_m": mean,
        "step_over_x_step": mean / x_step,
        "spread": spread,
    }


def check_lifting(component, controls, items, polylines):
    """Measure the lifting-surface spacing rules.

    ``items`` lists ``{"family", "u", "v", "surface"}`` per structured patch (``u`` and ``v`` the
    edge tags); ``polylines`` maps an edge tag to its ``(n, 3)`` realized nodes. Trailing-edge
    strips are skipped. Returns ``(rows, failures)``.

    Parameters
    ----------
    component : dict
        Resolved lifting-surface component; its span sets the tip position.
    controls : dict
        Mesh controls whose chord growth, tip cells and tip ratio the edges are measured against.
    items : sequence of dict
        One entry per structured patch, as described above.
    polylines : dict
        Edge tag to its ``(n, 3)`` realized nodes.

    Returns
    -------
    tuple
        ``(rows, failures)``: one row per measured edge, and the rows that failed.
    """
    half = component["planform"]["span"] / 2
    result, failures = [], []
    for item in items:
        if item["family"].startswith("trailing"):
            continue
        for role, edges in (("span", item["v"]), ("chord", item["u"])):
            for edge in edges:
                points = polylines[abs(edge)]
                if role == "span":
                    ok, detail = spanwise(points, controls, half)
                else:
                    ok, detail = chordwise(component, points, controls)
                row = {
                    "surface": item["surface"],
                    "family": item["family"],
                    "role": role,
                    "edge": abs(edge),
                    "ok": bool(ok),
                    **detail,
                }
                result.append(row)
                if not ok:
                    failures.append(row)
    return result, failures


def check_body(component, controls, items, polylines):
    """Measure the body spacing rules; ``items`` also carry ``axial`` (set on a band).

    Parameters
    ----------
    component : dict
        Resolved body component; its length sets the axial step.
    controls : dict
        Mesh controls whose ``x_cells`` and ``azimuth_cells`` set the steps.
    items : sequence of dict
        One entry per structured patch with its ``family``, ``u`` and ``v`` edge tags,
        ``surface`` and ``axial`` flag.
    polylines : dict
        Edge tag to its ``(n, 3)`` realized nodes.

    Returns
    -------
    tuple
        ``(rows, failures)``: one row per measured edge, and the rows that failed.
    """
    x_step = component["length"] / controls["x_cells"]
    az_step = 180.0 / controls["azimuth_cells"]
    result, failures = [], []
    for item in items:
        banded = item["axial"]
        for edge in item["u"] + item["v"]:
            kind = ("azimuth" if edge in item["u"] else "axial") if banded else "pole"
            ok, detail = body_edge(polylines[abs(edge)], kind, x_step, az_step)
            row = {
                "surface": item["surface"],
                "family": item["family"],
                "role": kind,
                "edge": abs(edge),
                "ok": bool(ok),
                **detail,
            }
            result.append(row)
            if not ok:
                failures.append(row)
    return result, failures

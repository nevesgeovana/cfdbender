"""Planform-relative control spans and explicit bounding-section splits.

Pipeline role: translate control spans before the pinned lifting model is
built. Host-derived limits must be supplied by the assembly boundary.
"""

from __future__ import annotations

import copy
import math

from cfdbender._errors import InputValueError

__all__ = ["prepare_spans", "span_limits"]


def span_limits(entry, fraction, previous):
    """Validate one interval and express it in the skin's normalized span.

    Parameters
    ----------
    entry : dict
        Resolved plain control with planform-relative eta and gap_span.
    fraction : float
        Fraction of the planform semispan occupied by the skin before its cap.
    previous : float
        End of the preceding control in normalized skin coordinates.

    Returns
    -------
    tuple
        Inboard limit, outboard limit and end gap.

    Raises
    ------
    InputValueError
        If the kind is unsupported, intervals overlap or the gap consumes the span.
    """
    if entry["kind"] not in ("plain_flap", "aileron", "fowler", "elevator", "rudder"):
        raise InputValueError("control.kind: unsupported applied control")
    a, b = (float(entry[k]) / fraction for k in ("eta_in", "eta_out"))
    gap = float(entry["gap_span"]) / fraction
    if not math.isfinite(gap) or gap < 0 or 2 * gap >= b - a:
        raise InputValueError("control.gap_span: must leave a positive moving span")
    if a < previous:
        raise InputValueError("control.eta: overlapping controls")
    return a, b, gap


def _exposed(entry, semispan):
    if "host" in (entry["eta_in"], entry["eta_out"]):
        raise InputValueError("control.eta: host limits require the measured host boundary")
    if entry.get("span_basis", "semi_span") == "semi_span":
        return entry
    root = entry.get("exposed_root_m", "host")
    if not isinstance(root, (int, float)) or not math.isfinite(root) or not 0 <= root < semispan:
        raise InputValueError("control.exposed_root_m: supply a measured root inside the semispan")
    fraction = root / semispan
    return {
        **entry,
        "eta_in": fraction + entry["eta_in"] * (1 - fraction),
        "eta_out": fraction + entry["eta_out"] * (1 - fraction),
        "gap_span": entry["gap_span"] * (1 - fraction),
        "span_basis": "semi_span",
    }


def _split(entry, bounds, warnings):
    intervals = [(entry["eta_in"], entry["eta_out"])]
    for name in entry.get("split_around", []):
        if name not in bounds:
            warnings.append(
                f"control {entry['name']}: split_around {name!r} has no bounding sections; "
                "continuous piece retained"
            )
            continue
        low, high = bounds[name]
        kept = []
        for a, b in intervals:
            if high <= a or low >= b:
                kept.append((a, b))
            else:
                kept.extend((start, end) for start, end in ((a, low), (high, b)) if start < end)
        intervals = kept
    if not intervals:
        raise InputValueError(f"control {entry['name']}: split_around removes the whole span")
    result = []
    for index, (a, b) in enumerate(intervals):
        suffix = ("inboard", "outboard")[index] if len(intervals) == 2 else f"part{index + 1}"
        label = entry["name"] if len(intervals) == 1 else f"{entry['name']}-{suffix}"
        result.append({**entry, "name": label, "eta_in": a, "eta_out": b, "split_around": []})
    return result


def prepare_spans(component):
    """Return a component with planform-relative spans and its split warnings.

    Parameters
    ----------
    component : dict
        Resolved lifting surface. Named engine bounding sections supply the
        excluded intervals; they are explicit inputs, not a geometric guess.

    Returns
    -------
    tuple
        Copied component and warning strings for names without bounds.

    Raises
    ------
    InputValueError
        If a host-derived limit is missing, the exposed root is invalid or
        the excluded intervals consume a control completely.
    """
    result = copy.deepcopy(component)
    bounds = {
        row["component"]: (row["eta_in"], row["eta_out"])
        for row in component["sections"].get("nacelle_bounds", [])
    }
    warnings, entries = [], []
    for entry in component.get("control_surfaces", []):
        entry = _exposed(entry, component["planform"]["span"] / 2)
        entries.extend(_split(entry, bounds, warnings))
    result["control_surfaces"] = entries
    return result, warnings

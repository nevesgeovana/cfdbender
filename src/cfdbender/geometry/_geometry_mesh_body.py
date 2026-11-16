"""Strategy 1 body rows: the azimuth rule on the generated route and its aspect band.

Pipeline role: workspace mesh planning and acceptance for bodies (MESH_STRATEGY 2.5.1). The
half perimeters are measured on the generated body bands; the realized band quads are measured
on the mesh, independently of the requested controls.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cfdbender.geometry.stages import Mesh, azimuth_count, body, check_entry, evaluate_patch

__all__ = ["BODY_BANDS", "azimuth_from_rule", "body_aspect_band", "body_half_perimeters"]


def _check(name: str, value: float, limit: float = 0.0, *, where=None) -> dict:
    return check_entry(name, "passed" if value <= limit else "failed", value, limit, where=where)


def body_half_perimeters(spec: Mapping) -> list[tuple[float, float]]:
    """Return (x, half perimeter) at the start of every axial band of the generated body, m."""
    stations: dict[float, float] = {}
    for patch in body(spec):
        if patch.bay is None:
            continue
        # Native u is azimuth and v is axial in the body bands.
        ring = evaluate_patch(patch, 41)[:, 0, :]
        x = round(float(ring[0, 0]), 12)
        stations[x] = stations.get(x, 0.0) + float(
            np.sum(np.linalg.norm(np.diff(ring, axis=0), axis=1))
        )
    return sorted(stations.items())


def azimuth_from_rule(spec: Mapping, rule: Mapping, x_cells: float) -> dict:
    """Derive the half-body azimuth count of the strategy 1 route (MESH_STRATEGY 2.5.1).

    The count from the measured half perimeters and the axial step is raised to a multiple of
    the generated half-sector layout, which the body bands need.
    """
    step = spec["length"] / x_cells
    rows = [(x, p, step) for x, p in body_half_perimeters(spec)]
    count, reference = azimuth_count(rows, rule)
    sectors = 2 * (len(spec["azimuths"]) - 1)
    count = sectors * -(-count // sectors)
    return {
        "azimuth_cells": int(count),
        "azimuth_rule": {
            "reference_x": reference,
            "aspect_band": list(rule.get("aspect_band", [0.67, 1.5])),
        },
    }


def body_aspect_band(mesh: Mesh, families: Mapping, controls: Mapping) -> list[dict]:
    """Measure the realized body quad aspect against ``aspect_band`` where the rule is used.

    Over the quads of the body bands (families ``front`` and ``middle``) of every component
    with ``azimuth_rule``, the aspect is the mean circumferential over the mean axial edge
    length (an axial edge is the one along x).
    """
    checks = []
    tables = controls.get("components", {})
    body_default = controls.get("defaults", {}).get("body", {})
    for name, table in sorted(tables.items()):
        rule = table.get("azimuth_rule", body_default.get("azimuth_rule"))
        if rule is None or table.get("type") not in (None, "body"):
            continue
        low, high = rule.get("aspect_band", [0.67, 1.5])
        ratios = []
        for cell, holder, surface in zip(mesh.cells, mesh.components, mesh.surfaces, strict=True):
            if holder != name or len(cell) != 4 or families.get(int(surface)) not in BODY_BANDS:
                continue
            p = mesh.xyz[list(cell)]
            edges = np.roll(p, -1, axis=0) - p
            axial = np.abs(edges[:, 0]) >= np.linalg.norm(edges[:, 1:], axis=1)
            if axial.all() or not axial.any():
                continue
            lengths = np.linalg.norm(edges, axis=1)
            ratios.append(float(lengths[~axial].mean() / lengths[axial].mean()))
        if not ratios:
            continue
        outside = sum(1 for r in ratios if not low <= r <= high)
        checks.append(
            _check(
                "mesh.body.aspect_band",
                outside,
                where={
                    "component": name,
                    "band": [low, high],
                    "min": min(ratios),
                    "max": max(ratios),
                    "cells": len(ratios),
                },
            )
        )
    return checks


#: Body families that are axial bands of constant x step (the poles are not).
BODY_BANDS = ("front", "middle")

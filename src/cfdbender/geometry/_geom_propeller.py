"""Propeller sub-step adapters for the geometry runner and its persistent log.

Pipeline role: measure pure blade construction, then inspect the actual IGES
trim curves and sewn topology. Kernel work stays behind the guarded adapter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.geometry import Propeller
from cfdbender.geometry._geom_assembly import Assembly
from cfdbender.geometry.stages import propeller_measurement as measure
from cfdbender.geometry.stages import stage_row, unpack
from cfdbender.mesh._gmsh_worker import _ENV as WORKER_ENV
from cfdbender.mesh._gmsh_worker import run_isolated

PURE_CHECKS = {
    "S1.P": ("propeller.section_fit",),
    "S4.P": (
        "propeller.solidity",
        "propeller.activity_factor",
        "propeller.beta_reference",
        "propeller.beta_0p7R",
    ),
    "S5.P": ("propeller.section_plane", "propeller.handedness"),
    "S6.P": ("propeller.loft_exact", "propeller.loft_c2"),
    "S8.P": ("propeller.closure",),
}
PROPELLER_STAGES = {*PURE_CHECKS, "S11.PU", "S11.PC", "S11.PS"}


class PropellerAssembly(Assembly):
    """Count geometrical boundaries of cap poles consistently with sewn imports."""

    def edge_counts(self, faces):
        counts = super().edge_counts(faces)
        result = {}
        for edge, count in counts.items():
            points = self._samples(edge)
            if float(np.linalg.norm(points - points[0], axis=1).max()) > 1e-12:
                result[edge] = count
        return result


def applicable_stages(planned, resolved, members):
    """Select the propeller-only surface workflow; mixed assemblies keep their full plan."""
    components = unpack(resolved)["components"]
    propellers = [
        n for n in members if components[n]["type"] == "propeller" and "definition" in components[n]
    ]
    if not propellers:
        return [stage for stage in planned if stage.row.stage not in PROPELLER_STAGES]
    if len(propellers) != len(members):
        return planned
    keep = {"S0", "S11", "S11.V", "S11.P", "S11.R", "S11.S"} | PROPELLER_STAGES
    return [stage for stage in planned if stage.row.stage in keep]


def _models(context):
    if "propellers" not in context.state:
        components = unpack(context.resolved)["components"]
        context.state["propellers"] = {
            n: Propeller(components[n]["definition"])
            for n in context.members
            if components[n]["type"] == "propeller" and "definition" in components[n]
        }
    return context.state["propellers"]


def stage_result(context, stage):
    """Measure one registered propeller sub-step, retaining every input in its record."""
    models = _models(context)
    if not models:
        from cfdbender.geometry.stages import check_entry

        return {
            "checks": [
                check_entry(cid, "skipped", note="no propeller definition in this run")
                for cid, _ in stage_row(stage).checks
            ]
        }
    inputs = {name: model.config for name, model in models.items()}
    if stage in PURE_CHECKS:
        checks = []
        for name, model in models.items():
            blade = model.blade()
            candidates = model.report()["checks"] + [
                measure("propeller.loft_exact", blade.info["loft_error_mm"], 1e-8, "mm"),
                measure("propeller.loft_c2", blade.info["loft_c2_violations"], 0, "knots"),
            ]
            checks += [
                {**c, "where": {"component": name}}
                for c in candidates
                if c["id"] in PURE_CHECKS[stage]
            ]
        return {"checks": checks, "outputs": {"inputs": inputs}}
    source = context.state["export"]["iges"]
    if stage == "S11.PU":
        checks = unit_checks(context.state["export"], models, len(models) == len(context.members))
    elif stage == "S11.PC":
        checks = trim_checks(source)
    else:
        expected = sum(
            model.report()["closure"]["expected_open_edges"]
            * (
                1
                if model.config["output"]["blades"] == "one"
                else model.config["propeller"]["blades"]
            )
            for model in models.values()
        )
        arguments = {"file": str(source), "expected": expected}
        checks = (
            sewn_checks(**arguments)
            if os.environ.get(WORKER_ENV) == "1"
            else run_isolated(f"{__name__}:sewn_checks", arguments, sewn_checks)
        )
    return {"checks": checks, "outputs": {"inputs": {"iges": source, "definitions": inputs}}}


def unit_checks(source, names, complete):
    """Measure the written units of each propeller file, including isolated face files."""
    from cfdbender.geometry._iges_names import read_iges_names

    folder = Path(source["manifest"]).parent
    manifest = json.loads(Path(source["manifest"]).read_text(encoding="utf-8"))
    files = [Path(source["iges"])] if complete else []
    for piece in manifest["pieces"]:
        if piece["component"] in names:
            files.append(folder / piece["file"])
            files.extend(folder / face["iges"] for face in piece["face_records"])
    wrong = [
        str(file)
        for file in dict.fromkeys(files)
        if float(METRES_PER_UNIT.get(read_iges_names(file).unit or "", 0)) != 0.001
    ]
    return [measure("propeller.output_units", len(wrong), 0, "files", note=", ".join(wrong))]


def sewn_checks(file, expected):
    """Count actual sewn edge adjacencies, permitting only declared uncapped rims."""
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("checking propeller closure")
    try:
        topology = kernel.inspect_file(Path(file), 1e-5)
        free = sum(len(faces) == 1 for _, faces in topology.edges)
        triple = sum(len(faces) > 2 for _, faces in topology.edges)
        return [
            measure(
                "propeller.sewn_free_edges",
                abs(free - expected),
                0,
                "unexpected edges",
                note=f"{free} free; {expected} declared by uncapped rims",
            ),
            measure("propeller.sewn_triple_edges", triple, 0, "edges"),
        ]
    finally:
        kernel.close()


def _image(point, entity, by_id):
    transform = int(entity["d1"][6])
    if not transform:
        return point
    matrix = np.asarray(
        [float(v.replace("D", "E")) for v in by_id[transform]["parameters"][1:13]]
    ).reshape(3, 4)
    return matrix[:, :3] @ point + matrix[:, 3]


def _curve_distance(point, curves):
    best = float("inf")
    for curve in curves:

        def error(t, curve=curve):
            return float(np.linalg.norm(curve.at(t) - point))

        grid = np.linspace(curve.low, curve.high, 33)
        j = min(range(len(grid)), key=lambda i: error(grid[i]))
        low, high = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
        opt = minimize_scalar(error, bounds=(low, high), method="bounded", options={"xatol": 1e-14})
        best = min(best, error(curve.low), error(curve.high), float(opt.fun))
    return best


def trim_checks(file):
    """Compare every written pcurve image against its spatial trim, with PREF 1 required."""
    from cfdbender.geometry._geom_gate import _boundary_curves, _pieces, _Surface
    from cfdbender.geometry._iges_names import read_iges_names
    from cfdbender.geometry._iges_write import read_iges

    document = read_iges(file)
    by_id = {entity["id"]: entity for entity in document["entities"]}
    boundaries = [e for e in document["entities"] if e["type"] == 142]
    invalid, worst = int(not boundaries), 0.0
    for boundary in boundaries:
        parametric, spatial = _boundary_curves(boundary)
        invalid += int(not parametric or not spatial or int(boundary["parameters"][5]) != 1)
        if not parametric or not spatial:
            continue
        uv = _pieces(by_id[parametric], by_id)
        xyz = _pieces(by_id[spatial], by_id)
        if not uv or not xyz:
            invalid += 1
            continue
        surface_entity = by_id[int(boundary["parameters"][2])]
        surface = _Surface(surface_entity["parameters"])
        for curve in uv:
            for t in np.linspace(curve.low, curve.high, 17):
                u, v = curve.at(t)[:2]
                point = _image(surface.at(float(u), float(v)), surface_entity, by_id)
                worst = max(worst, _curve_distance(point, xyz))
    units = read_iges_names(file).unit or "m"
    scale_mm = 1000 * METRES_PER_UNIT[units]
    return [
        measure("propeller.trim_preference", invalid, 0, "boundaries"),
        measure("propeller.pcurve_3d", worst * scale_mm, 1e-4, "mm"),
    ]

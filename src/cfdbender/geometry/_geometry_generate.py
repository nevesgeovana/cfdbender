"""Generating a named IGES file from a resolved parametric configuration (FR-1220).

Pipeline role: the workspace row of the geometry generation. The pure geometry
package gives the placed patches of every component and the inside predicate of
each; this module adds them to one OpenCASCADE model, fragments them against
each other, drops the faces buried in another component, audits the edges
(free edges off the symmetry plane, non-manifold edges, the transition ring where
components cross) and writes one IGES file whose faces carry one 406 form 15 NAME
per component. Coordinates are meters and the file declares them so.

gmsh is a dependency of the package; :func:`generate_iges` raises
``KernelUnavailableError``, naming the library, when it cannot load.
The pure geometry package never imports this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.geometry._geom_gate import check_profile
from cfdbender.geometry._iges_write import write_iges
from cfdbender.geometry.config import load_config, resolve_config
from cfdbender.geometry.stages import Patch, assembly_regions, generate_components
from cfdbender.mesh._gmsh import session

__all__ = ["add_occ_patch", "assemble", "generate_iges", "write_iges"]


def add_occ_patch(occ: Any, patch: Patch) -> int:
    """Add one patch to the model as an OpenCASCADE B-spline surface and return its tag.

    No initialization, export or global model change happens here.
    """
    controls = np.asarray(patch.controls)
    points = [occ.addPoint(*point) for point in controls.transpose(1, 0, 2).reshape(-1, 3)]
    return occ.addBSplineSurface(
        points,
        controls.shape[0],
        degreeU=patch.degree_u,
        degreeV=patch.degree_v,
        knotsU=list(patch.knots_u),
        knotsV=list(patch.knots_v),
        multiplicitiesU=list(patch.multiplicities_u),
        multiplicitiesV=list(patch.multiplicities_v),
    )


def fragment_surfaces(model, groups):
    entries = [(name, tag) for name, tags in groups.items() for tag in tags]
    entities = [(2, tag) for _, tag in entries]
    _, mapping = model.occ.fragment(entities[:1], entities[1:], removeObject=True, removeTool=True)
    model.occ.synchronize()
    result = {name: set() for name in groups}
    for (name, _), descendants in zip(entries, mapping, strict=True):
        result[name].update(tag for dim, tag in descendants if dim == 2)
    return result


def interior_point(model, tag):
    centre = np.asarray(model.occ.getCenterOfMass(2, tag))
    bounds = model.getParametrizationBounds(2, tag)
    candidates = []
    for u in np.linspace(bounds[0][0], bounds[1][0], 11)[1:-1]:
        for v in np.linspace(bounds[0][1], bounds[1][1], 11)[1:-1]:
            if model.isInside(2, tag, [u, v], parametric=True):
                candidates.append(np.asarray(model.getValue(2, tag, [u, v])))
    if not candidates:
        point, _ = model.getClosestPoint(2, tag, centre.tolist())
        return np.asarray(point)
    return min(candidates, key=lambda point: np.linalg.norm(point - centre))


def exposed_faces(model, groups, inside):
    result, remove = {}, set()
    for name, tags in groups.items():
        result[name] = []
        for tag in sorted(tags):
            point = interior_point(model, tag)
            if any(predicate(point) for other, predicate in inside.items() if other != name):
                remove.add(tag)
            else:
                result[name].append(tag)
    retained = {tag for tags in result.values() for tag in tags}
    remove.difference_update(retained)
    model.occ.remove([(2, tag) for tag in sorted(remove)], recursive=False)
    model.occ.synchronize()
    return result


def is_degenerate_edge(model, edge, tolerance=1e-12):
    """Whether an edge is a point: the collapsed leading-edge end of a planar cap face."""
    return model.occ.getMass(1, edge) <= tolerance


def is_planar_face(model, tag, tolerance=1e-9):
    """Whether a face lies in one plane: the one-face tip cap, whose boundary has many edges."""
    bounds = model.getParametrizationBounds(2, tag)
    grid = np.linspace(0.0, 1.0, 7)
    points = np.asarray(
        [
            model.getValue(
                2,
                tag,
                [
                    bounds[0][0] + u * (bounds[1][0] - bounds[0][0]),
                    bounds[0][1] + v * (bounds[1][1] - bounds[0][1]),
                ],
            )
            for u in grid
            for v in grid
        ]
    ).reshape(-1, 3)
    centred = points - points.mean(axis=0)
    size = float(np.ptp(points, axis=0).max())
    if size <= 0.0:
        return False
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return float(np.abs(centred @ normal).max()) <= tolerance * size


def edge_audit(model, groups, tolerance=1e-7, intersection_owners=None):
    surfaces = {tag for tags in groups.values() for tag in tags}
    free, nonmanifold, shared, nonquad = [], [], [], []
    edges = set()
    for tag in surfaces:
        boundary = [
            (d, e)
            for d, e in model.getBoundary([(2, tag)], oriented=False)
            if not is_degenerate_edge(model, e)
        ]
        # A planar face is the tip cap, one face on the whole outline; its boundary is divided
        # where the skin and the trailing-edge strips are, so the four-edge rule is not its own.
        if len(boundary) != 4 and not is_planar_face(model, tag):
            nonquad.append({"surface": tag, "edges": len(boundary)})
        edges.update(edge for _, edge in boundary)
    holders = {
        tag: sorted(name for name, tags in groups.items() if tag in tags) for tag in surfaces
    }
    intersections, ring, details = [], set(), []
    for edge in sorted(edges):
        parents, _ = model.getAdjacencies(1, edge)
        parents = set(parents).intersection(surfaces)
        components = sorted({name for tag in parents for name in holders[tag]})
        original_components = (intersection_owners or {}).get(edge, components)
        if len(original_components) > 1:
            intersections.append(
                {
                    "edge": edge,
                    "components": original_components,
                    "faces": sorted(int(tag) for tag in parents),
                    "shared": len(parents) == 2,
                }
            )
            ring.update(parents)
        if len(parents) == 2:
            shared.append(edge)
        elif len(parents) != 1:
            nonmanifold.append(edge)
        else:
            bounds = model.getParametrizationBounds(1, edge)
            xyz = np.asarray(
                model.getValue(1, edge, np.linspace(bounds[0][0], bounds[1][0], 17))
            ).reshape(-1, 3)
            if np.max(abs(xyz[:, 1])) > tolerance:
                free.append(edge)
        if edge in free or edge in nonmanifold:
            details.append(
                {
                    "edge": edge,
                    "kind": "free" if edge in free else "nonmanifold",
                    "location_m": list(model.occ.getCenterOfMass(1, edge)),
                    "length_m": model.occ.getMass(1, edge),
                    "components": components,
                    "faces": [
                        {
                            "surface": int(tag),
                            "components": holders[tag],
                            "area_m2": model.occ.getMass(2, int(tag)),
                        }
                        for tag in sorted(parents)
                    ],
                }
            )
    return {
        "free_edges_off_symmetry": free,
        "nonmanifold_edges": nonmanifold,
        "shared_edges": shared,
        "nonquadrilateral_faces": nonquad,
        "intersection_curves": intersections,
        "unshared_intersection_curves": [row for row in intersections if not row["shared"]],
        "transition_ring_faces": sorted(int(tag) for tag in ring),
        "transition_ring_area_m2": sum(model.occ.getMass(2, int(tag)) for tag in ring),
        "non_ring_nonquadrilateral_faces": [row for row in nonquad if row["surface"] not in ring],
        "defect_locations": details,
    }


def assemble(model, groups, inside):
    fragmented = fragment_surfaces(model, groups)
    intersection_owners = {}
    for name, tags in fragmented.items():
        for tag in tags:
            for _, edge in model.getBoundary([(2, tag)], oriented=False):
                intersection_owners.setdefault(edge, set()).add(name)
    intersection_owners = {
        edge: sorted(names) for edge, names in intersection_owners.items() if len(names) > 1
    }
    exposed = exposed_faces(model, fragmented, inside)
    # A tangent contact can map one descendant to more than one source. Keep
    # that face once. Later configuration members own coincident overlays.
    holder = {tag: name for name, tags in exposed.items() for tag in tags}
    overlaps = [
        {
            "surface": tag,
            "sources": [name for name, tags in exposed.items() if tag in tags],
            "holder": name,
            "area_m2": model.occ.getMass(2, tag),
        }
        for tag, name in holder.items()
        if sum(tag in tags for tags in exposed.values()) > 1
    ]
    exposed = {name: [tag for tag in tags if holder[tag] == name] for name, tags in exposed.items()}
    audit = edge_audit(model, exposed, intersection_owners=intersection_owners)
    audit["coincident_ownership"] = overlaps
    audit["component_faces"] = exposed
    return exposed, audit


def generate_iges(
    config: Any, path: str | Path, *, configuration: str | None = None
) -> dict[str, Any]:
    """Generate the exposed skin of a configuration and write it as a named IGES file.

    Parameters
    ----------
    config : mapping or path
        A configuration (resolved or not) or the path of its TOML file.
    path : str or Path
        The IGES file to write.
    configuration : str, optional
        Name of an entry of the configuration's ``configurations`` table to generate;
        every component is generated when omitted.

    Returns
    -------
    dict
        ``path``, the number of exposed ``faces`` of each component, the edge ``audit``, and
        the ``warnings``: every key the configuration skipped, then each finding of the reader
        profile gate on the written file (the run continues past them).

    Raises
    ------
    InputValueError
        For a refused configuration, an unknown ``configuration`` or a name the IGES format
        cannot carry; nothing is written.
    cfdbender._errors.InputArtifactError
        When the written file holds a solid entity; the file is removed.
    cfdbender.mesh._extras.MissingExtraError
        When the gmsh library cannot load (``KernelUnavailableError``).
    """
    resolved = load_config(config) if isinstance(config, (str, Path)) else resolve_config(config)
    every = generate_components(resolved)
    regions = assembly_regions(resolved, every)
    placed = every
    if configuration is not None:
        members = resolved["configurations"].get(configuration)
        if members is None:
            raise InputValueError(
                f"configuration {configuration!r}: unknown; available: "
                f"{sorted(resolved['configurations'])}"
            )
        placed = {name: every[name] for name in members if name in every}
    inside = {name: regions[name] for name in placed}
    with session("generating an IGES file from a parametric configuration") as gmsh:
        gmsh.model.add("generate")
        groups = {
            name: [add_occ_patch(gmsh.model.occ, patch) for patch in patches]
            for name, patches in placed.items()
        }
        gmsh.model.occ.synchronize()
        exposed, audit = assemble(gmsh.model, groups, inside)
        write_iges(gmsh, exposed, path)
    try:
        gate = check_profile(path)
    except InputArtifactError:
        Path(path).unlink(missing_ok=True)
        raise
    gate_warnings = [
        f"reader profile, {f.check}"
        + (f" (entity {f.entity})" if f.entity else "")
        + f": {f.detail}"
        for f in gate.findings
    ]
    return {
        "path": Path(path),
        "faces": {n: len(t) for n, t in exposed.items()},
        "audit": audit,
        "warnings": [*resolved["warnings"], *gate_warnings],
    }

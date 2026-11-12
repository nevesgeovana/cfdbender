"""Build a public NACA 0012 mesh for the serial ESI simpleFoam POC.

Chord is 1 m, farfield radius 20 m about mid-chord, span 0.1 m, one extruded
recombined layer. The plane mesh is a structured O-grid of two transfinite
blocks (upper and lower), so the written bytes depend only on the parameters.
The first wall cell is 5e-4 chord (y+ near 25 at Re 1e6, inside the Spalding
wall-function range) and grows geometrically to the farfield.
Run this script with an output .msh path; it prints the volume cell count.
There is no mesh-independence or accuracy claim.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Nodes along each airfoil side (bump-clustered at the leading and trailing edges),
# nodes across the O-grid, and the geometric growth that puts the first cell at 5e-4.
SURFACE_NODES = 121
RADIAL_NODES = 101
FIRST_CELL = 5e-4
FARFIELD = 20.0


def _growth(first: float, total: float, cells: int) -> float:
    """Return the ratio r with first * (r**cells - 1) / (r - 1) == total (bisection)."""
    low, high = 1.0 + 1e-9, 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if first * (mid**cells - 1) / (mid - 1) < total:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def generate(destination: Path) -> int:
    """Write repeatable gmsh 2.2 ASCII bytes and return the volume cell count.

    Parameters
    ----------
    destination : Path
        Output .msh file; its parent must exist.

    Returns
    -------
    int
        Number of one-layer volume elements.
    """
    import gmsh  # type: ignore[import-untyped]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("openfoam_naca0012")
        geo = gmsh.model.geo
        xs = [0.5 * (1 - math.cos(math.pi * i / 160)) for i in range(161)]

        def thickness(x: float) -> float:
            return 0.6 * (
                0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
            )

        leading = geo.addPoint(0, 0, 0)
        trailing = geo.addPoint(1, 0, 0)
        upper = geo.addSpline(
            [leading, *[geo.addPoint(x, thickness(x), 0) for x in xs[1:-1]], trailing]
        )
        lower = geo.addSpline(
            [trailing, *[geo.addPoint(x, -thickness(x), 0) for x in reversed(xs[1:-1])], leading]
        )
        center = geo.addPoint(0.5, 0, 0)
        right = geo.addPoint(0.5 + FARFIELD, 0, 0)
        top = geo.addPoint(0.5, FARFIELD, 0)
        left = geo.addPoint(0.5 - FARFIELD, 0, 0)
        bottom = geo.addPoint(0.5, -FARFIELD, 0)
        arc_rt = geo.addCircleArc(right, center, top)
        arc_tl = geo.addCircleArc(top, center, left)
        arc_lb = geo.addCircleArc(left, center, bottom)
        arc_br = geo.addCircleArc(bottom, center, right)
        wake = geo.addLine(trailing, right)
        front = geo.addLine(leading, left)
        upper_block = geo.addPlaneSurface([geo.addCurveLoop([upper, wake, arc_rt, arc_tl, -front])])
        lower_block = geo.addPlaneSurface([geo.addCurveLoop([lower, front, arc_lb, arc_br, -wake])])
        ratio = _growth(FIRST_CELL, FARFIELD - 0.5, RADIAL_NODES - 1)
        arc_nodes = (SURFACE_NODES + 1) // 2
        for curve in (upper, lower):
            geo.mesh.setTransfiniteCurve(curve, SURFACE_NODES, "Bump", 0.15)
        for curve in (arc_rt, arc_tl, arc_lb, arc_br):
            geo.mesh.setTransfiniteCurve(curve, arc_nodes)
        for curve in (wake, front):
            geo.mesh.setTransfiniteCurve(curve, RADIAL_NODES, "Progression", ratio)
        geo.mesh.setTransfiniteSurface(upper_block, cornerTags=[leading, trailing, right, left])
        geo.mesh.setTransfiniteSurface(lower_block, cornerTags=[trailing, leading, left, right])
        for surface in (upper_block, lower_block):
            geo.mesh.setRecombine(2, surface)
        extrusion = geo.extrude(
            [(2, upper_block), (2, lower_block)], 0, 0, 0.1, numElements=[1], recombine=True
        )
        geo.synchronize()
        volumes = [tag for dim, tag in extrusion if dim == 3]
        wall_surfaces, far_surfaces, backs = [], [], []
        for dim, tag in extrusion:
            if dim != 2:
                continue
            curves = {abs(t) for _, t in gmsh.model.getBoundary([(2, tag)], oriented=False)}
            if curves & {upper, lower}:
                wall_surfaces.append(tag)
            elif curves & {arc_rt, arc_tl, arc_lb, arc_br}:
                far_surfaces.append(tag)
        for i, (dim, tag) in enumerate(extrusion):
            if dim == 2 and i + 1 < len(extrusion) and extrusion[i + 1][0] == 3:
                backs.append(tag)
        for dim, tags, name in (
            (2, wall_surfaces, "airfoil"),
            (2, far_surfaces, "farfield"),
            (2, [upper_block, lower_block], "front"),
            (2, backs, "back"),
            (3, volumes, "fluid"),
        ):
            if not tags:
                raise ValueError(f"missing physical group {name}")
            group = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, group, name)
        gmsh.model.mesh.generate(3)
        cells = sum(len(tags) for tags in gmsh.model.mesh.getElements(3)[1])
        gmsh.write(str(destination))
        return cells
    finally:
        gmsh.finalize()


def main() -> None:
    """Generate the named mesh and print its measured volume cell count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"OpenFOAM NACA 0012: {generate(args.destination)} volume cells")


if __name__ == "__main__":
    main()

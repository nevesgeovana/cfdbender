"""Generate a neutral, coarse, first-order 2D inviscid NACA0012 mesh with gmsh."""

import math
from pathlib import Path


def generate(destination: Path) -> int:
    """Export a small SU2 mesh with airfoil and farfield physical markers."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.add("tiny_naca0012")
        geo = gmsh.model.geo
        count = 40
        xs = [0.5 * (1 - math.cos(math.pi * i / count)) for i in range(count + 1)]

        def thickness(x):
            return 0.6 * (
                0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
            )

        leading = geo.addPoint(0, 0, 0, 0.04)
        trailing = geo.addPoint(1, 0, 0, 0.04)
        upper = [leading, *[geo.addPoint(x, thickness(x), 0, 0.04) for x in xs[1:-1]], trailing]
        lower = [
            trailing,
            *[geo.addPoint(x, -thickness(x), 0, 0.04) for x in reversed(xs[1:-1])],
            leading,
        ]
        walls = [geo.addSpline(upper), geo.addSpline(lower)]
        center = geo.addPoint(0.5, 0, 0, 1.5)
        outer = [
            geo.addPoint(0.5 + 8 * math.cos(i * math.pi / 2), 8 * math.sin(i * math.pi / 2), 0, 1.5)
            for i in range(4)
        ]
        arcs = [geo.addCircleArc(outer[i], center, outer[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(arcs), geo.addCurveLoop(walls)])
        geo.synchronize()
        for dimension, entities, name in (
            (1, walls, "airfoil"),
            (1, arcs, "farfield"),
            (2, [surface], "fluid"),
        ):
            group = gmsh.model.addPhysicalGroup(dimension, entities)
            gmsh.model.setPhysicalName(dimension, group, name)
        gmsh.model.mesh.generate(2)
        cells = sum(len(tags) for tags in gmsh.model.mesh.getElements(2)[1])
        gmsh.write(str(destination))
        # Canonical bytes keep generated whitespace independent of host and checkout.
        lines = destination.read_text(encoding="ascii").splitlines()
        destination.write_bytes((chr(10).join(line.rstrip() for line in lines) + chr(10)).encode())
        return cells
    finally:
        gmsh.finalize()

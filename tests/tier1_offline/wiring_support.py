"""Small synthetic VTU files with SU2's field names, for the declared-products tests."""

from __future__ import annotations

from pathlib import Path

from cfdbender.solvers.mock._fields import _vtu


def box_surface(
    path: Path, p_inf: float, *, chord: float = 1.0, span: float = 1.0, shift_x: float = 0.0
) -> None:
    """A closed box wing (x chord, y span, z thickness 0.1), outward winding, SU2 fields.

    ``Pressure = p_inf - 1000 z``, so a section along y carries ``Fz = 100 * chord`` N/m.
    """
    xs = [shift_x + chord * i / 4 for i in range(5)]
    ys = [span * j / 4 for j in range(5)]
    zs = [-0.05, 0.0, 0.05]
    points: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def at(x: float, y: float, z: float) -> int:
        if (x, y, z) not in index:
            index[(x, y, z)] = len(points)
            points.append((x, y, z))
        return index[(x, y, z)]

    quads = []
    for j in range(4):
        y0, y1 = ys[j], ys[j + 1]
        for i in range(4):
            x0, x1 = xs[i], xs[i + 1]
            quads.append((at(x0, y0, 0.05), at(x1, y0, 0.05), at(x1, y1, 0.05), at(x0, y1, 0.05)))
            quads.append(
                (at(x0, y1, -0.05), at(x1, y1, -0.05), at(x1, y0, -0.05), at(x0, y0, -0.05))
            )
        for k in range(2):
            z0, z1 = zs[k], zs[k + 1]
            quads.append(
                (at(xs[0], y0, z0), at(xs[0], y0, z1), at(xs[0], y1, z1), at(xs[0], y1, z0))
            )
            quads.append(
                (at(xs[4], y1, z0), at(xs[4], y1, z1), at(xs[4], y0, z1), at(xs[4], y0, z0))
            )
    fields = {
        "Pressure": [(p_inf - 1000.0 * z,) for _, _, z in points],
        "Density": [(1.2,) for _ in points],
        "Momentum": [(12.0, 0.0, 0.0) for _ in points],
        "Mach": [(0.1,) for _ in points],
        "Pressure_Coefficient": [(-1000.0 * z / 60.0,) for _, _, z in points],
        "Skin_Friction_Coefficient": [(0.003, 0.004, 0.0) for _ in points],
    }
    path.write_text(_vtu(points, quads, 9, fields), encoding="ascii")


def block_volume(path: Path, p_inf: float, *, shift_x: float = 0.0) -> None:
    """Hexahedra over x -1..2 (moved by ``shift_x``), y -1..2, z -1..1.

    ``Momentum / Density = (10 + x, y, z)``.
    """
    xs, ys, zs = (
        [shift_x + x for x in (-1.0, 0.0, 1.0, 2.0)],
        [-1.0, 0.0, 1.0, 2.0],
        [-1.0, 0.0, 1.0],
    )
    points = [(x, y, z) for z in zs for y in ys for x in xs]

    def node(i: int, j: int, k: int) -> int:
        return (k * len(ys) + j) * len(xs) + i

    cells = [
        (
            node(i, j, k),
            node(i + 1, j, k),
            node(i + 1, j + 1, k),
            node(i, j + 1, k),
            node(i, j, k + 1),
            node(i + 1, j, k + 1),
            node(i + 1, j + 1, k + 1),
            node(i, j + 1, k + 1),
        )
        for k in range(2)
        for j in range(3)
        for i in range(3)
    ]
    fields = {
        "Density": [(2.0,) for _ in points],
        "Momentum": [(2.0 * (10.0 + x), 2.0 * y, 2.0 * z) for x, y, z in points],
        "Pressure": [(p_inf + 100.0 * x,) for x, _, _ in points],
        "Mach": [(0.1 + 0.01 * x,) for x, _, _ in points],
        "Pressure_Coefficient": [(0.5 * x,) for x, _, _ in points],
    }
    path.write_text(_vtu(points, cells, 12, fields), encoding="ascii")

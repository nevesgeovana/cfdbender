"""Small surfaces with known pressure, built for the sectioning tests."""

from __future__ import annotations

import numpy as np

from cfdbender.post.sections import Surface


def plate(
    chord: float,
    span: float,
    nx: int,
    ny: int,
    *,
    p0: float = 0.0,
    slope: float = 0.0,
    quads: bool = False,
    y0: float = 0.0,
    family: int = 0,
    fields: dict[str, np.ndarray] | None = None,
) -> Surface:
    """A plate in z = 0, x in [0, chord], y in [y0, y0 + span], normal +z, p = p0 + slope x."""
    xs = np.linspace(0.0, chord, nx + 1)
    ys = np.linspace(y0, y0 + span, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    node = lambda i, j: i * (ny + 1) + j  # noqa: E731
    cells = []
    for i in range(nx):
        for j in range(ny):
            cells.append([node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)])
    arr = np.array(cells, dtype=np.int64)
    pressure = p0 + slope * points[:, 0]
    if quads:
        q, t = arr, np.zeros((0, 3), dtype=np.int64)
    else:
        q = np.zeros((0, 4), dtype=np.int64)
        t = np.concatenate([arr[:, [0, 1, 2]], arr[:, [0, 2, 3]]])
    return Surface(
        points=points,
        triangles=t,
        quads=q,
        pressure=pressure,
        fields=fields or {},
        triangle_family=np.full(len(t), family, dtype=np.int64),
        quad_family=np.full(len(q), family, dtype=np.int64),
        families=("A", "B"),
    )


def cylinder(radius: float, length: float, sides: int, rings: int, p0: float, p1: float) -> Surface:
    """A cylinder about the y axis, outward winding, p = p0 + p1 cos(theta) at the nodes."""
    theta = 2.0 * np.pi * np.arange(sides) / sides
    ys = np.linspace(0.0, length, rings + 1)
    pts, press = [], []
    for y in ys:
        for th in theta:
            pts.append([radius * np.cos(th), y, radius * np.sin(th)])
            press.append(p0 + p1 * np.cos(th))
    cells = []
    for r in range(rings):
        for k in range(sides):
            k2 = (k + 1) % sides
            a, b = r * sides + k, r * sides + k2
            c, d = (r + 1) * sides + k2, (r + 1) * sides + k
            cells.append([a, d, c, b])
    return Surface(
        points=np.array(pts),
        triangles=np.zeros((0, 3), dtype=np.int64),
        quads=np.array(cells, dtype=np.int64),
        pressure=np.array(press),
    )


def wing(
    root_chord: float,
    tip_chord: float,
    span: float,
    sweep_tan: float,
    p0: float,
    slope: float,
) -> Surface:
    """A trapezoid planform in z = 0 as two triangles, normal +z, p = p0 + slope x."""
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [root_chord, 0.0, 0.0],
            [span * sweep_tan + tip_chord, span, 0.0],
            [span * sweep_tan, span, 0.0],
        ]
    )
    return Surface(
        points=pts,
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        quads=np.zeros((0, 4), dtype=np.int64),
        pressure=p0 + slope * pts[:, 0],
    )

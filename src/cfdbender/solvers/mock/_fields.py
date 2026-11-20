"""The mock's surface and volume files, for the products a p file declares.

When the p file declares ``[sections]``, the mock writes ``surface.vtu``: a closed box wing
of chord 1 m along x, span 1 m along y and thickness 0.1 m along z, one family ``wing``,
whose static pressure falls linearly with height, ``p = P_INF - 1000 z`` (Pa), so that every
section along y carries ``Fz = 2 * 1000 * 0.05 * 1 = 100`` N/m and ``Fx = 0``. When it
declares ``[volume_section]`` or ``[[probes]]`` the mock writes ``volume.vtu``: a block of
hexahedra over x from -1 to 2, y from -1 to 2 and z from -1 to 1 whose fields are linear,
``Velocity = (10 + x, y, z)`` m/s, ``Pressure = P_INF + 100 x`` Pa, ``Mach = 0.1 + 0.01 x``,
so that an interpolated value is exact. The numbers are made up; they test the post.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cfdbender.core.contract import SURFACE, VOLUME, FreeStream, ProductSupport

P_INF = 101325.0
#: The mock's free stream: density and speed of the Cp reference.
DENSITY, SPEED = 1.2, 10.0
SURFACE_FILE = "surface.vtu"
VOLUME_FILE = "volume.vtu"
_NO_LAYER = "the mock writes no boundary layer"


def _velocity(fields: Mapping[str, object], free: FreeStream) -> object:
    return fields["Velocity"]


def _component(index: int) -> Callable[[Mapping[str, object], FreeStream], object]:
    return lambda fields, free: fields["Velocity"][:, index]  # type: ignore[index]


def _cp_free(fields: Mapping[str, object], free: FreeStream) -> object:
    return (fields["Pressure"] - free.p_inf) / free.q_ref  # type: ignore[operator]


PRODUCTS: Mapping[str, ProductSupport] = {
    "sections": ProductSupport(
        SURFACE,
        {
            "pressure": "Pressure",
            "Mach": "Mach",
            "vx": _component(0),
            "vy": _component(1),
            "vz": _component(2),
        },
        {name: _NO_LAYER for name in ("CF", "Theta", "Delta*", "Delta", "H")},
    ),
    "volume_section": ProductSupport(VOLUME, {"Velocity": _velocity}),
    "probes": ProductSupport(
        VOLUME,
        {
            "CP_FREE": _cp_free,
            "MACH": "Mach",
            "VX": _component(0),
            "VY": _component(1),
            "VZ": _component(2),
        },
        {"CP_REF": "the mock states no reference speed", "BL_SKIN_FRICTION": _NO_LAYER},
    ),
}


def _vtu(
    points: Sequence[Sequence[float]],
    cells: Sequence[Sequence[int]],
    kind: int,
    fields: Mapping[str, Sequence[Sequence[float]]],
) -> str:
    offsets, total = [], 0
    for cell in cells:
        total += len(cell)
        offsets.append(total)
    lines = [
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
        "<UnstructuredGrid>",
        f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">',
        '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">',
        " ".join(repr(float(c)) for p in points for c in p),
        "</DataArray></Points>",
        "<Cells>",
        '<DataArray type="Int64" Name="connectivity" format="ascii">',
        " ".join(str(i) for cell in cells for i in cell),
        "</DataArray>",
        '<DataArray type="Int64" Name="offsets" format="ascii">',
        " ".join(str(o) for o in offsets),
        "</DataArray>",
        '<DataArray type="UInt8" Name="types" format="ascii">',
        " ".join(str(kind) for _ in cells),
        "</DataArray>",
        "</Cells>",
        "<PointData>",
    ]
    for name, values in fields.items():
        width = len(values[0])
        lines.append(
            f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{width}" format="ascii">'
        )
        lines.append(" ".join(repr(float(v)) for row in values for v in row))
        lines.append("</DataArray>")
    lines += ["</PointData>", "</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    return chr(10).join(lines) + chr(10)


def _box_surface() -> str:
    xs = [i / 4 for i in range(5)]
    ys = [j / 4 for j in range(5)]
    zs = [-0.05, 0.0, 0.05]
    points: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def at(x: float, y: float, z: float) -> int:
        key = (x, y, z)
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    quads: list[tuple[int, int, int, int]] = []
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
            quads.append((at(0.0, y0, z0), at(0.0, y0, z1), at(0.0, y1, z1), at(0.0, y1, z0)))
            quads.append((at(1.0, y1, z0), at(1.0, y1, z1), at(1.0, y0, z1), at(1.0, y0, z0)))
    fields: dict[str, Sequence[Sequence[float]]] = {
        "Pressure": [(P_INF - 1000.0 * z,) for _, _, z in points],
        "Mach": [(0.1,) for _ in points],
        "Velocity": [(SPEED, 0.0, 0.0) for _ in points],
    }
    return _vtu(points, quads, 9, fields)


def _block_volume() -> str:
    xs, ys, zs = [-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]
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
        for k in range(len(zs) - 1)
        for j in range(len(ys) - 1)
        for i in range(len(xs) - 1)
    ]
    fields: dict[str, Sequence[Sequence[float]]] = {
        "Velocity": [(SPEED + x, y, z) for x, y, z in points],
        "Pressure": [(P_INF + 100.0 * x,) for x, _, _ in points],
        "Mach": [(0.1 + 0.01 * x,) for x, _, _ in points],
    }
    return _vtu(points, cells, 12, fields)


def write_fields(folder: Path, post: Mapping[str, object]) -> None:
    """Write the files the p file's declared products are read from, and only those."""
    if "sections" in post:
        (folder / SURFACE_FILE).write_text(_box_surface(), encoding="ascii")
    if "volume_section" in post or "volume_sections" in post or "probes" in post:
        (folder / VOLUME_FILE).write_text(_block_volume(), encoding="ascii")

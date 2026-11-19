"""Small structured volume grids of every searched cell type, with known fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cfdbender.post.vtu import UnstructuredGrid

HEX, TET, WEDGE, PYR = 12, 10, 13, 14


def node_array(
    xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Nodes of a structured lattice, k fastest, then j, then i."""
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _index(shape: tuple[int, int, int]) -> NDArray[np.int64]:
    return np.arange(int(np.prod([n + 1 for n in shape])), dtype=np.int64).reshape(
        tuple(n + 1 for n in shape)
    )


def _grid(
    points: NDArray[np.float64],
    cells: list[tuple[int, list[int]]],
    point_data: dict[str, NDArray[np.float64]] | None = None,
) -> UnstructuredGrid:
    sizes = np.array([len(c[1]) for c in cells], dtype=np.int64)
    return UnstructuredGrid(
        points=points,
        connectivity=np.array([n for c in cells for n in c[1]], dtype=np.int64),
        offsets=np.cumsum(sizes),
        types=np.array([c[0] for c in cells], dtype=np.uint8),
        point_data=point_data or {},
    )


def cube_cells(
    kind: str,
    shape: tuple[int, int, int],
    base: int,
    index: NDArray[np.int64],
    extra_node: int,
) -> tuple[list[tuple[int, list[int]]], int]:
    """The cells of one cell type filling every cube of ``shape``; pyramids add centre nodes."""
    cells: list[tuple[int, list[int]]] = []
    nx, ny, nz = shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [
                    int(index[i, j, k]),
                    int(index[i + 1, j, k]),
                    int(index[i + 1, j + 1, k]),
                    int(index[i, j + 1, k]),
                    int(index[i, j, k + 1]),
                    int(index[i + 1, j, k + 1]),
                    int(index[i + 1, j + 1, k + 1]),
                    int(index[i, j + 1, k + 1]),
                ]
                if kind == "hex":
                    cells.append((HEX, c))
                elif kind == "wedge":
                    cells.append((WEDGE, [c[0], c[1], c[2], c[4], c[5], c[6]]))
                    cells.append((WEDGE, [c[0], c[2], c[3], c[4], c[6], c[7]]))
                elif kind == "tet":
                    for path in ((1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)):
                        code = 0
                        chain = [c[0]]
                        for bit in path:
                            code |= bit
                            chain.append(_corner(c, code))
                        cells.append((TET, chain))
                elif kind == "pyramid":
                    apex = extra_node
                    extra_node += 1
                    for face in (
                        (0, 1, 2, 3),
                        (4, 5, 6, 7),
                        (0, 1, 5, 4),
                        (1, 2, 6, 5),
                        (2, 3, 7, 6),
                        (3, 0, 4, 7),
                    ):
                        cells.append((PYR, [c[f] for f in face] + [apex]))
    return cells, extra_node


def _corner(c: list[int], code: int) -> int:
    """Node of a cube by its (x=1, y=2, z=4) bit code."""
    x, y, z = code & 1, (code >> 1) & 1, (code >> 2) & 1
    return c[
        {
            (0, 0, 0): 0,
            (1, 0, 0): 1,
            (1, 1, 0): 2,
            (0, 1, 0): 3,
            (0, 0, 1): 4,
            (1, 0, 1): 5,
            (1, 1, 1): 6,
            (0, 1, 1): 7,
        }[(x, y, z)]
    ]


def box_grid(
    kind: str,
    shape: tuple[int, int, int] = (3, 3, 3),
    lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    xs: NDArray[np.float64] | None = None,
) -> UnstructuredGrid:
    """A box filled with ``kind`` cells ('hex', 'tet', 'wedge' or 'pyramid')."""
    axes = [
        np.linspace(lo[a], hi[a], shape[a] + 1) if not (a == 0 and xs is not None) else xs
        for a in range(3)
    ]
    points = node_array(*axes)
    index = _index(shape)
    cells, n_nodes = cube_cells(kind, shape, 0, index, points.shape[0])
    if kind == "pyramid":
        centres = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    centres.append(
                        [
                            (axes[0][i] + axes[0][i + 1]) / 2,
                            (axes[1][j] + axes[1][j + 1]) / 2,
                            (axes[2][k] + axes[2][k + 1]) / 2,
                        ]
                    )
        points = np.vstack([points, np.array(centres)])
        assert n_nodes == points.shape[0]
    return _grid(points, cells)


def mixed_grid() -> UnstructuredGrid:
    """Hexahedra for x in [0, 1], wedges for [1, 2] and pyramids for [2, 3], conforming."""
    parts: list[NDArray[np.float64]] = []
    cells: list[tuple[int, list[int]]] = []
    previous: dict[tuple[float, float, float], int] = {}
    for slab, kind in enumerate(("hex", "wedge", "pyramid")):
        grid = box_grid(kind, (1, 2, 2), (float(slab), 0.0, 0.0), (float(slab + 1), 1.0, 1.0))
        remap = np.zeros(grid.points.shape[0], dtype=np.int64)
        for n, p in enumerate(grid.points):
            key = (round(p[0], 12), round(p[1], 12), round(p[2], 12))
            if key not in previous:
                previous[key] = len(parts)
                parts.append(p)
            remap[n] = previous[key]
        starts = np.concatenate(([0], grid.offsets[:-1]))
        for t, s, e in zip(grid.types, starts, grid.offsets, strict=True):
            cells.append((int(t), [int(remap[v]) for v in grid.connectivity[s:e]]))
    # parts holds points in index order because each new key gets the next index
    return _grid(np.array(parts), cells)


def distort(grid: UnstructuredGrid, amplitude: float, seed: int = 7) -> UnstructuredGrid:
    """Move the nodes by a smooth-ish random amount, interior and boundary alike."""
    rng = np.random.default_rng(seed)
    moved = grid.points + rng.uniform(-amplitude, amplitude, grid.points.shape)
    return UnstructuredGrid(
        moved, grid.connectivity, grid.offsets, grid.types, dict(grid.point_data)
    )


def with_fields(
    grid: UnstructuredGrid, scalar: object, vector: object | None = None
) -> UnstructuredGrid:
    """Attach ``f`` (and ``v``) evaluated at the nodes."""
    data = {"f": np.asarray(scalar(grid.points), dtype=float)}  # type: ignore[operator]
    if vector is not None:
        data["v"] = np.asarray(vector(grid.points), dtype=float)  # type: ignore[operator]
    return UnstructuredGrid(grid.points, grid.connectivity, grid.offsets, grid.types, data)


def big_hex_grid(n: int) -> UnstructuredGrid:
    """``n`` cubed hexahedra in the unit cube, built with array operations."""
    axis = np.linspace(0.0, 1.0, n + 1)
    points = node_array(axis, axis, axis)
    index = _index((n, n, n))
    i0 = index[:-1, :-1, :-1].ravel()
    di, dj, dk = (n + 1) * (n + 1), n + 1, 1
    conn = np.stack(
        [
            i0,
            i0 + di,
            i0 + di + dj,
            i0 + dj,
            i0 + dk,
            i0 + di + dk,
            i0 + di + dj + dk,
            i0 + dj + dk,
        ],
        axis=1,
    )
    n_cells = conn.shape[0]
    return UnstructuredGrid(
        points,
        conn.ravel().astype(np.int64),
        (np.arange(1, n_cells + 1, dtype=np.int64) * 8),
        np.full(n_cells, HEX, dtype=np.uint8),
        {"f": points[:, 0] + 2.0 * points[:, 1] - points[:, 2]},
    )


def write_vtu_ascii(path: object, grid: UnstructuredGrid) -> None:
    """Write ``grid`` as an ascii XML VTU file, for the tests that read a file."""
    from pathlib import Path

    def block(name: str, kind: str, values: NDArray[np.float64], comps: int = 1) -> str:
        text = " ".join(
            repr(float(v)) if kind.startswith("Float") else str(int(v)) for v in values.ravel()
        )
        return (
            f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{comps}" '
            f'format="ascii">{text}</DataArray>'
        )

    pdata = "".join(
        block(k, "Float64", v, 1 if v.ndim == 1 else 3) for k, v in grid.point_data.items()
    )
    xml = (
        '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" '
        'byte_order="LittleEndian"><UnstructuredGrid>'
        f'<Piece NumberOfPoints="{len(grid.points)}" NumberOfCells="{grid.n_cells}">'
        f"<PointData>{pdata}</PointData><CellData></CellData><Points>"
        + block("Points", "Float64", grid.points, 3)
        + "</Points><Cells>"
        + block("connectivity", "Int64", grid.connectivity)
        + block("offsets", "Int64", grid.offsets)
        + block("types", "UInt8", grid.types)
        + "</Cells></Piece></UnstructuredGrid></VTKFile>\n"
    )
    Path(str(path)).write_text(xml, encoding="utf-8")

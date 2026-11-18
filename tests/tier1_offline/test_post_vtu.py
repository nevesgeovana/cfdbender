"""The VTU reader reads every encoding the format has and refuses the rest by name."""

from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

from cfdbender.post.vtu import VtuError, parse_vtu, read_vtu

POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.5], [2.0, 0.0, 0.25]]
)
CONN = np.array([0, 1, 2, 3, 1, 4, 2], dtype=np.int64)
OFFSETS = np.array([4, 7], dtype=np.int64)
TYPES = np.array([9, 5], dtype=np.uint8)
PRESSURE = np.array([1.5, -2.0, 3.25, 0.0, 7.0])
VECTOR = np.arange(15, dtype=np.float64).reshape(5, 3) / 7.0
CELL_ID = np.array([10, 20], dtype=np.int32)

_TYPE_NAMES = {"<f8": "Float64", "<f4": "Float32", "<i8": "Int64", "u1": "UInt8", "<i4": "Int32"}


def _payload(values: np.ndarray, dtype: str) -> bytes:
    return np.ascontiguousarray(values, dtype=dtype).tobytes()


def _blocks(data: bytes, header: str, compress: bool, block: int = 16) -> bytes:
    """Header plus data of one binary array, as the format lays them out."""
    code = "<u4" if header == "UInt32" else "<u8"
    if not compress:
        return np.array([len(data)], dtype=code).tobytes() + data
    chunks = [data[i : i + block] for i in range(0, len(data), block)] or [b""]
    packed = [zlib.compress(c) for c in chunks]
    last = len(chunks[-1]) if len(data) % block else 0
    head = np.array([len(chunks), block, last, *[len(p) for p in packed]], dtype=code)
    return head.tobytes() + b"".join(packed)


def _inline(data: bytes, header: str, compress: bool) -> str:
    """Inline base64: the header and the data are encoded apart."""
    code = "<u4" if header == "UInt32" else "<u8"
    width = 4 if header == "UInt32" else 8
    whole = _blocks(data, header, compress)
    if compress:
        n = int(np.frombuffer(whole[:width], dtype=code)[0])
        cut = (3 + n) * width
    else:
        cut = width
    return base64.b64encode(whole[:cut]).decode() + base64.b64encode(whole[cut:]).decode()


def write_vtu(
    path: Path,
    fmt: str,
    header: str = "UInt32",
    compress: bool = False,
    byte_order: str = "LittleEndian",
    compressor: str | None = None,
) -> None:
    arrays: list[tuple[str, str, str, np.ndarray, int]] = [
        ("Points", "", "<f8", POINTS, 3),
        ("connectivity", "", "<i8", CONN, 1),
        ("offsets", "", "<i8", OFFSETS, 1),
        ("types", "", "u1", TYPES, 1),
        ("Pressure", "PointData", "<f8", PRESSURE, 1),
        ("Velocity", "PointData", "<f4", VECTOR, 3),
        ("ID", "CellData", "<i4", CELL_ID, 1),
    ]
    appended = b""
    nodes: list[str] = []
    for name, _sec, dtype, values, ncomp in arrays:
        head = (
            f'<DataArray type="{_TYPE_NAMES[dtype]}" Name="{name}" '
            f'NumberOfComponents="{ncomp}" format="{fmt}"'
        )
        data = _payload(values, dtype)
        if fmt == "ascii":
            text = " ".join(repr(float(v)) for v in np.asarray(values).ravel())
            nodes.append(f"{head}>{text}</DataArray>")
        elif fmt == "binary":
            nodes.append(f"{head}>{_inline(data, header, compress)}</DataArray>")
        else:
            nodes.append(f'{head} offset="{len(appended)}"/>')
            if fmt == "appended-raw":
                appended += _blocks(data, header, compress)
            else:
                appended += _inline(data, header, compress).encode()
    node = dict(zip([a[0] for a in arrays], nodes, strict=True))
    fmt_attr = "appended" if fmt.startswith("appended") else fmt
    for key in node:
        node[key] = node[key].replace(f'format="{fmt}"', f'format="{fmt_attr}"')
    comp = compressor or ("vtkZLibDataCompressor" if compress else None)
    attrs = f'byte_order="{byte_order}" header_type="{header}"'
    if comp:
        attrs += f' compressor="{comp}"'
    xml = (
        f'<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" {attrs}>\n'
        f'<UnstructuredGrid><Piece NumberOfPoints="5" NumberOfCells="2">\n'
        f"<PointData>{node['Pressure']}{node['Velocity']}</PointData>\n"
        f"<CellData>{node['ID']}</CellData>\n<Points>{node['Points']}</Points>\n"
        f"<Cells>{node['connectivity']}{node['offsets']}{node['types']}</Cells>\n"
        "</Piece></UnstructuredGrid>\n"
    )
    out = xml.encode()
    if fmt.startswith("appended"):
        enc = "raw" if fmt == "appended-raw" else "base64"
        out += f'<AppendedData encoding="{enc}">\n   _'.encode() + appended + b"\n</AppendedData>\n"
    path.write_bytes(out + b"</VTKFile>\n")


def _check(grid) -> None:
    np.testing.assert_array_equal(grid.points, POINTS)
    np.testing.assert_array_equal(grid.connectivity, CONN)
    np.testing.assert_array_equal(grid.offsets, OFFSETS)
    np.testing.assert_array_equal(grid.types, TYPES)
    np.testing.assert_array_equal(grid.point_data["Pressure"], PRESSURE)
    np.testing.assert_array_equal(
        grid.point_data["Velocity"], VECTOR.astype(np.float32).astype(np.float64)
    )
    np.testing.assert_array_equal(grid.cell_data["ID"], CELL_ID)


@pytest.mark.parametrize("header", ["UInt32", "UInt64"])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("fmt", ["ascii", "binary", "appended-raw", "appended-base64"])
def test_every_encoding_reads_the_same_grid(tmp_path, fmt, compress, header):
    if fmt == "ascii" and (compress or header == "UInt64"):
        pytest.skip("ascii has no header and no compression")
    path = tmp_path / "g.vtu"
    write_vtu(path, fmt, header=header, compress=compress)
    _check(read_vtu(path))


def test_cells_of_type_gives_the_rows_of_each_kind(tmp_path):
    path = tmp_path / "g.vtu"
    write_vtu(path, "binary")
    grid = read_vtu(path)
    quads, qi = grid.cells_of_type(9)
    tris, ti = grid.cells_of_type(5)
    np.testing.assert_array_equal(quads, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(tris, [[1, 4, 2]])
    assert qi.tolist() == [0] and ti.tolist() == [1]
    assert grid.cells_of_type(10)[0].shape[0] == 0


def test_other_compressor_is_refused_by_name(tmp_path):
    path = tmp_path / "g.vtu"
    write_vtu(path, "binary", compressor="vtkLZ4DataCompressor")
    with pytest.raises(VtuError, match="vtkLZ4DataCompressor"):
        read_vtu(path)


def test_big_endian_is_refused_by_name(tmp_path):
    path = tmp_path / "g.vtu"
    write_vtu(path, "binary", byte_order="BigEndian")
    with pytest.raises(VtuError, match="big-endian"):
        read_vtu(path)


def test_a_polydata_file_is_not_an_unstructured_grid():
    with pytest.raises(VtuError, match="PolyData"):
        parse_vtu(b'<VTKFile type="PolyData"><PolyData/></VTKFile>')


def test_garbage_is_refused():
    with pytest.raises(VtuError, match="XML"):
        parse_vtu(b"not xml at all")

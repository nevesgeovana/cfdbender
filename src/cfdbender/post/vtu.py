"""Read an unstructured-grid ``.vtu`` file with numpy and the standard library.

The reader takes the XML flavour of the format. The arrays may be ascii, inline
base64 or appended (raw or base64), zlib compressed or not, with a ``UInt32`` or
``UInt64`` header, little endian. What it does not read is refused by name:
another compressor, a big-endian file, a file that is not an unstructured grid.
"""

from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["UnstructuredGrid", "VtuError", "parse_vtu", "read_vtu"]

_DTYPES: dict[str, str] = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "<i2",
    "UInt16": "<u2",
    "Int32": "<i4",
    "UInt32": "<u4",
    "Int64": "<i8",
    "UInt64": "<u8",
    "Float32": "<f4",
    "Float64": "<f8",
}
_HEADER_BYTES = {"UInt32": 4, "UInt64": 8}
_ZLIB = "vtkZLibDataCompressor"


class VtuError(ValueError):
    """A VTU file that this reader does not read, or that is malformed."""


@dataclass(frozen=True)
class UnstructuredGrid:
    """Points, cells and named data of one unstructured grid.

    Attributes
    ----------
    points : numpy.ndarray
        Coordinates, shape ``(n_points, 3)``.
    connectivity : numpy.ndarray
        Point indices of all cells, concatenated.
    offsets : numpy.ndarray
        End position of each cell in ``connectivity``.
    types : numpy.ndarray
        VTK cell type of each cell (5 triangle, 9 quadrilateral).
    point_data, cell_data : dict of str to numpy.ndarray
        Named arrays; a vector has shape ``(n, 3)``.
    """

    points: NDArray[np.float64]
    connectivity: NDArray[np.int64]
    offsets: NDArray[np.int64]
    types: NDArray[np.uint8]
    point_data: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    cell_data: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return int(self.offsets.size)

    def cells_of_type(self, vtk_type: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the point indices and the cell numbers of the cells of one type.

        The first array has one row per cell. The type must have one size.
        """
        index = np.flatnonzero(self.types == vtk_type)
        if index.size == 0:
            return np.zeros((0, 0), dtype=np.int64), index.astype(np.int64)
        starts = np.concatenate(([0], self.offsets[:-1]))[index]
        sizes = (self.offsets - np.concatenate(([0], self.offsets[:-1])))[index]
        if np.any(sizes != sizes[0]):
            raise VtuError(f"cells of type {vtk_type} have different sizes")
        rows = self.connectivity[starts[:, None] + np.arange(int(sizes[0]))[None, :]]
        return rows.astype(np.int64), index.astype(np.int64)


def read_vtu(path: str | Path) -> UnstructuredGrid:
    """Read the unstructured grid in ``path``."""
    return parse_vtu(Path(path).read_bytes())


def _decode_block(blob: bytes, header_type: str, compressed: bool) -> bytes:
    """Return the payload of one binary array given its header and data bytes."""
    width = _HEADER_BYTES[header_type]
    code = "<u4" if width == 4 else "<u8"
    if not compressed:
        size = int(np.frombuffer(blob[:width], dtype=code)[0])
        return blob[width : width + size]
    n_blocks = int(np.frombuffer(blob[:width], dtype=code)[0])
    header_len = (3 + n_blocks) * width
    sizes = np.frombuffer(blob[3 * width : header_len], dtype=code)
    out = bytearray()
    cursor = header_len
    for size in sizes:
        out += zlib.decompress(blob[cursor : cursor + int(size)])
        cursor += int(size)
    return bytes(out)


def _split_inline_base64(text: str, header_type: str, compressed: bool) -> bytes:
    """Decode an inline or appended base64 array, header and data encoded apart."""
    text = "".join(text.split())
    width = _HEADER_BYTES[header_type]
    if not compressed:
        head_chars = -(-width // 3) * 4
        head = base64.b64decode(text[:head_chars])
        size = int(np.frombuffer(head, dtype="<u4" if width == 4 else "<u8")[0])
        data_chars = -(-size // 3) * 4
        data = base64.b64decode(text[head_chars : head_chars + data_chars])
        return _decode_block(head + data, header_type, False)
    first_chars = width * 4
    first = np.frombuffer(
        base64.b64decode(text[:first_chars]), dtype="<u4" if width == 4 else "<u8"
    )
    n_blocks = int(first[0])
    head_chars = -(-((3 + n_blocks) * width) // 3) * 4
    head = base64.b64decode(text[:head_chars])
    sizes = np.frombuffer(head[3 * width :], dtype="<u4" if width == 4 else "<u8")
    data_chars = -(-int(sizes.sum()) // 3) * 4
    data = base64.b64decode(text[head_chars : head_chars + data_chars])
    return _decode_block(head + data, header_type, True)


def _raw_appended(blob: bytes, offset: int, header_type: str, compressed: bool) -> bytes:
    """Decode one raw appended array starting at ``offset``."""
    width = _HEADER_BYTES[header_type]
    code = "<u4" if width == 4 else "<u8"
    if not compressed:
        size = int(np.frombuffer(blob[offset : offset + width], dtype=code)[0])
        return _decode_block(blob[offset : offset + width + size], header_type, False)
    first = np.frombuffer(blob[offset : offset + 3 * width], dtype=code)
    n_blocks = int(first[0])
    head_len = (3 + n_blocks) * width
    sizes = np.frombuffer(blob[offset + 3 * width : offset + head_len], dtype=code)
    end = offset + head_len + int(sizes.sum())
    return _decode_block(blob[offset:end], header_type, True)


def parse_vtu(content: bytes) -> UnstructuredGrid:
    """Parse the bytes of an unstructured-grid ``.vtu`` file."""
    appended = b""
    appended_encoding = ""
    marker = re.search(rb"<AppendedData[^>]*encoding=\"(raw|base64)\"[^>]*>", content)
    xml_bytes = content
    if marker:
        appended_encoding = marker.group(1).decode()
        underscore = content.index(b"_", marker.end())
        end = content.rindex(b"</AppendedData>")
        appended = content[underscore + 1 : end]
        xml_bytes = content[: underscore + 1] + content[end:]
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise VtuError(f"not an XML VTU file: {exc}") from exc
    if root.tag != "VTKFile" or root.get("type") != "UnstructuredGrid":
        raise VtuError(f"not an unstructured grid (type {root.get('type')!r})")
    if root.get("byte_order", "LittleEndian") != "LittleEndian":
        raise VtuError("big-endian VTU files are not read")
    header_type = root.get("header_type", "UInt32")
    if header_type not in _HEADER_BYTES:
        raise VtuError(f"header type {header_type!r} is not read")
    compressor = root.get("compressor")
    if compressor not in (None, _ZLIB):
        raise VtuError(f"compressor {compressor!r} is not read (only {_ZLIB})")
    compressed = compressor == _ZLIB
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise VtuError("no Piece in the unstructured grid")

    def array(node: ET.Element) -> NDArray[np.float64]:
        kind = node.get("type", "")
        if kind not in _DTYPES:
            raise VtuError(f"data type {kind!r} of array {node.get('Name')!r} is not read")
        dtype = np.dtype(_DTYPES[kind])
        fmt = node.get("format", "ascii")
        if fmt == "ascii":
            values = np.array((node.text or "").split(), dtype=np.float64).astype(dtype)
        elif fmt == "binary":
            values = np.frombuffer(
                _split_inline_base64(node.text or "", header_type, compressed), dtype=dtype
            )
        elif fmt == "appended":
            offset = int(node.get("offset", "0"))
            if appended_encoding == "raw":
                payload = _raw_appended(appended, offset, header_type, compressed)
            elif appended_encoding == "base64":
                payload = _split_inline_base64(
                    appended[offset:].decode("ascii"), header_type, compressed
                )
            else:
                raise VtuError("appended array without an AppendedData section")
            values = np.frombuffer(payload, dtype=dtype)
        else:
            raise VtuError(f"array format {fmt!r} is not read")
        n_comp = int(node.get("NumberOfComponents", "1"))
        return values.reshape(-1, n_comp) if n_comp > 1 else values
    def named(section: str) -> dict[str, NDArray[np.float64]]:
        out: dict[str, NDArray[np.float64]] = {}
        parent = piece.find(section)
        if parent is not None:
            for node in parent.findall("DataArray"):
                out[node.get("Name", "")] = array(node)
        return out

    points_node = piece.find("./Points/DataArray")
    cells_node = piece.find("./Cells")
    if points_node is None or cells_node is None:
        raise VtuError("the piece has no Points or Cells")
    by_name = {n.get("Name"): n for n in cells_node.findall("DataArray")}
    for need in ("connectivity", "offsets", "types"):
        if need not in by_name:
            raise VtuError(f"the Cells section has no {need!r} array")
    points = array(points_node).astype(np.float64).reshape(-1, 3)
    return UnstructuredGrid(
        points=points,
        connectivity=array(by_name["connectivity"]).astype(np.int64),
        offsets=array(by_name["offsets"]).astype(np.int64),
        types=array(by_name["types"]).astype(np.uint8),
        point_data=named("PointData"),
        cell_data=named("CellData"),
    )

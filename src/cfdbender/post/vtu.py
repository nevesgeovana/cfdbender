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


def _header(blob: bytes | memoryview, header_type: str, count: int) -> tuple[int, ...]:
    """Read a complete binary header without trusting a short slice."""
    if header_type not in _HEADER_BYTES:
        raise VtuError(f"header type {header_type!r} is not read")
    width = _HEADER_BYTES[header_type]
    if len(blob) < count * width:
        raise VtuError("truncated VTU binary header")
    return tuple(int.from_bytes(blob[i * width : (i + 1) * width], "little") for i in range(count))


def _decode_block(blob: bytes | memoryview, header_type: str, compressed: bool) -> bytes:
    """Return a complete binary array, checking every declared block length."""
    first = _header(blob, header_type, 3 if compressed else 1)
    width = _HEADER_BYTES[header_type]
    if not compressed:
        size = first[0]
        if len(blob) < width + size:
            raise VtuError("truncated VTU binary payload")
        return bytes(blob[width : width + size])
    n_blocks, block_size, last_size = first
    header = _header(blob, header_type, 3 + n_blocks)
    if last_size > block_size or (n_blocks and block_size == 0):
        raise VtuError("invalid VTU compressed block sizes")
    cursor = (3 + n_blocks) * width
    out = bytearray()
    for index, size in enumerate(header[3:]):
        if len(blob) < cursor + size:
            raise VtuError("truncated VTU compressed payload")
        expected = last_size if index == n_blocks - 1 and last_size else block_size
        decoder = zlib.decompressobj()
        try:
            chunk = decoder.decompress(blob[cursor : cursor + size], expected + 1)
        except (zlib.error, OverflowError) as error:
            raise VtuError(f"invalid VTU compressed payload: {error}") from error
        if len(chunk) != expected or not decoder.eof or decoder.unused_data:
            raise VtuError("VTU compressed block does not match its declared size")
        out += chunk
        cursor += size
    return bytes(out)


def _base64(text: str) -> bytes:
    """Decode one complete base64 fragment, reporting malformed input by format."""
    try:
        return base64.b64decode(text, validate=True)
    except ValueError as error:
        raise VtuError(f"invalid VTU base64 payload: {error}") from error


def _split_inline_base64(text: str, header_type: str, compressed: bool) -> bytes:
    """Decode an array whose header and data are encoded together or separately."""
    text = "".join(text.split())
    if header_type not in _HEADER_BYTES:
        raise VtuError(f"header type {header_type!r} is not read")
    width = _HEADER_BYTES[header_type]
    first_bytes = (3 if compressed else 1) * width
    first_chars = -(-first_bytes // 3) * 4
    first = _header(_base64(text[:first_chars]), header_type, 3 if compressed else 1)
    head_bytes = (3 + first[0]) * width if compressed else width
    head_chars = -(-head_bytes // 3) * 4
    head = _base64(text[:head_chars])
    header = _header(head, header_type, head_bytes // width)
    size = sum(header[3:]) if compressed else header[0]
    if len(head) > head_bytes:
        # No padding separated the header: this is one continuous base64 stream.
        whole_chars = -(-(head_bytes + size) // 3) * 4
        blob = _base64(text[:whole_chars])
    else:
        data_chars = -(-size // 3) * 4
        blob = head + _base64(text[head_chars : head_chars + data_chars])
    return _decode_block(blob, header_type, compressed)


def _raw_appended(blob: bytes, offset: int, header_type: str, compressed: bool) -> bytes:
    """Decode a raw appended array without copying the remaining file."""
    if offset < 0 or offset >= len(blob):
        raise VtuError(f"VTU appended offset {offset} is outside the payload")
    return _decode_block(memoryview(blob)[offset:], header_type, compressed)


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
            if offset < 0 or offset >= len(appended):
                raise VtuError(f"VTU appended offset {offset} is outside the payload")
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

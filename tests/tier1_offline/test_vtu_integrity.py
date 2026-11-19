"""Binary VTU headers must describe exactly the data that is decoded."""

from __future__ import annotations

import base64
import struct
import zlib

import pytest
from test_post_vtu import _blocks, write_vtu

from cfdbender.post.vtu import (
    VtuError,
    _decode_block,
    _raw_appended,
    _split_inline_base64,
    parse_vtu,
)


@pytest.mark.parametrize("blob", [b"", b"\x01", struct.pack("<I", 16) + b"12345678"])
def test_truncated_blocks_are_refused(blob):
    try:
        _decode_block(blob, "UInt32", False)
    except Exception as error:
        assert isinstance(error, VtuError), type(error).__name__
    else:
        pytest.fail("truncated block accepted")


def test_compressed_block_must_match_declared_uncompressed_size():
    packed = zlib.compress(b"12345678")
    blob = struct.pack("<4I", 1, 16, 16, len(packed)) + packed
    with pytest.raises(VtuError):
        _decode_block(blob, "UInt32", True)


@pytest.mark.parametrize("offset", [-1, 999])
def test_appended_offset_must_reach_a_header(offset):
    try:
        _raw_appended(struct.pack("<I", 0), offset, "UInt32", False)
    except Exception as error:
        assert isinstance(error, VtuError), type(error).__name__
    else:
        pytest.fail("invalid offset accepted")


def test_a_truncated_inline_array_cannot_become_a_short_grid(tmp_path):
    path = tmp_path / "mesh.vtu"
    write_vtu(path, "binary")
    content = path.read_bytes()
    start = content.index(b'format="binary">') + len(b'format="binary">')
    end = content.index(b"</DataArray>", start)
    # PointData Pressure declares five doubles, but supplies four whole doubles.
    header = base64.b64encode(struct.pack("<I", 40))
    payload = base64.b64encode(struct.pack("<4d", 1, 2, 3, 4))
    with pytest.raises(VtuError):
        parse_vtu(content[:start] + header + payload + content[end:])


@pytest.mark.parametrize("header", ["UInt32", "UInt64"])
@pytest.mark.parametrize("compressed", [False, True])
def test_continuous_base64_header_and_payload_are_read(header, compressed):
    payload = struct.pack("<3d", 1, 2, 3)
    encoded = base64.b64encode(_blocks(payload, header, compressed)).decode()
    try:
        decoded = _split_inline_base64(encoded, header, compressed)
    except Exception as error:
        pytest.fail(f"valid continuous base64 was refused: {error}")
    assert decoded == payload

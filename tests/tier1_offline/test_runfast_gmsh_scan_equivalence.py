"""The streamed Gmsh scan equals the whole-file result for any bytes and any block size."""

from __future__ import annotations

import random

import pytest

from cfdbender.core import plan as plan_module
from cfdbender.core.plan import InputFiles
from cfdbender.solvers._host import _gmsh
from cfdbender.solvers._host._gmsh import GmshScan
from cfdbender.solvers._host._groups import marker_names, tracked_markers

BLOCKS = (1, 2, 7, plan_module._BLOCK)
HEADERS = {
    "2.2": (b"$MeshFormat\n2.2 0 8\n", b"$MeshFormat\r\n2.2 0 8\r\n"),
    "gmsh": (b"$MeshFormat\n", b"$MeshFormat\r\n"),
}


def whole_header(data: bytes) -> str:
    """The header kind by the whole-file startswith checks the adapters used."""
    if data.startswith(HEADERS["2.2"]):
        return "2.2"
    if data.startswith(HEADERS["gmsh"]):
        return "gmsh"
    return "other"


def scan(data: bytes, block: int) -> tuple[str, ...]:
    scanner = GmshScan()
    for start in range(0, len(data), block):
        scanner.feed(data[start : start + block])
    return scanner.result()


def check(data: bytes, block: int) -> None:
    found = scan(data, block)
    assert found[0] == whole_header(data), (data, block)
    if found[1] == "overflow":
        assert (
            b"$PhysicalNames" in data
            and b"$EndPhysicalNames" not in data.partition(b"$PhysicalNames")[2]
        )
        return
    assert found[1] == "names" and found[2:] == marker_names(data, "x.msh"), (data, block)


def generated(rng: random.Random) -> bytes:
    pieces = [
        b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n",
        b"$MeshFormat\r\n2.2 0 8\r\n$EndMeshFormat\r\n",
        b"$MeshFormat\n4.1 0 8\n",
        b"$PhysicalNames\n",
        b"$EndPhysicalNames\n",
        b"$PhysicalNames",
        b"$EndPhysicalNames",
        b"$Phys",
        b"icalNames\n",
        b"$End",
        b"3\n",
        b'2 1 "wing"\n',
        b'  1\t7  "tail edge"\n',
        b'2 2 "wing"\n',
        b'3 4 "solid"\n',
        b'2 5 "multi\nline"\n',
        b'\n\n  2 9 "after blank"\r\n',
        b'2 3 "caf\xe9"\n',
        b'2 3 "\xff\xfe"\n',
        b'2 3 ""\n',
        b'2 3 "open',
        b'\x1c\x1f2 8 "ws"\n',
        b"$Nodes\n1 0.5 0.25 0.125\n",
        b"\n",
        b"\r\n",
        b"junk " * rng.randint(1, 20),
        b'"',
    ]
    data = b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 40)))
    if rng.random() < 0.5:
        data = rng.choice(pieces[:3]) + data
    return data


@pytest.mark.parametrize("block", BLOCKS)
def test_the_scan_equals_the_whole_file_result_for_seeded_random_bytes(block):
    rng = random.Random(20260901)
    for _ in range(400):
        check(generated(rng), block)


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"$MeshFormat",
        b"$MeshFormat\n2.2 0 8\n",
        b"$MeshFormat\n2.2 0 8",
        b'$MeshFormat\r\n2.2 0 8\r\n$PhysicalNames\n1\n2 1 "a"\n',  # no end marker
        b"$PhysicalNames$EndPhysicalNames",
        b'$PhysicalNames\n2 1 "a"\n$EndPhysicalNames\n$PhysicalNames\n2 2 "b"\n$EndPhysicalNames\n',
        b'$EndPhysicalNames\n$PhysicalNames\n2 1 "a"\n',
        b'x$PhysicalNames\n2 1 "mid"\n$EndPhysicalNames',
    ],
)
def test_the_boundary_cases_equal_the_whole_file_result(data, block):
    check(data, block)


@pytest.mark.parametrize("block", BLOCKS)
def test_a_marker_split_over_every_block_boundary_is_found(block):
    data = b'$MeshFormat\n2.2 0 8\n$PhysicalNames\n1\n2 1 "w"\n$EndPhysicalNames\n$Nodes\n'
    for pad in range(0, 40):
        check(b"%" * pad + b"\n" + data, block)


def test_a_section_over_the_bound_sends_the_caller_to_the_exact_route(monkeypatch):
    monkeypatch.setattr(_gmsh, "_SECTION_LIMIT", 64)
    data = b"$PhysicalNames\n" + b'2 1 "w"\n' * 100 + b"$EndPhysicalNames\n"
    assert scan(data, 7) == ("other", "overflow")
    assert marker_names(data, "x.msh") == ("w",)


@pytest.mark.parametrize("size", [0, 5])
def test_the_exact_route_gives_the_whole_file_names_when_the_section_is_over_the_bound(
    tmp_path, monkeypatch, size
):
    monkeypatch.setattr(_gmsh, "_SECTION_LIMIT", 64)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", size)
    monkeypatch.setattr(plan_module, "_BLOCK", 16)
    data = b"$PhysicalNames\n" + b"".join(b'2 %d "n%d"\n' % (i, i) for i in range(30))
    data += b"$EndPhysicalNames\n"
    (tmp_path / "m.msh").write_bytes(data)
    found = tracked_markers(InputFiles(tmp_path.resolve()), "m.msh")
    assert found == marker_names(data, "m.msh") and len(found) == 30


@pytest.mark.parametrize("size", [0, 5])
def test_the_tracked_reader_gives_the_whole_file_names_for_any_threshold(
    tmp_path, monkeypatch, size
):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", size)
    monkeypatch.setattr(plan_module, "_BLOCK", 7)
    rng = random.Random(7)
    for index in range(60):
        data = generated(rng)
        (tmp_path / f"m{index}.msh").write_bytes(data)
        found = tracked_markers(InputFiles(tmp_path.resolve()), f"m{index}.msh")
        assert found == marker_names(data, "m.msh"), data


@pytest.mark.parametrize(
    "bad", [[], ["gmsh"], ["bad", "names"], ["2.2", "bogus"], ["2.2", "overflow", "wing"]]
)
def test_corrupt_gmsh_cache_fact_is_recomputed(tmp_path, monkeypatch, bad):
    import json
    import time

    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 0)
    monkeypatch.setattr(plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9)
    (tmp_path / "m.msh").write_bytes(
        b'$MeshFormat\n2.2 0 8\n$PhysicalNames\n2 1 "wing"\n$EndPhysicalNames'
    )
    assert tracked_markers(InputFiles(tmp_path), "m.msh") == ("wing",)
    cache = tmp_path / plan_module.DigestCache.NAME
    document = json.loads(cache.read_text())
    document["entries"]["m.msh"]["scans"]["gmsh-info"] = bad
    cache.write_text(json.dumps(document))
    assert tracked_markers(InputFiles(tmp_path), "m.msh") == ("wing",)


def test_unterminated_names_section_requests_exact_route():
    assert scan(b'$PhysicalNames\n2 1 "wing"\n', 3) == ("other", "overflow")

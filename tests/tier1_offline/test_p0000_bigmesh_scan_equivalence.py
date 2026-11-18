"""BIGMESH1-002: the streamed SU2 marker scan equals the whole-file marker_names for any bytes."""

import random

import pytest

from cfdbender.core import plan as plan_module
from cfdbender.core.plan import InputFiles
from cfdbender.solvers._host import _groups
from cfdbender.solvers._host._groups import _Su2Markers, marker_names, tracked_markers


def _tracked(tmp_path, data):
    (tmp_path / "m.su2").write_bytes(data)
    return tracked_markers(InputFiles(tmp_path.resolve()), "m.su2")


@pytest.mark.parametrize(
    ("data", "block"),
    [
        (b" " * ((1 << 20) + 1) + b"MARKER_TAG= wall\n", 700_000),
        (b"MARKER_TAG=\nwall\n", 12),
        (b"MARKER_TAG\n=\nwall\n", 11),
        (b"MARKER_TAG=\n" + b" " * (5 << 20) + b"\nwall\n", 1 << 20),
    ],
)
def test_bigmesh1_002_reproducers_equal_the_whole_file_scan(tmp_path, monkeypatch, data, block):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 8)
    monkeypatch.setattr(plan_module, "_BLOCK", block)
    assert marker_names(data, "m.su2") == ("wall",)
    assert _tracked(tmp_path, data) == ("wall",)


def _generated(rng):
    ws = [b" ", b"\t", b"\r", b"\x0b", b"\x0c", b"\x1c", b"\x1f"]
    pieces = [
        lambda: b"1 2 3\n",
        lambda: b"  4.5e3 7 -8\r\n",
        lambda: b"% comment MARKER_TAG= c\n",
        lambda: b"MARKER_TAG= wall\n",
        lambda: b"MARKER_TAG=wall2\r\n",
        lambda: b"marker_tag= low\n",
        lambda: b"Marker_Tag= Mix\n",
        lambda: b"MARKER_TAG=\nnamed\n",
        lambda: b"MARKER_TAG\n=\nsplit\n",
        lambda: b"MARKER_TAG =\n \n\t\n%c\nafter\n",
        lambda: b"MARKER_\nTAG= torn\n",
        lambda: b"\tMARKER_TAG\t=\tx\r\n",
        lambda: b"x MARKER_TAG= no\n",
        lambda: b"MARKER_TAG=\n" + b"".join(rng.choice(ws) for _ in range(rng.randint(0, 20))),
        lambda: b"MARKER_TAG= a % b\n",
        lambda: b"MARKER_TAG= 5\n",
        lambda: b"\n",
        lambda: b"\r\n",
        lambda: b"".join(rng.choice(ws) for _ in range(rng.randint(0, 120))) + b"\n",
        lambda: b"".join(rng.choice(ws) for _ in range(rng.randint(0, 120))) + b"MARKER_TAG= far\n",
        lambda: b"7" * rng.randint(1, 150) + b"\n",
        lambda: b"q" * rng.randint(1, 150),
        lambda: b"\xff\xfeMARKER_TAG= bad\n",
        lambda: b"MARKER_TAG= caf\xe9\n",
        lambda: b"MARKER_TAG=\n\xff\n",
        lambda: b"MARKER_TAG",
        lambda: b"=",
        lambda: b"\x00\n",
    ]
    count = rng.choice([0, 1, 2, 3, 6, 12])
    data = b"".join(rng.choice(pieces)() for _ in range(count))
    if rng.random() < 0.3:
        data = data.rstrip(b"\n")
    return data


@pytest.mark.parametrize("limits", [(1 << 20, 1 << 26), (40, 90)])
def test_bigmesh1_002_streamed_scan_equals_the_whole_file_function(monkeypatch, limits):
    monkeypatch.setattr(_groups, "_LONG_LINE", limits[0])
    monkeypatch.setattr(_groups, "_REDUCED_LIMIT", limits[1])
    rng = random.Random(20261118)
    cases = 0
    fallbacks = 0
    for _ in range(2500):
        data = _generated(rng)
        expected = marker_names(data, "x.su2")
        for block in (1, 2, 7, 1 << 20):
            scanner = _Su2Markers()
            for start in range(0, len(data), block):
                scanner.feed(data[start : start + block])
            found = scanner.result()
            if found is None:
                fallbacks += 1
            else:
                assert found == expected, (data, block)
            cases += 1
    assert cases >= 10000
    if limits[0] > 1000:
        assert fallbacks == 0

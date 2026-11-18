"""XFOIL and QPROP read a large input once per command, not once per call and point.

A threshold of one KiB stands in for the 8 MiB one, so the airfoil file and the blade OBJ count
as large. Every open of the input by any code path is counted through the audit hook.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import qprop_workspace as qw
import test_p0000_xfoil_execute as xfoil_case

from cfdbender.core import plan as plan_module
from cfdbender.matrix.cli import main

WATCHED = {"profile.dat", "blade.obj"}
_COUNTING: dict[str, object] = {"on": False, "opens": []}


def _hook(event: str, info: tuple) -> None:
    if _COUNTING["on"] and event == "open" and Path(str(info[0])).name in WATCHED:
        _COUNTING["opens"].append(Path(str(info[0])).name)  # type: ignore[attr-defined]


sys.addaudithook(_hook)


@pytest.fixture(autouse=True)
def small_threshold_and_settled_clock(monkeypatch):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024)
    monkeypatch.setattr(plan_module, "_BLOCK", 512)
    monkeypatch.setattr(
        plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9, raising=False
    )


@pytest.fixture
def opens():
    seen: list[str] = _COUNTING["opens"]  # type: ignore[assignment]
    seen.clear()
    _COUNTING["on"] = True
    yield seen
    _COUNTING["on"] = False


def xfoil(root: Path, count: int) -> None:
    xfoil_case.workspace(
        root,
        setup='airfoil_file="inputs/airfoils/profile.dat"\nreynolds = 1000000\n',
        sweep=",".join(str(2 * n) for n in range(count)),
    )
    (root / "inputs/airfoils").mkdir()
    rows = b"0.5 0.1\n" * 400
    (root / "inputs/airfoils/profile.dat").write_bytes(b"NACA 0012\n1 0\n" + rows + b"1 0\n")


def qprop(root: Path, count: int) -> None:
    qw.workspace(
        root,
        condition="V:sweep,RPM:6000,BETA07:30,RHO:1.2,MU:1.8e-5,A:340",
        sweep=",".join(str(10 + 2 * n) for n in range(count)),
    )


def run(root: Path, *words: str) -> int:
    return main([words[0], "study", "--workspace", str(root), *words[1:]])


@pytest.mark.parametrize(("build", "mesh"), [(xfoil, "profile.dat"), (qprop, "blade.obj")])
def test_the_reads_of_a_large_input_do_not_grow_with_the_points(tmp_path, opens, build, mesh):
    counts = []
    for index, count in enumerate((1, 5)):
        root = tmp_path / f"w{index}"
        root.mkdir()
        build(root, count)
        opens.clear()
        assert run(root, "run", "--local") == 0
        counts.append(opens.count(mesh))
        opens.clear()
        assert run(root, "collect") == 0
        counts.append(opens.count(mesh))
    assert counts[0] == counts[2] and counts[1] == counts[3], counts
    assert counts[0] <= 3 and counts[1] <= 3, counts

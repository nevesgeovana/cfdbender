"""P0000: a committed reference mesh and opt-in real wrapper acceptance.

CFDBENDER_REAL_PYSU2=1 enables the real 40-iteration inviscid reference case.
The test consumes the committed mesh and does not require gmsh in WSL.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from cfdbender.solvers import su2
from cfdbender.solvers.su2 import _ram

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/su2"
MESH = FIXTURES / "naca0012_small.su2"


def test_p0000_item3g_committed_mesh_contract():
    assert MESH.is_file(), "commit the small SU2 mesh for gmsh-free WSL tests"
    data = MESH.read_bytes()
    assert len(data) < 300_000
    assert b"\r" not in data, "the committed mesh must use portable canonical LF bytes"
    assert all(line == line.rstrip() for line in data.splitlines())
    assert b"NELEM= 1280" in data
    assert b"NDIME= 2" in data
    assert b"MARKER_TAG= airfoil" in data and b"MARKER_TAG= farfield" in data


def test_p0000_item3g_mesh_reproduces_generator(tmp_path):
    pytest.importorskip("gmsh")
    assert MESH.is_file(), "the generator's output must be committed"
    _ram.require_ram(0.25)
    spec = importlib.util.spec_from_file_location("tiny_case", FIXTURES / "tiny_case.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = tmp_path / "reproduced.su2"
    assert module.generate(generated) == 1280
    assert b"\r" not in generated.read_bytes(), "generator output must be platform independent"
    assert all(line == line.rstrip() for line in generated.read_bytes().splitlines())
    if sys.platform == "win32":
        # The fixture was generated on Windows with the pinned gmsh: exact bytes.
        assert generated.read_bytes() == MESH.read_bytes()
    else:
        # gmsh's Delaunay mesher triangulates differently across platforms (first hosted
        # Linux run differed in connectivity at byte 22978), so require a strict signature.
        assert _signature(generated.read_bytes()) == _signature(MESH.read_bytes())
    _ram.require_ram(0)


def test_p0000_cifix3_signature_rejects_a_different_mesh():
    data = MESH.read_bytes()
    reference = _signature(data)
    assert reference["NELEM"] == 1280 and reference["markers"] and reference["types"]
    lines = data.decode("ascii").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("NELEM="))
    dropped = [*lines[:start], "NELEM= 1279", *lines[start + 1 : start + 1280]]
    dropped += lines[start + 1281 :]
    assert _signature("\n".join(dropped).encode()) != reference
    renamed = data.replace(b"MARKER_TAG= airfoil", b"MARKER_TAG= wing")
    assert _signature(renamed) != reference


def _signature(data: bytes) -> dict[str, object]:
    """Header counts, element type histogram and marker sizes of an SU2 mesh."""
    lines = data.decode("ascii").splitlines()
    signature: dict[str, object] = {}
    types: dict[str, int] = {}
    markers: dict[str, int] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(("NDIME=", "NELEM=", "NPOIN=", "NMARK=")):
            key, value = line.split("=")
            signature[key] = int(value)
            if key == "NELEM":
                for row in lines[index + 1 : index + 1 + int(value)]:
                    types[row.split()[0]] = types.get(row.split()[0], 0) + 1
        elif line.startswith("MARKER_TAG="):
            markers[line.split("=")[1].strip()] = int(lines[index + 1].split("=")[1])
        index += 1
    signature["types"] = types
    signature["markers"] = markers
    return signature


@pytest.mark.skipif(
    os.environ.get("CFDBENDER_REAL_PYSU2") != "1",
    reason="set CFDBENDER_REAL_PYSU2=1 with a Linux/WSL wrapper install",
)
def test_p0000_item3g_real_pysu2_naca0012(tmp_path):
    from cfdbender.core.adapter import load_point
    from cfdbender.core.run import execute_plan
    from cfdbender.solvers.su2._reference import prepare

    _ram.require_ram(0.25)
    proposal = prepare(tmp_path / "real", MESH, "pysu2")
    records = execute_plan(proposal, su2)
    point = load_point(proposal.root, proposal.points[0], solver="su2")
    folder = proposal.root / point.point.directory
    result = su2.collect(point)
    print("real pysu2:", result.classification, "history:", folder / "history.csv")
    assert result.classification == "iteration_limit"
    assert records[0].classification == result.classification
    assert result.to_dict()["route"] == "pysu2"
    assert (folder / "history.csv").is_file() and len(result.history) == 40
    assert {"CL", "CD"} <= result.coefficients.keys()

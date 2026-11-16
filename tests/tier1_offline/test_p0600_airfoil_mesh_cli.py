"""P0600 airfoil-2d: the cfd-mesh airfoil-2d verb, its output folder and its provenance sidecar.

``cfd-mesh airfoil-2d --naca 0012 --reynolds 6e6`` writes ``inputs/mesh/naca0012.su2`` and
``inputs/mesh/naca0012.msh`` in the workspace, each with a ``<file name>.provenance.json``
sidecar that records the command, the sources with their SHA-256, every option after defaults,
the measured quality and the SHA-256 of the mesh. A refusal exits 2, names the fault and
writes nothing.
"""

from __future__ import annotations

import argparse
import json

import pytest
from kernel_support import kernel_or_skip

from cfdbender._digest import file_sha256
from cfdbender.geometry._airfoils import normalized_naca
from cfdbender.mesh import _airfoil2d as af
from cfdbender.mesh.cli import build_parser, main


def test_p0600_airfoil_is_registered_in_the_public_mesh_parser():
    parser = build_parser()
    verbs = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert "airfoil-2d" in verbs.choices


def _run(argv):
    return main(argv)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    kernel_or_skip()
    root = tmp_path_factory.mktemp("ws")
    code = _run(["--workspace", str(root), "airfoil-2d", "--naca", "0012", "--reynolds", "6e6"])
    return code, root


def test_p0600_verb_writes_both_meshes_into_inputs_mesh(made):
    code, root = made
    assert code == 0
    names = sorted(p.name for p in (root / "inputs/mesh").iterdir())
    assert names == [
        "naca0012.msh",
        "naca0012.msh.provenance.json",
        "naca0012.su2",
        "naca0012.su2.provenance.json",
    ]


def test_p0600_sidecar_records_every_parameter_and_the_hashes(made):
    _, root = made
    for kind in ("su2", "msh"):
        mesh = root / f"inputs/mesh/naca0012.{kind}"
        record = json.loads((root / f"inputs/mesh/naca0012.{kind}.provenance.json").read_text())
        assert record["schema"] == "cfdbender-mesh-provenance/1"
        assert record["command"] == "cfd-mesh airfoil-2d"
        assert record["mesh"] == f"inputs/mesh/naca0012.{kind}"
        assert record["sha256"] == file_sha256(mesh)
        assert record["sources"] == []
        options = record["options"]
        assert options["naca"] == "0012"
        assert options["reynolds"] == 6e6
        assert options["yplus"] == 1.0
        assert options["first_cell"] == pytest.approx(af.first_cell(1.0, 6e6), rel=1e-12)
        assert options["level"] == "coarse"
        assert (options["wall_cells"], options["normal_cells"], options["wake_cells"]) == (
            64,
            64,
            48,
        )
        assert options["normal_growth"] == pytest.approx(1.3118, abs=2e-4)
        assert options["farfield"] == 500.0
        assert options["trailing_edge"] == "sharp"
        assert options["gap_removed"] > 0.0025
        assert options["span"] == 1.0
        assert options["gmsh"]
        outputs = record["outputs"]
        assert outputs["cells"] == 224 * 64
        assert outputs["duplicate_nodes"] == 0
        assert outputs["unmarked_boundary_edges"] == 0
        assert set(outputs["files"]) == {"su2", "msh"}
        for other, path in outputs["files"].items():
            assert outputs["sha256"][other] == file_sha256(root / path)


def test_p0600_coordinates_file_is_recorded_with_its_hash(tmp_path):
    kernel_or_skip()
    upper, lower, _ = normalized_naca("0012", count=301)
    section = tmp_path / "inputs/sections/n0012.dat"
    section.parent.mkdir(parents=True)
    rows = [*upper[::-1], *lower[1:]]
    section.write_text("".join(f"{x:.17g} {y:.17g}\n" for x, y in rows), encoding="utf-8")
    argv = ["--workspace", str(tmp_path), "airfoil-2d", "--coordinates", str(section)]
    argv += ["--reynolds", "1e6", "--trailing-edge", "blunt", "--out", "blunt0012"]
    assert _run(argv) == 0
    record = json.loads((tmp_path / "inputs/mesh/blunt0012.su2.provenance.json").read_text())
    assert record["sources"] == [
        {"path": "inputs/sections/n0012.dat", "sha256": file_sha256(section)}
    ]
    assert record["options"]["trailing_edge"] == "blunt"
    assert record["options"]["coordinates"] == "inputs/sections/n0012.dat"
    assert record["options"]["first_cell"] == pytest.approx(af.first_cell(1.0, 1e6), rel=1e-12)


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (["--naca", "9"], "NACA 4 or 5 digit"),
        (["--naca", "0012", "--yplus", "0.05", "--reynolds", "1e9"], "1.35"),
        (["--naca", "0012", "--out", "wing.stl"], "suffix"),
        (["--coordinates", "absent.dat"], "not found"),
    ],
)
def test_p0600_refusal_exits_2_and_writes_nothing(tmp_path, capsys, extra, named):
    argv = ["--workspace", str(tmp_path), "airfoil-2d", *extra]
    if "--reynolds" not in extra:
        argv += ["--reynolds", "6e6"]
    assert _run(argv) == 2
    assert named in capsys.readouterr().err
    assert not (tmp_path / "inputs/mesh").exists() or not any((tmp_path / "inputs/mesh").iterdir())


def test_p0600_reynolds_is_required(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run(["--workspace", str(tmp_path), "airfoil-2d", "--naca", "0012"])
    assert "--reynolds" in capsys.readouterr().err


def test_p0600_one_section_input_is_required(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run(["--workspace", str(tmp_path), "airfoil-2d", "--reynolds", "6e6"])
    with pytest.raises(SystemExit):
        argv = ["--workspace", str(tmp_path), "airfoil-2d", "--reynolds", "6e6"]
        _run([*argv, "--naca", "0012", "--coordinates", "a.dat"])

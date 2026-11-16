"""Remesh an OBJ sector through the generic surface remesher, without a solver.

Usage: cfd-mesh --workspace campaign sector-remesh sector.obj 1.5 --quiet
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import sheet_with_strips, write_source

from cfdbender.mesh._refine._obj import read_obj
from cfdbender.mesh.cli import main


def _invoke(*args):
    try:
        return main(list(args))
    except SystemExit as error:
        return error.code


def _source(tmp_path):
    return write_source(tmp_path / "original", "sector", sheet_with_strips(()))


@pytest.mark.parametrize("factor", ["1", "2", "0.5"])
def test_p060_generic_rest_sector_remesh_uses_remesh_even_for_a_grid(tmp_path, capsys, factor):
    source = _source(tmp_path)
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    workspace = tmp_path / "campaign"
    code = _invoke("--workspace", str(workspace), "sector-remesh", str(source), factor, "--quiet")
    assert code == 0, capsys.readouterr().err
    report = json.loads(capsys.readouterr().out)
    assert report["families"] == {"G": "remesh"}
    obj = Path(report["obj"])
    assert obj.is_relative_to(workspace / "inputs" / "mesh")
    assert Path(str(obj) + ".boundaries.toml").is_file()
    mesh = read_obj(obj)
    assert set(mesh.families) == {"G"}
    assert all(len(face) == 3 for face in mesh.families["G"])
    assert np.isfinite(mesh.verts).all()
    source_count = sum(map(len, read_obj(source).families.values()))
    if float(factor) < 1:
        assert len(mesh.families["G"]) < source_count
    elif float(factor) > 1:
        assert len(mesh.families["G"]) > source_count
    provenance = json.loads(Path(report["provenance"]).read_text(encoding="utf-8"))
    assert provenance["command"] == "cfd-mesh sector-remesh"
    assert provenance["options"]["factor"] == float(factor)
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before


def test_p060_generic_rest_sector_remesh_honors_element_mode_and_explicit_folder(tmp_path, capsys):
    source = _source(tmp_path)
    config = tmp_path / "remesh.toml"
    config.write_text('[refine]\nelements = "quad-dominant"\n', encoding="utf-8")
    target = tmp_path / "levels"
    assert (
        _invoke(
            "sector-remesh",
            str(source),
            "1",
            "--config",
            str(config),
            "--out",
            str(target),
            "--quiet",
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    obj = Path(report["obj"])
    assert obj.parent.parent == target
    assert report["families"] == {"G": "remesh"}
    assert any(len(face) == 4 for face in read_obj(obj).families["G"])


@pytest.mark.parametrize("factor", ["0", "-1", "nan"])
def test_p060_generic_rest_sector_remesh_refuses_bad_factors(tmp_path, capsys, factor):
    source = _source(tmp_path)
    target = tmp_path / "levels"
    assert _invoke("sector-remesh", str(source), factor, "--out", str(target), "--quiet") == 2
    assert "factor" in capsys.readouterr().err.lower()
    assert not list(target.rglob("*.obj"))


def test_p060_generic_rest_sector_remesh_refuses_an_existing_level(tmp_path, capsys):
    source = _source(tmp_path)
    args = ("sector-remesh", str(source), "1", "--out", str(tmp_path / "levels"), "--quiet")
    assert _invoke(*args) == 0
    report = json.loads(capsys.readouterr().out)
    obj = Path(report["obj"])
    before = obj.read_bytes()
    assert _invoke(*args) == 2
    assert "overwrite" in capsys.readouterr().err
    assert obj.read_bytes() == before


def test_p060_generic_rest_sector_remesh_keeps_unselected_family_coordinates(tmp_path, capsys):
    source = write_source(
        tmp_path / "original", "sector", sheet_with_strips((("S", "quad", (0.4, 0.4)),))
    )
    assert (
        _invoke(
            "sector-remesh",
            str(source),
            "2",
            "--families",
            "G",
            "--out",
            str(tmp_path / "levels"),
            "--quiet",
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    before, after = read_obj(source), read_obj(report["obj"])

    def coordinates(mesh):
        return [[tuple(mesh.verts[v]) for v in face] for face in mesh.families["S"]]

    assert coordinates(after) == coordinates(before)

"""Derived OBJ commands write into the selected workspace, leaving external inputs alone."""

import json
from pathlib import Path

from mesh_fixtures import sheet_with_strips, write_source
from test_mesh_thin_blade import _blade, _write_obj

from cfdbender.mesh.cli import main


def test_p060_generic_rest_prepared_mesh_lands_in_workspace(tmp_path, capsys):
    source = write_source(tmp_path / "original", "wing", sheet_with_strips(()))
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    workspace = tmp_path / "campaign"
    assert main(["--workspace", str(workspace), "prepare-to-refine", str(source)]) == 0
    record = json.loads(capsys.readouterr().out)
    obj = workspace / "inputs/mesh/wing_patches.obj"
    assert str(obj) in record["files"]
    assert obj.is_file()
    assert obj.with_suffix(".refine.toml").is_file()
    assert Path(record["provenance"]).parent == obj.parent
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before


def test_p060_generic_rest_thin_blade_lands_in_workspace(tmp_path, capsys):
    folder = tmp_path / "original"
    folder.mkdir()
    source = _write_obj(folder / "blade.obj", *_blade())
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    workspace = tmp_path / "campaign"
    assert (
        main(["--workspace", str(workspace), "degenerate", str(source), "--root-offset", "0.05"])
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    obj = workspace / "inputs/mesh/blade_thin_blade.obj"
    assert Path(record["mesh"]) == obj
    assert obj.is_file()
    assert Path(record["inventory"]).is_file()
    assert Path(record["provenance"]).parent == obj.parent
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before

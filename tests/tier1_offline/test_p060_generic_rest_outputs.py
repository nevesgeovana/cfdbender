"""Refinement levels belong to the workspace even when the source lives elsewhere.

Usage: cfd-mesh --workspace campaign refine original.obj 0.5
"""

import json
from pathlib import Path

from mesh_fixtures import sheet_with_strips, write_source

from cfdbender.mesh.cli import main


def test_p060_generic_rest_refine_defaults_to_workspace_mesh_folder(tmp_path, capsys):
    source = write_source(tmp_path / "original", "wing", sheet_with_strips(()))
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    workspace = tmp_path / "campaign"
    assert main(["--workspace", str(workspace), "refine", str(source), "0.5", "--quiet"]) == 0
    record = json.loads(capsys.readouterr().out)
    folder = workspace / "inputs" / "mesh" / "wing_R0p5"
    assert Path(record["level"]) == folder
    assert (folder / "wing_R0p5.obj").is_file()
    provenance = json.loads(Path(record["provenance"]).read_text(encoding="utf-8"))
    assert provenance["mesh"] == "inputs/mesh/wing_R0p5/wing_R0p5.obj"
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before

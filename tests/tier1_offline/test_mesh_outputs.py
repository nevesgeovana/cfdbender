"""Where a cfd-mesh verb writes, and the sidecar that records how each mesh was made."""

from __future__ import annotations

import json
from pathlib import Path

from cfdbender.core.mesh import MESH_FOLDER, mesh_candidates
from cfdbender.mesh._cli_support import (
    PROVENANCE_SUFFIX,
    mesh_target,
    provenance_path,
    write_provenance,
)


def test_mesh_outputs_a_bare_name_lands_in_the_workspace_mesh_folder(tmp_path):
    target = mesh_target(tmp_path, "wing-fine.obj")
    assert target == tmp_path / "inputs" / "mesh" / "wing-fine.obj"
    assert target.parent.is_dir()


def test_mesh_outputs_a_path_with_a_folder_is_used_as_written(tmp_path):
    named = tmp_path / "elsewhere" / "wing.obj"
    assert mesh_target(tmp_path, str(named)) == named
    assert mesh_target(tmp_path, "sub/wing.obj") == tmp_path / "sub" / "wing.obj"


def test_mesh_outputs_the_sidecar_records_command_sources_and_options(tmp_path):
    source = tmp_path / "inputs" / "mesh" / "wing.obj"
    source.parent.mkdir(parents=True)
    source.write_text("v 0 0 0\n", encoding="utf-8")
    out = source.with_name("wing-fine.obj")
    out.write_text("v 1 0 0\n", encoding="utf-8")
    sidecar = write_provenance(
        out, command="refine", sources=[source], options={"factor": 2.0}, workspace=tmp_path
    )
    assert sidecar == provenance_path(out) == out.with_name("wing-fine.obj" + PROVENANCE_SUFFIX)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["command"] == "cfd-mesh refine"
    assert record["mesh"] == f"{MESH_FOLDER}/wing-fine.obj"
    assert record["sources"] == [
        {"path": f"{MESH_FOLDER}/wing.obj", "sha256": record["sources"][0]["sha256"]}
    ]
    assert len(record["sources"][0]["sha256"]) == 64
    assert record["options"] == {"factor": 2.0}
    assert len(record["sha256"]) == 64
    assert record["cfdbender"]


def test_mesh_outputs_a_sidecar_is_never_offered_as_a_mesh(tmp_path):
    folder = tmp_path / "inputs" / "mesh"
    folder.mkdir(parents=True)
    (folder / "wing.obj").write_text("v 0 0 0\n", encoding="utf-8")
    write_provenance(
        folder / "wing.obj", command="mesh", sources=[], options={}, workspace=tmp_path
    )
    assert mesh_candidates(tmp_path, "wing") == [f"{MESH_FOLDER}/wing.obj"]


def test_mesh_outputs_paths_outside_the_workspace_are_recorded_absolute(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    out = outside / "m.obj"
    out.write_text("v 0 0 0\n", encoding="utf-8")
    sidecar = write_provenance(
        out, command="scale", sources=[], options={}, workspace=tmp_path / "ws"
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert Path(record["mesh"]).is_absolute()

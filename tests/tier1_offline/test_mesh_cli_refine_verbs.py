"""Tier 1: the cfd-mesh verbs of the refine group on a small OBJ.

Each verb is run through ``main([...])``: it exits 0 or 2, prints its record as JSON, writes
what it names and leaves a ``<file name>.provenance.json`` beside the main output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pytest
from mesh_fixtures import sheet_with_strips, write_source

from cfdbender.mesh import cli
from cfdbender.mesh._refine._obj import read_node_list, read_obj


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    output = ["--out", str(Path(argv[1]).parent)] if argv[0] == "prepare-to-refine" else []
    code = cli.main([*argv, *output])
    seen = capsys.readouterr()
    return code, seen.out, seen.err


def _source(tmp_path: Path, *, strips=()) -> Path:
    return write_source(tmp_path / "src", "wing", sheet_with_strips(strips))


def test_inventory_writes_the_counts_of_each_group_and_never_rewrites(tmp_path, capsys):
    mesh = tmp_path / "wing.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\ng A\nf 1 2 3\ng B\nf 1 2 3\nf 1 3 2\n")
    code, out, err = _run(capsys, "inventory", str(mesh))
    assert code == 0, err
    record = json.loads(out)
    sidecar = tmp_path / "wing.obj.boundaries.toml"
    assert record["inventory"] == str(sidecar) and Path(record["provenance"]).is_file()
    text = sidecar.read_text(encoding="utf-8")
    assert '"A"' in text and "mesh_faces = 3" in text and "boundary_faces = [1, 2]" in text
    before = sidecar.read_bytes()
    code, out, err = _run(capsys, "inventory", str(mesh), "--overwrite")
    assert code == 2 and out == "" and "never rewritten" in err
    assert sidecar.read_bytes() == before


def test_inventory_refuses_a_file_that_is_not_an_obj(tmp_path, capsys):
    stl = tmp_path / "wing.stl"
    stl.write_text("solid w\nendsolid w\n", encoding="utf-8")
    code, out, err = _run(capsys, "inventory", str(stl))
    assert code == 2 and out == "" and "not an OBJ" in err


def test_refine_writes_the_level_under_out_with_a_provenance_record(tmp_path, capsys):
    src = _source(tmp_path)
    code, out, err = _run(capsys, "refine", str(src), "2", "--out", str(tmp_path / "levels"))
    assert code == 0, err
    record = json.loads(out)
    level = tmp_path / "levels" / "wing_R2"
    assert record["level"] == str(level) and record["families"] == {"G": "grid"}
    assert (level / "wing_R2.obj").is_file() and (level / "wing_R2.audit.json").is_file()
    provenance = json.loads(Path(record["provenance"]).read_text(encoding="utf-8"))
    assert provenance["command"] == "cfd-mesh refine"
    assert provenance["options"]["factor"] == 2.0
    assert [s["path"] for s in provenance["sources"]][0].endswith("wing.obj")


def test_refine_with_a_factor_below_one_coarsens(tmp_path, capsys):
    src = _source(tmp_path)
    code, out, err = _run(capsys, "refine", str(src), "0.5", "--out", str(tmp_path / "levels"))
    assert code == 0, err
    level = Path(json.loads(out)["level"])
    before = sum(len(f) for f in read_obj(src).families.values())
    after = sum(len(f) for f in read_obj(level / "wing_R0p5.obj").families.values())
    assert after < before


def test_refine_without_a_factor_or_a_file_is_refused(tmp_path, capsys):
    src = _source(tmp_path)
    code, out, err = _run(capsys, "refine", str(src), "--out", str(tmp_path / "levels"))
    assert code == 2 and out == "" and "no factor was given" in err
    assert not (tmp_path / "levels").exists() or not any((tmp_path / "levels").iterdir())


def test_prepare_to_refine_writes_patches_a_refine_file_and_a_node_list(tmp_path, capsys):
    src = _source(tmp_path, strips=(("S", "quad", (0.4, 0.4)),))
    code, out, err = _run(capsys, "prepare-to-refine", str(src))
    assert code == 0, err
    record = json.loads(out)
    assert list(record["patches"]) == ["G", "S"] or "G" in record["patches"]
    obj = src.with_name("wing_patches.obj")
    assert str(obj) in record["files"] and Path(record["provenance"]).is_file()
    patches = read_obj(obj)
    assert sum(len(f) for f in patches.families.values()) == sum(
        len(f) for f in read_obj(src).families.values()
    )
    nodes = read_node_list(src.with_name("wing_patches.nodes.txt"))
    assert len(nodes.ids) > 0 and numpy.all(numpy.isfinite(nodes.points))
    assert src.with_name("wing_patches.refine.toml").is_file()
    code, out, err = _run(capsys, "prepare-to-refine", str(src))
    assert code == 2 and "overwrite" in err


def test_audit_writes_the_audit_beside_the_mesh_and_records_it(tmp_path, capsys):
    src = _source(tmp_path)
    code, out, err = _run(capsys, "audit", str(src))
    record = json.loads(out)
    assert code in (0, 1) and (record["passed"] is (code == 0))
    assert str(src.with_suffix(".audit.json")) in record["files"]
    assert Path(record["provenance"]).name == "wing.audit.json.provenance.json"


def test_audit_of_a_missing_mesh_is_refused(tmp_path, capsys):
    code, out, err = _run(capsys, "audit", str(tmp_path / "none.obj"))
    assert code == 2 and out == "" and "no such OBJ file" in err


@pytest.mark.parametrize(
    "verb", ["refine", "prepare-to-refine", "audit", "inventory", "degenerate", "merge"]
)
def test_every_verb_is_listed_with_a_help_text(verb, capsys):
    with pytest.raises(SystemExit) as helped:
        cli.main([verb, "--help"])
    assert helped.value.code == 0
    assert verb in capsys.readouterr().out

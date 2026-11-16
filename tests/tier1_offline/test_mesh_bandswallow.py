"""a neighbour the band covers whole is rebuilt whole at factor 1.

Every source is synthetic (tests/p0380_mesh_fixtures.py): the cambered sheet
grid ``G`` with its trailing-edge points file, and ``T``, a strip of
triangles sharing G's tip station node for node. G is refined at factor 2
with T unselected, so T's curve with G changes and T is an unchanged
neighbour whose band of two face layers is remeshed. A strip of
two rows is all band: 0.38.0 cut an empty remainder from it and raised on the
trailing-edge lookup of that empty family. The control strip has four rows,
so its band leaves a remainder and the cut of stays.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from mesh_fixtures import (
    face_coordinates,
    open_loop_count,
    read_mesh,
    sheet_with_strips,
    write_source,
)

from cfdbender._errors import GeometryWarning
from cfdbender.mesh._refine import refine_mesh

pytest.importorskip("trimesh")

TWO_ROWS = (("T", "tri", (0.03, 0.04)),)
FOUR_ROWS = (("T", "tri", (0.03, 0.04, 0.05, 0.065)),)
STATED = '[families.G]\nfactor = 2\n[families.T]\nfactor = 1\nmethod = "remesh"\n'


def _refine(tmp_path: Path, strips, name: str, *, stated: bool = False):
    """Refine G at factor 2, T unselected or stated at factor 1; return the level and warnings."""
    mesh = sheet_with_strips(strips)
    assert len(mesh.te), "the fixture must carry a trailing-edge points file"
    src = write_source(tmp_path / name, "wing", mesh, refine_toml=STATED if stated else None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if stated:
            config = src.with_suffix(".refine.toml")
            level = refine_mesh(src, config=config, out_dir=tmp_path / f"o{name}")
        else:
            level = refine_mesh(src, 2, families=["G"], out_dir=tmp_path / f"o{name}")
    rebuilt = [
        str(w.message)
        for w in caught
        if issubclass(w.category, GeometryWarning) and "rebuilt whole" in str(w.message)
    ]
    return mesh, src, level, rebuilt


def _record(level) -> dict:
    return json.loads(next(Path(level.folder).glob("*.refine.json")).read_text(encoding="utf-8"))


def _gate(level, name: str):
    return next(item for item in level.audit.gates if item.name == name)


def test_mesh_bandswallow_bandswallow_two_rows_are_rebuilt_whole_and_warned(tmp_path):
    """a neighbour two layers deep: a level and one warning."""
    mesh, _, level, rebuilt = _refine(tmp_path, TWO_ROWS, "swallow")
    assert level.obj.is_file()
    faces = len(mesh.families["T"])
    assert len(rebuilt) == 1, rebuilt
    message = rebuilt[0]
    assert "family T" in message
    assert f"all {faces} faces" in message
    assert "rebuilt whole at factor 1" in message
    verts, fams = read_mesh(level.obj)
    assert fams["T"], "the rebuilt family is in the level"
    g_nodes = {v for f in fams["G"] for v in f}
    t_nodes = {v for f in fams["T"] for v in f}
    # T's nodes on the shared curve are the grid's new nodes (factor 2 doubles the 19 nodes)
    assert len(g_nodes & t_nodes) == 2 * (len(mesh.rows["G"][-1]) - 1) + 1


def test_mesh_bandswallow_bandswallow_record_states_the_rebuild(tmp_path):
    """refine.json records method, reason, band and faces."""
    mesh, _, level, _ = _refine(tmp_path, TWO_ROWS, "record")
    entry = _record(level)["families"]["T"]
    faces = len(mesh.families["T"])
    assert entry["method"] == "rebuilt_whole"
    assert entry["reason"] == "band_covers_family"
    assert entry["band"] == faces
    assert entry["faces"] == faces
    assert "T" not in _record(level)["specs"], "the user did not state T"


def test_mesh_bandswallow_bandswallow_equals_the_stated_remesh_at_factor_1(tmp_path):
    """the swallowed neighbour equals T stated at factor 1."""
    _, _, swallowed, _ = _refine(tmp_path, TWO_ROWS, "swallowed")
    _, _, stated, rebuilt = _refine(tmp_path, TWO_ROWS, "stated", stated=True)
    assert rebuilt == [], "a stated family is not rebuilt by"
    stated_entry = _record(stated)["families"]["T"]
    assert stated_entry["method"] == "remesh", "a stated family keeps its method"
    v1, f1 = read_mesh(swallowed.obj)
    v2, f2 = read_mesh(stated.obj)
    assert len({v for f in f1["T"] for v in f}) == len({v for f in f2["T"] for v in f})
    assert face_coordinates(v1, f1["T"]) == face_coordinates(v2, f2["T"])
    assert face_coordinates(v1, f1["G"]) == face_coordinates(v2, f2["G"])


def test_mesh_bandswallow_bandswallow_level_keeps_the_guarantees_of_fr425(tmp_path):
    """open loops as the source's, G1 and G3 pass, G as asked."""
    mesh, src, level, _ = _refine(tmp_path, TWO_ROWS, "loops")
    _, src_fams = read_mesh(src)
    _, fams = read_mesh(level.obj)
    assert open_loop_count([f for fs in fams.values() for f in fs]) == open_loop_count(
        [f for fs in src_fams.values() for f in fs]
    )
    assert _gate(level, "G1").verdict == "pass"
    assert _gate(level, "G3").verdict == "pass"
    assert len(fams["G"]) == 4 * len(mesh.families["G"])


def test_mesh_bandswallow_bandswallow_control_four_rows_keep_the_band(tmp_path):
    """four layers deep: cut as, with no warning."""
    mesh, _, level, rebuilt = _refine(tmp_path, FOUR_ROWS, "control")
    assert rebuilt == []
    entry = _record(level)["families"]["T"]
    assert entry["method"] == "unchanged"
    assert 0 < entry["band"] < entry["faces"] == len(mesh.families["T"])
    assert "rebuilt_whole" not in json.dumps(_record(level))

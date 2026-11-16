"""The visual-inspect verb and the VTK preview writers, on a small OBJ."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from cfdbender._errors import InputArtifactError
from cfdbender.mesh._geometry_vtk import write_preview_vtk, write_sweep_vtk
from cfdbender.mesh.cli import build_parser, main

OBJ = """# two patches
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 2 1 0
g Wing
f 1 2 3 4
g Flap
f 2 5 6 3
"""


@pytest.fixture
def obj(tmp_path):
    path = tmp_path / "wing.obj"
    path.write_text(OBJ, encoding="utf-8")
    return path


def test_the_verb_writes_the_pdf_a_record_and_a_provenance(obj, tmp_path, capsys):
    out = tmp_path / "views" / "wing.pdf"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = main(
            ["visual-inspect", str(obj), "--mesh-unit", "METER", "--dpi", "40", "--out", str(out)]
        )
    assert code == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["pdf"] == str(out)
    assert out.read_bytes()[:5] == b"%PDF-"
    assert (tmp_path / "views" / "wing.pdf.provenance.json").is_file()


def test_the_default_pdf_sits_beside_the_mesh_and_an_existing_one_is_refused(obj, capsys):
    argv = ["visual-inspect", str(obj), "--mesh-unit", "METER", "--dpi", "40"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(argv) == 0
        capsys.readouterr()
        assert (obj.parent / "wing.inspect.pdf").is_file()
        assert main(argv) == 2
        assert "--overwrite" in capsys.readouterr().err
        assert main([*argv, "--overwrite"]) == 0
    capsys.readouterr()


def test_the_png_flag_writes_the_pages_of_views(obj, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (
            main(["visual-inspect", str(obj), "--mesh-unit", "METER", "--dpi", "40", "--png"]) == 0
        )
    capsys.readouterr()
    names = sorted(p.name for p in obj.parent.glob("wing.inspect*.png"))
    views = [f"wing.inspect.{v}.png" for v in ("front", "iso", "side", "top")]
    assert names == sorted([*views, "wing.inspect.png"])


def test_a_mesh_that_is_missing_is_refused_with_exit_two(tmp_path, capsys):
    assert main(["visual-inspect", str(tmp_path / "none.obj")]) == 2
    assert capsys.readouterr().err


def test_the_nodes_and_te_flags_can_repeat():
    args = build_parser().parse_args(
        ["visual-inspect", "wing.obj", "--te", "a", "--te", "b", "--nodes", "n", "m"]
    )
    assert [str(x) for x in args.te] == ["a", "b"]
    assert [str(x) for x in args.nodes] == ["n", "m"]


def test_the_preview_vtk_colours_each_family_by_its_position(obj, tmp_path):
    target = write_preview_vtk(tmp_path / "wing.preview.vtk", obj=obj, names=["Wing", "Flap"])
    lines = Path(target).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# vtk DataFile")
    assert "POINTS 6 double" in lines
    assert "POLYGONS 2 10" in lines
    start = lines.index("LOOKUP_TABLE default")
    assert lines[start + 1 : start + 3] == ["1", "2"]
    assert lines[-3:] == ["names 1 2 string", "Wing", "Flap"]


def test_the_sweep_vtk_overlays_one_surface_of_each_variant(obj, tmp_path):
    target = write_sweep_vtk(tmp_path / "a.sweep.vtk", [(1, "a", obj), (3, "c", obj)], "Flap")
    assert target is not None
    text = Path(target).read_text(encoding="utf-8")
    assert "POLYGONS 2 10" in text
    assert "SCALARS variant int 1" in text
    assert write_sweep_vtk(tmp_path / "none.sweep.vtk", [], "Flap") is None


def test_a_sweep_variant_without_the_surface_is_refused(obj, tmp_path):
    with pytest.raises(InputArtifactError, match="no faces of control surface Aileron"):
        write_sweep_vtk(tmp_path / "x.sweep.vtk", [(1, "a", obj)], "Aileron")

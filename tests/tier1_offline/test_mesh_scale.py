"""P0570-GSCALE: scale an OBJ mesh uniformly into a new output folder ().

Usage: cfd-mesh scale 0.5 --mesh part.obj --out part_half.obj
"""

import hashlib
import json
from pathlib import Path

import pytest

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.mesh.cli import main

OBJ = (
    b"# mesh with comments, v 9 9 9\n"
    b"mtllib part.mtl\n"
    b"g wing\n"
    b"v 1.0 2.0 3.0\n"
    b"v -1 0 4\n"
    b"v 3 -2 0\n"
    b"vn 0 0 1\n"
    b"vt 0.5 0.5\n"
    b"f 1//1 2//1 3//1\n"
    b"g tail\n"
    b"v 2 2 2 \n"
    b"f 4 3 2\n"
    b"s off\n"
)


@pytest.fixture
def mesh(tmp_path):
    path = tmp_path / "part.obj"
    path.write_bytes(OBJ)
    return path


def vertices(data):
    rows = [ln.split() for ln in data.decode().splitlines() if ln.startswith("v ")]
    return [[float(c) for c in r[1:4]] for r in rows]


def non_vertex(data):
    return [ln for ln in data.split(b"\n") if not ln.startswith(b"v ")]


def test_factor_one_half_halves_every_coordinate_and_the_box(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    out = tmp_path / "half"
    record = scale_mesh(mesh, 0.5, out=out)
    before, after = vertices(OBJ), vertices((out / "part.obj").read_bytes())
    assert after == [[c * 0.5 for c in v] for v in before]
    assert record["bounding_box_before"] == {"min": [-1.0, -2.0, 0.0], "max": [3.0, 2.0, 4.0]}
    assert record["bounding_box_after"] == {"min": [-0.5, -1.0, 0.0], "max": [1.5, 1.0, 2.0]}


def test_everything_but_the_vertex_lines_is_byte_identical(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    out = tmp_path / "o"
    scale_mesh(mesh, 2, out=out)
    data = (out / "part.obj").read_bytes()
    assert non_vertex(data) == non_vertex(OBJ)
    assert data.count(b"\n") == OBJ.count(b"\n")
    assert b"# mesh with comments, v 9 9 9\n" in data
    assert b"v 4.0 4.0 4.0 \n" in data


def test_crlf_line_ends_survive(tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    src = tmp_path / "w.obj"
    src.write_bytes(b"g a\r\nv 1 1 1\r\nf 1 1 1\r\n")
    scale_mesh(src, 3, out=tmp_path / "o")
    assert (tmp_path / "o" / "w.obj").read_bytes() == b"g a\r\nv 3.0 3.0 3.0\r\nf 1 1 1\r\n"


def test_about_scales_about_that_point(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    record = scale_mesh(mesh, 0.5, out=tmp_path / "o", about="1,2,3")
    got = vertices((tmp_path / "o" / "part.obj").read_bytes())
    assert got[0] == [1.0, 2.0, 3.0]
    assert got[1] == [0.0, 1.0, 3.5]
    assert record["about"] == [1.0, 2.0, 3.0]


def test_the_input_is_never_modified(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    scale_mesh(mesh, 0.5, out=tmp_path / "o")
    assert mesh.read_bytes() == OBJ


@pytest.mark.parametrize("bad", [0, 0.0, "0", -1, "-0.5", "abc", "", "nan", "inf", None])
def test_a_bad_factor_is_refused_by_name(mesh, tmp_path, bad):
    from cfdbender.mesh.scale import scale_mesh

    with pytest.raises(InputValueError, match=r"scale: the factor"):
        scale_mesh(mesh, bad, out=tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_a_missing_file_is_refused_by_name(tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    with pytest.raises(InputArtifactError, match=r"scale: the mesh file .* does not exist"):
        scale_mesh(tmp_path / "nope.obj", 0.5, out=tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_an_existing_output_is_refused_unless_overwrite(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    out = tmp_path / "o"
    scale_mesh(mesh, 0.5, out=out)
    first = (out / "part.obj").read_bytes()
    with pytest.raises(InputValueError, match=r"already exists in the output folder.*--overwrite"):
        scale_mesh(mesh, 0.25, out=out)
    assert (out / "part.obj").read_bytes() == first
    scale_mesh(mesh, 0.25, out=out, overwrite=True)
    assert (out / "part.obj").read_bytes() != first


def test_the_input_folder_as_output_never_overwrites_the_input(mesh):
    from cfdbender.mesh.scale import scale_mesh

    for overwrite in (False, True):
        with pytest.raises(InputValueError, match="never overwritten"):
            scale_mesh(mesh, 0.5, out=mesh.parent, overwrite=overwrite)
    assert mesh.read_bytes() == OBJ


def test_the_record_holds_hash_factor_counts_and_boxes(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    out = tmp_path / "o"
    scale_mesh(mesh, "0.5", out=out)
    record = json.loads((out / "part.obj.scale.json").read_text())
    assert record["input_sha256"] == hashlib.sha256(OBJ).hexdigest()
    assert record["factor"] == 0.5
    assert record["about"] == [0.0, 0.0, 0.0]
    assert (record["vertices"], record["faces"]) == (4, 2)
    assert set(record["bounding_box_before"]) == {"min", "max"}


def test_the_default_output_folder_is_beside_the_input(mesh):
    from cfdbender.mesh.scale import scale_mesh

    record = scale_mesh(mesh, 0.5)
    folder = mesh.parent / "part_scaled0.5"
    assert (folder / "part.obj").is_file() and (folder / "part.obj.scale.json").is_file()
    assert Path(record["output"]) == folder / "part.obj"


def test_the_output_folder_is_created_with_its_parents(mesh, tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    scale_mesh(mesh, 0.5, out=tmp_path / "a" / "b" / "c")
    assert (tmp_path / "a" / "b" / "c" / "part.obj").is_file()


def test_the_cli_writes_the_obj_a_record_and_a_provenance(mesh, tmp_path, capsys):
    out = tmp_path / "cli_out" / "half.obj"
    code = main(["scale", "0.5", "--mesh", str(mesh), "--out", str(out)])
    assert code == 0
    shown = capsys.readouterr().out
    assert "bounding_box_before" in shown and "bounding_box_after" in shown
    assert out.is_file()
    assert (tmp_path / "cli_out" / "half.obj.scale.json").is_file()
    assert (tmp_path / "cli_out" / "half.obj.provenance.json").is_file()


def test_the_cli_bare_name_lands_in_the_workspace_mesh_folder(mesh, tmp_path, capsys):
    ws = tmp_path / "ws"
    code = main(["--workspace", str(ws), "scale", "2", "--mesh", str(mesh)])
    assert code == 0
    assert (ws / "inputs" / "mesh" / "part_scaled2.obj").is_file()
    capsys.readouterr()


def test_the_cli_refuses_by_name_with_exit_two(mesh, tmp_path, capsys):
    code = main(["scale", "-0", "--mesh", str(mesh), "--out", str(tmp_path / "o.obj")])
    assert code == 2
    assert "scale: the factor" in capsys.readouterr().err
    code = main(
        ["scale", "0.5", "--mesh", str(tmp_path / "x.obj"), "--out", str(tmp_path / "y.obj")]
    )
    assert code == 2


def test_a_saved_simulation_is_refused_naming_what_is_accepted(tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    sim = tmp_path / "part.fsm"
    sim.write_bytes(b"x")
    with pytest.raises(InputValueError, match="Give an OBJ"):
        scale_mesh(sim, 0.5, out=tmp_path / "o")


def test_r27_scale_moves_indented_vertices_and_preserves_their_prefixes(tmp_path):
    from cfdbender.mesh.scale import scale_mesh

    source = tmp_path / "indent.obj"
    original = b"v 1 0 0\r\n  v 0 2 0\r\n\tv 0 0 3\r\n  f 1 2 3\r\n"
    source.write_bytes(original)
    record = scale_mesh(source, 2, out=tmp_path / "scaled")
    assert (tmp_path / "scaled" / source.name).read_bytes() == (
        b"v 2.0 0.0 0.0\r\n  v 0.0 4.0 0.0\r\n\tv 0.0 0.0 6.0\r\n  f 1 2 3\r\n"
    )
    assert record["vertices"] == 3 and record["faces"] == 1
    assert source.read_bytes() == original

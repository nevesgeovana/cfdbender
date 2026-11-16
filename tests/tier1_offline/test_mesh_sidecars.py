"""Tier 1: the sidecar beside an OBJ, its boundary inventory and its declarations.

What these tests hold:

* the reader of an OBJ's group names: one boundary per ``o`` or ``g`` group that holds
  a face, named by the group, in the order of the file, and its refusals of the shapes it
  does not define, each naming its line and the route by hand;
* the inventory written beside an OBJ, with the face counts taken at the same time, and
  that an existing sidecar is never rewritten, one that disagrees with the groups being
  cited with a warning naming both lists;
* the readers of the ``[import]`` table and of the named trailing-edge and base-region
  declarations, with their refusals;
* the TOML writers the verbs share.
"""

from __future__ import annotations

import tomllib
import warnings

import pytest

from cfdbender._digest import file_sha256
from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh import _sidecars as sidecars

#: Three vertices and the face over them, which every group below reuses.
TRIANGLE = "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
FACE = "f 1 2 3\n"

#: The variants of a group list, by the names given. The names are not in alphabetical
#: order, so an order by name and one by file differ.
O_ONE = "o Wing\n" + TRIANGLE + FACE
O_THREE = "o ZETA\n" + TRIANGLE + FACE + "o ALPHA\n" + FACE + "o MID\n" + FACE
O_G = O_THREE.replace("o ", "g ")
O_EMPTY = "o ZETA\n" + TRIANGLE + FACE + "o EMPTY\no ALPHA\n" + FACE + "o MID\n" + FACE

THREE = ("ZETA", "ALPHA", "MID")


def _mesh(tmp_path, text, name="wing.obj"):
    mesh = tmp_path / name
    mesh.write_text(text, encoding="utf-8")
    return mesh


# --- the reader --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "names"),
    [(O_ONE, ("Wing",)), (O_THREE, THREE), (O_G, THREE), (O_EMPTY, THREE)],
    ids=["O_ONE", "O_THREE", "O_G", "O_EMPTY"],
)
def test_mesh_sidecars_names_are_the_groups_holding_a_face_in_file_order(tmp_path, text, names):
    """One boundary per group with a face, named by the group, in file order."""
    assert sidecars.obj_boundary_names(_mesh(tmp_path, text)) == names


@pytest.mark.parametrize(
    ("text", "needles"),
    [
        ("g ZETA\n" + TRIANGLE + FACE + "o ALPHA\n" + FACE, ("line 6", "line 1", "`g`", "`o`")),
        (
            "o A\n" + TRIANGLE + FACE + "o B\n" + FACE + "o A\n" + FACE,
            ("line 8", "line 1", "'A'", "two places"),
        ),
        (TRIANGLE + FACE + "o Wing\n" + FACE, ("line 4", "before the first")),
        ("o\n" + TRIANGLE + FACE, ("line 1", "no group")),
        ("o Left Wing\n" + TRIANGLE + FACE, ("line 1", "several words")),
        ("o Wing\n" + TRIANGLE, ("no face",)),
    ],
    ids=[
        "o-and-g-mixed",
        "a-group-in-two-places",
        "a-face-before-the-first-group",
        "a-group-with-no-name",
        "a-name-of-two-words",
        "no-face",
    ],
)
def test_mesh_sidecars_an_undefined_shape_is_refused_naming_the_line(tmp_path, text, needles):
    """Each shape the reader does not define is refused by name, pointing at the list by hand."""
    mesh = _mesh(tmp_path, text)
    with pytest.raises(InputArtifactError) as caught:
        sidecars.obj_boundary_names(mesh)
    message = str(caught.value)
    for needle in (*needles, "wing.obj", "wing.boundaries.toml", "by hand"):
        assert needle in message, f"the refusal does not name {needle!r}: {message}"


def test_mesh_sidecars_a_comment_on_a_group_line_is_not_part_of_the_name(tmp_path):
    mesh = _mesh(tmp_path, "# exported\no Wing # the main wing\n" + TRIANGLE + FACE)
    assert sidecars.obj_boundary_names(mesh) == ("Wing",)


def test_mesh_sidecars_a_g_statement_of_several_names_uses_the_first_with_a_warning(tmp_path):
    mesh = _mesh(tmp_path, "g Left Right\n" + TRIANGLE + FACE)
    with pytest.warns(GeometryWarning, match="multiple OBJ groups"):
        assert sidecars.obj_boundary_names(mesh) == ("Left",)


def test_mesh_sidecars_face_counts_follow_the_groups_with_a_face(tmp_path):
    mesh = _mesh(tmp_path, O_EMPTY + FACE)
    assert sidecars.obj_face_counts(mesh) == (1, 1, 2)


# --- the inventory -----------------------------------------------------------------


def test_mesh_sidecars_inventory_sidecar_names(tmp_path):
    mesh = tmp_path / "wing.obj"
    assert sidecars.inventory_sidecar(mesh, "obj").name == "wing.obj.boundaries.toml"
    assert sidecars.inventory_sidecar(mesh).name == "wing.boundaries.toml"
    (tmp_path / "wing.obj.boundaries.toml").write_text("boundaries = []\n", encoding="utf-8")
    assert sidecars.inventory_sidecar(mesh).name == "wing.obj.boundaries.toml"


def test_mesh_sidecars_write_inventory_writes_names_and_counts_and_never_rewrites(tmp_path):
    mesh = _mesh(tmp_path, O_EMPTY)
    sidecar = tmp_path / "wing.obj.boundaries.toml"
    assert sidecars.write_inventory(mesh) == sidecar
    # The empty group holds no face and is no boundary.
    assert tomllib.loads(sidecar.read_text(encoding="utf-8")) == {
        "boundaries": list(THREE),
        "mesh_faces": 3,
        "boundary_faces": [1, 1, 1],
        "mesh_sha256": file_sha256(mesh),
    }
    assert sidecars.read_inventory(sidecar) == THREE
    sidecar.write_text('boundaries = ["edited"]\n', encoding="utf-8")
    before = sidecar.read_bytes()
    for overwrite in (False, True):
        with pytest.raises(InputArtifactError, match="never rewritten"):
            sidecars.write_inventory(mesh, overwrite=overwrite)
        assert sidecar.read_bytes() == before


def test_mesh_sidecars_write_inventory_refuses_what_is_not_an_obj(tmp_path):
    stl = tmp_path / "wing.stl"
    stl.write_text("solid Wing\nendsolid Wing\n", encoding="utf-8")
    with pytest.raises(InputArtifactError, match="not an OBJ"):
        sidecars.write_inventory(stl)
    with pytest.raises(InputArtifactError, match="is not a file"):
        sidecars.write_inventory(tmp_path / "missing.obj")


def test_mesh_sidecars_ensure_inventory_keeps_a_sidecar_and_warns_when_it_disagrees(tmp_path):
    mesh = _mesh(tmp_path, O_THREE)
    sidecar = sidecars.inventory_sidecar(mesh, "obj")
    sidecar.write_text('boundaries = ["ALPHA", "ZETA", "MID"]\n', encoding="utf-8")
    before = sidecar.read_bytes()
    with pytest.warns(GeometryWarning, match="lists the boundaries of wing.obj"):
        assert sidecars.ensure_inventory(mesh, kind="obj") == sidecar
    assert sidecar.read_bytes() == before
    sidecar.write_text('boundaries = ["ZETA", "ALPHA", "MID"]\n', encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sidecars.ensure_inventory(mesh, kind="obj") == sidecar


def test_mesh_sidecars_ensure_inventory_refuses_a_sidecar_with_no_list(tmp_path):
    mesh = _mesh(tmp_path, O_THREE)
    sidecar = sidecars.inventory_sidecar(mesh, "obj")
    sidecar.write_text("[import]\nunits = 'METER'\n", encoding="utf-8")
    with pytest.raises(InputArtifactError, match="non-empty list of strings"):
        sidecars.ensure_inventory(mesh, kind="obj")


def test_mesh_sidecars_ensure_inventory_leaves_other_files_alone(tmp_path):
    stl = tmp_path / "wing.stl"
    stl.write_text("solid Wing\nendsolid Wing\n", encoding="utf-8")
    assert sidecars.ensure_inventory(stl) == tmp_path / "wing.boundaries.toml"
    assert not (tmp_path / "wing.boundaries.toml").exists()


def test_mesh_sidecars_face_count_lines_need_the_same_bytes_as_the_sealed_digest(tmp_path):
    mesh = _mesh(tmp_path, O_THREE)
    lines = sidecars.face_count_lines(mesh, 3)
    assert f"{sidecars.MESH_FACES_KEY} = 3" in lines
    assert f"{sidecars.BOUNDARY_FACES_KEY} = [1, 1, 1]" in lines
    assert f'{sidecars.MESH_SHA256_KEY} = "{file_sha256(mesh)}"' in lines
    # A boundary count that does not match the groups drops the per-boundary line only.
    fewer = sidecars.face_count_lines(mesh, 2)
    assert not any(line.startswith(sidecars.BOUNDARY_FACES_KEY) for line in fewer)
    # A digest that differs from the sealed one writes nothing.
    assert sidecars.face_count_lines(mesh, 3, sealed="0" * 64) == []
    assert sidecars.face_count_lines(tmp_path / "missing.obj", 3) == []


def test_mesh_sidecars_read_inventory_names_a_bad_file(tmp_path):
    bad = tmp_path / "x.boundaries.toml"
    bad.write_text("not toml [", encoding="utf-8")
    with pytest.raises(InputArtifactError, match="cannot be read as a geometry sidecar"):
        sidecars.read_inventory(bad)


# --- the [import] table ------------------------------------------------------------


def _sidecar(tmp_path, text):
    path = tmp_path / "wing.obj.boundaries.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_mesh_sidecars_read_mesh_import_units_and_operations(tmp_path):
    path = _sidecar(
        tmp_path,
        'boundaries = ["naca"]\n[import]\nunits = "mm"\n[[import.operations]]\n'
        'op = "rename"\nsurface = "naca"\nto = "Wing"\n',
    )
    stated = sidecars.read_mesh_import(path)
    assert stated is not None
    assert stated.units == "MM"
    assert [(o.op, o.surface, o.to) for o in stated.operations] == [("rename", "naca", "Wing")]
    assert stated.names_after_renames(["naca"]) == ("Wing",)
    assert sidecars.read_mesh_import(_sidecar(tmp_path, 'boundaries = ["a"]\n')) is None


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("[import]\nunits = 'M'\nfoo = 1\n", "refused"),
        (
            "[import]\nunits = 'M'\n[[import.operations]]\nop = 'scale'\nfactors = [1, 0, 1]\n",
            "operation 1",
        ),
        ("import = 3\n", "as a int"),
        ("[import]\nunits = ' '\n", "empty"),
    ],
)
def test_mesh_sidecars_read_mesh_import_refuses_a_malformed_table(tmp_path, body, needle):
    with pytest.raises(InputArtifactError, match=needle):
        sidecars.read_mesh_import(_sidecar(tmp_path, body))


def test_mesh_sidecars_read_mesh_import_without_units_says_so(tmp_path):
    body = "[import]\n[[import.operations]]\nop = 'mirror'\nsurface = 'a'\nplane = 'XZ'\n"
    with pytest.raises(InputArtifactError, match="does not state `units`"):
        sidecars.read_mesh_import(_sidecar(tmp_path, body))


# --- the declarations --------------------------------------------------------------

BOUNDARIES = 'boundaries = ["Wing", "Base"]\n'


def test_mesh_sidecars_declarations_read_named_entries_in_order(tmp_path):
    path = _sidecar(
        tmp_path,
        BOUNDARIES
        + '[trailing_edges.main]\nfile = "wing.te.txt"\ntype = "STANDARD"\n'
        + '[trailing_edges.aux]\ndetect = "auto"\ntype = "STANDARD"\n'
        + '[base_regions.Base]\ntrailing_edges = "auto"\nmodel = "EMPIRICAL"\n',
    )
    declared = sidecars.read_boundary_declarations(path)
    assert [entry.label for entry in declared.trailing_edges] == ["main", "aux"]
    first = declared.trailing_edges[0].marking
    assert first.route == "file"
    assert first.edge_type == "STANDARD"
    assert first.points_file == str(tmp_path / "wing.te.txt")
    assert declared.trailing_edges[1].marking.route == "detect"
    assert [region.family for region in declared.base_regions] == ["Base"]
    assert declared.base_regions[0].trailing_edges == "auto"
    assert not declared.no_trailing_edges


def test_mesh_sidecars_declarations_none_and_empty(tmp_path):
    none = sidecars.read_boundary_declarations(
        _sidecar(tmp_path, BOUNDARIES + "[trailing_edges]\nnone = true\n")
    )
    assert none.no_trailing_edges and none.trailing_edges == ()
    empty = sidecars.read_boundary_declarations(_sidecar(tmp_path, BOUNDARIES))
    assert not empty.no_trailing_edges and empty.trailing_edges == () and empty.base_regions == ()


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ('[trailing_edges]\nfile = "a.te.txt"\ntype = "STANDARD"\n', "replace unnamed"),
        ('[base_regions]\ndetect = "auto"\n', "replace"),
        ("[trailing_edges.a]\nfile = 'a.te.txt'\n", "requires type"),
        ("[trailing_edges.a]\ntype = 'STANDARD'\n", "neither file nor detect"),
        ("[trailing_edges.a]\nfile = '../a.te.txt'\ntype = 'STANDARD'\n", "not beside the sidecar"),
        ("[trailing_edges.a]\ndetect = 'auto'\ntype = 'RELAXED'\n", "only type STANDARD"),
        ("[trailing_edges.a]\nfile = 'a'\ntype = 'STANDARD'\nbogus = 1\n", "does not read"),
        ("[base_regions.Nope]\ntrailing_edges = 'auto'\n", "must exist in boundaries"),
        ("[base_regions.Base]\ntrailing_edges = 'auto'\nmerge_into = 'Base'\n", "another existing"),
        ("[trailing_edges]\nnone = true\n[trailing_edges.a]\n", "cannot coexist"),
    ],
)
def test_mesh_sidecars_declarations_refuse_malformed_tables(tmp_path, body, needle):
    with pytest.raises(InputArtifactError, match=needle):
        sidecars.read_boundary_declarations(_sidecar(tmp_path, BOUNDARIES + body))


def test_mesh_sidecars_read_trailing_edges_detect_forms(tmp_path):
    path = tmp_path / "wing.obj.boundaries.toml"
    marking = sidecars.read_trailing_edges(
        path, {"detect": {"surfaces": ["Wing"], "sweep_angle": 10}}
    )
    assert marking.route == "detect"
    assert marking.detect_surfaces == ("Wing",)
    assert marking.sweep_angle_deg == 10.0
    assert sidecars.read_trailing_edges(path, {"detect": {"surfaces": "all"}}).detect_surfaces == ()
    with pytest.raises(InputArtifactError, match="neither"):
        sidecars.read_trailing_edges(path, {})
    with pytest.raises(InputArtifactError, match="beside detect"):
        sidecars.read_trailing_edges(path, {"detect": "auto", "type": "STANDARD"})
    with pytest.raises(InputArtifactError, match="tolerance"):
        sidecars.read_trailing_edges(path, {"file": "a", "type": "STANDARD", "tolerance": "x"})


def test_mesh_sidecars_read_trailing_edges_files_list(tmp_path):
    path = tmp_path / "wing.obj.boundaries.toml"
    files = [{"file": "a.te.txt", "type": "STANDARD"}, {"file": "b.te.txt", "type": "RELAXED"}]
    marking = sidecars.read_trailing_edges(path, {"files": files})
    assert [s.edge_type for s in marking.sets] == ["STANDARD", "RELAXED"]
    with pytest.raises(InputArtifactError, match="listed twice"):
        sidecars.read_trailing_edges(path, {"files": [files[0], files[0]]})


# --- the TOML writers --------------------------------------------------------------


def test_mesh_sidecars_toml_values_read_back_as_written():
    table = {
        "name": 'a "quoted" \\ name',
        "flag": True,
        "number": 1.5,
        "list": [1, 2],
        "inline": {"a b": 1},
    }
    text = "\n".join(f"{sidecars.toml_key(k)} = {sidecars.toml_value(v)}" for k, v in table.items())
    assert tomllib.loads(text) == table
    assert sidecars.toml_key("plain-key_1") == "plain-key_1"
    assert sidecars.toml_key("two words") == '"two words"'


def test_mesh_sidecars_toml_table_nests_lists_of_tables():
    lines = sidecars.toml_table(
        "import", {"units": "METER", "operations": [{"op": "rename", "surface": "a", "to": "b"}]}
    )
    assert tomllib.loads("\n".join(lines)) == {
        "import": {
            "units": "METER",
            "operations": [{"op": "rename", "surface": "a", "to": "b"}],
        }
    }

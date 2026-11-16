"""repitch keeps the face order, refuses a rotor with no axis in words.

Usage: cfd-mesh repitch sector.obj --morphing morphing.toml --out pitched.obj
"""

import datetime
import tomllib
import warnings
from pathlib import Path

import pytest
from test_mesh_morph import STATION, build, record
from test_mesh_morph import control as morph_control
from test_mesh_morph import invoke as morph_invoke
from test_mesh_repitch import control, invoke, source_mesh

from cfdbender._errors import GeometryWarning
from cfdbender.mesh._refine._obj import read_obj, read_sidecar
from cfdbender.mesh._sidecars import toml_value


def face_lines(path: Path) -> list[str]:
    """Return the group and face lines of an OBJ, in file order, as written."""
    return [
        " ".join(line.split())
        for line in path.read_text().splitlines()
        if line.split()[:1] in (["g"], ["f"], ["o"])
    ]


def test_mesh_repitch_bugs_keeps_the_face_order_of_an_obj_that_returns_to_a_family(
    tmp_path,
):
    source = source_mesh(tmp_path)
    lines = source.read_text().splitlines()
    first = lines.index("g Blade1")
    # Blade1's last face moves behind the Spinner block, under a second "g Blade1".
    moved = lines.pop(lines.index("g Spinner") - 1)
    assert lines[first] == "g Blade1"
    lines += ["g Blade1", moved]
    source.write_text("\n".join(lines) + "\n")
    out = tmp_path / "pitched.obj"
    assert invoke(source, control(tmp_path, 1.0, 'hub_method = "rings"'), out) == 0
    assert face_lines(out) == face_lines(source)
    assert read_obj(out).families == read_obj(source).families


def test_mesh_repitch_bugs_repitch_of_one_block_per_family_is_written_as_before(tmp_path):
    source = source_mesh(tmp_path)
    out = tmp_path / "pitched.obj"
    assert invoke(source, control(tmp_path, 1.0, 'hub_method = "rings"'), out) == 0
    assert face_lines(out) == face_lines(source)


def _interleaved_hub(folder: Path) -> tuple[Path, list[list[float]]]:
    """The quad-dominant hub of the morph tests, its first face moved behind the nacelle.

    The hub returns after the nacelle under a second ``g Spinner``; returns the OBJ and the
    places of the moved face's nodes.
    """
    source = build(folder, nacelle=True)
    lines = source.read_text().splitlines()
    moved = lines.pop(lines.index("g Spinner") + 1)
    lines += ["g Spinner", moved]
    source.write_text("\n".join(lines) + "\n")
    mesh = read_obj(source)
    return source, [mesh.verts[v].tolist() for v in mesh.families["Spinner"][-1]]


def group_lines(path: Path) -> list[str]:
    """Return the group lines of an OBJ, in file order."""
    return [line for line in face_lines(path) if not line.startswith("f ")]


def test_mesh_repitch_bugs_the_remesh_route_keeps_the_runs_of_a_family_that_returns(
    tmp_path,
):
    """auto takes the remesh and writes the source's runs, not a refusal."""
    source, moved = _interleaved_hub(tmp_path)
    text = f'[repitch]\nangle_deg = 3.0\nrotor_axis = "x"\nreference_point = {STATION}\n'
    out = tmp_path / "pitched.obj"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        assert morph_invoke("repitch", source, morph_control(tmp_path, text), out) == 0
    assert record(out, "repitch")["blades"]["Blade1"]["hub_route"] == "remesh"
    assert group_lines(out) == group_lines(source)
    written = read_obj(out)
    last = [int(v) - 1 for v in face_lines(out)[-1].split()[1:]]
    assert [written.verts[v].tolist() for v in last] == moved
    assert written.families["Nacelle"] == [
        [written.verts.tolist().index(read_obj(source).verts[v].tolist()) for v in f]
        for f in read_obj(source).families["Nacelle"]
    ]


def test_mesh_repitch_bugs_hub_rings_keeps_the_runs_of_a_family_that_returns(tmp_path):
    """hub-rings writes the source's runs of a returning hub, not a refusal."""
    source, moved = _interleaved_hub(tmp_path)
    out = tmp_path / "rings.obj"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        cfg = morph_control(tmp_path, "[hub_rings]\nrings = 2\n")
        assert morph_invoke("hub-rings", source, cfg, out) == 0
    assert group_lines(out) == group_lines(source)
    written = read_obj(out)
    last = [int(v) - 1 for v in face_lines(out)[-1].split()[1:]]
    assert [written.verts[v].tolist() for v in last] == moved


def test_mesh_repitch_bugs_a_control_file_with_a_byte_order_mark_is_read(tmp_path, capsys):
    """a mark is read past; a file that is not TOML is refused by name."""
    source = source_mesh(tmp_path)
    config = control(tmp_path, 1.0)
    config.write_bytes(b"\xef\xbb\xbf" + config.read_bytes())
    assert invoke(source, config, tmp_path / "marked.obj") == 0
    config.write_text("[repitch\nangle_deg = 1.0\n")
    capsys.readouterr()
    assert invoke(source, config, tmp_path / "broken.obj") == 2
    err = capsys.readouterr().err
    assert f"{config}: not TOML" in err and "No file was written" in err
    assert not list(tmp_path.glob("broken.*"))


@pytest.mark.parametrize(
    "extra, words",
    [
        ("angle_deg = [1.0]", "angle_deg must be a number"),
        ("max_angle_deg = [5]", "max_angle_deg must be a number"),
        ("axes = [1]", "repitch.axes must be a table"),
        ("keep = 3", "repitch.keep must be a list"),
        ("reference = 5", "repitch.reference must be a name"),
        ('reference = "rotor.toml"\nrotor = ["R"]', "repitch.rotor must be a name"),
    ],
    ids=["P0560-R1-MORPH-005-" + k for k in ("angle", "max", "axes", "keep", "ref", "rotor")],
)
def test_mesh_repitch_bugs_a_value_of_the_wrong_kind_is_refused_in_words(
    tmp_path, capsys, extra, words
):
    """exit 2 naming the key, not a TypeError or AttributeError."""
    source = source_mesh(tmp_path)
    (tmp_path / "rotor.toml").write_text('[R]\nkind = "rotor"\naxis = "z"\n')
    text = control(tmp_path, 1.0).read_text().replace("angle_deg = 1.0\n", "")
    config = tmp_path / "wrong.toml"
    config.write_text(text.replace("max_angle_deg = 5.0\n", "") + extra + "\n")
    if "angle_deg" not in extra:
        config.write_text(config.read_text() + "angle_deg = 1.0\n")
    assert invoke(source, config, tmp_path / "bad.obj") == 2
    assert words in capsys.readouterr().err
    assert not list(tmp_path.glob("bad.*"))


@pytest.mark.parametrize("key", ["angle_deg", "max_angle_deg"])
def test_mesh_repitch_bugs_003_a_boolean_angle_is_refused_not_read_as_one_degree(
    tmp_path, capsys, key
):
    """P0560-R2-003: angle_deg = true turned the blades one degree and wrote the mesh."""
    source = source_mesh(tmp_path)
    text = control(tmp_path, 1.0).read_text().replace("max_angle_deg = 5.0\n", "")
    if key == "angle_deg":
        text = text.replace("angle_deg = 1.0\n", "")
    config = tmp_path / "wrong.toml"
    config.write_text(text + f"{key} = true\n")
    assert invoke(source, config, tmp_path / "bad.obj") == 2
    assert f"{key} must be a number of degrees, not True" in capsys.readouterr().err
    assert not list(tmp_path.glob("bad.*"))


def test_mesh_repitch_bugs_a_boolean_ring_width_is_refused_naming_the_key(tmp_path, capsys):
    """band_width = true is no length; it is refused, not read as 1."""
    source = build(tmp_path)
    cfg = morph_control(tmp_path, "[hub_rings]\nrings = 2\nband_width = true\n")
    assert morph_invoke("hub-rings", source, cfg, tmp_path / "rings.obj") == 2
    assert "hub_rings.band_width must be a positive length" in capsys.readouterr().err
    assert not list(tmp_path.glob("rings.*"))


def test_mesh_repitch_bugs_repitch_refuses_a_rotor_with_no_axis_in_words(tmp_path, capsys):
    source = source_mesh(tmp_path)
    (tmp_path / "rotor.toml").write_text('[R]\nkind = "rotor"\n')
    config = control(tmp_path, 1.0, 'reference = "rotor.toml"')
    out = tmp_path / "bad.obj"
    assert invoke(source, config, out) == 2
    err = capsys.readouterr().err
    assert "states no 'axis'" in err and "KeyError" not in err
    assert not list(tmp_path.glob("bad.*"))


def test_mesh_repitch_bugs_refine_sidecar_reader_accepts_a_byte_order_mark(tmp_path):
    mesh = source_mesh(tmp_path)
    side = tmp_path / "sector.boundaries.toml"
    side.write_bytes(b"\xef\xbb\xbf" + b'file = "sector.obj"\n')
    found = read_sidecar(mesh)
    assert found is not None and found[1] == {"file": "sector.obj"}


@pytest.mark.parametrize(
    "value",
    [
        datetime.date(2026, 11, 5),
        datetime.datetime(2026, 11, 5, 14, 30, 0),
        datetime.datetime(2026, 11, 5, 14, 30, 0, 250000, tzinfo=datetime.UTC),
        datetime.time(14, 30, 0),
    ],
)
def test_mesh_repitch_bugs_toml_value_writes_a_date_or_time_that_reads_back(value):
    text = toml_value({"when": value, "list": [value]})
    assert tomllib.loads(f"x = {text}")["x"] == {"when": value, "list": [value]}

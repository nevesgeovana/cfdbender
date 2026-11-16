"""Tier 1, 0.34.0: the thin-blade command and its --boundary option.

Pipeline role: quality gate on ``cfd-mesh degenerate --kind thin-blade``: the parser of the family, the command that runs it, its
refusals by name, the selection of the blade among the bodies of one file, and
the two places that must name it (the generated command-line reference and the
cheatsheet). The geometry itself is the library core's, held by
``test_p0340_thin_blade.py``; what is held here is the wiring, so the blades
are the synthetic ones that file builds, and the one saved simulation is the
tier-3 library's pusher, which holds three boundaries in one file.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pytest
from test_mesh_thin_blade import _blade, _read_obj, _sha, _write_obj

from cfdbender._errors import InputArtifactError
from cfdbender.mesh import _degenerate, cli
from cfdbender.mesh._degenerate import derive_thin_blade, thin_blade_path
from cfdbender.mesh._sidecars import read_inventory

#: A box of side 0.3 about the origin: a spinner the blade grows out of.
_BOX = [[x, y, z] for x in (-0.15, 0.15) for y in (-0.15, 0.15) for z in (-0.15, 0.15)]
_BOX_FACES = [
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
]  # fmt: skip


def _with_spinner(path: Path, *, boundary: str = "Blade1") -> Path:
    """Write an OBJ of a spinner first and the blade after it, two groups, shared numbering."""
    vertices, faces = _blade()
    lines = ["o Spinner"]
    lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in _BOX]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in _BOX_FACES]
    lines += [f"o {boundary}"]
    lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in vertices.tolist()]
    lines += [f"f {a + 9} {b + 9} {c + 9}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    """Run ``cfd-mesh`` on ``argv``; return its exit code and what it printed."""
    output = [] if "--out" in argv else ["--out", str(Path(argv[1]).parent)]
    code = cli.main([*argv, *output])
    seen = capsys.readouterr()
    return code, seen.out, seen.err


def test_the_command_derives_the_thin_blade_beside_the_source(tmp_path, capsys):
    """: the command writes what the library writes,
    beside the source, names both files, and leaves the source's bytes alone."""
    # , cfd-mesh degenerate GEOMETRY --kind thin-blade --root-offset.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    before = _sha(source)
    code, out, err = _run(
        capsys, "degenerate", str(source), "--kind", "thin-blade", "--root-offset", "0.05"
    )
    assert code == 0, err
    mesh = thin_blade_path(source)
    record = json.loads(out)
    assert record["mesh"] == str(mesh)
    assert record["inventory"] == str(mesh.with_name("blade_thin_blade.boundaries.toml"))
    assert Path(record["provenance"]).is_file()
    assert "the root moved outward by 0.05" in err
    assert _sha(source) == before and read_inventory(tmp_path / "blade_thin_blade.boundaries.toml")
    # The same bytes as the library function writes for the same offset.
    other = tmp_path / "again"
    other.mkdir()
    twin = _write_obj(other / "blade.obj", vertices, faces)
    assert _sha(derive_thin_blade(twin, root_offset=0.05).mesh) == _sha(mesh)


def test_the_kind_defaults_to_thin_blade_and_no_other_kind_is_accepted(tmp_path, capsys):
    """: thin-blade is the one kind and the default of --kind."""
    # , --kind thin-blade, the default; a later kind is a new choice.
    with pytest.raises(SystemExit) as helped:
        cli.main(["degenerate", "--help"])
    assert helped.value.code == 0
    shown = " ".join(capsys.readouterr().out.split())
    assert "--kind {thin-blade}" in shown and "(default: thin-blade)" in shown
    assert "--root-offset LENGTH" in shown and "--boundary NAME" in shown
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    with pytest.raises(SystemExit) as refused:
        cli.main(["degenerate", str(source), "--kind", "fat-blade", "--root-offset", "0.05"])
    assert refused.value.code == 2
    assert "invalid choice: 'fat-blade'" in capsys.readouterr().err
    assert not thin_blade_path(source).exists()


def test_a_missing_or_unusable_root_offset_is_refused_by_name(tmp_path, capsys):
    """: the offset is required, and one that is not a
    positive length is refused naming the file, with nothing written."""
    # , the command requires --root-offset, a positive length.
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    with pytest.raises(SystemExit) as missing:
        cli.main(["degenerate", str(source)])
    assert missing.value.code == 2 and "--root-offset" in capsys.readouterr().err
    for bad in ("-0.1", "0", "nan"):
        code, out, err = _run(capsys, "degenerate", str(source), "--root-offset", bad)
        assert code == 2 and out == ""
        assert str(source) in err and "not a positive length" in err, err
    code, _, err = _run(capsys, "degenerate", str(source), "--root-offset", "5")
    assert code == 2 and str(source) in err and "a length shorter than the blade" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blade.obj"]


def test_a_file_holding_several_bodies_is_refused_until_the_blade_is_named(tmp_path, capsys):
    """: without --boundary the file with a spinner is
    refused naming its groups and the option; with it, the blade alone is derived, equal to the
    sheet of the blade file alone, under a name that carries the boundary."""
    # , --boundary NAME selects the blade of a file of several bodies.
    source = _with_spinner(tmp_path / "rotor.obj")
    code, out, err = _run(capsys, "degenerate", str(source), "--root-offset", "0.05")
    assert code == 2 and out == ""
    assert "Spinner, Blade1" in err and "--boundary" in err and str(source) in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rotor.obj"]
    code, out, err = _run(
        capsys, "degenerate", str(source), "--root-offset", "0.05", "--boundary", "Blade1"
    )
    assert code == 0, err
    mesh = tmp_path / "rotor_Blade1_thin_blade.obj"
    assert json.loads(out)["mesh"] == str(mesh) == str(thin_blade_path(source, "Blade1"))
    assert read_inventory(mesh.with_name("rotor_Blade1_thin_blade.boundaries.toml")) == ("Blade1",)
    alone = tmp_path / "alone"
    alone.mkdir()
    vertices, faces = _blade()
    reference = derive_thin_blade(
        _write_obj(alone / "rotor.obj", vertices, faces), root_offset=0.05
    )
    assert numpy.allclose(_read_obj(mesh), _read_obj(reference.mesh), rtol=0, atol=1e-12)
    # The name differs only by the boundary, so a second blade of one file does not collide.
    assert thin_blade_path(source) != thin_blade_path(source, "Blade1")
    assert thin_blade_path(source, "Blade 2/x").name == "rotor_Blade_2_x_thin_blade.obj"


@pytest.mark.parametrize("existing", [False, True])
def test_mesh_thin_blade_cli_sidecar_build_failure_preserves_published_mesh(
    tmp_path, monkeypatch, existing
):
    """derive_thin_blade must build both payloads before publishing either."""
    import builtins

    source = _write_obj(tmp_path / "blade.obj", *_blade())
    mesh = thin_blade_path(source)
    sidecar = mesh.with_suffix(".boundaries.toml")
    if existing:
        mesh.write_bytes(b"old mesh")
        sidecar.write_bytes(b"old inventory")

    original_open = builtins.open

    def fail(path, mode="r", *args, **kwargs):
        if Path(path) == source and mode == "rb":
            raise OSError("injected inventory digest failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail)
    with pytest.raises(OSError, match="injected"):
        derive_thin_blade(source, root_offset=0.05, overwrite=existing)
    assert mesh.exists() == existing
    assert sidecar.exists() == existing
    if existing:
        assert mesh.read_bytes() == b"old mesh"
        assert sidecar.read_bytes() == b"old inventory"


def test_mesh_thin_blade_cli_swept_sections_keep_their_chord_direction(tmp_path):
    """spanwise translation cannot swap leading and trailing edges."""
    vertices, faces = _blade()
    vertices[:, 2] *= 100.0
    vertices[vertices[:, 2] > 60.0, 0] += 0.4
    source = _write_obj(tmp_path / "swept.obj", vertices, faces)
    result = derive_thin_blade(source, root_offset=5.0)
    grid = _read_obj(result.mesh).reshape(result.stations, -1, 3)
    assert numpy.all(grid[:, -1, 0] - grid[:, 0, 0] > 0.0)


def test_f_316_an_unused_vertex_of_the_implicit_group_does_not_move_the_blade(tmp_path):
    """F2-0203: F-316, the implicit single group is cut to the vertices its
    faces use, as a named one is, so a stray vertex cannot skew the axis or the stations."""
    vertices, faces = _blade()
    clean = _write_obj(tmp_path / "clean.obj", vertices, faces)
    stray = _write_obj(tmp_path / "stray.obj", vertices, faces)
    with stray.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("v 4.0 -3.0 25.0\n")
    expected = _read_obj(derive_thin_blade(clean, root_offset=0.05).mesh)
    got = _read_obj(derive_thin_blade(stray, root_offset=0.05).mesh)
    assert got.shape == expected.shape
    assert numpy.allclose(got, expected, atol=1e-12)


def test_f_317_an_offset_that_leaves_no_section_beyond_the_root_is_refused(tmp_path):
    """F2-0203: F-317, an offset within a thousand tolerances of the last ring
    would write a sheet with no faces; it is refused before any write."""
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    length = float(vertices[:, 2].max() - vertices[:, 2].min())
    near_tip = length - 100 * _degenerate.SPAN_TOLERANCE * length
    with pytest.raises(InputArtifactError, match="root offset"):
        derive_thin_blade(source, root_offset=near_tip)
    assert [p.name for p in tmp_path.iterdir()] == ["blade.obj"]


def _thin_grid(blade_path: Path, root_offset: float) -> numpy.ndarray:
    """Derive the thin blade of an OBJ and read its vertices as (station, point, xyz)."""
    result = derive_thin_blade(blade_path, root_offset=root_offset)
    rows = [
        [float(word) for word in line.split()[1:4]]
        for line in result.mesh.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    return numpy.asarray(rows).reshape(result.stations, -1, 3)


def test_f_318_the_chord_ends_are_the_farthest_pair_of_the_loop(tmp_path):
    """F2-0203: F-318, three farthest-point sweeps can stop short of the pair."""
    points = numpy.asarray(
        [
            [0.7834221408903144, 0.17032587978181613],
            [-0.057380669636337256, 0.5465540192976328],
            [-0.9393079846750576, 0.413930191311247],
            [-0.2515123330430584, -0.8182945729914843],
            [0.3210001348557896, 0.862927709482709],
            [-0.5856176638379975, 0.26018039957068595],
        ]
    )
    centre = points.mean(axis=0)
    loop = points[numpy.argsort(numpy.arctan2(*(points - centre).T[::-1]))] * 0.1
    spans = [0.2, 0.4, 0.6, 0.8, 1.0]
    vertices = numpy.asarray([[x, y, z] for z in spans for x, y in loop])
    count = len(loop)
    faces = []
    for k in range(len(spans) - 1):
        for i in range(count):
            a, b = k * count + i, k * count + (i + 1) % count
            c, d = a + count, b + count
            faces += [[a, b, d], [a, d, c]]
    source = _write_obj(tmp_path / "blade.obj", vertices, numpy.asarray(faces))
    grid = _thin_grid(source, 0.05)
    farthest = numpy.linalg.norm(loop[:, None] - loop[None], axis=2).max()
    assert numpy.linalg.norm(grid[0, 0] - grid[0, -1]) == pytest.approx(farthest, abs=1e-12)


def test_f_319_distinct_stations_stay_distinct_in_the_obj_at_a_large_origin(tmp_path):
    """F2-0203: F-319, the OBJ text reads back as the same doubles."""
    # RB4-03: at 1e14 the stations differ below the 15th significant digit, so the
    # former 15-digit text wrote fewer distinct vertices (124 of 451, measured).
    vertices, faces = _blade()
    far = vertices + numpy.asarray([0.0, 0.0, 1e14])
    source = _write_obj(tmp_path / "blade.obj", far, faces)
    grid = _thin_grid(source, 0.05).reshape(-1, 3)
    assert len({tuple(row) for row in grid.tolist()}) == len(grid)


def test_f_948_the_same_call_writes_the_same_bytes(tmp_path):
    """F2-0203: identical arguments give byte-identical files."""
    vertices, faces = _blade()
    written = []
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        result = derive_thin_blade(
            _write_obj(folder / "blade.obj", vertices, faces), root_offset=0.05
        )
        written.append((result.mesh.read_bytes(), result.sidecar.read_bytes()))
    assert written[0] == written[1]


def test_f1502_an_oversized_root_offset_is_refused_as_an_input_artifact(tmp_path):
    vertices, faces = _blade()
    source = _write_obj(tmp_path / "blade.obj", vertices, faces)
    with pytest.raises(InputArtifactError, match="not a number"):
        derive_thin_blade(source, root_offset=10**400)
    assert not thin_blade_path(source).exists()


def test_x_f0421_an_obj_group_name_with_a_space_reads_back_whole(tmp_path):
    vertices, faces = _blade()
    lines = ["o my blade"] + [f"v {x!r} {y!r} {z!r}" for x, y, z in vertices.tolist()]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    source = tmp_path / "blade.obj"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    assert derive_thin_blade(source, root_offset=0.05).boundary == "my blade"


def test_a_mesh_that_is_not_an_obj_is_refused_by_name_and_nothing_is_written(tmp_path, capsys):
    stl = tmp_path / "blade.stl"
    stl.write_text("solid blade\nendsolid blade\n", encoding="utf-8")
    code, out, err = _run(capsys, "degenerate", str(stl), "--root-offset", "0.05")
    assert code == 2 and out == ""
    assert str(stl) in err and "read from an OBJ" in err, err
    assert [p.name for p in tmp_path.iterdir()] == ["blade.stl"]
    code, _, err = _run(
        capsys, "degenerate", str(tmp_path / "missing.obj"), "--root-offset", "0.05"
    )
    assert code == 2 and "not a file" in err

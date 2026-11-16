"""and the findings of the independent pre-tag review.

Each test reproduces one finding of the two readings (IND-01 to IND-11 and
IND2-01 to IND2-03) on a synthetic source (tests/p0380_mesh_fixtures.py)
and asserts the requirement it broke.
"""
# ruff: noqa: E501

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import (  # noqa: F401
    HAS_INSPECT,
    Composite,
    Fixture,
    body_of_revolution,
    diag,
    sheet_with_strips,
    tube_pole_quads,
    tube_zipper,
    write_source,
)

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh import cli
from cfdbender.mesh._refine import audit_mesh, refine_mesh
from cfdbender.mesh._refine._grid import Grid, recover_grid, refine_grid


def _source(folder: Path, stem: str = "wing", refine_toml: str | None = None) -> Path:
    """Write the cambered sheet grid ``G`` as ``folder/<stem>.obj`` and return its path."""
    return write_source(folder, stem, sheet_with_strips((), te=True), refine_toml=refine_toml)


def _tree(folder: Path, *, ignore: Path | None = None) -> dict[str, bytes]:
    """Return every file under ``folder`` by its relative path, with its bytes."""
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p != ignore
    }


def _sentinel(folder: Path) -> dict[str, bytes]:
    """Plant an unrelated file in ``folder`` and return the tree to compare later."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "keep.txt").write_bytes(b"not the refinement's")
    return _tree(folder)


# ------------------------------------------------- IND-01 owned folders only


@pytest.mark.parametrize("suffix", [".partial", ".previous"])
def test_ind01_a_source_in_a_reserved_name_is_never_removed(tmp_path, suffix):
    """(IND-01): a source inside ``<level>.partial`` or ``.previous`` stays.

    The source sits in ``study/wing_R1<suffix>/``, so its default level is
    ``study/wing_R1`` and the reserved staging or backup name is the source's
    own folder. Refining at factor 1 (and again with ``overwrite``, which
    moves the first level aside) writes the level and leaves the source's
    folder byte for byte. Control: the level is written.
    """
    src = _source(tmp_path / "study" / f"wing_R1{suffix}")
    before = _tree(src.parent)
    level = refine_mesh(src, 1.0)
    assert level.folder == tmp_path / "study" / "wing_R1" and level.obj.is_file()
    assert src.parent.is_dir() and _tree(src.parent) == before
    level = refine_mesh(src, 1.0, overwrite=True)
    assert level.obj.is_file() and _tree(src.parent) == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_ind01_unrelated_reserved_folders_are_kept_and_the_run_leaves_none(tmp_path, overwrite):
    """(IND-01): unrelated ``.partial`` and ``.previous`` folders stay.

    Both hold a file the refinement did not write. A run (and with
    ``overwrite`` a run replacing an existing level) keeps both byte for byte
    and leaves no folder of its own beside the level. Control: the level is
    published, and with ``overwrite`` a file planted in the old level is gone.
    """
    src = _source(tmp_path / "src")
    partial = _sentinel(tmp_path / "wing_R2.partial")
    previous = _sentinel(tmp_path / "wing_R2.previous")
    if overwrite:
        refine_mesh(src, 2.0)
        (tmp_path / "wing_R2" / "planted.txt").write_bytes(b"x")
    level = refine_mesh(src, 2.0, overwrite=overwrite)
    assert level.obj.is_file() and not (level.folder / "planted.txt").exists()
    assert _tree(tmp_path / "wing_R2.partial") == partial
    assert _tree(tmp_path / "wing_R2.previous") == previous
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["src", "wing_R2", "wing_R2.partial", "wing_R2.previous"]


def test_ind01_a_failed_publish_removes_only_its_own_folder(tmp_path, monkeypatch):
    """(IND-01): a failed publish keeps every folder it did not make.

    The publish fails after the level was written and audited (an occupying file
    is refused before any work, X-F-0599, so the failure is made at the publish).
    The existing level and the unrelated reserved folders are kept and the run's
    own staging folder is gone.
    """
    src = _source(tmp_path / "src")
    partial = _sentinel(tmp_path / "wing_R2.partial")
    previous = _sentinel(tmp_path / "wing_R2.previous")
    level = _sentinel(tmp_path / "wing_R2")
    _failing_renames(monkeypatch, fail_put_back=False)
    with pytest.raises(OSError):
        refine_mesh(src, 2.0, overwrite=True)
    assert _tree(tmp_path / "wing_R2") == level
    assert _tree(tmp_path / "wing_R2.partial") == partial
    assert _tree(tmp_path / "wing_R2.previous") == previous
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["src", "wing_R2", "wing_R2.partial", "wing_R2.previous"]


def test_ind01_a_level_holding_the_source_is_refused_before_any_work(tmp_path, capsys):
    """(IND-01): ``overwrite`` never replaces the source's folder.

    The source is ``study/wing_R2/inner/wing.obj`` and ``out_dir`` is
    ``study``, so the level ``study/wing_R2`` holds the source's folder. Both
    routes refuse before writing and the source's tree is kept.
    """
    src = _source(tmp_path / "study" / "wing_R2" / "inner")
    before = _tree(tmp_path / "study")
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0, out_dir=tmp_path / "study", overwrite=True)
    text = str(caught.value)
    pdf = src.with_suffix(".refine-refusal.pdf")
    assert "holds the source's folder" in text and text.endswith(diag(pdf).lstrip())
    assert not HAS_INSPECT or pdf.read_bytes().startswith(b"%PDF")
    code = cli.main(["refine", str(src), "2", "--out", str(tmp_path / "study"), "--overwrite"])
    assert code == 2 and text in capsys.readouterr().err
    assert _tree(tmp_path / "study", ignore=pdf) == before


def test_ind01_a_level_inside_the_sources_folder_is_refused(tmp_path):
    """(IND-01): no level is written into the source's folder.

    ``out_dir`` is the source's own folder; the refinement is refused and
    the source files are kept byte for byte, beside the diagnostic PDF.
    """
    src = _source(tmp_path / "src")
    before = _tree(src.parent)
    with pytest.raises(InputArtifactError, match="lies inside the source's folder") as caught:
        refine_mesh(src, 2.0, out_dir=src.parent)
    pdf = src.with_suffix(".refine-refusal.pdf")
    assert str(caught.value).endswith(diag(pdf).lstrip())
    assert not HAS_INSPECT or pdf.read_bytes().startswith(b"%PDF")
    assert _tree(src.parent, ignore=pdf) == before


@pytest.mark.parametrize("tag", ["../up", "a/b", "a\\b", "..", "."])
def test_ind01_a_tag_that_is_not_one_folder_name_is_refused(tmp_path, tag):
    """(IND-01): ``[refine] tag`` names one folder beside the source's.

    A tag holding a path separator, or ``.`` or ``..``, is refused naming the
    file, the table and the key. Control: ``fine`` writes ``wing_fine``.
    """
    escaped = tag.replace("\\", "\\\\")
    src = _source(tmp_path / "src", refine_toml=f'[refine]\ntag = "{escaped}"\n')
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0)
    text = str(caught.value)
    assert f"{src.with_suffix('.refine.toml')} [refine] tag" in text
    assert "one folder name" in text and text.endswith(
        diag(src.with_suffix(".refine-refusal.pdf")).lstrip()
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]
    src.with_suffix(".refine.toml").write_bytes(b'[refine]\ntag = "fine"\n')
    assert refine_mesh(src, 2.0).folder == tmp_path / "wing_fine"


# --------------------------------------------- IND-02 each cap face's own start


def _recovered(fx: Fixture) -> Grid:
    grid = recover_grid(fx.verts, fx.faces, fx.te)
    assert isinstance(grid, Grid), grid
    return grid


def _same_faces(points, faces, fx: Fixture) -> bool:
    """Return whether the level's faces are the source's, in order and vertex for vertex."""
    return len(faces) == len(fx.faces) and all(
        np.array_equal(points[f], fx.verts[g]) for f, g in zip(faces, fx.faces, strict=True)
    )


def _rotated(face: list[int], by: int = 1) -> list[int]:
    return face[by:] + face[:by]


def test_ind02_a_fan_triangle_rotated_alone_keeps_its_start_at_factor_one():
    """(IND-02): at factor 1 one rotated fan triangle keeps its own start.

    The tube's fan triangles start at the pole, except one that starts at a
    ring node. The level at factor 1 equals the source face for face and
    vertex for vertex. Control: the source unchanged passes the same check.
    """
    fx = tube_pole_quads()
    plain = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=1.0, spanwise=1.0)
    assert _same_faces(plain.points, plain.faces, fx)
    fan = fx.lateral + 5
    assert fx.faces[fan][0] == fx.pole
    fx.faces[fan] = _rotated(fx.faces[fan])
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=1.0, spanwise=1.0)
    assert _same_faces(level.points, level.faces, fx)


def test_ind02_mixed_cap_rotations_interleaved_with_lateral_faces_are_kept_at_factor_one():
    """(IND-02): zipper caps of mixed starts, in among the lateral faces.

    One cap quad and one cap triangle start one vertex later than the other
    cap faces, and one cap face is moved between two lateral cells. At
    factor 1 every face, cap faces included, is the source's in order and
    starts at the source face's start vertex.
    """
    fx = tube_zipper()
    caps = list(range(fx.lateral, len(fx.faces)))
    quad = next(p for p in caps if len(fx.faces[p]) == 4)
    tri = next(p for p in caps if len(fx.faces[p]) == 3)
    fx.faces[quad] = _rotated(fx.faces[quad])
    fx.faces[tri] = _rotated(fx.faces[tri], 2)
    moved = fx.faces.pop(caps[-1])
    fx.faces.insert(fx.lateral // 2, moved)
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=1.0, spanwise=1.0)
    assert _same_faces(level.points, level.faces, fx)


def test_ind02_above_factor_one_each_fan_triangle_starts_like_its_nearest_source_triangle():
    """(IND-02): a cap face starts as its nearest source face does.

    The first 8 of the 24 fan triangles start at a ring node and the other 16
    at the pole. At factor 2 the 48 new fan triangles follow the source
    triangle nearest to each: about a third start at a ring node, where a
    majority rule would start all 48 at the pole.

    PUSH-IND-QA-5: the count is exact, derived here from the fixture by the
    rule (the source fan triangle whose centroid is nearest the new one's),
    and each new triangle starts on the vertex role that source triangle
    starts on (the pole's place in the face, the faces being oriented alike);
    a rule that changes one triangle's start fails.
    """
    fx = tube_pole_quads()
    for p in range(fx.lateral, fx.lateral + 8):
        fx.faces[p] = _rotated(fx.faces[p])
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=2.0, spanwise=2.0)
    pole = len(level.points) - 1
    fans = [f for f in level.faces if pole in f]
    assert len(fans) == 48
    sources = [f for f in fx.faces if fx.pole in f]
    centroids = np.array([fx.verts[f].mean(axis=0) for f in sources])
    nearest = [
        sources[int(np.argmin(np.linalg.norm(centroids - level.points[f].mean(axis=0), axis=1)))]
        for f in fans
    ]
    assert [f.index(pole) for f in fans] == [s.index(fx.pole) for s in nearest]
    expected = sum(s[0] != fx.pole for s in nearest)
    ring_first = sum(f[0] != pole for f in fans)
    assert ring_first == expected == 16, ring_first


# ------------------------------------- IND-03 counts a tube's caps can carry


@pytest.mark.parametrize(
    ("make", "factor", "said"),
    [
        (tube_zipper, 1 / 24, "1 interval(s) on each half"),
        (tube_zipper, 0.05, "1 interval(s) on each half"),
        (tube_zipper, 0.1, "1 interval(s) on each half"),
        (tube_pole_quads, 1 / 24, "2 node(s) around"),
        (tube_pole_quads, 0.07, "2 node(s) around"),
    ],
    ids=["zipper-1/m", "zipper-0.05", "zipper-0.1", "pole-1/m", "pole-0.07"],
)
def test_ind03_a_coarsening_the_caps_cannot_carry_is_refused(make, factor, said):
    """(IND-03): too few nodes around a tube is refused, not a crash.

    The factor passes the 1/m limit of 24 intervals, but leaves a zipper
    section one interval per half (its end caps need two) or a pole tube two
    nodes around (a section needs three). The refusal names the family, the
    direction and the value and ends "Nothing was written.".
    """
    fx = make()
    with pytest.raises(InputArtifactError) as caught:
        refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=factor, spanwise=1.0, family="B")
    text = str(caught.value)
    assert text.startswith(f"family B: the normal factor is {factor:g}, ")
    assert said in text and text.endswith("Nothing was written.")


@pytest.mark.parametrize(
    ("make", "factor", "nodes"),
    [(tube_zipper, 2 / 12, 4), (tube_pole_quads, 3 / 24, 4)],
    ids=["zipper-2/half", "pole-3/n"],
)
def test_ind03_the_smallest_accepted_coarsening_builds_valid_caps(make, factor, nodes):
    """(IND-03): just above the limit the tube is coarsened.

    Control of the refusal: the smallest factor the message asks for builds
    the section's nodes and closes the ends with faces of distinct nodes.
    """
    fx = make()
    level = refine_grid(fx.verts, fx.faces, _recovered(fx), chordwise=factor, spanwise=1.0)
    assert level.report["grid"].endswith(f"x{nodes}")
    assert all(len(set(f)) == len(f) for f in level.faces)


def test_ind03_a_smooth_tube_of_two_nodes_around_is_refused_by_both_routes(tmp_path, capsys):
    """(IND-03): a smooth tube keeps three nodes around.

    The body has 24 nodes around; chordwise 2/24 leaves two, which
    ``refine_mesh`` and ``cfd-mesh refine`` refuse before any work with the
    same text. Control: 3/24 writes the level.
    """
    body = body_of_revolution(pole=True)
    mesh = Composite(body.verts, {"S": body.faces}, np.zeros((0, 3)), {})
    src = write_source(tmp_path / "src", "body", mesh)
    src.with_suffix(".te.txt").write_text("METER\n", encoding="utf-8")
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, normal=2 / 24, force_factors=False)
    text = str(caught.value)
    assert "family S: the normal factor is 0.0833333, " in text
    assert "2 node(s) around" in text and text.endswith(
        diag(src.with_suffix(".refine-refusal.pdf")).lstrip()
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]
    code = cli.main(["refine", str(src), "--normal", str(2 / 24), "--strict-factors"])
    assert code == 2 and text in capsys.readouterr().err
    assert refine_mesh(src, normal=3 / 24).obj.is_file()


# ------------------------------------------ IND-04 and IND-05 malformed inputs


def _both_commands_refuse(capsys, src: Path, said: str, *, refine: bool = True) -> None:
    """Assert ``refine`` and ``audit-mesh`` exit 2 printing ``said``, writing nothing."""
    before = _tree(src.parent.parent)
    for argv in (
        (["refine", str(src), "2"] if refine else []),
        ["audit", str(src)],
    ):
        if not argv:
            continue
        capsys.readouterr()
        assert cli.main(argv) == 2, argv
        err = capsys.readouterr().err
        assert said in err and "Nothing was written." in err, err
    assert _tree(src.parent.parent) == before


TRIANGLE = ["v 0 0 0", "v 1 0 0", "v 0 1 0", "g S", "f 1 2 3"]


def _lines(lines: list[str]) -> bytes:
    """Return the lines as a text file's bytes, each ended by a line feed."""
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


@pytest.mark.parametrize(
    ("line", "text", "said"),
    [
        (1, "v bad 0 0", "a vertex whose coordinate 'bad' is not a number"),
        (2, "v 1 0 0x", "a vertex whose coordinate '0x' is not a number"),
        (1, "v nan 0 0", "a vertex whose coordinate 'nan' is not finite"),
        (3, "v 0 inf 0", "a vertex whose coordinate 'inf' is not finite"),
        (2, "v 1e999 0 0", "a vertex whose coordinate '1e999' is not finite"),
        (5, "f 1 2 x", "a face whose vertex index 'x' is not an integer"),
        (5, "f 1 2.0 3", "a face whose vertex index '2.0' is not an integer"),
        (5, "f 1 2 9", "a face with fewer than three vertices or a missing vertex"),
    ],
    ids=["word", "trailing-text", "nan", "inf", "overflow", "word-index", "float-index", "range"],
)
def test_ind04_a_malformed_obj_is_refused_naming_file_and_line(tmp_path, capsys, line, text, said):
    """and (IND-04): a bad OBJ line is a refusal.

    A coordinate that is not a number or not finite, and a face index that is
    not an integer or names no vertex, is refused as InputArtifactError
    naming the file and the line, ending "Nothing was written."; the function
    and both commands refuse it (exit 2) and nothing is written. Control:
    the well-formed triangle is audited.
    """
    folder = tmp_path / "src"
    folder.mkdir()
    src = folder / "tri.obj"
    src.write_bytes(_lines(TRIANGLE))
    assert audit_mesh(src).passed
    src.with_suffix(".audit.json").unlink()
    lines = list(TRIANGLE)
    lines[line - 1] = text
    src.write_bytes(_lines(lines))
    for call in (lambda: refine_mesh(src, 2.0), lambda: audit_mesh(src)):
        with pytest.raises(InputArtifactError) as caught:
            call()
        message = str(caught.value)
        assert message.startswith(f"{src}: line {line} is {said}"), message
        assert message.endswith("Nothing was written.")
    _both_commands_refuse(capsys, src, f"{src}: line {line} is {said}")


@pytest.mark.parametrize(
    ("rows", "row", "said"),
    [
        (["1,2", "3,4", "5,6"], 2, "2 number(s)"),
        (["1,2"], 2, "2 number(s)"),
        (["1,2,3,4", "5,6,7,8"], 2, "4 number(s)"),
        (["0.5,0,0", "1,2"], 3, "2 number(s)"),
        (["0.5,0,0", "1,2,nan"], 3, "a number that is not finite"),
        (["0.5,0,0", "1,x,0"], 3, "a word that is not a number"),
    ],
    ids=["two-columns", "one-short-row", "four-columns", "ragged", "nan", "word"],
)
def test_ind05_each_trailing_edge_row_holds_three_numbers(tmp_path, capsys, rows, row, said):
    """and ( G4, IND-05): x,y,z on every row.

    A points file whose rows hold two or four numbers, or a word, or a number
    that is not finite, is refused naming the file and the row before any
    work, by the audit and its command, and is never reshaped into other
    points. Control: the fixture's own points file is accepted. The refinement
    no longer reads this file (its trailing edge comes from a perimeter file).
    """
    src = _source(tmp_path / "src")
    te = src.with_name("wing.te.txt")
    assert audit_mesh(src).passed
    src.with_suffix(".audit.json").unlink()
    te.write_bytes(_lines(["METER", *rows]))
    for call in (lambda: audit_mesh(src),):
        with pytest.raises(InputArtifactError) as caught:
            call()
        message = str(caught.value)
        assert message.startswith(f"{te}: line {row} ") and said in message, message
        assert message.endswith("Nothing was written.")
    _both_commands_refuse(capsys, src, f"{te}: line {row} ", refine=False)


# --------------------------------------------- IND-06 --out names no input


def _csv_case(tmp_path: Path) -> tuple[Path, Path]:
    """Return a mesh and a copy of it as its source, each with its points and boundaries."""
    src = _source(tmp_path / "src")
    shutil.copytree(src.parent, tmp_path / "level")
    level = tmp_path / "level" / "wing.obj"
    (tmp_path / "level" / "wing.boundaries.toml").write_bytes(
        _lines(['boundaries = ["G"]', "[trailing_edges]", 'file = "pts.txt"'])
    )
    (tmp_path / "level" / "wing.te.txt").rename(tmp_path / "level" / "pts.txt")
    return level, src


@pytest.mark.parametrize(
    "target",
    [
        "level/wing.obj",
        "level/wing.audit.json",
        "level/wing.boundaries.toml",
        "level/pts.txt",
        "src/wing.obj",
        "src/wing.te.txt",
        "src/wing.boundaries.toml",
        "level/../level/wing.obj",
        "link.obj",
    ],
    ids=[
        "mesh",
        "audit-json",
        "mesh-boundaries",
        "named-points",
        "against",
        "against-points",
        "against-boundaries",
        "spelled-otherwise",
        "hard-link",
    ],
)
def test_ind06_csv_naming_a_file_the_audit_reads_or_writes_is_refused(tmp_path, capsys, target):
    """(IND-06): ``--csv`` never overwrites an input or the audit JSON.

    The mesh, the source given by ``--against``, their boundaries and points
    files, the audit JSON, a path spelled otherwise and a hard link to the
    mesh are each refused before the audit runs: exit 2, the refusal names
    the file, and every file is kept byte for byte with no audit written.
    Control: a CSV elsewhere is written and the command exits 0.
    """
    level, src = _csv_case(tmp_path)
    if target == "link.obj":
        try:
            os.link(level, tmp_path / "link.obj")
        except OSError:
            pytest.skip("hard links are not available here")
    before = _tree(tmp_path)
    csv = tmp_path / target
    capsys.readouterr()
    code = cli.main(["audit", str(level), "--against", str(src), "--out", str(csv)])
    err = capsys.readouterr().err
    assert code == 2, err
    said = [line for line in err.splitlines() if line.startswith(f"{csv}: --out names ")]
    assert len(said) == 1 and said[0].endswith("Nothing was written."), err
    assert _tree(tmp_path) == before
    code = cli.main(["audit", str(level), "--against", str(src), "--out", str(tmp_path / "f.csv")])
    assert code == 0 and (tmp_path / "f.csv").is_file()


# ------------------------------------------ IND-07 when the audit warns, said


REPO = Path(__file__).resolve().parents[2]


def test_ind07_a_paired_square_mesh_passes_the_relative_checks(tmp_path):
    """(IND-07): quad-dominant is not a warning by construction.

    Two right triangles per square of a planar grid, paired into the square
    quadrilaterals, lower the skewness (0.25 to 0) and keep every area ratio
    at 1: the audit against the triangulated source passes every gate and
    check with no warning. Control: the same quads with one square moved off
    the plane fail the warp check.
    """
    import warnings

    nodes = [f"v {x} {y} 0" for y in range(3) for x in range(3)]

    def at(x: int, y: int) -> int:
        return 3 * y + x + 1

    squares = [
        (at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)) for y in range(2) for x in range(2)
    ]
    tris = [f"f {a} {b} {c}" for a, b, c, d in squares] + [
        f"f {a} {c} {d}" for a, b, c, d in squares
    ]
    quads = [f"f {a} {b} {c} {d}" for a, b, c, d in squares]
    (tmp_path / "src").mkdir()
    (tmp_path / "lvl").mkdir()
    source = tmp_path / "src" / "plate.obj"
    level = tmp_path / "lvl" / "plate.obj"
    source.write_bytes(_lines([*nodes, "g S", *tris]))
    level.write_bytes(_lines([*nodes, "g S", *quads]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        audit = audit_mesh(level, against=source)
    assert audit.passed
    nodes[at(2, 2) - 1] = "v 2 2 0.5"
    level.write_bytes(_lines([*nodes, "g S", *quads]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bent = audit_mesh(level, against=source)
    assert any(i.name == "warp" and not i.passed for i in bent.checks)


def _failing_renames(monkeypatch, fail_put_back):
    """Make the publish rename fail, and with ``fail_put_back`` the put-back too."""
    real = Path.rename

    def rename(self, target):
        target = Path(target)
        moving_staging = self.parent.name.startswith("wing_R2.partial-") and self.name == "wing_R2"
        putting_back = self.name.endswith(".previous") and target.name == "wing_R2"
        if moving_staging or (fail_put_back and putting_back):
            raise OSError("simulated rename failure")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)


def test_ind01_a_failed_publish_puts_the_previous_level_back(tmp_path, monkeypatch):
    """(IND-01): the old level survives a publish that fails.

    An existing level is set aside, the rename of the new one fails, and the
    old level is back at its path byte for byte, with no folder of the run
    left. Control: without the failure the level is replaced.
    """
    src = _source(tmp_path / "src")
    refine_mesh(src, 2.0)
    (tmp_path / "wing_R2" / "planted.txt").write_bytes(b"old")
    old = _tree(tmp_path / "wing_R2")
    _failing_renames(monkeypatch, fail_put_back=False)
    with pytest.raises(OSError):
        refine_mesh(src, 2.0, overwrite=True)
    assert _tree(tmp_path / "wing_R2") == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "wing_R2"]
    monkeypatch.undo()
    refine_mesh(src, 2.0, overwrite=True)
    assert not (tmp_path / "wing_R2" / "planted.txt").exists()


def test_ind01_a_level_that_cannot_be_put_back_is_kept_and_named(tmp_path, monkeypatch):
    """(IND-01): when the old level cannot be put back, it is not removed.

    Both the publish and the put-back fail: the run's folder is kept with the
    old level inside it, byte for byte, and a warning names where it is.
    """
    src = _source(tmp_path / "src")
    refine_mesh(src, 2.0)
    (tmp_path / "wing_R2" / "planted.txt").write_bytes(b"old")
    old = _tree(tmp_path / "wing_R2")
    _failing_renames(monkeypatch, fail_put_back=True)
    with pytest.warns(GeometryWarning, match="put back"), pytest.raises(OSError):
        refine_mesh(src, 2.0, overwrite=True)
    kept = [p for p in tmp_path.iterdir() if p.name.startswith("wing_R2.partial-")]
    assert len(kept) == 1 and _tree(kept[0] / "wing_R2.previous") == old


# ------------------------------------------- IND-09 any whitespace separates


def _tabbed(text: str) -> str:
    """Return the OBJ text with the keyword of every other v, f and g line ended by a tab."""
    out = []
    for number, line in enumerate(text.splitlines()):
        keyword, _, rest = line.partition(" ")
        if keyword in ("v", "f", "g") and number % 2:
            line = f"{keyword}\t{rest}" if number % 4 == 1 else f"{keyword} \t {rest}"
        out.append(line)
    return "\n".join(out) + "\n"


def test_ind09_statements_separated_by_tabs_are_read_whole(tmp_path):
    """and (IND-09): a tab separates too.

    The fixture's OBJ is written again with every other ``v``, ``f`` and
    ``g`` line's keyword followed by a tab, or by spaces and a tab. The mesh
    read holds every vertex, face and family of the space-separated one, the
    audit gives the same figures, and the factor-1 level is the same bytes.
    Control: the space-separated source itself.
    """
    from cfdbender.mesh._refine._obj import read_obj

    spaced = _source(tmp_path / "a")
    tabbed = _source(tmp_path / "b")
    tabbed.write_bytes(_tabbed(spaced.read_text(encoding="utf-8")).encode("utf-8"))
    assert "\t" in tabbed.read_text(encoding="utf-8")
    one, two = read_obj(spaced), read_obj(tabbed)
    assert np.array_equal(one.verts, two.verts) and one.families == two.families
    assert one.vertex_text == two.vertex_text
    assert audit_mesh(tabbed).as_json()["figures"] == audit_mesh(spaced).as_json()["figures"]
    first = refine_mesh(spaced, 1.0, out_dir=tmp_path / "la")
    second = refine_mesh(tabbed, 1.0, out_dir=tmp_path / "lb")
    assert first.obj.read_bytes() == second.obj.read_bytes()


# ------------------------------- IND-01 (second part) and IND2-02 family tags


UNSAFE_FAMILIES = ["A/../../victim", "A\\..\\..\\victim", "A..B", "C:victim", "A<B", "A|B", "A?B"]


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("family", UNSAFE_FAMILIES)
def test_ind01_a_family_name_that_is_not_one_folder_name_is_refused(
    tmp_path, capsys, family, overwrite
):
    """(IND-01, IND2-02): a family name never leaves out_dir.

    The family whose name holds a path separator, ``..``, a drive colon or a
    reserved character is the only one refined, so its name would enter the
    level's tag ``R-<family><factor>``. ``study/victim1``, where the path
    ``study/levels/wing_R-A/../../victim1`` resolves, holds an unrelated
    file. Both routes refuse before any work, naming the family, with and
    without ``overwrite``; every file under the study is kept byte for byte.
    Control: the same mesh with the family named ``A`` writes
    ``study/levels/wing_R-A1``.
    """
    mesh = sheet_with_strips([("B", "quad", [0.1, 0.1])], grid=family)
    src = write_source(tmp_path / "study" / "src", "wing", mesh, boundaries=False)
    _sentinel(tmp_path / "study" / "victim1")
    levels = tmp_path / "study" / "levels"
    before = _tree(tmp_path / "study")
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 1.0, families=[family], out_dir=levels, overwrite=overwrite)
    text = str(caught.value)
    assert f"the family {family!r}" in text and "one folder name" in text, text
    assert text.endswith(diag(src.with_suffix(".refine-refusal.pdf")).lstrip())
    argv = ["refine", str(src), "1", "--families", family, "--out", str(levels)]
    capsys.readouterr()
    assert cli.main(argv + (["--overwrite"] if overwrite else [])) == 2
    assert text in capsys.readouterr().err
    pdf = src.with_suffix(".refine-refusal.pdf")
    assert not HAS_INSPECT or pdf.read_bytes().startswith(b"%PDF")
    assert _tree(tmp_path / "study", ignore=pdf) == before and not levels.exists()
    plain = sheet_with_strips([("B", "quad", [0.1, 0.1])], grid="A")
    src = write_source(tmp_path / "plain", "wing", plain)
    assert refine_mesh(src, 1.0, families=["A"], out_dir=levels).folder == levels / "wing_R-A1"


@pytest.mark.parametrize("tag", ["a<b", "a|b", "a?b", "a*b", "fine.", "fine ", "a..b"])
def test_ind01_a_tag_holding_a_reserved_character_is_refused(tmp_path, tag):
    """(IND2-02): ``[refine] tag`` is one portable folder name.

    A tag holding a character Windows reserves, ``..``, or ending with a dot
    or a space (which Windows drops, naming another folder) is refused naming
    the file, the table and the key, before any work. Control: ``fine``.
    """
    src = _source(tmp_path / "src", refine_toml=f'[refine]\ntag = "{tag}"\n')
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0)
    text = str(caught.value)
    assert f"{src.with_suffix('.refine.toml')} [refine] tag" in text
    assert "one folder name" in text and text.endswith(
        diag(src.with_suffix(".refine-refusal.pdf")).lstrip()
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]
    src.with_suffix(".refine.toml").write_bytes(b'[refine]\ntag = "fine"\n')
    assert refine_mesh(src, 2.0).folder == tmp_path / "wing_fine"


@pytest.mark.parametrize("overwrite", [False, True])
def test_ind01_a_level_path_resolving_outside_out_dir_is_refused(tmp_path, overwrite):
    """(IND-01): the level is a folder directly inside out_dir.

    ``study/levels/wing_R2`` is a directory junction to the unrelated
    ``study/victim``, so the level path resolves outside ``out_dir``. The
    refinement is refused before any work, naming ``out_dir``, with and
    without ``overwrite``; the junction and the folder it reaches are kept
    byte for byte. Control: a level beside it is written. Windows only (a
    junction needs no privilege there).
    """
    winapi = pytest.importorskip("_winapi")
    src = _source(tmp_path / "study" / "src")
    victim = _sentinel(tmp_path / "study" / "victim")
    levels = tmp_path / "study" / "levels"
    levels.mkdir()
    winapi.CreateJunction(str(tmp_path / "study" / "victim"), str(levels / "wing_R2"))
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0, out_dir=levels, overwrite=overwrite)
    text = str(caught.value)
    assert f"is not a folder directly inside out_dir {levels}" in text, text
    assert text.endswith(diag(src.with_suffix(".refine-refusal.pdf")).lstrip())
    assert (levels / "wing_R2").is_junction()
    assert _tree(tmp_path / "study" / "victim") == victim
    assert sorted(p.name for p in levels.iterdir()) == ["wing_R2"]
    assert refine_mesh(src, 1.0, out_dir=levels).folder == levels / "wing_R1"


# ------------------------------------- IND2-03 a level that appears meanwhile


def _level_appears_once_the_run_is_under_way(monkeypatch, folder: Path) -> None:
    """Make another writer create ``folder``, holding a file, once the run makes its own folder.

    The run makes its own staging folder after it checked that the level did
    not exist and before it resamples, writes and audits the level.
    """
    real = tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        made = real(*args, **kwargs)
        if Path(made).name.startswith(f"{folder.name}.partial-"):
            _sentinel(folder)
        return made

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)


def test_ind2_03_a_level_that_appears_during_the_run_is_not_replaced(tmp_path, monkeypatch):
    """(IND2-03): without overwrite, a level made meanwhile is kept.

    The level folder does not exist when the run starts, and another writer
    creates it, holding a file, while the run writes and audits its staged
    level. The publish refuses, naming the folder and ``overwrite``: the
    folder is kept byte for byte and the run leaves no folder of its own.
    Control: the same race with ``overwrite`` replaces it with the level.
    """
    src = _source(tmp_path / "src")
    folder = tmp_path / "wing_R2"
    _level_appears_once_the_run_is_under_way(monkeypatch, folder)
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0)
    text = str(caught.value)
    assert text.startswith(f"{folder}: the level appeared while the run wrote it"), text
    assert "overwrite=True (--overwrite)" in text
    assert _tree(folder) == {"keep.txt": b"not the refinement's"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "wing_R2"]
    shutil.rmtree(folder)
    level = refine_mesh(src, 2.0, overwrite=True)
    assert level.obj.is_file() and not (folder / "keep.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "wing_R2"]


def test_mesh_refine_review_a_file_that_takes_the_level_name_during_the_run_is_a_refusal(
    tmp_path, monkeypatch
):
    """with overwrite, a FILE that appears at the level path while the run
    writes is refused as _place refuses one up front (InputArtifactError naming the path),
    never a bare FileExistsError; the file is kept."""
    src = _source(tmp_path / "src")
    folder = tmp_path / "wing_R2"
    real = tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        made = real(*args, **kwargs)
        if Path(made).name.startswith(f"{folder.name}.partial-"):
            folder.write_text("not a level", encoding="utf-8")
        return made

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, 2.0, overwrite=True)
    assert str(caught.value).startswith(f"{folder}: a file took the level folder's name")
    assert folder.read_text(encoding="utf-8") == "not a level"


# ------------------------------------- IND-11 the audit JSON is never an input


def _link(target: Path, existing: Path) -> None:
    """Make ``target`` a hard link to ``existing``, or skip where links are not available."""
    try:
        os.link(existing, target)
    except OSError:
        pytest.skip("hard links are not available here")


def _json_over_an_input(case: str, level: Path, src: Path) -> tuple[Path, Path]:
    """Arrange the audit JSON of ``level`` to be a file the audit reads; return it and against."""
    json_path = level.with_name("wing.audit.json")
    if case == "against":
        shutil.copyfile(src, json_path)
        return json_path, json_path
    if case == "named-points":
        (level.parent / "pts.txt").rename(json_path)
        (level.parent / "wing.boundaries.toml").write_bytes(
            _lines(['boundaries = ["G"]', "[trailing_edges]", 'file = "wing.audit.json"'])
        )
        return json_path, src
    linked = {
        "mesh-link": level,
        "record-link": level.with_name("wing.refine.json"),
        "against-link": src,
        "against-points-link": src.with_name("wing.te.txt"),
    }[case]
    if case == "record-link":
        linked.write_bytes(b'{"schema_version": 1, "families": {}}\n')
    _link(json_path, linked)
    return linked, src


@pytest.mark.parametrize(
    "case",
    ["against", "named-points", "mesh-link", "record-link", "against-link", "against-points-link"],
)
def test_ind11_the_audit_json_never_overwrites_a_file_the_audit_reads(tmp_path, capsys, case):
    """(IND-11): ``<stem>.audit.json`` is refused when it is an input.

    The audit JSON beside the level is, by its own path or as a hard link,
    the source of ``against``, the points file the level's boundaries file
    names, the level's OBJ, its ``refine.json``, the source's OBJ or the
    source's points file. The function and the command refuse before
    writing, naming both paths and ending "Nothing was written."; every file
    is kept byte for byte. Control: the same level with no such collision
    writes its audit and passes.
    """
    level, src = _csv_case(tmp_path)
    taken, against = _json_over_an_input(case, level, src)
    json_path = level.with_name("wing.audit.json")
    before = _tree(tmp_path)
    with pytest.raises(InputArtifactError) as caught:
        audit_mesh(level, against=against)
    text = str(caught.value)
    assert text.startswith(f"{json_path}: the audit JSON would be written over "), text
    assert str(taken) in text and text.endswith("Nothing was written.")
    assert _tree(tmp_path) == before
    capsys.readouterr()
    assert cli.main(["audit", str(level), "--against", str(against)]) == 2
    assert text in capsys.readouterr().err
    assert _tree(tmp_path) == before
    plain, plain_src = _csv_case(tmp_path / "control")
    assert audit_mesh(plain, against=plain_src).passed
    assert plain.with_name("wing.audit.json").is_file()


# ------------------------------- IND2-01 --csv never names the level's record


@pytest.mark.parametrize("target", ["level/wing.refine.json", "level/../level/wing.refine.json"])
def test_ind2_01_csv_naming_the_levels_refine_json_is_refused(tmp_path, capsys, target):
    """(IND2-01): ``--csv`` never truncates the record the audit reads.

    The audited level holds ``<stem>.refine.json``, which the audit reads
    for its components and unchanged grids. ``--csv`` naming it, directly or
    spelled otherwise, is refused before the audit: exit 2, the refusal names
    the record, and every file is kept byte for byte. Control: a CSV
    elsewhere is written and the command exits 0.
    """
    level, src = _csv_case(tmp_path)
    level.with_name("wing.refine.json").write_bytes(b'{"schema_version": 1, "families": {}}\n')
    before = _tree(tmp_path)
    csv = tmp_path / target
    capsys.readouterr()
    code = cli.main(["audit", str(level), "--against", str(src), "--out", str(csv)])
    err = capsys.readouterr().err
    assert code == 2, err
    said = [line for line in err.splitlines() if line.startswith(f"{csv}: --out names ")]
    assert len(said) == 1 and "refinement record" in said[0], err
    assert said[0].endswith("Nothing was written.")
    assert _tree(tmp_path) == before
    code = cli.main(["audit", str(level), "--against", str(src), "--out", str(tmp_path / "f.csv")])
    assert code == 0 and (tmp_path / "f.csv").is_file()


# ------------------------------------- IND-08 a sheet of split quadrilaterals


def _split(quad: list[int], diagonal: int) -> list[list[int]]:
    """Return a quadrilateral ``a b c d`` as two triangles sharing a-c (0) or b-d (1)."""
    a, b, c, d = quad
    return [[a, b, c], [a, c, d]] if diagonal == 0 else [[a, b, d], [b, c, d]]


SPLITS = {
    "a-c": lambda k, i: 0,
    "b-d": lambda k, i: 1,
    "checkerboard": lambda k, i: (k + i) % 2,
    "halves": lambda k, i: int(2 * i >= 18),
}


def _split_sheet(rule: str) -> tuple[Composite, Fixture]:
    """Return the cambered sheet ``G`` with every quadrilateral split by ``rule``, and its grid.

    Each cell keeps its place and its trailing-edge points; the rule gives
    the diagonal of the cell at station k and chordwise cell i.
    """
    from mesh_fixtures import sheet_quads

    fx = sheet_quads()
    mesh = sheet_with_strips(())
    faces = []
    for quad in mesh.families["G"]:
        k, i = (min(fx.place[v][j] for v in quad) for j in (0, 1))
        faces += _split(quad, SPLITS[rule](k, i))
    mesh.families["G"] = faces
    return mesh, fx


def _cell_triangles(grid: Grid) -> list[frozenset[int]]:
    """Return the two triangles of each cell of a split sheet, as the grid states them."""
    out = []
    for (k, i), diagonal in np.ndenumerate(grid.cell_diag):
        a, b = int(grid.ids[k, i]), int(grid.ids[k, i + 1])
        c, d = int(grid.ids[k + 1, i + 1]), int(grid.ids[k + 1, i])
        out += [frozenset(t) for t in _split([a, b, c, d], int(diagonal))]
    return out


@pytest.mark.parametrize("rule", sorted(SPLITS))
def test_ind08_a_sheet_of_split_quadrilaterals_is_recovered_as_its_grid(rule):
    """( GRID FAMILY,, IND-08): corners come from the cells.

    Every quadrilateral of the 9 by 19 sheet is split into two triangles,
    along one diagonal, the other, a checkerboard of both, or one per half,
    so two or more of the sheet's corners lie in two triangles. The grid is
    recovered with the unsplit sheet's stations and chordwise nodes (the
    trailing edge at chordwise index 0), and the two triangles it states for
    each cell are the family's faces. Control: the unsplit sheet.
    """
    mesh, fx = _split_sheet(rule)
    te = {v for v, (k, i) in fx.place.items() if i == 0}
    grid = recover_grid(mesh.verts, mesh.families["G"], te)
    assert isinstance(grid, Grid), grid
    quads = recover_grid(fx.verts, fx.faces, fx.te)
    assert isinstance(quads, Grid) and quads.cell_diag is None
    assert grid.layout == "sheet" and grid.ends == ("edge", "edge")
    assert np.array_equal(grid.ids, quads.ids) and grid.cell_diag is not None
    assert sorted(map(sorted, _cell_triangles(grid))) == sorted(
        sorted(f) for f in mesh.families["G"]
    )


@pytest.mark.parametrize("rule", ["a-c", "checkerboard"])
def test_ind08_a_split_sheet_refines_as_a_grid_through_refine_mesh(tmp_path, rule):
    """(IND-08): the split sheet keeps its grid behavior.

    Through ``refine_mesh`` under ``auto``: at factor 1 the level is the
    source's faces in order and its audit passes with G6 judged; at factor 2
    the grid's intervals double in both directions (18 by 8 to 36 by 16,
    each cell two triangles); and ``chordwise`` alone doubles one direction.
    Each is reported as a grid of triangles.
    """
    import json

    from mesh_fixtures import read_mesh

    mesh, _ = _split_sheet(rule)
    src = write_source(tmp_path / "src", "wing", mesh)
    one = refine_mesh(src, 1.0)
    report = one.report["G"]
    assert report["method"] == "grid" and report["split"] == "triangles", report
    record = json.loads(one.folder.joinpath("wing_R1.refine.json").read_text(encoding="utf-8"))
    assert one.audit.passed and record["unchanged_grids"] == ["G"]
    assert read_mesh(one.obj)[1]["G"] == mesh.families["G"]
    two = refine_mesh(src, 2.0).report["G"]
    assert two["method"] == "grid", two
    assert two["intervals"] == {"chordwise": [18, 36], "spanwise": [8, 16]}
    assert two["faces"] == [2 * 18 * 8, 2 * 36 * 16]
    chord = refine_mesh(src, normal=2.0).report["G"]
    assert chord["method"] == "grid"
    assert chord["intervals"] == {"chordwise": [18, 36], "spanwise": [8, 8]}


def test_ind08_triangles_that_are_not_the_cells_are_no_grid():
    """( GRID FAMILY,, IND-08): triangles across two cells are no grid.

    In the sheet split along a-c, the side shared by two neighboring cells
    is flipped into the diagonal of the two triangles beside it: the nodes,
    the boundary and the face count are the sheet's, but two triangles span
    two cells. No grid is recovered and the reason is named. Control: the
    unflipped sheet is a grid.
    """
    mesh, fx = _split_sheet("a-c")
    te = {v for v, (k, i) in fx.place.items() if i == 0}
    assert isinstance(recover_grid(mesh.verts, mesh.families["G"], te), Grid)
    at = {place: v for v, place in fx.place.items()}
    a, b, c, f = at[(3, 5)], at[(3, 6)], at[(4, 6)], at[(4, 7)]
    faces = [list(t) for t in mesh.families["G"]]
    first = faces.index([a, b, c])
    second = next(j for j, t in enumerate(faces) if set(t) == {b, f, c})
    faces[first], faces[second] = [a, b, f], [a, f, c]
    found = recover_grid(mesh.verts, faces, te)
    assert isinstance(found, str) and found, found


# ------------------------------------------- IND-10 a pole fan at each end


def _volume(verts: np.ndarray, faces: list[list[int]]) -> float:
    """Return the signed volume the faces enclose, each face fanned from its first vertex."""
    total = 0.0
    for f in faces:
        for j in range(1, len(f) - 1):
            a, b, c = verts[f[0]], verts[f[j]], verts[f[j + 1]]
            total += float(np.dot(a, np.cross(b, c))) / 6.0
    return total


def _tube_with_poles(root: bool, tip: bool, split: bool = False) -> tuple[Composite, Fixture]:
    """Return the wing tube ``W``, each end open or closed by a fan around a pole, and its grid.

    The lateral grid is ``tube_pole_triangles``' (split) or
    ``tube_pole_quads``' with its trailing edge; the tip fan is the
    fixture's own, and the root fan is wound like it, around a pole set
    0.05 of the span inboard of the root section's center.
    """
    from mesh_fixtures import Fixture as Fx
    from mesh_fixtures import build_grid, tube_pole_triangles

    if tip:
        fx = tube_pole_triangles() if split else tube_pole_quads()
    else:
        fx = build_grid(Fx("tube-open", True, 9, 24, ("open", "open"), None))
    at = {place: v for v, place in fx.place.items()}
    n = fx.nodes
    verts, faces = fx.verts, [list(f) for f in fx.faces]
    if root:
        ring = [at[(0, i)] for i in range(n)]
        center = verts[ring].mean(axis=0) - [0.0, 0.05, 0.0]
        verts = np.vstack([verts, center[None, :]])
        pole = len(verts) - 1
        faces += [[pole, ring[(i + 1) % n], ring[i]] for i in range(n)]
    if root and tip and _volume(verts, faces) < 0:
        faces = [f[::-1] for f in faces]  # a closed body encloses a positive volume (G2)
    column = [at[(k, 0)] for k in range(fx.stations)]
    te = 0.5 * (verts[column[:-1]] + verts[column[1:]])
    return Composite(verts, {"W": faces}, te, {}), fx


@pytest.mark.parametrize(
    ("root", "tip", "split"),
    [(False, True, False), (True, False, False), (True, True, False), (True, True, True)],
    ids=["open-pole", "pole-open", "pole-pole", "pole-pole-split"],
)
def test_ind10_a_tube_closed_by_a_pole_fan_at_each_end_is_a_grid(tmp_path, root, tip, split):
    """( GRID FAMILY,, IND-10): either end, or both, may be a
     pole fan.

    The wing tube of 9 stations by 24 chordwise nodes is closed by a fan of
    triangles around a pole at the root, at the tip, or at both, its lateral
    cells quadrilaterals or split into triangles. The grid is recovered with
    its 9 stations (the poles apart), and through ``refine_mesh`` under
    ``auto``: at factor 1 the level is the source's faces in order and its
    audit passes with G6 judged; at factor 2 the intervals double (24 by 8
    to 48 by 16), each pole closes its end with 48 triangles and the audit
    passes (every face oriented like its neighbors, the volume positive).
    Control: the open root and pole tip of ``tube_pole_quads``.
    """
    import json

    from mesh_fixtures import read_mesh

    mesh, fx = _tube_with_poles(root, tip, split)
    te = {v for v, (k, i) in fx.place.items() if i == 0}
    grid = recover_grid(mesh.verts, mesh.families["W"], te)
    assert isinstance(grid, Grid), grid
    assert grid.layout == "tube" and grid.ids.shape == (9, 24)
    assert sorted(grid.ends) == sorted(["pole" if root else "open", "pole" if tip else "open"])
    src = write_source(tmp_path / "src", "wing", mesh)
    one = refine_mesh(src, 1.0)
    assert one.report["W"]["method"] == "grid", one.report["W"]
    record = json.loads(one.folder.joinpath("wing_R1.refine.json").read_text(encoding="utf-8"))
    assert one.audit.passed and record["unchanged_grids"] == ["W"]
    assert read_mesh(one.obj)[1]["W"] == mesh.families["W"]
    two = refine_mesh(src, 2.0)
    report = two.report["W"]
    assert report["method"] == "grid" and two.audit.passed, report
    assert report["intervals"] == {"chordwise": [24, 48], "spanwise": [8, 16]}
    cells = 48 * 16 * (2 if split else 1)
    poles = int(root) + int(tip)
    assert report["faces"] == [len(mesh.families["W"]), cells + 48 * poles]


def test_ind3_01_an_unstructured_closed_triangulation_is_refused_at_once():
    """(IND3-01): a closed irregular triangulation offers no pole pair.

    The convex hull of 2402 random points on a sphere is a closed family of
    triangles whose nodes share a handful of fan sizes, so no node can be the
    pole of a two-pole tube: grid recovery refuses it in well under a second
    (before the fix it tried every node as a pole, for minutes). Control: a
    UV sphere, whose two poles are the only nodes of their fan size, is
    still recovered as a grid.
    """
    import time

    import numpy
    import scipy.spatial as spatial

    from cfdbender.mesh._refine._grid import recover_grid

    rng = numpy.random.default_rng(3)
    points = rng.normal(size=(2402, 3))
    points /= numpy.linalg.norm(points, axis=1)[:, None]
    faces = [[int(i) for i in s] for s in spatial.ConvexHull(points).simplices]
    started = time.perf_counter()
    found = recover_grid(points, faces, ())
    assert isinstance(found, str) and time.perf_counter() - started < 10.0
    n, m = 12, 24
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, n):
        th = numpy.pi * i / n
        verts += [
            (
                numpy.sin(th) * numpy.cos(2 * numpy.pi * j / m),
                numpy.sin(th) * numpy.sin(2 * numpy.pi * j / m),
                numpy.cos(th),
            )
            for j in range(m)
        ]
    verts.append((0.0, 0.0, -1.0))
    node = lambda i, j: 1 + (i - 1) * m + j % m  # noqa: E731
    uv = [[0, node(1, j), node(1, j + 1)] for j in range(m)]
    for i in range(1, n - 1):
        for j in range(m):
            a, b, c, d = node(i, j), node(i, j + 1), node(i + 1, j + 1), node(i + 1, j)
            uv += [[a, d, c], [a, c, b]]
    uv += [[len(verts) - 1, node(n - 1, j + 1), node(n - 1, j)] for j in range(m)]
    assert not isinstance(recover_grid(numpy.array(verts), uv, ()), str)

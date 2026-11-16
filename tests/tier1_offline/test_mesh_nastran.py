"""prepare-to-refine cuts an ANSA NASTRAN export along its FE perimeters; visual-inspect.

The export is built here: a flat sheet of 4 x 2 quadrilaterals in one property,
and a perimeter block naming the sides of the line that splits it in two 2 x 2
halves, in the layout ANSA writes (a base64 field holding a 4-byte length and
zlib data: a count, then a 32-bit element id and a 16-bit side per record).
"""
# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import struct
import tomllib
import warnings
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import (  # noqa: F401
    HAS_INSPECT,
    Composite,
    diag,
    face_coordinates,
    read_mesh,
    sheet_with_strips,
    write_source,
)

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh._refine import _blocks, _nastran, prepare_to_refine, refine_mesh
from cfdbender.mesh._refine._config import (
    FamilySpec,
    RefineRequest,
    canonical_selection,
    level_tag,
    tag_selection,
)

COLUMNS, ROWS = 4, 2


needs_inspect = pytest.mark.skipif(not HAS_INSPECT, reason="needs the visual-inspection module")


def _p0500_component_source(tmp_path, *, components=True, factor=2, connected=False):
    points, families = [], {}

    def vertex(point):
        if point not in points:
            points.append(point)
        return points.index(point)

    for name, x0, width, height in (
        ("A", 0, 2, 6),
        ("B", 2 if connected else 10, 2, 6),
        ("C", 20, 8, 2),
    ):
        ids = [
            [vertex((x0 + width * x / 4, height * y / 4, 0)) for y in range(5)] for x in range(5)
        ]
        families[name] = [
            [
                ids[x][y],
                ids[x + 1][y],
                ids[x + 1][y + 1],
                ids[x][y + 1],
            ]
            for x in range(4)
            for y in range(4)
        ]
    config = "[refine]\nkeep_within_component = true\n"
    config += "".join(
        f'[families.{n}]\nmethod = "grid"\nnormal_factor = {factor if n == "C" else 1}\n'
        f"parallel_factor = {factor if n == 'A' else 1}\n"
        for n in families
    )
    if components:
        config += '[components]\nwing = ["C", "B", "A"]\n'
    return write_source(
        tmp_path / "src",
        "components",
        Composite(np.array(points, dtype=float), families, np.zeros((0, 3)), {}),
        refine_toml=config,
    )


@pytest.mark.parametrize("factor", [2, 0.5])
@pytest.mark.parametrize("connected", [False, True])
def test_mesh_nastran_component_governors_cross_disconnected_layouts(tmp_path, factor, connected):
    """source-order tie, independent directions and distant followers."""
    source = _p0500_component_source(tmp_path, factor=factor, connected=connected)
    # Explicit perimeters fix the same P/N reference for all three plates.
    rows = ["# Id, Name, X, Y, Z"]
    from cfdbender.mesh._refine._obj import read_obj

    obj = read_obj(source)
    for n, fs in obj.families.items():
        ids = sorted({v for f in fs for v in f})
        xmax = max(obj.verts[v, 0] for v in ids)
        rows += [
            f"{v + 1}, {n}, " + ", ".join(map(str, obj.verts[v]))
            for v in ids
            if obj.verts[v, 0] == xmax
        ]
    source.with_suffix(".nodes.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = source.with_suffix(".refine.toml")
    config.write_text(
        config.read_text() + '\n[[perimeter]]\nfile = "components.nodes.txt"\n', encoding="utf-8"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        level = refine_mesh(source, accept_unknown=True)
    record = json.loads(level.obj.with_suffix(".refine.json").read_text())
    assert record.get("component_governors") == {"wing": {"parallel": "A", "normal": "C"}}
    assert record["faces"]["wing"] == 3 * round(4 * factor) ** 2
    provenance = "propagated_from:A" if connected else "own_orientation_unlinked"
    assert record["families"]["B"]["provenance"][provenance] == 4
    if not connected:
        assert any("unlinked" in str(w.message) for w in caught)


def test_rel1_001_component_member_without_a_table_neither_governs_nor_is_refined(tmp_path):
    """REL1-001: a kept member keeps its faces and never supplies the others' factors."""
    source = _p0500_component_source(tmp_path, connected=True)
    config = source.with_suffix(".refine.toml")
    config.write_text(
        "[refine]\nkeep_within_component = true\n"
        + "".join(
            f'[families.{n}]\nmethod = "grid"\nnormal_factor = 2\nparallel_factor = 2\n'
            for n in ("A", "B")
        )
        + '[components]\nwing = ["C", "B", "A"]\n',
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level = refine_mesh(source, accept_unknown=True)
    record = json.loads(level.obj.with_suffix(".refine.json").read_text())
    governors = record.get("component_governors", {}).get("wing", {})
    assert governors and set(governors.values()) <= {"A", "B"}, governors
    assert record["families"]["C"]["method"] == "copied"
    # A and B are refined 2 by 2 (16 faces to 64 each); C keeps its 16 faces.
    assert record["faces"]["wing"] == 2 * 64 + 16


@pytest.mark.parametrize(
    "own",
    ["", "normal_factor = 2\nparallel_factor = 2\n", "normal_factor = 3\nparallel_factor = 3\n"],
    ids=["states-nothing", "states-the-governors-factors", "states-other-factors"],
)
def test_mesh_nastran_component_member_factors_replaced_by_its_governors_warn(tmp_path, own):
    """A member's own stated factors given way to its governors' warn once, naming both."""
    source = _p0500_component_source(tmp_path, connected=True)
    config = source.with_suffix(".refine.toml")
    config.write_text(
        "[refine]\nkeep_within_component = true\n"
        + "".join(
            f'[families.{n}]\nmethod = "grid"\nnormal_factor = 2\nparallel_factor = 2\n'
            for n in ("A", "B")
        )
        + '[families.C]\nmethod = "grid"\n'
        + own
        + '[components]\nwing = ["A", "B", "C"]\n',
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        level = refine_mesh(source, accept_unknown=True)
    record = json.loads(level.obj.with_suffix(".refine.json").read_text())
    leads = record["component_governors"]["wing"]
    c_factor = float(own.split("= ")[1].splitlines()[0]) if own else 1.0
    stated = {"A": (2.0, 2.0), "B": (2.0, 2.0), "C": (c_factor, c_factor)}
    given = (stated[leads["normal"]][0], stated[leads["parallel"]][1])
    # A member warns when it stated factors of its own and they differ from its governors'.
    expected = {m for m, f in stated.items() if f != (1.0, 1.0) and f != given}
    replaced = [str(w.message) for w in caught if "are replaced by" in str(w.message)]
    warned = {m for m in stated for text in replaced if f"family {m}:" in text}
    assert warned == expected, (leads, replaced)
    assert len(replaced) == len(expected), replaced
    if c_factor == 3.0:
        assert expected, "a member stating 3 beside members stating 2 must be overridden"
    if c_factor == 2.0:
        assert not expected, "a member stating its governors' factors is not warned"


def test_mesh_nastran_component_without_components_is_byte_identical(tmp_path):
    """the switch without a component table changes no output bytes."""

    source = _p0500_component_source(tmp_path, components=False)
    config = source.with_suffix(".refine.toml")
    config.write_text(config.read_text().replace("keep_within_component = true\n", ""))
    before = refine_mesh(source)
    expected = {p.name: p.read_bytes() for p in before.files}
    after = refine_mesh(source, overwrite=True, keep_within_component=True)
    assert {p.name: p.read_bytes() for p in after.files} == expected


def test_mesh_nastran_prepare_component_governors_are_persisted(tmp_path):
    """prepare_to_refine(source, keep_within_component=True)."""
    import inspect

    source = _p0500_component_source(tmp_path)
    obj = source.read_text().replace("g B", "g A").replace("g C", "g A")
    source.write_text(obj, encoding="utf-8")
    options = (
        {"keep_within_component": True}
        if "keep_within_component" in inspect.signature(prepare_to_refine).parameters
        else {}
    )
    prepared = prepare_to_refine(source, **options)
    config = tomllib.loads(prepared.config.read_text())
    assert config["refine"].get("keep_within_component") is True
    tables = config["families"]
    assert sum(bool(s.get("governing_parallel")) for s in tables.values()) == 1
    assert sum(bool(s.get("governing_normal")) for s in tables.values()) == 1
    assert sum(s["method"] == "grid_auto" for s in tables.values()) == 1


@pytest.mark.parametrize("names", [("nodes", "other"), ("nodes", "nodes_family"), ("A", "A.fs")])
@pytest.mark.parametrize("declared", [True, False])
def test_mesh_nastran_family_output_stems_do_not_collide(tmp_path, names, declared):
    """Refine two sheets without overwriting aggregate or another family's edge files."""
    from cfdbender.mesh._refine._obj import read_te

    one = sheet_with_strips(())
    offset = np.array([0, 5, 0])
    mesh = Composite(
        np.vstack([one.verts, one.verts + offset]),
        {
            names[0]: one.families["G"],
            names[1]: [[v + len(one.verts) for v in f] for f in one.families["G"]],
        },
        np.vstack([one.te, one.te + offset]),
        {},
    )
    source = write_source(tmp_path / "src", "wing", mesh, boundaries=declared)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GeometryWarning)
        try:
            level = refine_mesh(source, 1)
        except InputArtifactError as error:
            pytest.fail(f"A colliding family output must be renamed, not refused: {error}")
    record = json.loads(level.obj.with_suffix(".refine.json").read_text(encoding="utf-8"))
    renamed = record.get("renamed_family_outputs", {})
    assert renamed, "The collision must be resolved and recorded"
    assert any("output stem" in str(w.message) for w in caught)
    assert list(read_mesh(level.obj)[1]) == list(names)
    assert len({p.name.casefold() for p in level.files}) == len(level.files)
    points = []
    for index, name in enumerate(names):
        stem = renamed.get(name, name)
        target = level.folder / f"{level.obj.stem}.{stem}.te.txt"
        _, own = read_te(target)
        assert len(own)
        assert np.all(own[:, 1] < 3) if index == 0 else np.all(own[:, 1] > 4)
        points.extend(map(tuple, own))
        assert (level.folder / f"{level.obj.stem}.{stem}.nodes.txt").is_file()
        listing = (level.folder / f"{level.obj.stem}.{stem}.nodes.txt").read_text()
        assert all(row.split(", ")[1] == name for row in listing.splitlines()[1:])
        if not declared:
            side = tomllib.loads(level.obj.with_suffix(".obj.boundaries.toml").read_text())
            assert side["trailing_edges"][name]["file"] == target.name
    _, combined = read_te(level.obj.with_suffix(".te.txt"))
    assert sorted(points) == sorted(map(tuple, combined))
    assert level.obj.with_suffix(".nodes.txt").read_text().startswith("#")


def _continuity_source(tmp_path, *, shape="side", factor=2, keep=False):
    """Write two synthetic families for refine_mesh(source), with only A refined."""
    points, families = [], {"A": [], "B": []}

    def vertex(x, y):
        point = (x, y, 0)
        if point not in points:
            points.append(point)
        return points.index(point)

    def cell(name, x, y, *, mixed=False):
        a, b, c, d = (vertex(x, y), vertex(x + 1, y), vertex(x + 1, y + 1), vertex(x, y + 1))
        families[name].extend([[a, b, c], [a, c, d]] if mixed else [[a, b, c, d]])

    height = 2 if shape == "both" else 6
    for x in range(2):
        for y in range(height):
            cell("A", x, y)
    if shape == "both":
        for x in range(4):
            for y in range(4):
                if x >= 2 or y >= 2:
                    cell("B", x, y)
    else:
        start, stop = (10, 12) if shape == "isolated" else (2, 8 if shape == "mixed" else 4)
        for x in range(start, stop):
            for y in range(height):
                cell("B", x, y, mixed=shape == "mixed")
    config = (
        f'[families.A]\nmethod = "grid"\nnormal_factor = {factor}\n'
        f"parallel_factor = {factor}\ngoverning_parallel = true\ngoverning_normal = true\n"
    )
    if keep:
        config += '\n[families.B]\nmethod = "keep"\n'
    return write_source(
        tmp_path / "src",
        "continuity",
        Composite(np.array(points, dtype=float), families, np.zeros((0, 3)), {}),
        refine_toml=config,
    )


def _continuity_level(source):
    """Assert the continuity contract accepts the request before checking its mesh."""
    level, error = None, None
    try:
        level = refine_mesh(source)
    except InputArtifactError as caught:
        error = str(caught)
    assert error is None, f"continuity must warn and skip, not refuse: {error}"
    assert level is not None
    return level


@pytest.mark.parametrize("governing", [False, True])
@pytest.mark.parametrize(
    "shape, directions", [("side", ["parallel"]), ("both", ["parallel", "normal"])]
)
def test_mesh_nastran_continuity_undeclared_grid_changes_only_crossing_bands(
    tmp_path, shape, directions, governing
):
    """one shared direction changes one axis; an L neighbour needs both."""
    source = _continuity_source(tmp_path, shape=shape)
    if not governing:
        config = source.with_suffix(".refine.toml")
        config.write_text(config.read_text().replace(" = true", " = false"), encoding="utf-8")
    level = _continuity_level(source)
    entry = level.report["B"]
    assert entry.get("continuity", {}).get("directions") == directions
    assert entry["continuity"]["because"] == ["A"]
    assert entry["continuity"]["bands"]
    points, families = read_mesh(level.obj)
    held = points[sorted({v for f in families["B"] for v in f})]
    if shape == "side":
        assert len(set(held[:, 0])) == 3
        assert len(set(held[:, 1])) == 13
    else:
        assert len(set(held[:, 0])) == len(set(held[:, 1])) == 7
    record = json.loads(next(p for p in level.files if p.name.endswith(".refine.json")).read_text())
    labels = {b["label"] for layout in record["layouts"] for b in layout["bands"]}
    assert set(entry["continuity"]["bands"]) <= labels
    assert "B" not in record["unchanged_grids"]


@pytest.mark.parametrize("shape, factor", [("side", 1), ("isolated", 2), ("side", 2), ("both", 2)])
def test_mesh_nastran_continuity_keep_and_omitted_tables_write_the_same_mesh(
    tmp_path, shape, factor
):
    """explicit keep equals omission, and unaffected families are exact copies."""
    source = _continuity_source(tmp_path / "implicit", shape=shape, factor=factor)
    implicit = _continuity_level(source)
    explicit = _continuity_level(
        _continuity_source(tmp_path / "explicit", shape=shape, factor=factor, keep=True)
    )
    a, af = read_mesh(implicit.obj)
    b, bf = read_mesh(explicit.obj)
    assert a.tobytes() == b.tobytes() and af == bf
    if factor == 1 or shape == "isolated":
        before, faces = read_mesh(source)
        assert face_coordinates(a, af["B"]) == face_coordinates(before, faces["B"])
        assert implicit.report["B"]["method"] in ("copied", "unchanged")
        assert "continuity" not in implicit.report["B"]


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("keep", [False, True])
def test_mesh_nastran_continuity_mixed_changes_only_the_neighbour_band(tmp_path, factor, keep):
    """mixed neighbours retain every face beyond the two-layer band."""
    source = _continuity_source(tmp_path, shape="mixed", factor=factor, keep=keep)
    level = _continuity_level(source)
    before, old = read_mesh(source)
    after, new = read_mesh(level.obj)
    old_faces, new_faces = face_coordinates(before, old["B"]), face_coordinates(after, new["B"])
    if factor == 1:
        assert old_faces == new_faces
        assert "continuity" not in level.report["B"]
    else:
        assert level.report["B"].get("continuity") == {"band": "remeshed", "because": ["A"]}
        assert level.report["B"]["band"] < len(old["B"])
        assert [f for f in old_faces if min(p[0] for p in f) >= 5] == [
            f for f in new_faces if min(p[0] for p in f) >= 5
        ]


def test_mesh_nastran_continuity_keep_factors_warn_and_are_skipped(tmp_path):
    """a factor in a keep table warns and cannot request refinement."""
    source = _continuity_source(tmp_path, shape="isolated", keep=True)
    config = source.with_suffix(".refine.toml")
    config.write_text(config.read_text() + "factor = 9\n", encoding="utf-8")
    with pytest.warns(GeometryWarning, match="keep.*skipped"):
        level = _continuity_level(source)
    before, old = read_mesh(source)
    after, new = read_mesh(level.obj)
    assert face_coordinates(after, new["B"]) == face_coordinates(before, old["B"])


def test_mesh_nastran_continuity_prepare_declares_followers_and_band_comments(tmp_path):
    """prepared followers explicitly request both governing factors."""
    prepared = prepare_to_refine(_continuity_source(tmp_path))
    text = prepared.config.read_text(encoding="utf-8")
    tables = tomllib.loads(text)["families"]
    assert tables.get("B_001_grid") == {"method": "grid_auto"}
    assert "# follows A_001_grid (parallel) and A_001_grid (normal)\n[families.B_001_grid]" in text
    assert "# leads " in text and "P1" in text and "N1" in text


def _node(i: int, j: int) -> int:
    return 1 + i + (COLUMNS + 1) * j


def _element(i: int, j: int) -> int:
    return 1 + i + COLUMNS * j


def _block(records: list[tuple[int, int]]) -> str:
    body = struct.pack(">I", len(records)) + b"".join(struct.pack(">IH", e, s) for e, s in records)
    return base64.b64encode(struct.pack(">I", len(body)) + zlib.compress(body)).decode()


def _ids_block(ids: list[int]) -> str:
    body = struct.pack(">I", len(ids)) + b"".join(struct.pack(">I", i) for i in ids)
    return base64.b64encode(struct.pack(">I", len(body)) + zlib.compress(body)).decode()


def _export(path: Path, *, wide: bool, perimeter: str | None, only: bool = False) -> Path:
    lines = ["$ANSA_VERSION;25.1.2;", "BEGIN BULK"]
    for j in range(ROWS + 1):
        for i in range(COLUMNS + 1):
            x, y = f"{1000.0 * i:.1f}", f"{1000.0 * j:.1f}"
            if wide:
                lines += [
                    f"GRID*   {_node(i, j):>16}{'':>16}{x:>16}{y:>16}*",
                    f"*       {'0.':>16}",
                ]
            else:
                lines.append(f"GRID    {_node(i, j):>8}{'':>8}{x:>8}{y:>8}{'0.':>8}")
    for j in range(ROWS):
        for i in range(COLUMNS):
            g = [_node(i, j), _node(i + 1, j), _node(i + 1, j + 1), _node(i, j + 1)]
            if wide:
                lines += [
                    f"CQUAD4* {_element(i, j):>16}{1:>16}{g[0]:>16}{g[1]:>16}*",
                    f"*       {g[2]:>16}{g[3]:>16}",
                ]
            else:
                lines.append("CQUAD4  " + "".join(f"{v:>8}" for v in (_element(i, j), 1, *g)))
    lines += ["$Sheet", "PSHELL         1       1      1."]
    if perimeter is not None:
        text = f"0;{perimeter};" + ("" if only else f"{_ids_block([1, 3])};1;{_block([])};")
        lines.append("$ANSA_FE_PERIMETER;" + text[:60])
        lines += ["$" + text[k : k + 70] for k in range(60, len(text), 70)]
    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _middle_line() -> str:
    # Side 2 of the elements in column 1 joins nodes 2 and 3: the line x = 2000 mm.
    return _block([(_element(1, j), 2) for j in range(ROWS)])


@pytest.mark.parametrize("wide", [False, True])
def test_the_families_are_cut_along_the_fe_perimeters(tmp_path: Path, wide: bool) -> None:
    source = _export(tmp_path / "sheet.nas", wide=wide, perimeter=_middle_line())
    prepared = prepare_to_refine(source)
    assert prepared.patches == {"Sheet": ["Sheet_001_grid", "Sheet_002_grid"]}
    text = prepared.obj.read_text(encoding="utf-8")
    assert text.count("\ng Sheet_00") == 2
    xs = [float(line.split()[1]) for line in text.splitlines() if line.startswith("v ")]
    assert max(xs) == pytest.approx(4.0)  # MILLIMETER written in METER


def test_without_a_perimeter_block_the_families_are_cut_at_sharp_edges(tmp_path: Path) -> None:
    source = _export(tmp_path / "flat.nas", wide=False, perimeter=None)
    with pytest.warns(GeometryWarning, match="FE perimeters"):
        prepared = prepare_to_refine(source)
    assert prepared.patches == {"Sheet": ["Sheet_001_grid"]}


def test_a_block_that_does_not_read_is_warned_and_skipped(tmp_path: Path) -> None:
    source = _export(tmp_path / "bad.nas", wide=False, perimeter="AAAA!!notzlib", only=True)
    with pytest.warns(GeometryWarning, match="does not read"):
        prepared = prepare_to_refine(source)
    assert prepared.patches == {"Sheet": ["Sheet_001_grid"]}


def test_the_reader_keeps_the_grid_order_and_converts_the_unit(tmp_path: Path) -> None:
    source = _export(tmp_path / "s.nas", wide=True, perimeter=_middle_line())
    read = _nastran.read_nastran(source, unit="MILLIMETER", obj_unit="MILLIMETER")
    assert read.mesh.verts[1].tolist() == [1000.0, 0.0, 0.0]
    assert list(read.mesh.families) == ["Sheet"]
    assert len(read.mesh.families["Sheet"]) == COLUMNS * ROWS
    assert read.perimeter == {(2, 7), (7, 12)}  # x = 2000 mm, vertex index = GRID id - 1


def test_nastran_reals_take_the_short_exponent() -> None:
    assert _nastran.real("1.5-3") == pytest.approx(1.5e-3)
    assert _nastran.real("-2.+4") == pytest.approx(-2.0e4)
    assert _nastran.real("3.25E+1") == pytest.approx(32.5)


@needs_inspect
def test_visual_inspect_writes_the_pdf_and_counts_the_marked_points(tmp_path: Path) -> None:
    from cfdbender.mesh._visual_inspect import marked_points, visual_inspect

    source = _export(tmp_path / "sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source)
    te = tmp_path / "sheet.te.txt"
    te.write_text("METER\n4.0,0.5,0.0\n4.0,1.5,0.0\n9.0,9.0,9.0\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = visual_inspect(prepared.obj, te=[te], mesh_unit="METER", dpi=60)
    assert pdf.read_bytes()[:5] == b"%PDF-"
    from cfdbender.mesh._visual_inspect import (
        Scene,
        patch_colors,
        read_obj,
        read_points_file,
    )

    mesh = read_obj(prepared.obj)
    names = list(mesh.patches)
    colors, _ = patch_colors(names, None)
    marked = marked_points(Scene(mesh, names, colors, None), [read_points_file(te, "METER")])
    assert marked.points == [0, 2] or marked.points == [2, 0]
    assert marked.total_points == 3
    assert marked.on_no_patch == 1


@needs_inspect
def test_visual_inspect_png_writes_the_views_page_and_one_file_per_view(tmp_path: Path) -> None:
    from cfdbender.mesh._visual_inspect import visual_inspect

    prepared = prepare_to_refine(
        _export(tmp_path / "sheet.nas", wide=False, perimeter=_middle_line())
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = visual_inspect(prepared.obj, mesh_unit="METER", dpi=40, png=True)
    pngs = sorted(path.name for path in pdf.parent.glob(f"{pdf.stem}*.png"))
    views = [f"{pdf.stem}.{view}.png" for view in ("front", "iso", "side", "top")]
    assert pngs == sorted([f"{pdf.stem}.png", *views])
    assert all((pdf.parent / name).read_bytes()[:4] == b"\x89PNG" for name in pngs)


def test_a_segment_of_another_layout_does_not_discard_the_side_block() -> None:
    body = struct.pack(">I", 1) + struct.pack(">IH", 7, 2)
    good = base64.b64encode(struct.pack(">I", len(body)) + zlib.compress(body)).decode()
    raw = base64.b64encode(b"\x00\x00\x00\x08rawbytes").decode()
    assert _nastran.decode_sides("PART_NAME_LONG;" + raw + ";" + good) == [(7, 2)]


def _captured_points(tmp_path: Path, **units: str) -> list:
    from cfdbender.mesh._visual_inspect import load_points

    nodes = tmp_path / "n.txt"
    nodes.write_text("#  Id, Name, X, Y, Z\n1, a, 1000.0, 0.0, 0.0\n", encoding="utf-8")
    points = load_points([nodes], [], **units)
    return [float(v) for v in points[0].xyz[0]]


@needs_inspect
def test_length_unit_names_are_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "u").mkdir()
    (tmp_path / "l").mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        upper = _captured_points(tmp_path / "u", mesh_unit="METER", points_unit="MILLIMETER")
        lower = _captured_points(tmp_path / "l", mesh_unit="meter", points_unit="millimeter")
    assert lower == upper == pytest.approx([1.0, 0.0, 0.0])


@needs_inspect
def test_an_unknown_unit_warns_and_draws_the_points_as_written(tmp_path: Path) -> None:
    with pytest.warns(GeometryWarning, match="(?i)furlong"):
        xyz = _captured_points(tmp_path, mesh_unit="METER", points_unit="furlong")
    assert xyz == [1000.0, 0.0, 0.0]


@needs_inspect
def test_an_unknown_nastran_unit_warns_and_reads_the_mesh_unscaled(tmp_path: Path) -> None:
    from cfdbender.mesh._visual_inspect import visual_inspect

    source = _export(tmp_path / "sheet.nas", wide=False, perimeter=_middle_line())
    with pytest.warns(GeometryWarning, match="(?i)furlong"):
        pdf = visual_inspect(source, nastran_unit="furlong", dpi=40)
    assert pdf.read_bytes()[:5] == b"%PDF-"


@needs_inspect
def test_a_latin1_obj_keeps_distinct_group_names_and_warns(tmp_path: Path) -> None:
    from cfdbender.mesh._visual_inspect import read_obj

    obj = tmp_path / "latin.obj"
    obj.write_bytes(b"v 0 0 0\nv 1 0 0\nv 0 1 0\ng wing\xe9\nf 1 2 3\ng wing\xe8\nf 1 2 3\n")
    with pytest.warns(GeometryWarning, match="not UTF-8"):
        mesh = read_obj(obj)
    assert len(mesh.patches) == 2


@needs_inspect
def test_a_malformed_obj_line_is_skipped_and_counted_not_raised(tmp_path: Path) -> None:
    from cfdbender.mesh._visual_inspect import read_obj

    obj = tmp_path / "bad.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 x 0\nf 1 2 3\nf 1 2 q\nf 1 2 4\n")
    with pytest.warns(GeometryWarning, match=r"3 line\(s\)"):
        mesh = read_obj(obj)
    assert [len(f) for f in mesh.patches.values()] == [1]


@needs_inspect
def test_fr3298_prepare_marks_one_governing_patch_per_touching_group(tmp_path):
    """one governing grid patch per touching group; the other grids are left out."""
    import tomllib

    from cfdbender.mesh._refine._prepare import prepare_to_refine

    src = tmp_path / "plate.obj"
    lines = []
    for j in range(3):
        for i in range(7):
            lines.append(f"v {i:.1f} {j:.1f} 0.0")
    lines.append("g A")
    for j in range(2):
        for i in range(3):
            a = j * 7 + i + 1
            lines.append(f"f {a} {a + 1} {a + 8} {a + 7}")
    lines.append("g B")
    for j in range(2):
        for i in range(3, 6):
            a = j * 7 + i + 1
            lines.append(f"f {a} {a + 1} {a + 8} {a + 7}")
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prepared = prepare_to_refine(src, visual_inspect=True)
    assert prepared.inspect == tmp_path / "plate_patches.inspect.pdf"
    pdf = prepared.inspect.read_bytes()
    assert pdf.startswith(b"%PDF") and pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") >= 2
    assert len(prepared.governing) == 1
    table = tomllib.loads(prepared.config.read_text(encoding="utf-8"))["families"]
    governing = [
        n for n, t in table.items() if t.get("governing_parallel") and t.get("governing_normal")
    ]
    grids = [p for ps in prepared.patches.values() for p in ps if p.endswith("_grid")]
    assert len(grids) == 2 and len(governing) == 1
    assert all(table[g] == {"method": "grid_auto"} for g in grids if g not in governing)


# ------------------------------------- governing families: governing-r1 cases


def _layout(specs):
    verts = np.array([(x, y, 0) for x in range(10) for y in range(3)], dtype=float)
    families = {}
    for name, start, end in (("A", 0, 4), ("B", 4, 6), ("C", 6, 9)):
        families[name] = [
            [x * 3 + y, (x + 1) * 3 + y, (x + 1) * 3 + y + 1, x * 3 + y + 1]
            for x in range(start, end)
            for y in range(2)
        ]
    if "D" in specs:
        # A corner connects the layout without adding D to the A/B/C shared band.
        verts = np.vstack([verts, [(9, 3, 0), (10, 2, 0), (10, 3, 0)]])
        families["D"] = [[29, 31, 32, 30]]
    grids = {
        n: _blocks.recover_family(
            verts, fs, {v for f in fs for v in f if verts[v, 1] == (2 if n == "D" else 0)}
        )
        for n, fs in families.items()
    }
    assert all(not isinstance(g, str) for g in grids.values())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layouts, late = _blocks.plan(
            verts, families, grids, specs, {n: set() for n in families}, where="plates"
        )
    assert late == []
    return layouts["A"], [str(w.message) for w in caught]


def test_mesh_nastran_continuity_band_labels_are_stable_and_name_the_dominant_arc():
    """source-order leads and their first arcs number each direction."""

    def planned():
        return _layout(
            {
                "A": FamilySpec(factor=2, governing_parallel=True),
                "B": FamilySpec(factor=3, governing_normal=True),
                "C": FamilySpec(factor=1),
            }
        )[0]

    first, second = planned(), planned()
    labels = getattr(first, "labels", {})
    assert set(labels) == set(range(len(first.blocks.arcs))), "the planner must label every band"
    assert labels == second.labels and first.leads == second.leads
    bands = _blocks.arc_bands(first.blocks, labels)
    assert len(set(labels.values())) == len(set(bands.values()))
    for arc, label in labels.items():
        lead, dominant = first.leads[label]
        mine = {a for p in first.patches_of(lead) for a, _ in first.blocks.sides[p]}
        assert dominant == min(a for a in mine if labels[a] == label)
        assert labels[dominant] == label
        assert all(labels[a] == label for a in labels if bands[a] == bands[arc])
    for prefix in ("P", "N"):
        ordered = sorted(
            (label for label in first.leads if label.startswith(prefix)),
            key=lambda label: (first.names.index(first.leads[label][0]), first.leads[label][1]),
        )
        assert ordered == [f"{prefix}{i}" for i in range(1, len(ordered) + 1)]


def test_mesh_nastran_continuity_json_records_label_direction_lead_and_count(tmp_path):
    """the written record retains the planner's band identifiers."""
    source = _topology_source(tmp_path)
    level = refine_mesh(source)
    record = next(p for p in level.files if p.name.endswith(".refine.json"))
    bands = json.loads(record.read_text(encoding="utf-8"))["layouts"][0]["bands"]
    assert bands and all("label" in band and "count" in band for band in bands)
    for band in bands:
        assert band["label"].startswith("P" if band["direction"] == "parallel" else "N")
        assert band["lead"] in ("A", "B")
        assert band["dominant_arc"] in band["arcs"]
        assert band["count"] > 0


def test_mesh_nastran_continuity_dominant_arc_supplies_the_selected_directional_count():
    """conflicting local directions retain the count-setting arc."""
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)])
    blocks = _blocks.Blocks(
        arcs=((2, 3), (0, 1), (0, 2), (1, 3), (1, 4), (3, 5)),
        sides=(
            ((1, False), (0, False), (2, False), (3, False)),
            ((1, False), (2, False), (4, False), (5, False)),
        ),
        patches=(np.array([[0, 1], [2, 3]]), np.array([[0, 2], [4, 5]])),
        cells=(np.array([[0]]), np.array([[1]])),
        te_arcs=frozenset(),
    )
    layouts, late = _blocks.plan(
        points,
        {"B": [[0, 1, 3, 2], [0, 2, 5, 4]]},
        {"B": blocks},
        {"B": FamilySpec(chordwise=3, spanwise=2, governing_normal=True)},
        {"B": {(0, 1)}},
        where="conflicting patches",
    )
    assert not late
    layout = layouts["B"]
    assert hasattr(layout, "labels"), "the planner must retain its band labels"
    assert layout.counts[0] == 3
    lead, dominant = layout.leads[layout.labels[0]]
    assert lead == "B" and dominant != 0
    assert any(
        told[dominant].direction == "normal"
        for told in layout.directions.values()
        if dominant in told
    )


def test_mesh_nastran_a_shared_band_crosses_three_families_with_one_count():
    """opposite sides and shared arcs propagate A's count."""
    layout, messages = _layout(
        {
            "A": FamilySpec(factor=2, governing=True),
            "B": FamilySpec(factor=3),
            "C": FamilySpec(factor=4),
        }
    )
    assert layout.names == ("A", "B", "C")
    shared = set(a for a, _ in layout.blocks.sides[0]) & set(a for a, _ in layout.blocks.sides[1])
    assert len(shared) == 1
    bands = _blocks.arc_bands(layout.blocks, range(len(layout.blocks.arcs)))
    shared_band = bands[next(iter(shared))]
    assert {layout.counts[a] for a, band in bands.items() if band == shared_band} == {4}
    for sides in layout.blocks.sides:
        a, b, c, d = (layout.counts[arc] for arc, _ in sides)
        assert a == b and c == d
    for name in ("B", "C"):
        message = next(m for m in messages if f"family {name} follows A" in m)
        assert "on 1 band(s)" in message, message


def test_mesh_nastran_declared_family_leads_larger_grid_auto_and_warning_explains_override():
    """a grid_auto offer never outranks a declared offer."""
    layout, messages = _layout(
        {
            "A": FamilySpec(factor=2, method="grid_auto"),
            "B": FamilySpec(factor=3),
            "C": FamilySpec(factor=4, method="grid_auto"),
            "D": FamilySpec(factor=2, governing=True),
        }
    )
    shared = set(a for a, _ in layout.blocks.sides[0]) & set(a for a, _ in layout.blocks.sides[1])
    assert {layout.counts[a] for a in shared} == {6}
    message = next(m for m in messages if "family A follows B" in m)
    assert 'method = "grid"' in message and "factors" in message, message


def test_mesh_nastran_prepare_declares_isolated_grid_beside_three_touching_grids(tmp_path):
    """touching followers are explicit grid_auto tables."""
    source = tmp_path / "plates.obj"
    lines = [f"v {x} {y} 0" for x in range(15) for y in range(3)]
    for name, start, end in (("A", 0, 2), ("B", 2, 5), ("C", 5, 9), ("D", 12, 14)):
        lines.append(f"g {name}")
        for x in range(start, end):
            for y in range(2):
                a = x * 3 + y + 1
                lines.append(f"f {a} {a + 3} {a + 4} {a + 1}")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prepared = prepare_to_refine(source)
    tables = tomllib.loads(prepared.config.read_text(encoding="utf-8"))["families"]
    a, b, c, d = (prepared.patches[n][0] for n in "ABCD")
    assert tables[a]["governing_parallel"] is True
    assert tables[c]["governing_normal"] is True
    assert tables[b] == {"method": "grid_auto"}
    assert d in tables, "the isolated grid must retain its own factor table"
    assert tables[d] == {"method": "grid", "factor": 1.0}
    level = refine_mesh(prepared.obj)
    record = next(p for p in level.files if p.name.endswith(".refine.json"))
    report = json.loads(record.read_text(encoding="utf-8"))["families"]
    assert report[b]["provenance"] == {f"propagated_from:{a}": 2, f"propagated_from:{c}": 2}


@pytest.mark.parametrize("declared", [False, True])
def test_mesh_nastran_governing_does_not_refine_an_unselected_isolated_grid(tmp_path, declared):
    """selecting A leaves isolated B unchanged."""
    verts = np.array([(x, y, 0) for x in (0, 1, 4, 5) for y in (0, 1)], dtype=float)
    families = {"A": [[0, 2, 3, 1]], "B": [[4, 6, 7, 5]]}
    mesh = Composite(verts, families, np.zeros((0, 3)), {})
    source = write_source(
        tmp_path / "src",
        "plates",
        mesh,
        refine_toml='[families.A]\nmethod = "grid"\nfactor = 2\ngoverning = true\n'
        + ("[families.B]\nfactor = 3\n" if declared else ""),
    )
    level = refine_mesh(source, families=["A"])
    out, faces = read_mesh(level.obj)
    assert face_coordinates(out, faces["B"]) == face_coordinates(verts, families["B"])
    assert level.report["B"]["method"] == "copied"
    assert len(faces["A"]) == 4


def test_mesh_nastran_governing_changes_a_short_level_tag():
    """swapping the leader cannot reuse one level folder."""
    first = RefineRequest(
        specs={"A": FamilySpec(factor=2, governing=True), "B": FamilySpec(factor=3)}
    )
    second = replace(
        first, specs={"A": FamilySpec(factor=2), "B": FamilySpec(factor=3, governing=True)}
    )
    assert canonical_selection(first, ["A", "B"]) != canonical_selection(second, ["A", "B"])
    assert level_tag(first, ["A", "B"]) != level_tag(second, ["A", "B"])
    assert tag_selection(first, ["A", "B"]) == ["A", "B"]


def test_mesh_nastran_grid_auto_tags_include_the_leaders_directional_factors():
    """inherited factors and grid_auto participate in the hash."""
    first = RefineRequest(
        specs={
            "A": FamilySpec(chordwise=2, spanwise=3, governing=True),
            "B": FamilySpec(method="grid_auto"),
        }
    )
    second = replace(first, specs={**first.specs, "A": replace(first.specs["A"], spanwise=4)})
    assert level_tag(first, ["A", "B"]) != level_tag(second, ["A", "B"])
    assert tag_selection(first, ["A", "B"]) == ["A", "B"]
    explicit = replace(first, tag="named")
    assert level_tag(explicit, ["A", "B"]) == "named"
    assert tag_selection(explicit, ["A", "B"]) is None
    plain = RefineRequest(specs={"A": FamilySpec(factor=2), "B": FamilySpec(factor=2)})
    assert level_tag(plain, ["A", "B"]) == "R2"


def test_govr1_q02_governing_on_grid_auto_is_warned_and_skipped():
    """(GOVR1-Q02): a grid_auto family has no factors of its own to lead with."""
    from cfdbender._errors import GeometryWarning
    from cfdbender.mesh._refine._config import grid_auto_spec

    with pytest.warns(GeometryWarning, match="cannot lead; governing is skipped"):
        spec = grid_auto_spec("[families.G]", {"method": "grid_auto"}, governing=True)
    assert spec.method == "grid_auto" and not spec.governing


@pytest.mark.parametrize("count", [21, 41, 121])
@needs_inspect
def test_patch_pdf_legend_stays_inside_its_column(tmp_path, monkeypatch, count):
    """GOVR2-001: long names and many patches must not clip the page's legend."""
    from matplotlib.backends.backend_pdf import PdfPages

    from cfdbender.mesh._visual_inspect import BOTTOM, RIGHT, TOP, patches_pdf

    source = tmp_path / "patches.obj"
    names = [f"MainSurface_{i:03d}_grid" for i in range(count)]
    lines = ["v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0"]
    for name in names:
        lines += [f"g {name}", "f 1 2 3 4"]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save = PdfPages.savefig
    pages = []

    def checked_save(pdf, fig, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = fig.legends[0]
        bounds = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        width, height = fig.get_size_inches()
        assert (width - RIGHT) / width <= bounds.x0 < bounds.x1 <= 1 - 0.1 / width
        assert BOTTOM / height <= bounds.y0 < bounds.y1 <= 1 - TOP / height
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == [f"{names[0]}  (GOVERNING PARALLEL, GOVERNING NORMAL)", *names[1:]]
        for ax in fig.axes[:4]:
            assert sum("GOVERNING" in t.get_text() for t in ax.texts) == 1
            assert any("GOVERNING PARALLEL\nGOVERNING NORMAL" in t.get_text() for t in ax.texts)
        pages.append(fig)
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", checked_save)
    target = patches_pdf(
        source, {"MainSurface": names}, {names[0]: (True, True)}, tmp_path / "patches.pdf", dpi=40
    )
    assert target.read_bytes().startswith(b"%PDF")
    assert len(pages) == 1


def _topology_source(tmp_path, *, corner=False, ring=False):
    """Synthetic plates: refine_mesh(source) must transfer directions through curves."""
    points, families = [], {}

    def vertex(x, y):
        point = (x, y, 0)
        if point not in points:
            points.append(point)
        return points.index(point)

    def plate(name, xs, ys):
        families[name] = [
            [vertex(x, y), vertex(u, y), vertex(u, v), vertex(x, v)]
            for x, u in zip(xs, xs[1:], strict=False)
            for y, v in zip(ys, ys[1:], strict=False)
        ]

    if ring:
        plate("A", [-1, 0, 1], [-1, 0, 1])
        edge = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        for side, name in enumerate("BCDE"):
            families[name] = [
                [
                    vertex(s * x, s * y),
                    vertex(s * u, s * v),
                    vertex((s + 1) * u, (s + 1) * v),
                    vertex((s + 1) * x, (s + 1) * y),
                ][::-1]
                for s in (1, 2)
                for (x, y), (u, v) in list(zip(edge, edge[1:] + edge[:1], strict=True))[
                    side * 2 : side * 2 + 2
                ]
            ]
    else:
        plate("A", [0, 1, 2], list(range(7)))
        plate("B", [2, 7, 12], list(range(7)))
        if corner:
            plate("C", [0, -1, -2], [6, 7, 8])
    mesh = Composite(np.array(points, dtype=float), families, np.zeros((0, 3)), {})
    return write_source(
        tmp_path / "src",
        "topology",
        mesh,
        refine_toml='[families.A]\nmethod = "grid"\nnormal_factor = 2\n'
        "parallel_factor = 3\ngoverning = true\n"
        + "".join(f'\n[families.{n}]\nmethod = "grid_auto"\n' for n in families if n != "A"),
    )


def test_mesh_nastran_topology_follower_uses_governing_directions(tmp_path):
    source = _topology_source(tmp_path)
    level = refine_mesh(source)
    record = next(p for p in level.files if p.name.endswith(".refine.json"))
    report = json.loads(record.read_text(encoding="utf-8"))["families"]
    assert report["B"]["blocks"] == [{"rows": [2, 4], "columns": [6, 18]}]
    assert report["B"]["grid_auto"]["directions"] == "from A"


@pytest.mark.parametrize("force", [False, True])
def test_mesh_nastran_coarsening_checks_propagated_directions(tmp_path, force):
    """follower limits use the planner's propagated axes."""
    source = _topology_source(tmp_path)
    config = source.with_suffix(".refine.toml")
    config.write_text(
        config.read_text()
        .replace("normal_factor = 2", "normal_factor = 1")
        .replace("parallel_factor = 3", "parallel_factor = 0.25"),
        encoding="utf-8",
    )
    level = refine_mesh(source, force_factors=force)
    assert level.report["B"]["grid_auto"]["factors"] == [1, 0.25]
    assert level.report["B"]["blocks"] == [{"rows": [2, 2], "columns": [6, 2]}]


@pytest.mark.parametrize("reason", ["conflict", "unlinked"])
def test_mesh_nastran_topology_fallback_warns_and_records_own_reference(tmp_path, reason):
    source = _topology_source(tmp_path, ring=reason == "conflict", corner=reason == "unlinked")
    name = "B" if reason == "conflict" else "C"
    with pytest.warns(GeometryWarning, match=f"family {name}.*{reason}"):
        level = refine_mesh(source)
    record = next(p for p in level.files if p.name.endswith(".refine.json"))
    report = json.loads(record.read_text(encoding="utf-8"))["families"]
    assert report[name]["grid_auto"]["directions"] == f"own reference: {reason}"
    assert set(report[name].get("provenance", {})) == {f"own_orientation_{reason}"}
    assert level.report[name]["method"] == "grid"
    config = source.with_suffix(".refine.toml")
    config.write_text(
        config.read_text(encoding="utf-8").replace(f'[families.{name}]\nmethod = "grid_auto"\n', "")
        + f'\n[families.{name}]\nmethod = "grid"\nnormal_factor = 2\nparallel_factor = 3\n',
        encoding="utf-8",
    )
    own = refine_mesh(source, out_dir=tmp_path / "own")
    assert level.report[name]["blocks"] == own.report[name]["blocks"]


def test_mesh_nastran_topology_prepare_declares_corner_only_patch(tmp_path):
    source = _topology_source(tmp_path, corner=True)
    with pytest.warns(GeometryWarning, match="C_001_grid.*unlinked"):
        prepared = prepare_to_refine(source)
    text = prepared.config.read_text(encoding="utf-8")
    tables = tomllib.loads(text)["families"]
    assert tables["A_001_grid"]["governing_parallel"] is True
    assert tables["B_001_grid"]["governing_normal"] is True
    assert tables["C_001_grid"] == {"method": "grid", "factor": 1.0}
    assert "# not a follower of A_001_grid, B_001_grid: unlinked\n[families.C_001_grid]" in text


def test_mesh_nastran_planner_factors_do_not_cross_a_mixed_bridge(tmp_path):
    """R2: each follower takes the governing factors of its grid layout."""
    points, families = [], {}

    def vertex(x, y):
        point = (x, y, 0)
        if point not in points:
            points.append(point)
        return points.index(point)

    def plate(name, xs):
        families[name] = [
            [vertex(x, y), vertex(u, y), vertex(u, y + 1), vertex(x, y + 1)]
            for x, u in zip(xs, xs[1:], strict=False)
            for y in range(6)
        ]

    plate("A", [0, 1, 2])
    plate("B", [2, 7, 12])
    plate("C", [30, 40, 50])
    plate("D", [50, 51, 52])
    families["bridge_mixed"] = [[vertex(12, 6), vertex(30, 6), vertex(20, 8)]]
    mesh = Composite(np.array(points, dtype=float), families, np.zeros((0, 3)), {})
    source = write_source(
        tmp_path / "src",
        "groups",
        mesh,
        refine_toml='[families.A]\nmethod = "grid"\nnormal_factor = 2\n'
        'parallel_factor = 3\ngoverning = true\n[families.C]\nmethod = "grid"\n'
        "normal_factor = 4\nparallel_factor = 5\ngoverning = true\n"
        '[families.B]\nmethod = "grid_auto"\n[families.D]\nmethod = "grid_auto"\n'
        "[families.bridge_mixed]\nfactor = 1\n",
    )
    level = refine_mesh(source)
    assert level.report["B"]["grid_auto"]["follows"] == "A"
    assert level.report["B"]["grid_auto"]["factors"] == [2, 3]
    assert level.report["D"]["grid_auto"]["follows"] == "C"
    assert level.report["D"]["grid_auto"]["factors"] == [4, 5]


def test_mesh_nastran_planner_governing_is_resolved_per_direction(tmp_path):
    """A leads parallel, C leads normal, and B inherits both factors."""
    from cfdbender.mesh._refine._config import resolve_request

    source = tmp_path / "plates.obj"
    source.with_suffix(".refine.toml").write_text(
        "[families.A]\nfactor = 2\ngoverning_parallel = true\n"
        '[families.B]\nmethod = "grid_auto"\n'
        "[families.C]\nfactor = 4\ngoverning_normal = true\n",
        encoding="utf-8",
    )
    request = None
    try:
        request = resolve_request(source, ["A", "B", "C"])
    except InputArtifactError:
        pass
    assert request is not None, "directional governing keys must be accepted"
    layout, _ = _layout(request.specs)
    shared = set(a for a, _ in layout.blocks.sides[0]) & set(a for a, _ in layout.blocks.sides[1])
    assert {layout.counts[a] for a in shared} == {8}
    assert request.specs["B"].directions == (4, 2)
    records = [d for p in layout.patches_of("B") for d in layout.directions[p].values()]
    assert {(d.direction, d.source_family) for d in records} == {("parallel", "A"), ("normal", "C")}


def test_mesh_nastran_planner_legacy_governing_warns_and_means_both(tmp_path):
    """the earlier spelling remains accepted as both directions."""
    from cfdbender.mesh._refine._config import resolve_request

    source = tmp_path / "plates.obj"
    source.with_suffix(".refine.toml").write_text(
        "[families.A]\nfactor = 2\ngoverning = true\n",
        encoding="utf-8",
    )
    with pytest.warns(GeometryWarning, match="earlier spelling"):
        request = resolve_request(source, ["A", "B"])
    assert getattr(request.specs["A"], "leads", ()) == (True, True)
    assert "B" not in request.specs


@pytest.mark.parametrize("key", ["governing_parallel", "governing_normal"])
@pytest.mark.parametrize("value", ['"true"', "1", "[]"])
def test_mesh_nastran_governing_flags_require_booleans(tmp_path, key, value):
    """malformed governing values need explicit permissive parsing."""
    from cfdbender.mesh._refine._config import resolve_request

    source = tmp_path / "plates.obj"
    source.with_suffix(".refine.toml").write_text(
        f"[families.A]\nfactor = 2\n{key} = {value}\n", encoding="utf-8"
    )
    with pytest.raises(InputArtifactError, match=f"{key}.*not true or false"):
        resolve_request(source, ["A"])
    with pytest.warns(GeometryWarning, match=f"{key}.*not true or false"):
        request = resolve_request(source, ["A"], accept_unknown=True)
    assert request.specs["A"].leads == (False, False)


def test_mesh_nastran_prepare_short_rearmost_line_refines_its_own_toml(tmp_path):
    """R1: the written reference cannot certify this sheet as a grid."""
    widths = [2, 4, 6, 8, 10, 10, 10, 8, 6, 4, 2]
    points = np.array([(i * widths[j] / 2, j, 0) for j in range(11) for i in range(3)])
    faces = [
        [j * 3 + i, j * 3 + i + 1, (j + 1) * 3 + i + 1, (j + 1) * 3 + i]
        for j in range(10)
        for i in range(2)
    ]
    source = write_source(
        tmp_path / "src", "short_reference", Composite(points, {"A": faces}, np.zeros((0, 3)), {})
    )
    prepared = prepare_to_refine(source)
    assert prepared.patches["A"] == ["A_001_grid"]
    error = None
    try:
        level = refine_mesh(prepared.obj)
    except InputArtifactError as caught:
        error = str(caught)
    assert error is None, f"prepare's own TOML must refine without refusal: {error}"
    assert level.obj.is_file()


def test_mesh_nastran_prepare_tube_touching_sheet_refines_its_own_toml(tmp_path):
    """R3: a tube excluded from the sheet's layout is declared auto."""
    from mesh_fixtures import body_of_revolution

    body = body_of_revolution()
    points = body.verts.copy()
    points[:, :2] = points[:, :2] @ np.array(
        [[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]]
    )
    corner = int(np.argmin(points[:, 0]))
    origin, sheet = points[corner].copy(), [corner]
    for j in range(3):
        for i in range(3):
            if i or j:
                sheet.append(len(points))
                points = np.vstack([points, origin + [-i, j, 0]])
    faces = [
        [sheet[j * 3 + i], sheet[j * 3 + i + 1], sheet[(j + 1) * 3 + i + 1], sheet[(j + 1) * 3 + i]]
        for j in range(2)
        for i in range(2)
    ]
    source = write_source(
        tmp_path / "src",
        "tube_sheet",
        Composite(points, {"A": faces, "B": body.faces}, np.zeros((0, 3)), {}),
    )
    prepared = prepare_to_refine(source, max_angle=0)
    error = None
    try:
        level = refine_mesh(prepared.obj)
    except InputArtifactError as caught:
        error = str(caught)
    assert error is None, f"prepare must not demand an unjoinable tube grid: {error}"
    tables = tomllib.loads(prepared.config.read_text(encoding="utf-8"))["families"]
    assert tables["B_001_grid"]["method"] == "auto"
    assert level.obj.is_file()
    record = next(p for p in level.files if p.name.endswith(".refine.json"))
    report = json.loads(record.read_text(encoding="utf-8"))["families"]
    assert report["A_001_grid"]["directions"] == {"parallel": 0, "normal": 0}
    assert report["A_001_grid"]["provenance"] == {}


@pytest.mark.parametrize("corner", [False, True])
def test_mesh_nastran_planner_json_records_directions_provenance_and_band_leads(tmp_path, corner):
    """JSON retains arc direction counts, their origins and each band's lead."""
    source = _topology_source(tmp_path, corner=corner)
    level = refine_mesh(source)
    record = next(p for p in level.files if p.name.endswith(".refine.json"))
    report = json.loads(record.read_text(encoding="utf-8"))
    families = report["families"]
    for name in ("A", "B"):
        assert families[name].get("directions") == {"parallel": 2, "normal": 2}
    assert families["A"].get("provenance") == {"longest_side": 4}
    assert families["B"].get("provenance") == {"propagated_from:A": 4}
    if corner:
        assert families["C"].get("directions") == {"parallel": 2, "normal": 2}
        assert families["C"].get("provenance") == {"own_orientation_unlinked": 4}
    layouts = report.get("layouts", [])
    assert len(layouts) == 1
    assert layouts[0]["families"] == ["A", "B", *(["C"] if corner else [])]
    bands = layouts[0]["bands"]
    assert any(
        b["lead"] == "A" and b["direction"] == "parallel" and len(b["arcs"]) == 3 for b in bands
    )
    assert all(b["lead"] in layouts[0]["families"] and b["arcs"] for b in bands)
    arcs = [a for band in bands for a in band["arcs"]]
    assert len(arcs) == len(set(arcs)), "each layout arc belongs to exactly one reported band"


@needs_inspect
def test_mesh_nastran_visual_prepare_has_patches_then_direction_colors(tmp_path, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import to_rgba

    pages = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        texts = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
        colors = [
            tuple(c) for ax in fig.axes[:4] for c in ax.collections for c in c.get_edgecolors()
        ]
        pages.append((texts, colors))
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    source = _export(tmp_path / "sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source, visual_inspect=True)
    assert len(pages) == 2 * len(prepared.patches)
    texts, colors = pages[1]
    assert "parallel" in texts and "normal" in texts
    assert to_rgba("#0072b2") in colors and to_rgba("#d55e00") in colors


@needs_inspect
def test_mesh_nastran_visual_dominant_label_is_bold_and_framed(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace

    from matplotlib.backends.backend_pdf import PdfPages

    from cfdbender.mesh._visual_inspect import patches_pdf

    assert "layouts" in inspect.signature(patches_pdf).parameters
    source = tmp_path / "sheet.obj"
    source.write_text("v 0 0 0\nv 2 0 0\nv 2 1 0\nv 0 1 0\ng A\nf 1 2 3 4\n")
    layout = SimpleNamespace(
        names=("A",),
        holder=(0,),
        blocks=SimpleNamespace(arcs=((0, 1), (1, 2), (2, 3), (3, 0))),
        directions={
            0: {
                a: _blocks.ArcDirection("parallel" if a % 2 == 0 else "normal", "A", "longest_side")
                for a in range(4)
            }
        },
        labels={0: "P1", 1: "N1", 2: "P1", 3: "N1"},
        leads={"P1": ("A", 0), "N1": ("A", 1)},
    )
    captured = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        captured.append([t for ax in fig.axes[:4] for t in ax.texts])
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    patches_pdf(
        source,
        {"A": ["A"]},
        {"A": (True, True)},
        tmp_path / "out.pdf",
        layouts={"A": layout},
        dpi=40,
    )
    labels = [t for t in captured[1] if t.get_text() in {"P1", "N1"}]
    assert len(labels) == 16
    assert sum(t.get_fontweight() == "bold" and t.get_bbox_patch() is not None for t in labels) == 8
    assert sum(t.get_fontweight() == "normal" and t.get_bbox_patch() is None for t in labels) == 8


@needs_inspect
def test_mesh_nastran_visual_refine_writes_pdf_beside_level(tmp_path):
    import inspect

    assert "visual_inspect" in inspect.signature(refine_mesh).parameters
    (tmp_path / "source").mkdir()
    source = _export(tmp_path / "source/sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source)
    level = refine_mesh(prepared.obj, 2, out_dir=tmp_path / "levels", visual_inspect=True)
    pdf = level.folder / f"{level.folder.name}.refine-governors.pdf"
    assert pdf in level.files and pdf.read_bytes().startswith(b"%PDF")
    assert level.audit.path.is_file() and level.report


def _p0502_pages(monkeypatch):
    """Record the texts of every page saved to a PDF, in order."""
    from matplotlib.backends.backend_pdf import PdfPages

    pages: list[str] = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        texts = [t.get_text() for t in fig.texts]
        for ax in fig.axes:
            texts += [t.get_text() for t in ax.texts]
            texts += [c.get_text().get_text() for tb in ax.tables for c in tb.get_celld().values()]
        pages.append("\n".join(texts))
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    return pages


@needs_inspect
def test_mesh_nastran_refine_governors_pdf_one_page_per_lead_and_direction(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    source = _export(tmp_path / "source/sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source)
    pages = _p0502_pages(monkeypatch)
    level = refine_mesh(prepared.obj, 2, out_dir=tmp_path / "levels", visual_inspect=True)
    name = level.folder.name
    pdf = level.folder / f"{name}.refine-governors.pdf"
    assert pdf in level.files and pdf.read_bytes().startswith(b"%PDF")
    assert not (level.folder / f"{name}.inspect.pdf").exists()
    assert not any(p.name.endswith(".inspect.pdf") for p in level.files)
    data = json.loads((level.folder / f"{name}.refine.json").read_text(encoding="utf-8"))
    pairs = {(b["lead"], b["direction"]) for lay in data["layouts"] for b in lay["bands"]}
    pairs |= {
        (lead, d) for g in data.get("component_governors", {}).values() for d, lead in g.items()
    }
    assert len(pairs) == 3  # the fixture's planner leads N1, N2 and P1
    assert "requested and achieved factors" in pages[1]
    pages.pop(1)  # the factor comparison sits between the summary and the governors
    governors = pages[1 : 1 + len(pairs)]
    assert "governing families" in pages[0]
    assert all(lead in pages[0] for lead, _ in pairs)
    drawn = set()
    for text in governors:
        found = [
            (lead, d) for lead, d in pairs if f"{lead} governs the {d.upper()} direction" in text
        ]
        assert len(found) == 1, text
        drawn |= set(found)
    assert drawn == pairs
    # The patches and directions pages follow the governor pages in the same file.
    assert pages[1 + len(pairs) :] and all(": family " in t for t in pages[1 + len(pairs) :])


@needs_inspect
def test_mesh_nastran_standalone_visual_inspect_keeps_its_own_name(tmp_path):
    from cfdbender.mesh._visual_inspect import visual_inspect

    (tmp_path / "source").mkdir()
    source = _export(tmp_path / "source/sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source)
    level = refine_mesh(prepared.obj, 2, out_dir=tmp_path / "levels", visual_inspect=True)
    governors = level.folder / f"{level.folder.name}.refine-governors.pdf"
    before = governors.read_bytes()
    written = visual_inspect(level.obj)
    assert written == level.obj.with_name(level.obj.stem + ".inspect.pdf")
    assert written.read_bytes().startswith(b"%PDF")
    assert governors.read_bytes() == before


@needs_inspect
def test_mesh_nastran_component_governor_governs_its_other_refined_grids(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cfdbender.mesh._inspect_pages import refined_pdf
    from cfdbender.mesh._refine._config import FamilySpec

    vertices = np.array([[x, y, 0.0] for x in range(5) for y in range(2)], dtype=float)
    quad = lambda i: [[2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1]]  # noqa: E731
    work = SimpleNamespace(
        mesh=SimpleNamespace(verts=vertices, families={n: quad(i) for i, n in enumerate("ABCK")}),
        bands={},
        parts={},
        specs={
            "A": FamilySpec(method="grid", spanwise=2.0),
            "B": FamilySpec(method="grid", chordwise=3.0),
            "C": FamilySpec(method="grid_auto"),
            "K": FamilySpec(method="keep"),
        },
        layouts={},
        te_unit="METER",
    )
    record = tmp_path / "level.refine.json"
    record.write_text(
        json.dumps(
            {
                "components": {"Wing": ["A", "B", "C", "K"]},
                "component_governors": {"Wing": {"parallel": "A", "normal": "B"}},
                "families": {},
                "layouts": [],
            }
        ),
        encoding="utf-8",
    )
    pages = _p0502_pages(monkeypatch)
    pdf = refined_pdf(work, tmp_path / "level.obj", record)
    assert pdf == tmp_path / "level.refine-governors.pdf" and pdf.is_file()
    assert "A governs the PARALLEL direction" in pages[1]
    assert "Parallel factor 2. It governs 2 other families" in " ".join(pages[1].split())
    assert "B governs the NORMAL direction" in pages[2]
    assert "Normal factor 3. It governs 2 other families" in " ".join(pages[2].split())
    # The title carries the direction; no framed label repeats it over the mesh.
    assert not any("GOVERNS " in page for page in pages[1:3])


@needs_inspect
def test_mesh_nastran_unreadable_factor_record_warns_and_still_writes_the_pdf(tmp_path):
    """The factor table is a reading aid: a record it cannot read leaves it out, never the PDF."""
    from types import SimpleNamespace

    from cfdbender._errors import GeometryWarning
    from cfdbender.mesh._inspect_pages import refined_pdf
    from cfdbender.mesh._refine._config import FamilySpec

    vertices = np.array([[x, y, 0.0] for x in range(3) for y in range(2)], dtype=float)
    work = SimpleNamespace(
        mesh=SimpleNamespace(verts=vertices, families={"A": [[0, 2, 3, 1]], "B": [[2, 4, 5, 3]]}),
        bands={},
        parts={},
        specs={"A": FamilySpec(method="grid", spanwise=2.0), "B": FamilySpec(method="grid_auto")},
        layouts={},
        te_unit="METER",
    )
    record = tmp_path / "level.refine.json"
    # A sheet whose intervals lack the chordwise axis cannot be tabulated.
    record.write_text(
        json.dumps(
            {
                "components": {"Wing": ["A", "B"]},
                "component_governors": {"Wing": {"parallel": "A"}},
                "families": {"A": {"intervals": {"spanwise": [1, 2]}, "faces": [1, 2]}},
                "layouts": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(GeometryWarning, match="factor table was left out"):
        pdf = refined_pdf(work, tmp_path / "level.obj", record)
    assert pdf.is_file()


@needs_inspect
def test_mesh_nastran_governor_pages_name_families_in_a_legend_of_distinct_colors(
    tmp_path, monkeypatch
):
    """No label over the views; a bold governor and one color per governed family."""
    from types import SimpleNamespace

    from matplotlib.backends.backend_pdf import PdfPages

    from cfdbender.mesh._inspect_pages import refined_pdf
    from cfdbender.mesh._refine._config import FamilySpec

    seen = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        legend = fig.legends[0] if fig.legends else None
        seen.append(
            {
                "view_texts": [t.get_text() for ax in fig.axes for t in ax.texts],
                "labels": [t.get_text() for t in legend.get_texts()] if legend else [],
                "weights": [t.get_fontweight() for t in legend.get_texts()] if legend else [],
                "colors": [tuple(p.get_facecolor()) for p in legend.get_patches()]
                if legend
                else [],
            }
        )
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    vertices = np.array([[x, y, 0.0] for x in range(5) for y in range(2)], dtype=float)
    quad = lambda i: [[2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1]]  # noqa: E731
    work = SimpleNamespace(
        mesh=SimpleNamespace(verts=vertices, families={n: quad(i) for i, n in enumerate("ABCK")}),
        bands={},
        parts={},
        specs={
            "A": FamilySpec(method="grid", spanwise=2.0),
            "B": FamilySpec(method="grid", chordwise=3.0),
            "C": FamilySpec(method="grid_auto"),
            "K": FamilySpec(method="keep"),
        },
        layouts={},
        te_unit="METER",
    )
    record = tmp_path / "level.refine.json"
    record.write_text(
        json.dumps(
            {
                "components": {"Wing": ["A", "B", "C", "K"]},
                "component_governors": {"Wing": {"parallel": "A", "normal": "B"}},
                "families": {},
                "layouts": [],
            }
        ),
        encoding="utf-8",
    )
    refined_pdf(work, tmp_path / "level.obj", record)
    page = seen[1]
    assert not set("ABCK") & set(page["view_texts"])
    assert page["labels"][0] == "A (GOVERNING)" and page["weights"][0] == "bold"
    assert page["labels"][1:3] == ["B", "C"]
    lead, *governed = page["colors"][:3]
    assert len({lead, *governed}) == 3


def _p0502_factor_tables(monkeypatch):
    """Record, per saved page, its title and its table rows keyed by (family, direction)."""
    from matplotlib.backends.backend_pdf import PdfPages

    pages: list[tuple[str, dict, dict]] = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        title = fig._suptitle.get_text() if fig._suptitle else ""
        rows, bold = {}, {}
        for table in (tb for ax in fig.axes for tb in ax.tables):
            cells = table.get_celld()
            for r in sorted({r for r, _ in cells if r > 0 and (r, 7) in cells}):
                row = [cells[r, c].get_text().get_text() for c in range(8)]
                rows[row[0], row[1]] = row
                bold[row[0], row[1]] = cells[r, 6].get_text().get_fontweight() == "bold"
        notes = " ".join(t.get_text() for t in fig.texts)
        pages.append((title + "\n" + notes, rows, bold))
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    return pages


def _p0502_factor_level(folder, monkeypatch, parallel):
    """Refine the three-plate component with A's parallel factor; return the saved tables."""
    from cfdbender.mesh._refine._obj import read_obj

    source = _p0500_component_source(folder, factor=2, connected=True)
    obj = read_obj(source)
    rows = ["# Id, Name, X, Y, Z"]
    for n, fs in obj.families.items():
        ids = sorted({v for f in fs for v in f})
        xmax = max(obj.verts[v, 0] for v in ids)
        rows += [
            f"{v + 1}, {n}, " + ", ".join(map(str, obj.verts[v]))
            for v in ids
            if obj.verts[v, 0] == xmax
        ]
    source.with_suffix(".nodes.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = source.with_suffix(".refine.toml")
    text = config.read_text().replace(
        "normal_factor = 1\nparallel_factor = 2", f"normal_factor = 1\nparallel_factor = {parallel}"
    )
    config.write_text(text + '\n[[perimeter]]\nfile = "components.nodes.txt"\n', encoding="utf-8")
    pages = _p0502_factor_tables(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refine_mesh(source, accept_unknown=True, visual_inspect=True)
    return pages


@needs_inspect
def test_mesh_nastran_governors_pdf_compares_requested_and_achieved_factors(tmp_path, monkeypatch):
    """one row per family and direction, after the summary, flagged off rounding."""
    pages = _p0502_factor_level(tmp_path / "even", monkeypatch, 2)
    assert "governing families" in pages[0][0]
    assert "requested and achieved factors" in pages[1][0]
    assert "intervals after / intervals before" in pages[1][0]
    assert "governs the" in pages[2][0]
    table = pages[1][1]
    # A asks 2 parallel and 1 normal, C 2 normal and 1 parallel, B 1 and 1; the
    # component's governors are A (parallel) and C (normal), so every grid reaches 2.
    expected = {
        ("A", "PARALLEL"): ["itself", "2", "2"],
        ("A", "NORMAL"): ["C", "1", "2"],
        ("B", "PARALLEL"): ["A", "1", "2"],
        ("B", "NORMAL"): ["C", "1", "2"],
        ("C", "PARALLEL"): ["A", "1", "2"],
        ("C", "NORMAL"): ["itself", "2", "2"],
    }
    assert set(table) == set(expected)
    for key, (governor, requested, governing) in expected.items():
        assert table[key][2:7] == [governor, requested, governing, "4 -> 8", "2.000"], key
        assert table[key][7] == "16 -> 64"
    assert list(table) == sorted(table, key=lambda k: (k[0], k[1] != "PARALLEL"))
    assert not any(pages[1][2].values())
    # A parallel factor of 0.1 takes 4 intervals to 0.4, raised to the least the grid
    # allows, 1: the achieved 0.25 departs from 0.1 by more than rounding explains.
    # A's parallel axis is its columns and C's its rows, so a swapped axis fails here.
    pages = _p0502_factor_level(tmp_path / "low", monkeypatch, 0.1)
    low = pages[1][1]
    for name in "ABC":
        assert low[name, "PARALLEL"][4:7] == ["0.1", "4 -> 1", "0.250 *"], name
        assert low[name, "NORMAL"][4:7] == ["2", "4 -> 8", "2.000"], name
    assert low["A", "PARALLEL"][3] == "0.1" and low["C", "PARALLEL"][3] == "1"
    assert pages[1][2] == {k: k[1] == "PARALLEL" for k in expected}
    assert "more than rounding to whole intervals" in pages[1][0]


@needs_inspect
def test_mesh_nastran_visual_strict_factor_refusal_has_pdf_and_location(tmp_path):
    (tmp_path / "source").mkdir()
    source = _export(tmp_path / "source/sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source)
    levels = tmp_path / "levels"
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(prepared.obj, 0.01, out_dir=levels, force_factors=False)
    pdf = prepared.obj.with_name(prepared.obj.stem + ".refine-refusal.pdf")
    assert pdf.is_file()
    assert str(caught.value).endswith(diag(pdf).lstrip())
    assert caught.value.where["families"]
    assert not levels.exists()


@needs_inspect
def test_mesh_nastran_visual_refusal_uses_structured_locations(tmp_path, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages

    from cfdbender.mesh import _inspect_pages as pages
    from cfdbender.mesh._visual_inspect import Mesh

    source = tmp_path / "mesh.obj"
    mesh = Mesh(
        source,
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        {"A": [[0, 1, 2]], "B": [[0, 2, 1]]},
    )
    error = InputArtifactError(
        "A name in prose does not select a family",
        where={"families": ["B"], "nodes": [1], "points": [[0.5, 0.5, 0.0]]},
    )
    seen = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        seen.append(fig.axes[0].collections)
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    pages.patches_pdf(
        mesh, {"all": ["A", "B"]}, {}, tmp_path / "locations.pdf", refusal=error, dpi=40
    )
    surfaces = seen[0][0].get_facecolors()
    assert sum(c[3] == 0.4 for c in surfaces) == 1
    assert sum(c[0] > c[1] and c[3] == 1 for c in surfaces) == 1
    points = seen[0][-1]
    np.testing.assert_array_equal(points.get_offsets(), [[0.5, 0.5], [2.0, 0.0]])
    np.testing.assert_array_equal(points.get_facecolors(), [[1.0, 0.0, 0.0, 1.0]])


@needs_inspect
def test_mesh_nastran_visual_drawing_failure_preserves_original_refusal(tmp_path, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages

    (tmp_path / "source").mkdir()
    source = _export(tmp_path / "source/sheet.nas", wide=False, perimeter=_middle_line())
    prepared = prepare_to_refine(source)

    def fail(*args, **kwargs):
        raise RuntimeError("drawing failed")

    monkeypatch.setattr(PdfPages, "savefig", fail)
    with (
        pytest.warns(UserWarning, match="could not write the diagnostic PDF"),
        pytest.raises(InputArtifactError) as caught,
    ):
        refine_mesh(prepared.obj, 0.01, force_factors=False, out_dir=tmp_path / "levels")
    assert "below 1/" in str(caught.value)
    assert "A diagnostic PDF" not in str(caught.value)
    assert not (tmp_path / "levels").exists()


@needs_inspect
def test_mesh_nastran_visual_refine_annotations_come_from_saved_record(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from matplotlib.backends.backend_pdf import PdfPages

    from cfdbender.mesh._inspect_pages import refined_pdf

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    work = SimpleNamespace(
        mesh=SimpleNamespace(verts=vertices, families={"A": [[0, 1, 2]], "B": [[0, 2, 1]]}),
        bands={},
        parts={},
        specs={},
        layouts={},
        te_unit="METER",
    )
    record = tmp_path / "level.refine.json"
    record.write_text(
        json.dumps(
            {
                "components": {"All": ["A", "B"]},
                "families": {
                    "A": {"continuity": {"directions": ["normal"]}},
                    "B": {"band": 7},
                },
            }
        ),
        encoding="utf-8",
    )
    texts = []
    save = PdfPages.savefig

    def capture(pdf, fig, **kwargs):
        texts.extend(t.get_text() for t in fig.texts)
        return save(pdf, fig, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    pdf = refined_pdf(work, tmp_path / "level.obj", record)
    assert pdf.is_file()
    assert any("A: continuity" in t and "normal" in t for t in texts)
    assert any("B: continuity remeshed band (7 source faces)" in t for t in texts)


def test_mesh_nastran_audit_off_skips_the_call_and_allows_a_later_audit(tmp_path, monkeypatch):
    """refine_mesh(source, audit=False) writes an auditable level."""
    import inspect

    from cfdbender.mesh._refine import _level, audit_mesh

    source = write_source(tmp_path / "src", "sheet", sheet_with_strips(()))
    calls = []
    original = _level.audit_mesh

    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_level, "audit_mesh", observed)
    ordinary = refine_mesh(source, 1)
    assert ordinary.audit is not None and len(calls) == 1
    calls.clear()
    progress = []
    options = {"audit": False} if "audit" in inspect.signature(refine_mesh).parameters else {}
    level = refine_mesh(source, 1, overwrite=True, progress=progress.append, **options)
    assert calls == [], "audit=False must not enter the audit"
    assert level.audit is None and not level.obj.with_suffix(".audit.json").exists()
    assert all(p.is_file() for p in level.files)
    record = json.loads(level.obj.with_suffix(".refine.json").read_text())
    assert record["audit"] == "skipped (--audit-off)"
    advice = [line for line in progress if "audit skipped" in line]
    assert len(advice) == 1 and "cfd-mesh audit" in advice[0] and "--against" in advice[0]
    later = audit_mesh(level.obj, against=source)
    assert later.path.is_file()


@pytest.mark.parametrize("case", ["coarsening", "closed_length"])
def test_mesh_nastran_component_tubes_use_closed_lengths_and_warn_at_the_minimum(tmp_path, case):
    """full rings select governors; inherited factors keep tubes writable."""
    import inspect

    from mesh_fixtures import Composite, body_of_revolution

    a = body_of_revolution(stations=7, nodes=24)
    b = body_of_revolution(stations=3 if case == "coarsening" else 7, nodes=12)
    av = a.verts * (2 if case == "coarsening" else 1)
    bv = b.verts.copy()
    if case == "closed_length":
        # The 12-sided ring at 1.02 radius is longer than the 24-sided ring at 1.
        bv[:, 1:] *= 1.02
    bv[:, 0] += 10
    mesh = Composite(
        np.vstack([av, bv]),
        {"A": a.faces, "B": [[v + len(av) for v in f] for f in b.faces]},
        np.zeros((0, 3)),
        {},
    )
    factor = 0.2 if case == "coarsening" else 2
    config = (
        "[refine]\nkeep_within_component = true\n"
        '[components]\nbody = ["A", "B"]\n'
        f'[families.A]\nmethod = "grid"\nfactor = {factor}\n'
        '[families.B]\nmethod = "grid"\nfactor = 1\n'
    )
    source = write_source(tmp_path / "src", "body", mesh, boundaries=False, refine_toml=config)
    options = {"audit": False} if "audit" in inspect.signature(refine_mesh).parameters else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        level = refine_mesh(source, accept_unknown=True, **options)
    data = json.loads(level.obj.with_suffix(".refine.json").read_text())
    if case == "closed_length":
        assert data.get("component_governors", {}).get("body", {}).get("normal") == "B"
    else:
        assert data["faces"] == {"body": 8}
        assert any("inherited" in str(w.message) and "raised" in str(w.message) for w in caught)

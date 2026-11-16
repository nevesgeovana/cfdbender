"""a level written by refine_mesh and by cfd-mesh refine.

Every source is synthetic (tests/p0380_mesh_fixtures.py): the cambered sheet
grid ``G`` alone, or followed by strips extruded from its tip station that
share its last row of nodes (``T``, triangles that are no grid; ``U``,
triangles beyond ``T``). The source sits in ``<tmp>/src/`` so the default
out-dir is ``<tmp>``, and every refusal is asserted by its message text and
by the absence of any folder beside ``src`` (no level, no ``.partial``).
Each behaviour is also run on an input it must reject, or with the step it
rests on taken away, so a check that accepts everything cannot pass.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tomllib
import warnings
from pathlib import Path

import numpy
import pytest
from mesh_fixtures import (
    Composite,
    body_of_revolution,
    l_sheet,
    read_mesh,
    sheet_with_strips,
    write_source,
)

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh import cli
from cfdbender.mesh._refine import RefinedMesh, _config, refine_mesh

#: The row spacings of the triangle strip T beyond the grid's tip station.
STRIP = (0.03, 0.04, 0.05, 0.065, 0.08)
WITH_T = (("T", "tri", STRIP),)
WITH_TU = (("T", "tri", STRIP), ("U", "tri", (0.1, 0.1)))
#: The audit warning the strip T draws beside a refined G (its growth check fails).
GROWTH_ON_T = "audit check growth failed on T"


def _source(tmp_path: Path, strips=(), *, te=True, boundaries=True, refine_toml=None) -> Path:
    """Write a synthetic source as ``<tmp>/src/wing.obj`` and return its path."""
    mesh = sheet_with_strips(strips, te=te)
    return write_source(
        tmp_path / "src", "wing", mesh, boundaries=boundaries, refine_toml=refine_toml
    )


def _left(root: Path) -> list[str]:
    """Return every entry beside the source's folder: the levels and any staging folder."""
    return sorted(p.name for p in root.iterdir() if p.name != "src")


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def _refused(root: Path, src: Path, *args, error=InputArtifactError, **kwargs) -> str:
    """Return the refusal text, checking only its diagnostic PDF was written."""
    pdf = src.with_name(src.stem + ".refine-refusal.pdf")
    before = _snapshot(src.parent)
    with pytest.raises(error) as caught:
        refine_mesh(src, *args, **kwargs)
    text = str(caught.value)
    assert text.endswith(f"A diagnostic PDF: {pdf}."), text
    assert pdf.read_bytes().startswith(b"%PDF")
    assert _left(root) == []
    after = _snapshot(src.parent)
    before.pop(pdf.name, None)
    after.pop(pdf.name)
    assert after == before
    return text


def _command(capsys, *argv: str) -> tuple[int, str, str]:
    """Run ``cfd-mesh refine`` in-process; return the exit status, stdout and stderr."""
    capsys.readouterr()
    output = [] if "--out" in argv else ["--out", str(Path(argv[0]).parent.parent)]
    code = cli.main(["refine", *argv, *output])
    out = capsys.readouterr()
    return code, out.out, out.err


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(level: RefinedMesh) -> Path:
    """Return the level's ``refine.json``, whatever the per-family files before it."""
    return next(p for p in level.files if p.name.endswith(".refine.json"))


# ---------------------------------------------------------------- factors


# ------------------------------------------------- where the factors come from


def _two_tubes(tmp_path: Path, refine_toml: str) -> Path:
    """Write two disconnected smooth tubes, families A and B, as ``<tmp>/src/tubes.obj``."""
    body = body_of_revolution()
    n = len(body.verts)
    mesh = Composite(
        numpy.vstack([body.verts, body.verts + [0.0, 4.0, 0.0]]),
        {"A": body.faces, "B": [[v + n for v in f] for f in body.faces]},
        numpy.zeros((0, 3)),
        {},
    )
    return write_source(tmp_path / "src", "tubes", mesh, refine_toml=refine_toml)


BOTH_TABLES = "[families.A]\nfactor = 2\n[families.B]\nfactor = 2\n"


def test_arch48_accept_empty_writes_nothing_when_no_trailing_edge_is_found(tmp_path, capsys):
    """no trailing edge found is refused; --accept-empty writes nothing, exit 0."""
    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._te_nodes import write_trailing_edges_identified

    src = _source(tmp_path, WITH_T)
    with pytest.warns(GeometryWarning, match="holds no family 'Nope'"):
        with pytest.raises(InputArtifactError, match="no trailing edge was found on Nope"):
            write_trailing_edges_identified(src, families=["Nope"])
    with pytest.warns(GeometryWarning, match="no trailing edge was found.*nothing is written"):
        assert write_trailing_edges_identified(src, families=["Nope"], accept_empty=True) == []
    with pytest.warns(GeometryWarning, match="nothing is written"):
        code = workspace_cli.main(["identify-te", str(src), "--families", "Nope", "--accept-empty"])
    assert code == 0 and json.loads(capsys.readouterr().out)["files"] == []
    assert not src.with_name(f"{src.stem}.Nope.fs.te.txt").exists()


# --------------------------------------------------------- the refinement file


# ---------------------------------------------------------- what a level holds


# ------------------------------------------------------ where a level goes


def _resampled_while(call) -> list[str]:
    """Run ``call()`` and return the grid resamplings it started (a profile, no patching)."""
    seen: list[str] = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("refine_grid", "refine_blocks"):
            seen.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(None)
    return seen


#: The same boundaries file spelled in six legal TOML ways (the fixture's own form first).
SIDECARS = {
    "double-quoted": (
        'boundaries = ["G", "T"]\n\n[trailing_edges]\nfile = "wing.te.txt"\ntype = "STANDARD"\n'
    ),
    "literal-strings": (
        "boundaries = ['G', 'T']\n[trailing_edges]\nfile = 'wing.te.txt'\ntype = 'STANDARD'\n"
    ),
    "multi-line-commented": (
        '# the boundaries\nboundaries = [\n  "G", # the grid\n  "T",\n] # end\n'
        '[trailing_edges]\nfile = "wing.te.txt" # the points\ntype = "STANDARD"\n'
    ),
    "inline-table-escapes": (
        'boundaries = ["\\u0047", "T"]\n'
        'trailing_edges = { file = "wing\\u002ete.txt", type = "STANDARD" }\n'
    ),
    "dotted-key": (
        "boundaries = ['G', 'T']\ntrailing_edges.file = 'wing.te.txt'\n"
        "trailing_edges.type = 'STANDARD'\n"
    ),
    "quoted-keys": (
        '"boundaries" = ["G", "T"]\n"trailing_edges"."file" = "wing.te.txt"\n'
        '"trailing_edges"."type" = "STANDARD"\n'
    ),
}


def _points_in(src: Path, unit: str, scale: float, mesh_unit: str | None) -> list[str]:
    """Rewrite the source's points file in ``unit`` (coordinates times ``scale``).

    The boundaries file gains ``[import] units = mesh_unit`` when one is given;
    return the rows written after the unit line.
    """
    side = src.with_name("wing.boundaries.toml")
    if mesh_unit is not None:
        side.write_text(
            side.read_text(encoding="utf-8") + f'\n[import]\nunits = "{mesh_unit}"\n',
            encoding="utf-8",
        )
    points = src.with_name("wing.te.txt")
    rows = points.read_text(encoding="utf-8").split()[1:]
    scaled = [",".join(repr(float(x) * scale) for x in row.split(",")) for row in rows]
    points.write_text("\n".join([unit, *scaled]) + "\n", encoding="utf-8")
    return scaled


def _g4(audit) -> dict:
    return dict(next(g for g in audit.gates if g.name == "G4").values)


# ----------------------------------------------------------------- face order


def _mesh_lines(path: Path, tags: tuple[str, ...] = ("v", "g", "f")) -> list[str]:
    """Return the OBJ's lines of the given statements, in order (the header aside)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.split(" ", 1)[0] in tags]


def _sheet_source(folder: Path, scale: float) -> tuple[Path, numpy.ndarray]:
    """Write the L-shaped multiblock sheet scaled by ``scale``, each coordinate round-trip exact."""
    sheet = l_sheet()
    verts = sheet.verts * scale
    lines = ["v " + " ".join(format(float(x), ".17g") for x in p) for p in verts] + ["g S"]
    lines += ["f " + " ".join(str(v + 1) for v in f) for f in sheet.faces]
    folder.mkdir(parents=True)
    src = folder / "plate.obj"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return src, verts


def _swap_two_faces(obj: Path) -> None:
    """Exchange the first two face lines of an OBJ in place."""
    lines = obj.read_text(encoding="utf-8").splitlines()
    a, b = [i for i, line in enumerate(lines) if line.startswith("f ")][:2]
    lines[a], lines[b] = lines[b], lines[a]
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- method


# ---------------------------------------------------------------- the band


# ------------------------------------------------------- the bytes written


def _digests(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


# ------------------------------------------------------------------- layering


def _imports(text: str, package: str) -> set[str]:
    """Return every module a source imports, at module level or deferred, relative ones resolved.

    ``from a.b import c`` yields both ``a.b`` and ``a.b.c`` (c may be a module).
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                base = ".".join([*parts, base] if base else parts)
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _reaching(found: set[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(m for m in found if any(m == p or m.startswith(p + ".") for p in prefixes))


REFINE = "cfdbender.mesh._refine"
#: The modules of the refinement package the audit may import: the shared helpers.
AUDIT_MAY_IMPORT = ("_audit", "_geometry", "_obj")


#: The shared engineering thresholds of the section, defined once in ``_geometry``.
SHARED_THRESHOLDS = (
    "DUPLICATE_FRACTION",
    "TE_POINT_FRACTION",
    "GRID_SURFACE_DISTANCE",
    "ZERO_AREA_FRACTION",
    "ON_CURVE_FRACTION",
    "OPENING_FRACTION",
)


def _threshold_literals(text: str, values: set[float]) -> list[tuple[int, float]]:
    """Return the line and value of every numeric literal of the source equal to a threshold."""
    return sorted(
        (node.lineno, float(node.value))
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and float(node.value) in values
    )


def _lock(folder: Path, locked: bool) -> None:
    """Make ``folder/held`` refuse removal (a read-only file; a sealed folder off Windows)."""
    held = folder / "held"
    if sys.platform == "win32":
        os.chmod(held / "held.txt", 0o444 if locked else 0o666)
    else:
        os.chmod(held, 0o500 if locked else 0o700)


# ------------------------------------------- the level's own inventory (0.44.0)

# a level's boundaries file states its own faces and digest.
#
# A boundaries file written by the package for an OBJ without one carries the
# mesh's inventory, ``mesh_faces``, ``boundary_faces`` and ``mesh_sha256``,
# of that OBJ. The level's boundaries file is the source's with the values a
# level changes rewritten, and these three are the LEVEL's: counted in the
# level's OBJ as written, in the level's boundaries order after the
# ``[components]`` merge, and its sha256. The source is synthetic
# (tests/p0380_mesh_fixtures.py): the sheet grid G, the triangle strips T and U.

INVENTORY_STRIPS = (("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)), ("U", "tri", (0.1, 0.1)))
INVENTORY_KEYS = ("mesh_faces", "boundary_faces", "mesh_sha256")


def _level_boundaries(folder) -> dict:
    return tomllib.loads(
        (folder / f"{folder.name}.obj.boundaries.toml").read_text(encoding="utf-8")
    )


def source(tmp_path: Path, config: str = "", *, perimeter: bool = True) -> Path:
    """Write the sheet source; its refine file carries ``config`` and, by default, the perimeter."""
    mesh = write_source(tmp_path / "src", "wing", sheet_with_strips(()), boundaries=False)
    if perimeter:
        config += '\n[[perimeter]]\nfile = "wing.te_nodes.txt"\nexport_te_per_family = true\n'
    mesh.with_suffix(".refine.toml").write_text(config, encoding="utf-8")
    return mesh


# ------------------------------------------------------ [[perimeter]] node lists


def _sheet_lines():
    """Return the sheet's trailing-edge nodes (u = 0) and its root station's, by vertex."""
    from mesh_fixtures import sheet_quads

    fx = sheet_quads()
    place = {p: v for v, p in fx.place.items()}
    te = [place[(k, 0)] for k in range(fx.stations)]
    root = [place[(0, i)] for i in range(fx.nodes)]
    return fx, te, root


def _same_level(a: Path, b: Path) -> bool:
    """Return whether two level OBJs hold the same nodes and the same families."""
    (va, fa), (vb, fb) = read_mesh(a), read_mesh(b)
    return numpy.array_equal(va, vb) and fa == fb


def _node_list(path: Path, verts, ids, *, first: int = 1001, extra=()) -> Path:
    """Write the ASCII list ANSA saves from a SET's node window, in a scrambled order."""
    rows = ["#  Id, Name, X , Y, Z"]
    order = list(ids)[1::2] + list(ids)[::2]
    rows += [f"{first + v}, , {x:.6f}, {y:.6f}, {z:.6f}" for v in order for x, y, z in [verts[v]]]
    rows += [f"{node}, , {x}, {y}, {z}" for node, (x, y, z) in extra]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_convert2midnodes_orders_a_scrambled_list_of_two_lines(tmp_path, capsys):
    """cfd-mesh convert2midnodes: one mid-point per neighbouring pair, per line.

    Two trailing edges in one list, out of order: the sheet's and a copy of it
    ten units above, as a second component's would be.
    """
    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine._obj import read_te

    fx, te, _ = _sheet_lines()
    verts = numpy.vstack([fx.verts, fx.verts[te] + (0.0, 0.0, 10.0)])
    second = list(range(len(fx.verts), len(verts)))
    listed = _node_list(tmp_path / "nodes.txt", verts, [*te, *second])
    out = tmp_path / "nodes.te.txt"
    code = workspace_cli.main(["convert2midnodes", str(listed), str(out), "--unit", "MILLIMETER"])
    assert code == 0 and json.loads(capsys.readouterr().out)["points"] == [str(out)]
    unit, points = read_te(out)
    expected = numpy.vstack([0.5 * (verts[ids[:-1]] + verts[ids[1:]]) for ids in (te, second)])
    assert unit == "MILLIMETER" and len(points) == len(expected)
    near = numpy.linalg.norm(points[:, None] - expected[None], axis=2).min(axis=1)
    assert near.max() < 1e-5


def test_drev48_convert2midnodes_reads_coincident_nodes_once(tmp_path):
    """a list naming every trailing-edge position twice (an open trailing
    edge's upper and lower nodes, two ids at one position) gives the edge mid-points.

    Each node's spacing was its twin's distance, zero, so every line was a twin
    pair and its mid-point the node itself. Control: the list with each
    position once gives the same points; the repeat is warned.
    """
    from cfdbender.mesh._refine._obj import read_te
    from cfdbender.mesh._te_nodes import write_trailing_edge_midnodes

    fx, te, _ = _sheet_lines()
    twins = [(5000 + v, tuple(f"{c:.6f}" for c in fx.verts[v])) for v in te]
    listed = _node_list(tmp_path / "twins.txt", fx.verts, te, extra=twins)
    with pytest.warns(GeometryWarning, match=r"9 node\(s\) repeat the position"):
        _, points = read_te(write_trailing_edge_midnodes(listed, tmp_path / "twins.te.txt"))
    once = _node_list(tmp_path / "once.txt", fx.verts, te)
    _, alone = read_te(write_trailing_edge_midnodes(once, tmp_path / "once.te.txt"))
    exact = 0.5 * (fx.verts[te[:-1]] + fx.verts[te[1:]])
    assert len(points) == len(alone) == len(exact)
    near = numpy.linalg.norm(points[:, None] - exact[None], axis=2).min(axis=1)
    assert near.max() < 1e-5


def test_convert2midnodes_per_family_writes_the_mesh_edges_of_each_family(tmp_path, capsys):
    """convert2midnodes --per-family MESH: one file per family, the mesh's own edge mid-points.

    The list holds the trailing edges of the sheet G and of the quadrilateral
    strip Q beside it, rounded and out of order; the strip T is untouched and
    gets no file.
    """
    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine._obj import read_te

    mesh = sheet_with_strips((("Q", "quad", (0.2, 0.2)), ("T", "tri", (0.1,))))
    src = write_source(tmp_path / "src", "wing", mesh)
    rows = mesh.rows
    lines = {"G": [r[0] for r in rows["G"]], "Q": [r[0] for r in rows["Q"]]}
    listed = _node_list(tmp_path / "nodes.txt", mesh.verts, sorted({*lines["G"], *lines["Q"]}))
    out = tmp_path / "nodes.te.txt"
    code = workspace_cli.main(["convert2midnodes", str(listed), str(out), "--per-family", str(src)])
    printed = json.loads(capsys.readouterr().out)["points"]
    assert code == 0 and printed == [str(tmp_path / f"nodes.{n}.te.txt") for n in ("G", "Q")]
    for name, ids in lines.items():
        unit, points = read_te(tmp_path / f"nodes.{name}.te.txt")
        expected = 0.5 * (mesh.verts[ids[:-1]] + mesh.verts[ids[1:]])
        assert unit == "METER" and len(points) == len(expected)
        near = numpy.linalg.norm(points[:, None] - expected[None], axis=2).min(axis=1)
        assert near.max() < 1e-8  # the OBJ holds nine decimals
    assert not out.exists()


# ------------------------------------------- one node list, te_families, identify-te


def _whole_boundary(fx) -> list[int]:
    """Return every boundary node of the sheet: trailing edge, leading edge, root and tip."""
    from cfdbender.mesh._refine._lines import boundary_sides

    return sorted({v for side in boundary_sides(fx.verts, fx.faces) for v in side})


def _te_families_level(tmp_path: Path, extra: str = ""):
    fx, te, _ = _sheet_lines()
    factors = "[families.G]\nparallel_factor = 2\nnormal_factor = 1.5\n"
    src = _source(tmp_path / "b", te=False)
    _node_list(src.with_name("nodes.txt"), fx.verts, _whole_boundary(fx))
    src.with_name("wing.refine.toml").write_text(
        factors + extra + '[[perimeter]]\nfile = "nodes.txt"\n' + 'te_families = ["Wing"]\n',
        encoding="utf-8",
    )
    return fx, te, factors, src


def test_identify_te_writes_the_solver_node_file_of_a_refined_merged_family(tmp_path, capsys):
    """identify-te: the refined level's Wing trailing edge, as a points file.

    The source's node list is read along a level whose nodes are not the listed
    ones (parallel factor 2); the file is the count, one unread coordinate line
    and one mid-point per trailing-edge edge, the points the refine wrote.
    """
    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine._obj import read_te

    _, _, _, src = _te_families_level(tmp_path, '[components]\nWing = ["G"]\n')
    level = refine_mesh(src, out_dir=tmp_path / "y")
    with pytest.warns(GeometryWarning, match="holds no family 'Nope'"):
        code = workspace_cli.main(
            [
                "identify-te",
                str(level.obj),
                str(src.with_name("nodes.txt")),
                "--families",
                "Wing,Nope",
                "--overwrite",  # the level's own <level>.Wing.te.txt is rewritten alike
            ]
        )
    package = level.obj.with_name(f"{level.obj.stem}.Wing.te.txt")
    assert code == 0 and json.loads(capsys.readouterr().out)["files"] == [str(package)]
    _, refined = read_te(package)
    assert len(refined) > 0


# ------------------------------------------------------------------ prepare-to-refine


def _one_family_wing(tmp_path: Path) -> Path:
    """Write Wing: a quad sheet and the quad strip beside it, triangles, then another quad strip."""
    mesh = sheet_with_strips(
        (("Q", "quad", (0.2, 0.2)), ("T", "tri", (0.1,)), ("R", "quad", (0.2, 0.2)))
    )
    faces = [f for name in ("G", "Q", "T", "R") for f in mesh.families[name]]
    wing = Composite(mesh.verts, {"Wing": faces}, numpy.zeros((0, 3)), {})
    return write_source(tmp_path / "src", "wing", wing, boundaries=False)


def test_prepare_to_refine_writes_patches_ordered_by_span_a_refine_file_and_a_node_list(
    tmp_path, capsys
):
    """cfd-mesh prepare-to-refine: patches, the refine file and the node list.

    Wing holds two quad pieces the triangles separate: the sheet with its strip
    (2.4 along y) and the far strip (0.4), each a grid patch numbered by its
    extent along y, and the triangles, a mixed patch. Each grid patch lists its
    rearmost line, the trailing edge (11 and 3 nodes), ordered along y; the
    refine file reads with no warning and refines.
    """
    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine import _config
    from cfdbender.mesh._refine._obj import read_node_list, read_obj

    src = _one_family_wing(tmp_path)
    code = workspace_cli.main(["prepare-to-refine", str(src), "--out", str(src.parent)])
    assert code == 0, capsys.readouterr()
    record = json.loads(capsys.readouterr().out)
    names = ["Wing_001_grid", "Wing_002_grid", "Wing_003_mixed"]
    assert record["patches"] == {"Wing": names}
    obj = src.with_name("wing_patches.obj")
    patches = read_obj(obj)
    assert list(patches.families) == names
    source = read_obj(src)
    assert sum(len(f) for f in patches.families.values()) == len(source.families["Wing"])
    spans = [numpy.ptp(patches.verts[patches.family_vertices(n), 1]) for n in names[:2]]
    assert spans[0] > 1.0 > spans[1]
    nodes = read_node_list(src.with_name("wing_patches.nodes.txt"))
    assert len(nodes.ids) == 11 + 3
    assert numpy.all(numpy.diff(nodes.points[:, 1]) > 0)
    assert numpy.allclose(nodes.points[:, 0], patches.verts[numpy.array(nodes.ids) - 1, 0])
    config = src.with_name("wing_patches.refine.toml")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stated = _config.read_refine_file(config, names)
    assert {n: s.method for n, s in stated["specs"].items()} == {
        "Wing_001_grid": "grid",
        "Wing_002_grid": "grid",
        "Wing_003_mixed": "remesh",
    }
    assert stated["components"] == {"Wing": names}
    assert [p.te_families for p in stated["perimeters"]] == [("Wing",)]
    text = config.read_text(encoding="utf-8").replace(
        '[families.Wing_001_grid]\nmethod = "grid"\nfactor = 1.0',
        '[families.Wing_001_grid]\nmethod = "grid"\nparallel_factor = 2.0',
    )
    config.write_text(text, encoding="utf-8")
    level = refine_mesh(obj, out_dir=tmp_path / "out")
    assert level.report["Wing_001_grid"]["method"] == "grid"
    te = level.folder / f"{level.obj.stem}.Wing.te.txt"
    from cfdbender.mesh._refine._obj import read_te

    assert len(read_te(te)[1]) == 2 * 10 + 1 + 2  # the grid at factor 2, T and R at 1


@pytest.mark.parametrize("closed", [False, True])
def test_identify_te_from_the_obj_alone_writes_the_trailing_edge_and_its_node_list(
    tmp_path, capsys, closed
):
    """identify-te OBJ, no node list: the trailing edge found from the mesh, solver and ANSA files.

    On an open sheet the trailing edge is a boundary; on a closed wing it is
    the sharp edge between the upper and the lower surface. The leading edge,
    the root and the tip are not written. Control: --families writes the same
    points in the family's own file.
    """
    from mesh_fixtures import sheet_quads, tube_pole_quads

    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine._obj import read_node_list

    fx = tube_pole_quads() if closed else sheet_quads()
    src = write_source(
        tmp_path / "src",
        "wing",
        Composite(fx.verts, {"Wing": fx.faces}, numpy.zeros((0, 3)), {}),
        boundaries=False,
    )
    assert workspace_cli.main(["identify-te", str(src)]) == 0
    listing = src.with_name("wing.nodes.txt")
    package = src.with_name("wing.te.txt")
    assert json.loads(capsys.readouterr().out)["files"] == [str(package), str(listing)]
    from cfdbender.mesh._refine._obj import read_te

    unit, stated = read_te(package)
    assert unit == "METER"
    place = {p: n for n, p in fx.place.items()}
    te = [place[(k, 0)] for k in range(fx.stations)]
    exact = 0.5 * (fx.verts[te[:-1]] + fx.verts[te[1:]])

    assert len(stated) == len(exact)
    assert numpy.allclose(numpy.sort(stated, axis=0), numpy.sort(exact, axis=0), atol=1e-9)
    nodes = read_node_list(listing)
    assert sorted(i - 1 for i in nodes.ids) == sorted(te)
    assert numpy.all(numpy.diff(nodes.points[:, 1]) > 0)
    code = workspace_cli.main(["identify-te", str(src), "--families", "Wing", "--overwrite"])
    assert code == 0
    capsys.readouterr()
    _, own = read_te(src.with_name("wing.Wing.te.txt"))
    assert numpy.allclose(numpy.sort(own, axis=0), numpy.sort(exact, axis=0), atol=1e-9)


def test_drev48_the_maximum_angle_is_taken_from_the_plane_across_the_aft_direction(tmp_path):
    """with --aft along y, the trailing edge runs along x and is found.

    The sheet is turned so its nose to tail direction is +y. The angle was
    measured from the yz plane whatever the aft direction, so the trailing
    edge, 90 degrees from that plane, was left out and identify-te refused.
    Control: the sheet as built, aft +x, gives the same edges.
    """
    from mesh_fixtures import sheet_quads

    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine._prepare import rearmost_lines

    fx = sheet_quads()
    turn = numpy.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # x to y
    turned = fx.verts @ turn
    control = rearmost_lines(fx.verts, fx.faces)
    assert len(control) == fx.stations - 1
    assert rearmost_lines(turned, fx.faces, aft=(0.0, 1.0, 0.0)) == control
    src = write_source(
        tmp_path / "src",
        "wing",
        Composite(turned, {"Wing": fx.faces}, numpy.zeros((0, 3)), {}),
        boundaries=False,
    )
    assert workspace_cli.main(["identify-te", str(src), "--aft", "0", "1", "0"]) == 0
    from cfdbender.mesh._refine._obj import read_te

    assert len(read_te(src.with_name("wing.te.txt"))[1]) == fx.stations - 1


def test_drev48_identify_te_units_of_its_three_files(tmp_path, capsys):
    """identify-te writes the solver file in --simulation-unit, converted, the
    package points file under the --unit line, unconverted, and the node list as read.

    The default unit is the mesh's ``[import] units``. Control: the same mesh
    with no unit stated writes METER and no conversion.
    """
    from mesh_fixtures import sheet_quads

    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine._obj import read_node_list, read_te

    fx = sheet_quads()
    src = write_source(
        tmp_path / "src",
        "wing",
        Composite(fx.verts, {"Wing": fx.faces}, numpy.zeros((0, 3)), {}),
        boundaries=False,
    )

    assert workspace_cli.main(["identify-te", str(src)]) == 0
    plain_unit, plain = read_te(src.with_name("wing.te.txt"))
    assert plain_unit == "METER"
    listed = read_node_list(src.with_name("wing.nodes.txt")).points
    command = ["identify-te", str(src), "--overwrite"]
    src.with_name("wing.obj.boundaries.toml").write_text(
        '[import]\nunits = "MILLIMETER"\n', encoding="utf-8"
    )
    assert workspace_cli.main(command) == 0, capsys.readouterr().err
    unit, stated = read_te(src.with_name("wing.te.txt"))
    assert unit == "MILLIMETER"
    assert numpy.allclose(numpy.sort(stated, axis=0), numpy.sort(plain, axis=0))
    assert numpy.allclose(read_node_list(src.with_name("wing.nodes.txt")).points, listed)


def test_drev48_an_aft_of_zeros_is_warned_and_plus_x_taken(tmp_path, capsys):
    """--aft 0 0 0 states no direction; it is warned and +x taken.

    A refine file's [[perimeter]] aft of zeros is warned and +x taken; the
    commands divided by its length and found no trailing edge (identify-te
    refused, prepare-to-refine wrote an empty node list). Control: the same
    files the default aft writes.
    """
    from mesh_fixtures import sheet_quads

    from cfdbender.mesh import cli as workspace_cli
    from cfdbender.mesh._refine import prepare_to_refine

    fx = sheet_quads()
    src = write_source(
        tmp_path / "src",
        "wing",
        Composite(fx.verts, {"Wing": fx.faces}, numpy.zeros((0, 3)), {}),
        boundaries=False,
    )
    with pytest.warns(GeometryWarning, match=r"aft .* not a direction.*\+x is taken"):
        code = workspace_cli.main(["identify-te", str(src), "--aft", "0", "0", "0"])
    assert code == 0, capsys.readouterr().err
    zeros = src.with_name("wing.te.txt").read_bytes()
    assert workspace_cli.main(["identify-te", str(src), "--overwrite"]) == 0
    assert src.with_name("wing.te.txt").read_bytes() == zeros
    with pytest.warns(GeometryWarning, match=r"aft .* not a direction.*\+x is taken"):
        prepared = prepare_to_refine(_one_family_wing(tmp_path / "p"), aft=(0.0, 0.0, 0.0))
    assert len(prepared.nodes.read_text(encoding="utf-8").splitlines()) == 1 + 11 + 3


WINDOWS_PATH_LIMIT = 260


RANDOM_CHARS = 8  # tempfile.mkdtemp's random part


LONG_STEM = "wing_with_a_long_descriptive_mesh_name"


def _staged_file_length(out_dir: str, run_folder_name: str, level: str, longest: int) -> int:
    """Length of the longest file path: the output folder, the run's folder, the level, a file."""
    return len(out_dir) + 1 + len(run_folder_name) + 1 + len(level) + 1 + longest


def _read(tmp_path: Path, text: str, names=("G", "T", "U")) -> dict:
    path = tmp_path / "w.refine.toml"
    path.write_text(text, encoding="utf-8")
    return _config.read_refine_file(path, list(names))

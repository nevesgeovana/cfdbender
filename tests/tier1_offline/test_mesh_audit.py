"""Tier 1, 0.38.0: the audit of a panel mesh, alone or against its source.

Pipeline role: quality gate on :func:`pyflightstream.workspace.audit_mesh` and
``cfd-mesh audit-mesh``. Every mesh is built here: a curved sheet of two
grid families (one in triangles, one in quadrilaterals) sharing a row of
nodes, a closed box beside it, and a trailing edge along one side. Each gate
and each relative check is shown failing on a mesh built to fail it (the
warning, the audit written, the exit status), each reported figure is shown
exceeded with no warning, and the same mesh audited against itself is the
control that passes.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import numpy
import pytest

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh import cli
from cfdbender.mesh._refine import MeshAudit, audit_mesh

NX, NY = 6, 6


#: A closed box beside the sheet, outward by the right-hand rule.
def _box(length: float = 0.2) -> list[list[float]]:
    """Return the corners of a box beside the sheet, ``length`` long along x."""
    return [[x, y, z] for x in (2.0, 2.0 + length) for y in (0.5, 0.7) for z in (0.0, 0.2)]


_BOX_FACES = [
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
]  # fmt: skip


def _node(i: int, j: int) -> int:
    """Return the index of the sheet node at column i and row j."""
    return i * (NY + 1) + j


class Mesh:
    """A synthetic mesh: vertices, families of 0-based faces, and the trailing-edge edges."""

    def __init__(self, xs=None, plane=False, box=0.2) -> None:
        xs = numpy.linspace(0.0, 1.0, NX + 1) if xs is None else numpy.asarray(xs, dtype=float)
        rows = []
        for x in xs:
            for j in range(NY + 1):
                t = j / NY
                rows.append(
                    [x, 0.0, 0.2 + t] if plane else [x, 0.2 + t, 0.05 * math.sin(math.pi * x)]
                )
        self.verts = numpy.array(rows + _box(box), dtype=float)
        a, b = [], []
        for i in range(NX):
            for j in range(NY):
                v00, v10 = _node(i, j), _node(i + 1, j)
                v11, v01 = _node(i + 1, j + 1), _node(i, j + 1)
                if j < NY // 2:
                    a += [[v00, v10, v11], [v00, v11, v01]]
                else:
                    b.append([v00, v10, v11, v01])
        box = len(rows)
        self.families = {"A": a, "B": b, "Pod": [[box + k for k in f] for f in _BOX_FACES]}
        self.te = [(_node(NX, j), _node(NX, j + 1)) for j in range(NY)]

    def add_vertex(self, point) -> int:
        """Append a vertex and return its index."""
        self.verts = numpy.vstack([self.verts, numpy.asarray(point, dtype=float)])
        return len(self.verts) - 1

    def te_points(self) -> numpy.ndarray:
        """Return the mid-point of every trailing-edge edge, at this mesh's coordinates."""
        return numpy.array([0.5 * (self.verts[p] + self.verts[q]) for p, q in self.te])

    def write(self, folder: Path, stem: str, *, te_points=None) -> Path:
        """Write ``<stem>.obj`` and ``<stem>.te.txt`` in ``folder`` and return the OBJ."""
        folder.mkdir(parents=True, exist_ok=True)
        lines = [f"v {x!r} {y!r} {z!r}" for x, y, z in self.verts.tolist()]
        for name, faces in self.families.items():
            lines.append(f"g {name}")
            lines += ["f " + " ".join(str(v + 1) for v in f) for f in faces]
        obj = folder / f"{stem}.obj"
        obj.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        points = self.te_points() if te_points is None else te_points
        rows = ["METER"] + [f"{x!r},{y!r},{z!r}" for x, y, z in points.tolist()]
        (folder / f"{stem}.te.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return obj


def _pair(tmp_path: Path, level: Mesh, source: Mesh | None = None, **write) -> tuple[Path, Path]:
    """Write a source (the base mesh unless given) and a level; return both OBJ paths."""
    src = (source or Mesh()).write(tmp_path / "src", "wing")
    out = level.write(tmp_path / "wing_R2", "wing_R2", **write)
    return out, src


def _audit(level: Path, source: Path | None = None, **kw) -> tuple[MeshAudit, list[str]]:
    """Audit and return the audit with the text of every package warning it raised."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        audit = audit_mesh(level, against=source, **kw)
    return audit, [str(w.message) for w in seen if issubclass(w.category, GeometryWarning)]


def _item(audit: MeshAudit, name: str, family: str = "(whole mesh)"):
    """Return the one gate or check of that name and family."""
    found = [i for i in audit.gates + audit.checks if i.name == name and i.family == family]
    assert len(found) == 1, (name, family, [i.line() for i in audit.gates + audit.checks])
    return found[0]


def _assert_fails(audit: MeshAudit, said: list[str], name: str, family: str, text: str) -> None:
    """Assert the item failed, the audit failed, was written, and a warning names it."""
    item = _item(audit, name, family)
    assert item.verdict == "fail", item.line()
    assert not audit.passed
    written = json.loads(audit.path.read_text(encoding="utf-8"))
    assert written["passed"] is False and written["schema_version"] == 1
    assert any(name in w and family in w and text in w for w in said), said


def test_a_level_equal_to_its_source_passes_every_gate_and_check(tmp_path):
    """the control: the source against itself passes, no
     warning.

    Every gate is judged (G6 with the triangle family named as a grid at factor 1), the
    closed box yields its positive volume, and the audit is written beside the level.
    """
    level, src = _pair(tmp_path, Mesh())
    audit, said = _audit(level, src, unchanged_grids=["A"])
    assert audit.passed and said == []
    assert [i.verdict for i in audit.gates] == ["pass"] * len(audit.gates)
    assert {i.name for i in audit.gates} == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert _item(audit, "G2", "Pod").values["enclosed_volume"] > 0
    # the line's length and its ends join the counts.
    assert _item(audit, "G4").values == {
        "points": NY, "points_on_edges": NY, "points_sharing_an_edge": 0, "chains": 1,
        "source_chains": 1,
        "length": 1.0, "source_length": 1.0, "source_ends": 2, "source_ends_missing": 0,
    }  # fmt: skip
    assert _item(audit, "G5", "A and B").values["shared_nodes"] == NX + 1
    assert {i.verdict for i in audit.checks} <= {"pass", "not judged"}
    assert audit.path == level.with_name("wing_R2.audit.json")
    written = json.loads(audit.path.read_text(encoding="utf-8"))
    assert written["schema_version"] == 1 and written["passed"] is True
    assert written["mesh"] == "wing_R2.obj" and written["source"] == "wing.obj"
    assert set(written["figures"]) == {"A", "B", "Pod", "(whole mesh)"}


def _nonmanifold(m: Mesh) -> None:
    v00, v11 = _node(2, 1), _node(3, 2)
    apex = m.add_vertex(0.5 * (m.verts[v00] + m.verts[v11]) + [0.0, 0.0, 0.3])
    m.families["A"].append([v00, v11, apex])


def _duplicate(m: Mesh) -> None:
    twin = m.add_vertex(m.verts[_node(0, 0)])
    m.families["A"].append([twin, m.add_vertex([-0.3, 0.0, 0.0]), m.add_vertex([-0.3, 0.1, 0.0])])


def _zero_area(m: Mesh) -> None:
    m.families["A"].append([m.add_vertex([5.0 + d, 5.0, 5.0]) for d in (0.0, 0.5, 1.0)])


@pytest.mark.parametrize(
    ("plant", "value"),
    [
        (_nonmanifold, "edges_of_more_than_two_faces 1"),
        (_duplicate, "node_pairs_at_one_position 1"),
        (_zero_area, "faces_of_zero_area 1"),
    ],
)
def test_g1_fails_on_each_of_its_three_defects(tmp_path, plant, value):
    """( G1): an edge of three faces, two nodes at one position,
    a face of zero area each fail G1, against the source and alone; the audit is written."""
    m = Mesh()
    plant(m)
    level, src = _pair(tmp_path, m)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G1", "(whole mesh)", value)
    alone, said = _audit(level)
    _assert_fails(alone, said, "G1", "(whole mesh)", value)


def test_g2_fails_on_a_flipped_face_and_on_an_inward_closed_family(tmp_path):
    """( G2): a face whose neighbours run the other way fails G2 on the
    whole mesh; a closed family turned inside out fails G2 on that family, alone too."""
    m = Mesh()
    m.families["A"][14] = m.families["A"][14][::-1]
    level, src = _pair(tmp_path, m)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G2", "(whole mesh)", "neighbours_of_opposite_orientation 3")
    inward = Mesh()
    inward.families["Pod"] = [f[::-1] for f in inward.families["Pod"]]
    level, src = _pair(tmp_path / "pod", inward)
    alone, said = _audit(level)
    _assert_fails(alone, said, "G2", "Pod", "enclosed_volume -")
    assert _item(alone, "G2").verdict == "pass"


def test_g3_fails_on_a_new_hole_and_on_a_hole_moved_elsewhere(tmp_path):
    """( G3): a level that opens a hole fails G3 by count;
    a level whose hole is not the source's opening fails by matching; alone, G3 is reported."""
    m = Mesh()
    del m.families["A"][14]  # cell (2, 1): every node of it is inside the sheet
    level, src = _pair(tmp_path, m)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G3", "(whole mesh)", "open_loops 2, source_open_loops 1")
    alone, said = _audit(level)
    assert _item(alone, "G3").verdict == "not judged" and alone.passed and said == []
    moved = Mesh()
    del moved.families["B"][4]  # cell (1, 4), far from cell (2, 1)
    level, src = _pair(tmp_path / "moved", moved, source=m)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G3", "(whole mesh)", "unmatched_loops 2")
    assert _item(audit, "G3").values["open_loops"] == 2


def _plate(path: Path, hole: tuple[int, int]) -> Path:
    """Write an 8 by 8 plate of unit quadrilaterals with the one cell ``hole`` left open."""
    nodes = [f"v {i} {j} 0" for j in range(9) for i in range(9)]
    cells = [
        f"f {9 * j + i + 1} {9 * j + i + 2} {9 * (j + 1) + i + 2} {9 * (j + 1) + i + 1}"
        for j in range(8)
        for i in range(8)
        if (i, j) != hole
    ]
    path.write_text("\n".join([*nodes, "g S", *cells]) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("hole", [(3, 2), (4, 3), (5, 5)], ids=["beside", "near", "far"])
def test_g3_fails_on_a_hole_moved_near_with_the_outer_boundary_unchanged(tmp_path, hole):
    """( G3;): a hole closed and another opened nearby fails G3.

    The plate's outer boundary is unchanged and the level holds as many loops
    as the source, so only the place of the opening tells them apart: a hole
    moved to the next cell, to a cell two away or far away is not the source's
    opening. Control: the same plate against itself passes G3 with no loop
    unmatched.

    Release review of 0.38.0: proves the fix of RV-03.
    """
    src = _plate(tmp_path / "source.obj", (2, 2))
    level = _plate(tmp_path / "level.obj", hole)
    same, said = _audit(src, src)
    assert _item(same, "G3").values["unmatched_loops"] == 0 and same.passed and said == []
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G3", "(whole mesh)", "unmatched_loops 2")
    assert _item(audit, "G3").values["open_loops"] == 2


def test_g4_fails_on_a_point_off_every_edge_and_on_a_split_chain(tmp_path):
    """( G4): a trailing-edge point off every mesh edge fails G4, alone
    too; a points file whose chains are not the source's in number fails G4 against it."""
    m = Mesh()
    off = m.te_points()
    off[2, 2] += 0.01
    level, src = _pair(tmp_path, m, te_points=off)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G4", "(whole mesh)", f"points {NY}, points_on_edges {NY - 1}")
    alone, said = _audit(level)
    _assert_fails(alone, said, "G4", "(whole mesh)", f"points_on_edges {NY - 1}")
    split = numpy.delete(m.te_points(), 2, axis=0)
    level, src = _pair(tmp_path / "split", Mesh(), te_points=split)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G4", "(whole mesh)", "chains 2, source_chains 1")


def test_g4_reads_the_points_file_the_sidecar_names_and_refuses_a_missing_one(tmp_path):
    """( G4): the points file is the one ``[trailing_edges] file`` names;
    a named file that does not exist is refused, and nothing is written."""
    level, src = _pair(tmp_path, Mesh())
    level.with_suffix(".te.txt").rename(level.with_name("edge.te.txt"))
    sidecar = level.with_name("wing_R2.boundaries.toml")
    sidecar.write_text(
        '[trailing_edges]\nfile = "edge.te.txt"\ntype = "STANDARD"\n', encoding="utf-8"
    )
    audit, _ = _audit(level, src)
    assert _item(audit, "G4").values["points_on_edges"] == NY and audit.passed
    audit.path.unlink()
    sidecar.write_text(
        '[trailing_edges]\nfile = "gone.te.txt"\ntype = "STANDARD"\n', encoding="utf-8"
    )
    with pytest.raises(InputArtifactError, match=r"gone\.te\.txt.*Nothing was written\."):
        audit_mesh(level, against=src)
    assert not audit.path.exists()


def test_g5_fails_when_two_families_no_longer_share_their_nodes(tmp_path):
    """( G5): families that shared nodes in the source and share none in
    the level fail G5 on that pair; alone, G5 is reported and not judged."""
    m = Mesh()
    seam = {_node(i, NY // 2) for i in range(NX + 1)}
    twins = {v: m.add_vertex(m.verts[v]) for v in sorted(seam)}
    m.families["B"] = [[twins.get(v, v) for v in f] for f in m.families["B"]]
    m.te = [(twins.get(p, p), q) for p, q in m.te]
    level, src = _pair(tmp_path, m)
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "G5", "A and B", f"shared_nodes 0, source_shared_nodes {NX + 1}")
    alone, _ = _audit(level)
    assert _item(alone, "G5").verdict == "not judged"


def test_g6_fails_when_a_factor_one_grid_family_moves_or_reorders(tmp_path):
    """( G6): a grid family named as unchanged fails G6 when
    one node moves by 1e-6 or its faces change order; unnamed or alone, G6 is not judged."""
    moved = Mesh()
    moved.verts[_node(2, 1)] += [0.0, 0.0, 1e-6]
    level, src = _pair(tmp_path, moved)
    audit, said = _audit(level, src, unchanged_grids=["A"])
    _assert_fails(audit, said, "G6", "A", "differing_faces 6")
    unnamed, _ = _audit(level, src)
    assert _item(unnamed, "G6").verdict == "not judged"
    alone, _ = _audit(level, unchanged_grids=["A"])
    assert _item(alone, "G6").verdict == "not judged"
    reordered = Mesh()
    reordered.families["A"] = reordered.families["A"][::-1]
    level, src = _pair(tmp_path / "order", reordered)
    audit, said = _audit(level, src, unchanged_grids=["A"])
    _assert_fails(audit, said, "G6", "A", "faces 36, source_faces 36")


def _sheared() -> Mesh:
    m = Mesh()
    for i in range(1, NX, 2):
        for j in range(NY + 1):
            m.verts[_node(i, j), 1] += 0.4 / NY
    return m


def _warped() -> Mesh:
    m = Mesh()
    for i in range(NX + 1):
        for j in range(NY + 1):
            m.verts[_node(i, j), 2] += 0.05 * ((i + j) % 2)
    return m


def _graded() -> Mesh:
    steps = 3.0 ** numpy.arange(NX)
    return Mesh(xs=numpy.concatenate([[0.0], numpy.cumsum(steps) / steps.sum()]))


@pytest.mark.parametrize(
    ("build", "check", "family"),
    [(_sheared, "skewness", "A"), (_warped, "warp", "B"), (_graded, "growth", "A")],
)
def test_each_relative_check_fails_on_a_mesh_built_to_fail_it(tmp_path, build, check, family):
    """a sheared grid fails the skewness check, a warped one
     the
    warp check, a graded one the size-growth check, per family and on the whole mesh, each a
    warning naming both 95th percentiles; alone, the check is reported and not judged."""
    level, src = _pair(tmp_path, build())
    audit, said = _audit(level, src)
    for where in (family, "(whole mesh)"):
        _assert_fails(audit, said, check, where, "p95 ")
        item = _item(audit, check, where)
        assert item.values["p95"] > item.values["limit"]
    assert any(f"source_p95 {_said(_item(audit, check, family).values['source_p95'])}" in w
               for w in said)  # fmt: skip
    alone, said = _audit(level)
    assert {i.verdict for i in alone.checks} == {"not judged"} and said == []


def _said(value: float) -> str:
    return f"{value:.6g}"


def test_the_relative_limits_are_the_margin_and_the_floors_of_r3(tmp_path):
    """skewness at most the source's plus 0.05; warp and growth at most
    the larger of the source's and 10 degrees, and of the source's and 2; the percentile is the
    95th with linear interpolation between ranks."""
    level, src = _pair(tmp_path, _graded())
    audit, _ = _audit(level, src)
    base = _item(audit, "skewness", "A").values
    assert base["limit"] == pytest.approx(base["source_p95"] + 0.05)
    growth = _item(audit, "growth", "A").values
    assert growth["limit"] == max(growth["source_p95"], 2.0) == 2.0
    warp = _item(audit, "warp", "B").values
    assert warp["limit"] == max(warp["source_p95"], 10.0)
    m = Mesh()
    m.verts[: (NX + 1) * (NY + 1), 2] += numpy.random.default_rng(0).uniform(
        -0.02, 0.02, (NX + 1) * (NY + 1)
    )
    aspects = []
    for f in m.families["A"]:
        p = m.verts[f]
        e = numpy.linalg.norm(numpy.roll(p, -1, axis=0) - p, axis=1)
        aspects.append(e.max() / e.min())
    alone, _ = _audit(m.write(tmp_path / "alone", "alone"))
    linear = numpy.percentile(aspects, 95, method="linear")
    assert linear != numpy.percentile(aspects, 95, method="nearest")
    assert alone.figures["A"]["aspect_p95"] == pytest.approx(linear, rel=1e-12)


def test_size_growth_across_a_corner_is_not_a_size_jump(tmp_path):
    """the size growth is read across edges whose dihedral is below 30
    degrees only, so a long box whose sides are ten times its ends grows by 1, and its growth
    check passes against a cube (its skewness check is the one that fails)."""
    level, src = _pair(tmp_path, Mesh(box=2.0))
    audit, said = _audit(level, src)
    assert audit.figures["Pod"]["growth_p95"] == pytest.approx(1.0)
    assert _item(audit, "growth", "Pod").verdict == "pass"
    assert not [w for w in said if "growth" in w]


def _slender() -> Mesh:
    return Mesh(xs=[0.0, 0.2, 0.4, 0.6, 0.8, 0.999, 1.0], plane=True)


def test_each_reported_figure_exceeded_gives_its_figure_and_no_warning(tmp_path):
    """a mesh in the plane y = 0 whose last column at the trailing
    edge is a thousandth wide exceeds every reported figure (aspect above 50, the
    pre-processor's thresholds, the face quality ratio above 2, trailing-edge triangles above
    50, faces on y = 0); against itself and alone it passes, with each figure and no warning."""
    level, src = _pair(tmp_path, _slender(), source=_slender())
    for audit, said in (_audit(level, src), _audit(level)):
        assert audit.passed and said == []
        whole = audit.figures["(whole mesh)"]
        assert whole["aspect_max"] > 50 and whole["aspect_over_50"] > 0
        # Only the thousandth-wide column breaks a threshold: its triangles in A, its quads in B.
        slim_triangles, slim_quads = 2 * (NY // 2), NY - NY // 2
        assert whole["aspect_over_8"] == whole["aspect_over_50"] == slim_triangles + slim_quads
        assert whole["tri_min_angle_under_30_degrees"] == slim_triangles
        assert whole["skewness_over_0p5"] == slim_triangles
        assert whole["quad_min_angle_under_45_degrees"] == whole["warp_over_10_degrees"] == 0
        assert whole["quality_ratio_over_2"] > 0 and whole["quality_ratio_p95"] is not None
        # Each trailing-edge edge has one face: a triangle in A, a quadrilateral in B.
        assert whole["trailing_edge_triangles_aspect_over_50"] == NY // 2
        assert audit.figures["B"]["trailing_edge_triangles_aspect_over_50"] == 0
        assert whole["faces_on_y0_plane"] == 2 * NX * (NY // 2) + NX * (NY - NY // 2)
        assert audit.figures["Pod"]["faces_on_y0_plane"] == 0
        written = json.loads(audit.path.read_text(encoding="utf-8"))
        assert written["figures"]["(whole mesh)"]["aspect_over_50"] == whole["aspect_over_50"]


def test_without_a_source_only_g1_g2_and_g4_are_judged(tmp_path):
    """alone, G1, G2 and G4 carry a verdict and G3, G5, G6 and every
    relative check are reported; the audit names no source."""
    audit, said = _audit(Mesh().write(tmp_path, "wing"))
    verdicts = {(i.name, i.family): i.verdict for i in audit.gates}
    assert {k for k, v in verdicts.items() if v != "not judged"} == {
        ("G1", "(whole mesh)"), ("G2", "(whole mesh)"), ("G2", "Pod"), ("G4", "(whole mesh)")
    }  # fmt: skip
    assert {i.verdict for i in audit.checks} == {"not judged"}
    assert audit.source is None and audit.passed and said == []
    assert json.loads(audit.path.read_text(encoding="utf-8"))["source"] is None


def _run(*argv: str) -> tuple[int, list[str]]:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        code = cli.main(list(argv))
    return code, [str(w.message) for w in seen if issubclass(w.category, GeometryWarning)]


def test_the_command_exits_0_when_every_gate_and_check_passes_and_writes_the_csv(tmp_path, capsys):
    """``audit OBJ --against SOURCE --out FILE`` exits 0, prints the summary and the files
    written as JSON, and the CSV holds one row per family and figure."""
    level, src = _pair(tmp_path, Mesh())
    table = tmp_path / "audit.csv"
    code, said = _run("audit", str(level), "--against", str(src), "--out", str(table))
    record = json.loads(capsys.readouterr().out)
    assert code == 0 and said == [] and record["passed"] is True
    assert any("every gate and check judged passed; " in line for line in record["summary"])
    assert any(" not judged yet" in line for line in record["summary"])
    assert str(level.with_suffix(".audit.json")) in record["files"]
    assert record["provenance"].endswith("wing_R2.audit.json.provenance.json")
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    figures = json.loads(level.with_name("wing_R2.audit.json").read_text(encoding="utf-8"))
    expected = sum(len(v) for v in figures["figures"].values())
    assert rows[0] == ["family", "figure", "value"] and len(rows) == 1 + expected
    assert {r[0] for r in rows[1:]} == {"A", "B", "Pod", "(whole mesh)"}
    assert (
        b"\r" not in table.read_bytes()
        and b"\r" not in level.with_name("wing_R2.audit.json").read_bytes()
    )


def test_the_command_exits_1_when_a_gate_fails_and_still_writes_the_audit(tmp_path, capsys):
    """a level that opens a hole exits 1 with the warning naming G3,
    the family and both values, and the audit is written."""
    m = Mesh()
    del m.families["A"][14]
    level, src = _pair(tmp_path, m)
    code, said = _run("audit", str(level), "--against", str(src))
    assert code == 1
    assert any("G3 failed on (whole mesh): open_loops 2, source_open_loops 1" in w for w in said)
    assert any(
        "the audit FAILED" in line for line in json.loads(capsys.readouterr().out)["summary"]
    )
    assert json.loads(level.with_name("wing_R2.audit.json").read_text())["passed"] is False


def test_the_command_exits_2_on_a_refusal_and_writes_nothing(tmp_path, capsys):
    """a missing OBJ, an OBJ with no face, a missing source and a CSV in
    a folder that does not exist each exit 2 with the refusal, and no audit is written."""
    level, src = _pair(tmp_path, Mesh())
    empty = tmp_path / "empty.obj"
    empty.write_text("v 0 0 0\n", encoding="utf-8")
    cases = [
        ([str(tmp_path / "none.obj")], "none.obj: no such OBJ file"),
        ([str(empty)], "the file holds no face"),
        ([str(level), "--against", str(tmp_path / "gone.obj")], "gone.obj: no such OBJ file"),
        ([str(level), "--out", str(tmp_path / "no" / "a.csv")], "--out names a folder"),
    ]
    for argv, refusal in cases:
        code, _ = _run("audit", *argv)
        err = capsys.readouterr().err
        assert code == 2 and refusal in err and "Nothing was written." in err, (argv, err)
    assert not list(tmp_path.rglob("*.audit.json"))


def test_the_command_refuses_a_csv_it_cannot_write_before_the_audit(tmp_path, capsys):
    """a CSV file that cannot be written is refused before any audit.

    The file exists and is read-only, so writing it would fail after the audit
    was written; the command refuses it first with the standard refusal (the
    file, the reason, what to do, "Nothing was written."), exits 2 and leaves
    no audit. Control: the same file made writable again is written and the
    command exits 0.
    """
    import os
    import stat

    level, src = _pair(tmp_path, Mesh())
    target = tmp_path / "figures.csv"
    target.write_text("kept\n", encoding="utf-8")
    os.chmod(target, stat.S_IREAD)
    try:
        if os.access(target, os.W_OK):
            pytest.skip("this account writes read-only files")
        code, _ = _run("audit", str(level), "--against", str(src), "--out", str(target))
        err = capsys.readouterr().err
        assert code == 2 and f"{target}: --out names a file that cannot be written" in err
        assert "and run again. Nothing was written." in err, err
        assert not list(tmp_path.rglob("*.audit.json")) and target.read_text() == "kept\n"
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
    code, _ = _run("audit", str(level), "--against", str(src), "--out", str(target))
    assert code == 0 and target.read_text(encoding="utf-8").startswith("family,figure,value")


def _graded_by(ratio: float) -> Mesh:
    steps = ratio ** numpy.arange(NX)
    return Mesh(xs=numpy.concatenate([[0.0], numpy.cumsum(steps) / steps.sum()]))


def test_a_check_whose_source_is_beyond_its_floor_is_judged_against_the_source(tmp_path):
    """(): the limit is the larger of the source's figure and the
    floor, so a level equal to a graded source (growth beyond 2) passes on growth, a level well
    beyond that source fails, and the same graded level against the even source fails (control)."""
    level, src = _pair(tmp_path / "same", _graded(), _graded())
    audit, said = _audit(level, src)
    growth = _item(audit, "growth", "A")
    assert growth.values["source_beyond_practice"] is True
    assert growth.verdict == "pass", growth.line()
    assert not any("growth" in w and " A:" in w for w in said)
    level, src = _pair(tmp_path / "beyond", _graded_by(9.0), _graded())
    audit, said = _audit(level, src)
    beyond = _item(audit, "growth", "A")
    assert beyond.values["source_beyond_practice"] is True
    assert beyond.values["limit"] == beyond.values["source_p95"] > 2.0
    _assert_fails(audit, said, "growth", "A", "p95 ")
    level, src = _pair(tmp_path / "even", _graded())
    audit, said = _audit(level, src)
    _assert_fails(audit, said, "growth", "A", "p95 ")


def test_mesh_audit_g5_fails_when_a_family_that_shared_nodes_is_gone(tmp_path):
    """F-1203: a source pair whose family is absent from the level fails G5 (it was not judged)."""
    gone = Mesh()
    del gone.families["B"]
    level, src = _pair(tmp_path, gone)
    audit, _ = _audit(level, src)
    assert _item(audit, "G5", "A and B").verdict == "fail"
    assert not audit.passed


def test_mesh_audit_a_duplicated_edge_does_not_erase_the_ends_of_the_line(tmp_path):
    """F-303: edges listed twice are one edge; the line keeps its two ends and its length."""
    m = Mesh()
    once, src = _pair(tmp_path / "once", m)
    twice_points = numpy.vstack([m.te_points()[:1], m.te_points()])
    twice, _ = _pair(tmp_path / "twice", m, te_points=twice_points)
    clean, _said = _audit(once, src)
    doubled, _said = _audit(twice, src)
    assert _item(clean, "G4").values["source_ends_missing"] == 0
    g4 = _item(doubled, "G4").values
    assert g4["points_sharing_an_edge"] == 1
    assert g4["source_ends_missing"] == 0
    assert g4["length"] == pytest.approx(_item(clean, "G4").values["length"])


def test_mesh_audit_g2_judges_a_closed_body_split_into_open_families(tmp_path):
    """F-304: an inward closed body held by two open families fails G2 on the whole mesh; the
    same body turned outward is the control."""
    for flip, verdict in ((True, "fail"), (False, "pass")):
        m = Mesh()
        pod = [f[::-1] if flip else f for f in m.families.pop("Pod")]
        m.families["P1"], m.families["P2"] = pod[:6], pod[6:]
        level, _ = _pair(tmp_path / verdict, m)
        audit, _said = _audit(level)
        assert _item(audit, "G2").verdict == verdict, _item(audit, "G2").line()


def test_a_record_of_another_source_does_not_group_this_sources_components(tmp_path):
    """(, F-1202): components are read only from a record that names this
    level and this source; a record of another refinement leaves the source's families alone."""
    level, src = _pair(tmp_path, Mesh())
    record = level.with_name(level.stem + ".refine.json")
    named = {"level": level.stem, "source": "other.obj", "components": {"AB": ["A", "B"]}}
    record.write_text(json.dumps(named), encoding="utf-8")
    audit, _ = _audit(level, src)
    assert _item(audit, "skewness", "A").verdict != "not judged"
    named["source"] = src.name
    record.write_text(json.dumps(named), encoding="utf-8")
    audit, _ = _audit(level, src)
    assert _item(audit, "skewness", "A").verdict == "not judged"


def test_files_read_is_the_one_list_with_the_record_and_the_source(tmp_path):
    """(, F-1513): files_read(mesh, against) names the refinement record
    and, with a source, the source's files; the CSV refusal reads the same list."""
    level, src = _pair(tmp_path, Mesh())
    alone = dict(MeshAudit.files_read(level))
    assert alone["its refinement record"] == level.with_name(level.stem + ".refine.json")
    both = MeshAudit.files_read(level, src)
    assert ("the source", src) in both
    assert ("the source's boundaries file", src.with_name(src.stem + ".boundaries.toml")) in both
    audit, _ = _audit(level, src)
    with pytest.raises(InputArtifactError, match="its refinement record"):
        audit.write_csv(level.with_name(level.stem + ".refine.json"))


def test_f321_the_enclosed_volume_does_not_depend_on_where_the_mesh_sits(tmp_path):
    """F-321: a closed family's volume is the same far from the origin."""
    near = Mesh()
    far = Mesh()
    far.verts[-8:] += 2.0**20
    near_audit, _said = _audit(near.write(tmp_path / "near", "wing"))
    far_audit, _said = _audit(far.write(tmp_path / "far", "wing"))
    volume = _item(near_audit, "G2", "Pod").values["enclosed_volume"]
    assert volume == pytest.approx(0.2**3, rel=1e-9)
    assert _item(far_audit, "G2", "Pod").values["enclosed_volume"] == pytest.approx(
        volume, rel=1e-6
    )


def test_f324_two_loops_touching_at_a_node_both_hold_it():
    """F-324: a node two boundary loops share is in both, so neither is one node short."""
    from cfdbender.mesh._refine._geometry import boundary_loops

    loops = boundary_loops([[0, 1, 2, 3], [2, 4, 5, 6]])
    assert sorted(sorted(loop) for loop in loops) == [[0, 1, 2, 3], [2, 4, 5, 6]]
    assert boundary_loops([[0, 1, 2, 3], [4, 5, 6, 7]]) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_f1595_a_standalone_audit_does_not_claim_every_gate_passed(tmp_path):
    """The summary never says every gate and check passed while some are not judged yet."""
    level, _ = _pair(tmp_path, Mesh())
    audit, _said = _audit(level)
    unjudged = [i for i in audit.gates + audit.checks if i.verdict == "not judged"]
    assert unjudged, "a standalone audit leaves the relative gates not judged"
    last = audit.summary()[-1]
    assert last == f"every gate and check judged passed; {len(unjudged)} not judged yet"


def test_x_f_0597_g4_names_a_missing_points_file(tmp_path):
    """With no trailing-edge points file G4 passes on 0 < 0, and the line says why."""
    level, _ = _pair(tmp_path, Mesh())
    level.with_suffix(".te.txt").unlink()
    audit, _said = _audit(level)
    g4 = _item(audit, "G4")
    assert g4.values["points"] == 0 and g4.values["points_file"] == "none found"
    assert "points_file none found" in g4.line()


def test_x_f_0596_an_unreadable_refinement_record_is_said(tmp_path):
    """G4 and G6 are judged on less information, and the audit says so."""
    level, _ = _pair(tmp_path, Mesh())
    level.with_name(level.stem + ".refine.json").write_text("{not json", encoding="utf-8")
    _audit_result, said = _audit(level)
    assert any("not readable as a refinement record" in text for text in said), said

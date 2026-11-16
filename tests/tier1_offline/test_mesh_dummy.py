"""a grid's shared nodes become its neighbours', and components.

Every source is synthetic (tests/p0380_mesh_fixtures.py): the cambered sheet
grid ``G``; ``T``, a strip of triangles (no grid) sharing G's tip station
node for node; ``U``, a strip of triangles sharing T's last row only, so it
touches T and never G; and for ``G2``, a second grid sharing G's tip
station, of quadrilaterals (resampled with G as one layout, ) or of
split quadrilaterals (which joins no layout and falls back to the rule of
the first grid). Each behaviour is also run with the step it rests on
taken away or on a planted input, so a check that accepts everything cannot
pass.
"""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import (  # noqa: F401
    HAS_INSPECT,
    diag,
    ending,
    face_coordinates,
    open_loop_count,
    open_loops,
    read_mesh,
    sheet_with_strips,
    write_source,
)

from cfdbender._errors import GeometryWarning, InputArtifactError
from cfdbender.mesh import cli
from cfdbender.mesh._refine import refine_mesh

#: The row spacings of T beyond G's tip station, growing so a moved node shows.
STRIP = (0.03, 0.04, 0.05, 0.065, 0.08)
GTU = (("T", "tri", STRIP), ("U", "tri", (0.1, 0.1)))
GG = (("G2", "quad", (0.05, 0.05, 0.05)),)
#: The y of G's tip station, the curve G shares with T (and with G2).
TIP = 2.0


def _source(tmp_path: Path, strips=GTU, *, refine_toml=None, first=None) -> Path:
    """Write ``<tmp>/src/wing.obj``; ``first`` names a family written before the others."""
    mesh = sheet_with_strips(strips)
    if first is not None:
        mesh.families = {first: mesh.families[first]} | {
            n: f for n, f in mesh.families.items() if n != first
        }
    return write_source(tmp_path / "src", "wing", mesh, refine_toml=refine_toml)


def _left(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.name != "src")


def _nodes(faces) -> set[int]:
    return {v for f in faces for v in f}


def _all(families) -> list[list[int]]:
    return [f for fs in families.values() for f in fs]


def _refused(tmp_path: Path, src: Path, capsys, *args, **kwargs) -> str:
    """Return the refusal text, the same from the command, with no level written."""
    with pytest.raises(InputArtifactError) as caught:
        refine_mesh(src, *args, **kwargs)
    text = str(caught.value)
    pdf = src.with_suffix(".refine-refusal.pdf")
    assert text.endswith(diag(pdf).lstrip()) and _left(tmp_path) == []
    assert not HAS_INSPECT or pdf.read_bytes().startswith(b"%PDF")
    capsys.readouterr()
    argv = [str(a) for a in args]
    if kwargs.get("families"):
        argv += ["--families", ",".join(kwargs["families"])]
    assert cli.main(["refine", str(src), *argv]) == 2
    assert text in capsys.readouterr().err and _left(tmp_path) == []
    return text


# --------------------------------------------------------- grids first

#: G2 as a sheet of split quadrilaterals: a grid, but not one of quadrilaterals.
GG_SPLIT = (("G2", "tri", (0.05, 0.05, 0.05)),)


def _split_source(tmp_path: Path, *, refine_toml=None, first=None) -> Path:
    """Write the source whose G2 is a sheet of split quadrilaterals with its trailing edge."""
    mesh = sheet_with_strips(GG_SPLIT)
    column = [row[0] for row in mesh.rows["G2"]]
    mesh.te = np.vstack([mesh.te, 0.5 * (mesh.verts[column[:-1]] + mesh.verts[column[1:]])])
    if first is not None:
        mesh.families = {first: mesh.families[first]} | {
            n: f for n, f in mesh.families.items() if n != first
        }
    return write_source(tmp_path / "src", "wing", mesh, refine_toml=refine_toml)


def test_r1_two_grids_of_quadrilaterals_sharing_nodes_are_both_grids(tmp_path):
    """(): G and G2, quadrilateral sheets sharing G's tip, stay grids.

    CHANGED IN 0.44.0 (, holder decision of 2026-10-22): this test replaced
    the rule it checked before, that G2 sharing G's nodes is remeshed. Grids
    of quadrilaterals that share nodes are now resampled as one layout: both are
    grids, each names the other under ``grouped_with``, and their shared station
    holds the same 37 nodes from both sides. control: selecting G2
    alone changes G only in the direction needed by their shared station.
    """
    src = _source(tmp_path, GG)
    level = refine_mesh(src, 2.0)
    report = level.report
    assert (report["G"]["method"], report["G2"]["method"]) == ("grid", "grid")
    assert (report["G"]["grouped_with"], report["G2"]["grouped_with"]) == (["G2"], ["G"])
    lv, lf = read_mesh(level.obj)
    assert len(_tip_nodes(lv, lf["G"])) == 37 and _nodes(lf["G"]) & _nodes(lf["G2"]) == (
        _tip_nodes(lv, lf["G"])
    )
    assert all(g.passed for g in level.audit.gates), [g.verdict for g in level.audit.gates]
    alone = refine_mesh(src, 2.0, families=["G2"]).report
    assert alone["G2"]["method"] == "grid" and alone["G2"]["grouped_with"] == ["G"]
    assert alone["G"]["continuity"]["directions"] == ["normal"]
    assert alone["G"]["continuity"]["because"] == ["G2"]


def test_r1_a_grid_that_cannot_join_the_layout_is_remeshed_under_auto(tmp_path):
    """a grid of split quadrilaterals sharing G's nodes is remeshed.

    The fallback of G2, a sheet of split quadrilaterals, joins no
    layout, so the first grid in the source's family order is a grid and the
    later one is remeshed, as before 0.44.0. Controls: G2 alone is refined as a
    grid (it is one), and with G2 written first in the source it is G that is
    remeshed, naming G2.
    """
    src = _split_source(tmp_path)
    report = refine_mesh(src, 2.0).report
    assert report["G"]["method"] == "grid"
    assert (report["G2"]["method"], report["G2"]["reason"]) == (
        "remesh",
        "shares nodes with the grid G",
    )
    alone = refine_mesh(src, 2.0, families=["G2"]).report
    assert alone["G2"]["method"] == "grid" and alone["G"]["method"] == "unchanged"
    swapped = refine_mesh(_split_source(tmp_path / "swapped", first="G2"), 2.0).report
    assert swapped["G2"]["method"] == "grid"
    assert swapped["G"]["reason"] == "shares nodes with the grid G2"


def test_r1_method_grid_on_the_second_grid_is_refused_naming_both(tmp_path, capsys):
    """``grid`` on a G2 that cannot join G's layout is refused.

    CHANGED IN 0.44.0 (): G2 is now a sheet of split quadrilaterals,
    which joins no layout; a quadrilateral G2 is resampled with G and accepted
    under ``grid``, which is the first control. Second control: ``grid`` on the
    split G2 with G not refined is accepted.
    """
    toml = '[families.G]\nfactor = 2\n\n[families.G2]\nfactor = 2\nmethod = "grid"\n'
    src = _split_source(tmp_path, refine_toml=toml)
    text = _refused(tmp_path, src, capsys)
    assert text == (
        f"{src} family G2: method is grid and it shares nodes with G, refined as a grid first. "
        f"{ending(src.with_suffix('.refine-refusal.pdf'))}"
    )
    quads = _source(tmp_path / "quads", GG, refine_toml=toml)
    assert refine_mesh(quads, out_dir=tmp_path / "q").report["G2"]["method"] == "grid"
    # ARCH48-R2: accept_remesh warns and remeshes the demanded grid instead.
    with pytest.warns(GeometryWarning, match="it is remeshed with factor 2"):
        accepted = refine_mesh(src, accept_remesh=True, out_dir=tmp_path / "acc")
    assert accepted.report["G2"]["method"] == "remesh"
    src.with_name("wing.refine.toml").write_bytes(
        b'[[perimeter]]\nfile = "wing.te_nodes.txt"\nexport_te_per_family = true\n'
        b'[families.G2]\nfactor = 2\nmethod = "grid"\n'
    )
    assert refine_mesh(src).report["G2"]["method"] == "grid"


# ------------------------------------------- the remeshed neighbour conforms


def _tip_nodes(verts: np.ndarray, faces) -> set[int]:
    """Return the nodes of the faces that lie on G's tip station."""
    return {v for v in _nodes(faces) if abs(verts[v, 1] - TIP) < 1e-12}


def test_r2_the_remeshed_neighbour_holds_exactly_the_grids_new_nodes(tmp_path):
    """T, remeshed, shares every new node of G's tip and no other.

    The level welds G and T on the 37 nodes of G's refined tip station, and
    its open loops are the source's. Control: the source's G and T share
    19 nodes, so 18 of the 37 the level's share are G's new ones; the mutant
    of the commit message that drops the hand-over in the code leaves T on
    the 19 and the level with 20 open loops, and fails this test.
    """
    src = _source(tmp_path)
    sv, sf = read_mesh(src)
    level = refine_mesh(src, 2.0, families=["G", "T"])
    lv, lf = read_mesh(level.obj)
    tip = _tip_nodes(lv, lf["G"])
    assert len(tip) == 2 * 18 + 1
    assert _nodes(lf["G"]) & _nodes(lf["T"]) == tip == _tip_nodes(lv, lf["T"])
    assert level.report["G"]["interfaces"] == {"T": "19 -> 37 nodes"}
    assert open_loop_count(_all(lf)) == open_loop_count(_all(sf)) == 1
    shared = {tuple(sv[v]) for v in _nodes(sf["G"]) & _nodes(sf["T"])}
    assert len(shared) == 19 and len({tuple(lv[v]) for v in tip} - shared) == 18


# ---------------------------------------- the band of an unchanged neighbour


def _band(faces, seed: set[int], layers: int) -> set[int]:
    """Return the faces within ``layers`` face layers of the seed nodes.

    Layer one is the faces with a node on the curve; each next layer the
    faces sharing a node with the layers before.
    """
    nodes, chosen = set(seed), set()
    for _ in range(layers):
        new = {i for i, f in enumerate(faces) if i not in chosen and nodes & set(f)}
        chosen |= new
        nodes |= {v for i in new for v in faces[i]}
    return chosen


def _outside(src: Path, layers: int) -> tuple[list[tuple], int]:
    """Return T's source faces outside the band, in order, and the band's face count."""
    sv, sf = read_mesh(src)
    band = _band(sf["T"], _nodes(sf["G"]) & _nodes(sf["T"]), layers)
    coords = face_coordinates(sv, sf["T"])
    return [c for i, c in enumerate(coords) if i not in band], len(band)


def _kept_in_order(level_faces: list[tuple], outside: list[tuple]) -> bool:
    """Return whether every face outside the band is in the level, unchanged and in order."""
    wanted = set(outside)
    return [f for f in level_faces if f in wanted] == outside


def test_r3_an_unchanged_neighbour_changes_only_in_its_two_layer_band(tmp_path):
    """T, not asked to change, is remeshed only in a band of two layers.

    Outside the band T's faces equal the source's in coordinates, first
    vertex and order; U, beyond T, is untouched; refine.json names the band's
    face count. Controls: one face outside the band rotated by a vertex fails
    the check, and so does the same check against a band of one layer (the
    second layer was remeshed); the mutant of the commit message that cuts
    three layers in the code fails it too.
    """
    src = _source(tmp_path)
    outside, count = _outside(src, 2)
    assert count == 2 * 2 * 18 and len(outside) == 180 - count
    level = refine_mesh(src, 2.0, families=["G"])
    lv, lf = read_mesh(level.obj)
    got = face_coordinates(lv, lf["T"])
    assert _kept_in_order(got, outside)
    # (0.39.0): the band beside a refined grid may also record shared_edge_target_m
    entry = dict(level.report["T"])
    target = entry.pop("shared_edge_target_mesh_units")
    assert set(target) == {"min", "mean", "max"}
    assert all(isinstance(v, float) and v > 0 for v in target.values())
    assert target["min"] <= target["mean"] <= target["max"]
    assert entry == {
        "method": "unchanged",
        "band": count,
        "faces": 180,
        "continuity": {"band": "remeshed", "because": ["G"]},
    }
    sv, sf = read_mesh(src)
    assert face_coordinates(lv, lf["U"]) == face_coordinates(sv, sf["U"])
    assert len(lf["T"]) > len(sf["T"])
    where = got.index(outside[len(outside) // 2])
    planted = [*got[:where], got[where][1:] + got[where][:1], *got[where + 1 :]]
    assert not _kept_in_order(planted, outside)
    assert not _kept_in_order(got, _outside(src, 1)[0])


@pytest.mark.parametrize(
    "kwargs",
    [{"factor": 1.0}, {"chordwise": 1.0, "spanwise": 2.0}],
    ids=["factor-1", "spanwise-only"],
)
def test_r3_when_the_grids_curve_nodes_do_not_change_nothing_of_the_neighbour_changes(
    tmp_path, kwargs
):
    """G's tip nodes unchanged leaves T and U as the source.

    At factor 1, and when only the spanwise direction is refined (the tip
    station is a knot and keeps its nodes), T and U are written face for face.
    The control is the band test above: chordwise 2 changes T.
    """
    src = _source(tmp_path)
    level = refine_mesh(src, families=["G"], **kwargs)
    sv, sf = read_mesh(src)
    lv, lf = read_mesh(level.obj)
    for name in ("T", "U"):
        assert face_coordinates(lv, lf[name]) == face_coordinates(sv, sf[name])
        assert level.report[name] == {"method": "copied"}
    assert "interfaces" not in level.report["G"]
    assert len(lf["G"]) == len(sf["G"]) * (2 if "spanwise" in kwargs else 1)


# ------------------------------------------------------------- components


def test_r4_a_component_writes_its_members_as_one_family_in_the_sources_order(tmp_path):
    """``Wing = ["T", "G"]`` writes Wing = G's faces then T's.

    The members are listed in reverse; the faces follow the source's family
    order. The boundaries file names Wing in place of G and T. Control: the
    listing's order (T then G) is not what is written. Refined, the
    component holds the level's G and T faces.
    """
    toml = '[components]\nWing = ["T", "G"]\n'
    src = _source(tmp_path, refine_toml=toml)
    level = refine_mesh(src, 1.0, families=["G"])
    sv, sf = read_mesh(src)
    lv, lf = read_mesh(level.obj)
    assert list(lf) == ["Wing", "U"]
    wing = face_coordinates(lv, lf["Wing"])
    assert wing == face_coordinates(sv, sf["G"] + sf["T"])
    assert wing != face_coordinates(sv, sf["T"] + sf["G"])
    side = (level.folder / f"{level.folder.name}.obj.boundaries.toml").read_text(encoding="utf-8")
    assert 'boundaries = ["Wing", "U"]' in side
    refined = refine_mesh(src, 2.0, families=["G", "T"])
    counts = {n: len(f) for n, f in read_mesh(refined.obj)[1].items()}
    t_after = refined.report["T"]["faces_after"]
    assert counts == {"Wing": 576 + t_after, "U": 72}


@pytest.mark.parametrize(
    ("toml", "said"),
    [
        (
            '[components]\nWing = ["G", "Fin"]\n',
            "[components] Wing: the mesh holds no family 'Fin'; it holds G, T, U",
        ),
        (
            '[components]\nWing = ["G"]\nBody = ["G", "T"]\n',
            "[components] Body: family 'G' is also in component 'Wing'",
        ),
        (
            '[components]\nT = ["G", "U"]\n',
            "[components] T: a family of the mesh already has this name and is not a member",
        ),
    ],
    ids=["missing-member", "member-in-two", "name-clash"],
)
def test_r4_a_bad_component_is_refused_naming_the_component_and_the_family(
    tmp_path, capsys, toml, said
):
    """each refusal by both routes, nothing written.

    Control: a component named after one of its own members is accepted.
    """
    src = _source(tmp_path, refine_toml=toml)
    text = _refused(tmp_path, src, capsys, 1.0, families=["G"])
    assert text.startswith(str(src.with_name("wing.refine.toml"))) and said in text
    src.with_name("wing.refine.toml").write_bytes(b'[components]\nT = ["T", "U"]\n')
    level = refine_mesh(src, 1.0, families=["G"])
    assert list(read_mesh(level.obj)[1]) == ["G", "T"]


# ------------------------------------------------------------ open loops


def _box(verts: np.ndarray, loop: set[int]) -> np.ndarray:
    return np.concatenate([verts[sorted(loop)].min(axis=0), verts[sorted(loop)].max(axis=0)])


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_r5_the_levels_open_loops_are_the_sources(tmp_path, factor):
    """refined or coarsened, the level has the source's open loops.

    Every family changes (G as a grid, T and U remeshed together and onto
    G's new nodes); the level has as many loops as the source, its loop
    spanning the source loop's box within 1e-3 of the size (a coarsened
    station drops the source's highest node). Control: one interior face
    taken out of the level opens a loop the same count sees.
    """
    src = _source(tmp_path)
    sv, sf = read_mesh(src)
    level = refine_mesh(src, factor)
    lv, lf = read_mesh(level.obj)
    source, got = open_loops(_all(sf)), open_loops(_all(lf))
    assert open_loop_count(_all(lf)) == open_loop_count(_all(sf)) == len(source) == 1
    size = float(np.linalg.norm(sv.max(axis=0) - sv.min(axis=0)))
    assert np.allclose(_box(lv, got[0]), _box(sv, source[0]), rtol=0.0, atol=1e-3 * size)
    faces = _all(lf)
    boundary = set().union(*got)
    inner = next(i for i, f in enumerate(faces) if not boundary & set(f))
    assert open_loop_count(faces[:inner] + faces[inner + 1 :]) == len(source) + 1


def test_r4_the_audit_compares_a_component_with_the_union_of_its_members(tmp_path):
    """a level written with ``Wing = ["G", "T"]`` at factor
    1 is audited against the source's G and T measured as one family, so no relative check
    of Wing fails and its source figure is the union's. Control: with the components removed
    from the level's refine.json the audit compares Wing with nothing of that name in the
    source and reports it not judged."""
    import json

    from cfdbender.mesh._refine import audit_mesh

    src = _source(tmp_path, refine_toml='[components]\nWing = ["G", "T"]\n')
    level = refine_mesh(src, 1.0)
    wing = [i for i in level.audit.checks if i.family == "Wing"]
    assert wing and all(i.verdict != "fail" for i in wing), [i.line() for i in wing]
    assert any(i.verdict == "pass" for i in wing)
    record = level.folder / f"{level.folder.name}.refine.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    data["components"] = {}
    record.write_text(json.dumps(data), encoding="utf-8")
    alone = audit_mesh(level.obj, against=src)
    assert {i.verdict for i in alone.checks if i.family == "Wing"} == {"not judged"}

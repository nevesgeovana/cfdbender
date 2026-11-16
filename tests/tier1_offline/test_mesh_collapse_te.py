"""collapse a blunt trailing edge of named blades on an OBJ.

Usage: cfd-mesh collapse-te sector.obj --morphing morphing.toml --out sharp.obj
"""
# ruff: noqa: E501

import dataclasses
import json
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest

from cfdbender._errors import InputValueError, NotApplicableWarning
from cfdbender.mesh import collapse_te as cte
from cfdbender.mesh._refine._obj import ObjMesh, obj_text, read_obj, read_te
from cfdbender.mesh._sidecars import read_boundary_declarations
from cfdbender.mesh.cli import main

M, STATIONS = 3, 4
AWAY = lambda v: v + np.array([0.0, 5.0, 0.0])  # noqa: E731


def ring_xy(m, h, *, blunt=True, wide=False, fold=False):
    x = np.linspace(0, 1, m + 1)[1:]
    top = 0.08 * np.sqrt(x) * (1 - x) + h * x
    up, lo = np.c_[x, top], np.c_[x, -top]
    if fold:
        up[-2], up[-1], lo[-1] = (0.9, -0.1), (1.0, 0.1), (1.0, -0.5)
    pts = [(0.0, 0.0), *up.tolist()]
    if blunt:
        pts += [(1.0, 0.0)] if wide else []
        pts += lo[::-1].tolist()
    else:
        pts += lo[::-1][1:].tolist()
    return np.array(pts)


def blade_parts(
    name,
    base,
    *,
    strip="same",
    blunt=True,
    wide=False,
    fold=False,
    layers=1,
    move=None,
    cap="polygon",
):
    """One blade of S stations with a one-cell base strip, a tip cap polygon and a quad hub."""
    h = lambda s: (0.01 + 0.002 * s) if blunt else 0.0  # noqa: E731
    rings = [ring_xy(M, h(s), blunt=blunt, wide=wide, fold=fold) for s in range(STATIONS + 1)]
    count = len(rings[0])
    node = lambda s, k: base + s * count + k % count  # noqa: E731
    verts = [np.c_[r, np.full(len(r), 0.5 * s)] for s, r in enumerate(rings)]
    centre = np.array([0.5, 0.0])
    for layer in range(1, layers + 1):
        xy = centre + (rings[0] - centre) * (1 + 0.6 * layer)
        verts.append(np.c_[xy, np.full(count, -0.25 * layer)])
    verts = np.vstack(verts)
    blade, own = [], []
    strip_ks = (M, M + 1) if wide else (M,)
    for s in range(STATIONS):
        for k in range(count):
            quad = [node(s, k), node(s, k + 1), node(s + 1, k + 1), node(s + 1, k)]
            if not (blunt and k in strip_ks):
                blade.append(quad)
            elif strip == "same":
                blade.append(quad)
            elif strip == "own":
                own.append(quad)
            else:
                own += [quad[:3], [quad[0], quad[2], quad[3]]]
    if cap == "fan":
        blade += [
            [node(STATIONS, 0), node(STATIONS, k), node(STATIONS, k + 1)]
            for k in range(1, count - 1)
        ]
    else:
        blade.append([node(STATIONS, k) for k in range(count)])
    hub_node = lambda layer, k: (  # noqa: E731
        node(0, k) if layer == 0 else base + (STATIONS + layer) * count + k % count
    )
    hub = [
        [hub_node(i, k), hub_node(i, k + 1), hub_node(i + 1, k + 1), hub_node(i + 1, k)]
        for i in range(layers)
        for k in range(count)
    ]
    families = {name: blade}
    if own:
        families[name + "Base"] = own
    families["Hub" + name[len("Blade") :]] = hub
    upper, lower = M, (M + 2 if wide else M + 1)
    if move is not None:
        verts = move(verts)
    mids = lambda k: [  # noqa: E731
        0.5 * (verts[node(s, k) - base] + verts[node(s + 1, k) - base]) for s in range(STATIONS)
    ]
    info = {"count": count, "upper": mids(upper), "lower": mids(lower), "node": node}
    return verts, families, info


def pts(path, rows):
    path.write_text(
        "METER\n" + "\n".join(",".join(format(float(v), ".17g") for v in r) for r in rows) + "\n"
    )


def make(folder, blades=None, *, entries="two", wake=True, sidecar=True, name="sector.obj", **kw):
    """Write a mesh of one or more blades and, with ``sidecar``, its boundaries file."""
    blades = blades or [("Blade1", {})]
    allv, fams, infos, base = [], {}, {}, 0
    for blade, extra in blades:
        v, f, info = blade_parts(blade, base, **{**kw, **extra})
        allv.append(v)
        fams.update(f)
        infos[blade] = info
        base += len(v)
    verts = np.vstack(allv)
    path = folder / name
    path.write_text(obj_text(ObjMesh(verts, fams)))
    if not sidecar:
        return path, infos
    lines = [
        'fs_build = "26.125"',
        f"boundaries = {json.dumps(list(fams))}",
        "[import]",
        'units = "METER"',
    ]
    for blade, extra in blades:
        info = infos[blade]
        node = info["node"]
        blunt = extra.get("blunt", kw.get("blunt", True))
        sets = [("U", info["upper"], verts[node(0, M)])]
        if blunt:
            lower = M + 2 if extra.get("wide", kw.get("wide", False)) else M + 1
            sets.append(("L", info["lower"], verts[node(0, lower)]))
        if entries == "one" and blunt:
            pts(folder / f"{blade}.te.txt", info["upper"] + info["lower"])
            pts(folder / f"{blade}.wake.txt", [s[2] for s in sets])
            lines += [
                f"[trailing_edges.{blade}]",
                f'file = "{blade}.te.txt"',
                'type = "STANDARD"',
            ]
            continue
        for tag, rows, root in sets:
            pts(folder / f"{blade}{tag}.te.txt", rows)
            pts(folder / f"{blade}{tag}.wake.txt", [root])
            lines += [
                f"[trailing_edges.{blade}{tag}]",
                f'file = "{blade}{tag}.te.txt"',
                'type = "STANDARD"',
            ]
    for family in fams:
        if family.endswith("Base"):
            lines += [f"[base_regions.{family}]", 'trailing_edges = "auto"']
    Path(str(path) + ".boundaries.toml").write_text("\n".join(lines) + "\n")
    return path, infos


def control(folder, blades='["Blade1"]', extra=""):
    path = folder / "morphing.toml"
    path.write_text(f"[collapse_te]\nblades = {blades}\n{extra}\n")
    return path


def invoke(source, config, out, verb="collapse-te"):
    try:
        return main([verb, str(source), "--morphing", str(config), "--out", str(out)])
    except SystemExit as error:
        return error.code


def key(p):
    return tuple(float(v) for v in p)


def by_place(mesh):
    return {key(p): i for i, p in enumerate(mesh.verts)}


def run(folder, **kw):
    source, infos = make(folder, **kw)
    out = folder / "sharp.obj"
    assert invoke(source, control(folder), out) == 0
    return source, out, infos


def expected_counts(strip_faces):
    return {"strip": strip_faces}


@pytest.mark.parametrize("strip", ["same", "own", "tri"])
def test_mesh_collapse_te_merges_rungs_at_midpoints_and_keeps_everything_else(tmp_path, strip):
    source, out, infos = run(tmp_path, strip=strip)
    old, new = read_obj(source), read_obj(out)
    pairs = [(infos["Blade1"]["node"](s, M), infos["Blade1"]["node"](s, M + 1)) for s in range(5)]
    assert len(new.verts) == len(old.verts) - 5
    place = by_place(new)
    removed = {b for _, b in pairs}
    for a, b in pairs:
        mid = 0.5 * (old.verts[a] + old.verts[b])
        assert key(mid) in place
        assert key(old.verts[a]) not in place and key(old.verts[b]) not in place
    far = [i for i in range(len(old.verts)) if i not in {n for p in pairs for n in p}]
    order = [place[key(old.verts[i])] for i in far]
    assert order == sorted(order)
    assert [new.vertex_text[j] for j in order] == [old.vertex_text[i] for i in far]
    del removed
    strip_faces = 4 if strip == "same" or strip == "own" else 8
    total_old = sum(len(f) for f in old.families.values())
    assert sum(len(f) for f in new.families.values()) == total_old - strip_faces - 0
    if strip != "same":
        assert "Blade1Base" not in new.families
    report = json.loads(Path(str(out) + ".collapse.json").read_text())
    blade = report["blades"]["Blade1"]
    assert blade["strip_faces_removed"] == strip_faces and blade["nodes_merged"] == 5
    assert blade["trailing_edge_edges"] == 4 and report["checks"]["passed"]
    assert blade["base_thickness_max"] == pytest.approx(2 * blade["largest_node_move"])
    assert blade["base_thickness_max"] == pytest.approx(2 * (0.01 + 0.008))


def test_mesh_collapse_te_quads_on_the_rung_become_triangles_and_face_order_is_kept(tmp_path):
    source, out, infos = run(tmp_path, strip="own")
    old, new = read_obj(source), read_obj(out)
    node = infos["Blade1"]["node"]
    on_rung = {node(s, k) for s in range(5) for k in (M, M + 1)}
    hub_old, hub_new = old.families["Hub1"], new.families["Hub1"]
    assert len(hub_old) == len(hub_new)
    sizes = [len(f) for f in hub_new]
    assert sizes.count(3) == 1 and sizes.index(3) == M
    cap_old, cap_new = old.families["Blade1"][-1], new.families["Blade1"][-1]
    assert len(cap_new) == len(cap_old) - 1
    place = by_place(new)
    for fam in ("Hub1", "Blade1"):
        assert len(old.families[fam]) == len(new.families[fam])
        for f_old, f_new in zip(old.families[fam], new.families[fam], strict=True):
            if not set(f_old) & on_rung:
                assert [place[key(old.verts[n])] for n in f_old] == f_new


def test_mesh_collapse_te_tip_cap_triangle_that_reaches_zero_area_is_deleted(tmp_path):
    source, infos = make(tmp_path, strip="own", cap="fan")
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    old, new = read_obj(source), read_obj(out)
    assert len(new.families["Blade1"]) == len(old.families["Blade1"]) - 1
    report = json.loads(Path(str(out) + ".collapse.json").read_text())
    assert report["blades"]["Blade1"]["faces_collapsed_to_nothing"] == 1


@pytest.mark.parametrize("swap", [False, True])
def test_mesh_collapse_te_the_first_declared_rim_keeps_its_place(tmp_path, swap):
    source, infos = make(tmp_path, strip="own", wake=False)
    side = Path(str(source) + ".boundaries.toml")
    if swap:
        text = side.read_text()
        text = text.replace("Blade1U.te.txt", "@").replace("Blade1L.te.txt", "Blade1U.te.txt")
        side.write_text(text.replace("@", "Blade1L.te.txt"))
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    old, new = read_obj(source).verts, read_obj(out).verts
    node = infos["Blade1"]["node"]
    low, high = node(0, M), node(0, M + 1)
    assert high == low + 1
    midpoint = 0.5 * (old[low] + old[high])
    # The lower-numbered node leaves when the other rim is declared first, and the rest close up.
    place = high - 1 if swap else low
    assert np.array_equal(new[place], midpoint)


def test_mesh_collapse_te_all_warns_and_skips_a_family_that_is_no_strip(tmp_path):
    source, infos = make(tmp_path, [("Blade1", {}), ("Blade2", {"move": AWAY, "wide": True})])
    out = tmp_path / "sharp.obj"
    with pytest.warns(NotApplicableWarning, match="Blade2"):
        assert invoke(source, control(tmp_path, '"all"'), out) == 0
    report = json.loads(Path(str(out) + ".collapse.json").read_text())
    assert set(report["blades"]) == {"Blade1"}
    assert "Blade2" in report["skipped"]
    assert "No file was written" not in report["skipped"]["Blade2"]
    assert not report["skipped"]["Blade2"].startswith("collapse_te:")


def test_mesh_collapse_te_two_blades_and_all(tmp_path):
    source, infos = make(tmp_path, [("Blade1", {}), ("Blade2", {"move": AWAY})], strip="own")
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path, '"all"'), out) == 0
    report = json.loads(Path(str(out) + ".collapse.json").read_text())
    assert set(report["blades"]) == {"Blade1", "Blade2"}
    data = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())
    assert list(data["trailing_edges"]) == ["Blade1U", "Blade2U"]
    assert len(read_obj(out).verts) == len(read_obj(source).verts) - 10


def test_mesh_collapse_te_base_route_without_sidecar_writes_the_points_file(tmp_path):
    source, infos = make(tmp_path, strip="own", sidecar=False)
    out = tmp_path / "sharp.obj"
    cfg = control(tmp_path, extra='base = ["Blade1Base"]')
    with pytest.warns(NotApplicableWarning, match="unit is unstated"):
        assert invoke(source, cfg, out) == 0
    assert not Path(str(out) + ".boundaries.toml").exists()
    unit, rows = read_te(tmp_path / "sharp.Blade1.te.txt")
    assert len(rows) == 4 and unit == "METER"


def test_mesh_collapse_te_base_route_names_the_group_for_a_strip_in_the_blade_group_it_cannot_find(
    tmp_path,
):
    source, _ = make(tmp_path, strip="same", sidecar=False)
    out = tmp_path / "sharp.obj"
    cfg = control(tmp_path, extra='base = ["Blade1"]')
    assert invoke(source, cfg, out) == 2
    assert not out.exists()


def refused(tmp_path, capsys, source, cfg, text):
    out = tmp_path / "bad.obj"
    assert invoke(source, cfg, out) == 2
    assert text in capsys.readouterr().err
    assert not out.exists() and not Path(str(out) + ".collapse.json").exists()
    assert not list(tmp_path.glob("bad.*"))


def test_mesh_collapse_te_missing_sidecar_without_base_is_refused(tmp_path, capsys):
    source, _ = make(tmp_path, sidecar=False)
    refused(tmp_path, capsys, source, control(tmp_path), "provide it, or name the strip group")


def test_mesh_collapse_te_unequal_rims_are_refused_naming_the_blade(tmp_path, capsys):
    source, infos = make(tmp_path, strip="own")
    (tmp_path / "Blade1L.te.txt").write_text(
        (tmp_path / "Blade1L.te.txt").read_text().rsplit("\n", 2)[0] + "\n"
    )
    refused(tmp_path, capsys, source, control(tmp_path), "blade 'Blade1': the rims hold 5 and 4")


def test_mesh_collapse_te_a_strip_wider_than_one_cell_is_refused(tmp_path, capsys):
    source, _ = make(tmp_path, strip="own", wide=True)
    refused(tmp_path, capsys, source, control(tmp_path), "more than one cell wide")


def test_mesh_collapse_te_nothing_to_collapse_exits_2_and_writes_nothing(tmp_path, capsys):
    source, _ = make(tmp_path, [("Blade1", {"blunt": False})])
    with pytest.warns(NotApplicableWarning, match="sharp"):
        refused(tmp_path, capsys, source, control(tmp_path), "nothing to collapse")


def test_mesh_collapse_te_unknown_and_foreign_tables_warn_only_for_unknown_keys(tmp_path, recwarn):
    source, _ = make(tmp_path, strip="own")
    cfg = tmp_path / "morphing.toml"
    cfg.write_text('[collapse_te]\nblades = ["Blade1"]\nunused = 1\n[repitch]\nangle_deg = 1\n')
    assert invoke(source, cfg, tmp_path / "sharp.obj") == 0
    texts = " ".join(str(w.message) for w in recwarn)
    assert "unused" in texts and "repitch" not in texts


def test_mesh_collapse_te_absent_family_and_existing_output_are_refused(tmp_path, capsys):
    source, _ = make(tmp_path, strip="own")
    refused(tmp_path, capsys, source, control(tmp_path, '["Nope"]'), "absent")
    (tmp_path / "bad.obj").write_text("x")
    assert invoke(source, control(tmp_path), tmp_path / "bad.obj") == 2


def collapsed(folder, **kw):
    """Run the library steps and return the plans' result, for the check arms."""
    source, _ = make(folder, **kw)
    mesh = read_obj(source)
    index = cte._Index(mesh)
    plans, *_ = cte._select({"blades": ["Blade1"]}, source, mesh, index)
    return cte._build(mesh, plans), plans


def test_mesh_collapse_te_the_check_arms_pass_on_the_collapse(tmp_path):
    result, _ = collapsed(tmp_path, strip="own")
    checks = cte._verify(result)
    assert checks["smallest_normal_cosine"] > 0.5
    assert checks["area"]["relative_change_against_source_minus_strip"] == pytest.approx(
        0, abs=5e-3
    )


def test_mesh_collapse_te_a_far_node_moved_by_1e12_is_refused(tmp_path):
    result, _ = collapsed(tmp_path, strip="own")
    result.mesh.verts[2 * 7 + 0, 0] += 1e-12
    with pytest.raises(InputValueError, match="away from the trailing edge"):
        cte._verify(result)


def test_mesh_collapse_te_one_far_face_rotated_is_refused(tmp_path):
    result, _ = collapsed(tmp_path, strip="own")
    face = result.mesh.families["Hub1"][0]
    result.mesh.families["Hub1"][0] = face[1:] + face[:1]
    with pytest.raises(InputValueError, match="differs from the source in place"):
        cte._verify(result)


def test_mesh_collapse_te_a_rim_pair_left_unmerged_is_refused(tmp_path):
    result, plans = collapsed(tmp_path, strip="own")
    plan = plans[0]
    short = dataclasses.replace(plan, rim_a=plan.rim_a[:-1], rim_b=plan.rim_b[:-1])
    partial = cte._build(result.source, [short])
    partial.plans = plans
    with pytest.raises(InputValueError, match="nodes, not"):
        cte._verify(partial)


def test_mesh_collapse_te_a_face_turned_over_by_the_merge_is_warned_and_written(tmp_path):
    # (): this test asserted the refusal; a face the merge turns over is a
    # quality finding, warned with its count, worst cosine and place, and the mesh is written.
    from cfdbender._errors import GeometryWarning

    source, _ = make(tmp_path, strip="own", fold=True)
    out = tmp_path / "bad.obj"
    with pytest.warns(
        GeometryWarning, match=r"\d+ face\(s\) near the trailing edge .*turned over.*cosine"
    ):
        assert invoke(source, control(tmp_path), out) == 0
    assert out.exists()
    checks = json.loads(Path(f"{out}.collapse.json").read_text())["checks"]
    assert checks["passed"] is False
    assert checks["turned_faces"]["count"] >= 1 and checks["turned_faces"]["worst_cosine"] <= 0.0
    assert any("turned over" in w for w in checks["warnings"])


def test_mesh_collapse_te_a_marking_on_the_removed_strip_is_refused(tmp_path, capsys):
    source, infos = make(tmp_path, strip="own")
    old = read_obj(source)
    node = infos["Blade1"]["node"]
    rung = 0.5 * (old.verts[node(2, M)] + old.verts[node(2, M + 1)])
    pts(tmp_path / "extra.te.txt", [rung])
    side = Path(str(source) + ".boundaries.toml")
    side.write_text(
        side.read_text() + '[trailing_edges.Extra]\nfile = "extra.te.txt"\ntype = "STANDARD"\n'
    )
    refused(tmp_path, capsys, source, control(tmp_path), "lies on an edge of the removed strip")


def wheel(folder):
    """Two blades 180 degrees apart about the Z axis, each with its strip in its own group."""
    shift = lambda v: v + np.array([1.0, 0.0, 0.0])  # noqa: E731
    flip = lambda v: shift(v) * np.array([-1.0, -1.0, 1.0])  # noqa: E731
    source, infos = make(
        folder, [("Blade1", {"move": shift}), ("Blade2", {"move": flip})], strip="own"
    )
    (folder / "rotor.toml").write_text(
        '[ROTOR]\nkind = "rotor"\naxis = "Z"\ndiameter_m = 6.0\n'
        'families_blades = ["Blade1", "Blade2"]\nblade1 = { zero = "X", azimuth_deg = 0.0 }\n'
    )
    return source


def test_mesh_collapse_te_periodicity_is_required_with_a_reference_and_kept_by_all_blades(tmp_path):
    source = wheel(tmp_path)
    cfg = tmp_path / "morphing.toml"
    cfg.write_text(
        '[collapse_te]\nblades = "all"\n[repitch]\nangle_deg = 0\nreference = "rotor.toml"\n'
    )
    assert invoke(source, cfg, tmp_path / "all.obj") == 0
    report = json.loads(Path(str(tmp_path / "all.obj") + ".collapse.json").read_text())
    inv = report["checks"]["periodicity"]
    assert inv["nodes_mapped"] == inv["nodes"] and inv["faces_mapped"] == inv["faces"]
    cfg.write_text(
        '[collapse_te]\nblades = ["Blade1"]\n[repitch]\nangle_deg = 0\nreference = "rotor.toml"\n'
    )
    assert invoke(source, cfg, tmp_path / "one.obj") == 2


def test_mesh_collapse_te_without_a_reference_periodicity_is_warned_as_not_applying(tmp_path):
    source, _ = make(tmp_path, strip="own")
    with pytest.warns(NotApplicableWarning, match="periodicity does not apply"):
        assert invoke(source, control(tmp_path), tmp_path / "sharp.obj") == 0


def test_mesh_collapse_te_collapse_then_repitch_names_the_working_order(tmp_path, capsys):
    source, _ = make(tmp_path, strip="same", layers=3)
    collapsed_obj = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), collapsed_obj) == 0
    rep = tmp_path / "rep.toml"
    rep.write_text(
        '[repitch]\nangle_deg = 1.0\nblades = ["Blade1"]\nhub = ["Hub1"]\n'
        "axis = { origin = [0.5, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }\nmax_angle_deg = 5.0\n"
    )
    capsys.readouterr()
    assert invoke(collapsed_obj, rep, tmp_path / "again.obj", "repitch") == 2
    err = capsys.readouterr().err
    assert "repitch first" in err and "collapse-te" in err
    assert not (tmp_path / "again.obj").exists()
    warnings.simplefilter("default")


REPITCH = (
    '[repitch]\nangle_deg = 1.0\nblades = ["Blade1"]\nhub = ["Hub1"]\n'
    "axis = { origin = [0.5, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }\nmax_angle_deg = 5.0\n"
)


@pytest.mark.parametrize("verb", ["collapse-te", "repitch"])
def test_mesh_collapse_te_a_name_with_a_space_keeps_the_output_sidecar_readable(tmp_path, verb):
    source, _ = make(tmp_path, strip="same" if verb == "repitch" else "own", layers=3)
    source.write_text(source.read_text().replace("g Hub1", "g Hub 1"))
    side = Path(str(source) + ".boundaries.toml")
    side.write_text(
        side.read_text().replace('"Hub1"', '"Hub 1"').replace(".Blade1U]", '."Blade1 U"]')
        + '[base_regions."Hub 1"]\ntrailing_edges = "auto"\n'
    )
    cfg = tmp_path / "rep.toml"
    cfg.write_text(REPITCH.replace('"Hub1"', '"Hub 1"'))
    cfg = cfg if verb == "repitch" else control(tmp_path)
    out = tmp_path / "out.obj"
    assert invoke(source, cfg, out, verb) == 0
    written = Path(str(out) + ".boundaries.toml")
    data = tomllib.loads(written.read_text())
    assert "Blade1 U" in data["trailing_edges"] and "Hub 1" in data["base_regions"]
    assert any(r.family == "Hub 1" for r in read_boundary_declarations(written).base_regions)


def test_mesh_collapse_te_the_base_route_keeps_another_marking_of_the_blade(tmp_path):
    source, infos = make(tmp_path, strip="own")
    old = read_obj(source)
    node = infos["Blade1"]["node"]
    flap = 0.5 * (old.verts[node(1, 1)] + old.verts[node(2, 1)])
    pts(tmp_path / "flap.te.txt", [flap])
    side = Path(str(source) + ".boundaries.toml")
    side.write_text(
        side.read_text() + '[trailing_edges.Flap]\nfile = "flap.te.txt"\ntype = "RELAXED"\n'
    )
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path, extra='base = ["Blade1Base"]'), out) == 0
    tables = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())["trailing_edges"]
    assert list(tables) == ["Blade1U", "Flap"]
    assert tables["Flap"]["type"] == "RELAXED"
    np.testing.assert_allclose(read_te(tmp_path / tables["Flap"]["file"])[1], [flap], atol=1e-15)
    assert len(read_te(tmp_path / tables["Blade1U"]["file"])[1]) == 4


BARE_SIDECAR = (
    'fs_build = "26.125"\nboundaries = ["Blade1", "Blade1Base", "Hub1"]\n'
    '[import]\nunits = "METER"\n'
)


@pytest.mark.parametrize(
    ("region", "kind"),
    [
        ("", "STANDARD"),
        ('[base_regions.Blade1Base]\ntrailing_edges = "auto"\n', "STANDARD"),
        (
            "[base_regions.Blade1Base]\n"
            'trailing_edges = { file = "rim.te.txt", type = "VORTEX_SHEDDING" }\n',
            "VORTEX_SHEDDING",
        ),
    ],
    ids=["no-region", "auto-region", "file-region"],
)
def test_mesh_collapse_te_the_base_route_declares_the_line_in_the_boundaries_file(
    tmp_path, capsys, region, kind
):
    source, infos = make(tmp_path, strip="own", sidecar=False)
    pts(tmp_path / "rim.te.txt", infos["Blade1"]["upper"])
    Path(str(source) + ".boundaries.toml").write_text(BARE_SIDECAR + region)
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path, extra='base = ["Blade1Base"]'), out) == 0
    written = Path(str(out) + ".boundaries.toml")
    tables = tomllib.loads(written.read_text())["trailing_edges"]
    assert list(tables) == ["Blade1"] and tables["Blade1"]["type"] == kind
    assert len(read_te(tmp_path / tables["Blade1"]["file"])[1]) == 4
    assert not (tmp_path / "sharp.Blade1.te.txt").exists()


@pytest.mark.parametrize(
    "sidecar",
    [
        BARE_SIDECAR + "[trailing_edges]\nnone = true\n",
        BARE_SIDECAR + '[trailing_edges.Found]\ndetect = "auto"\ntype = "STANDARD"\n',
    ],
    ids=["none", "detect"],
)
def test_mesh_collapse_te_a_line_the_boundaries_file_cannot_take_is_named_in_a_warning(
    tmp_path, sidecar
):
    source, _ = make(tmp_path, strip="own", sidecar=False)
    Path(str(source) + ".boundaries.toml").write_text(sidecar)
    out = tmp_path / "sharp.obj"
    with pytest.warns(NotApplicableWarning, match="sharp.Blade1.te.txt, which that file"):
        assert invoke(source, control(tmp_path, extra='base = ["Blade1Base"]'), out) == 0
    assert len(read_te(tmp_path / "sharp.Blade1.te.txt")[1]) == 4
    assert "sharp.Blade1.te.txt" not in Path(str(out) + ".boundaries.toml").read_text()


def test_mesh_collapse_te_a_hub_with_its_own_copy_of_the_root_nodes_is_not_cracked(
    tmp_path, capsys
):
    verts, families, info = blade_parts("Blade1", 0, strip="own")
    count = info["count"]
    copies = len(verts)
    families["Hub1"] = [[n + copies if n < count else n for n in f] for f in families["Hub1"]]
    source = tmp_path / "sector.obj"
    source.write_text(obj_text(ObjMesh(np.vstack([verts, verts[:count]]), families)))
    cfg = control(tmp_path, extra='base = ["Blade1Base"]')
    with pytest.warns(NotApplicableWarning):
        refused(tmp_path, capsys, source, cfg, "opens a crack")


def test_mesh_collapse_te_coincident_nodes_that_merge_or_stay_are_no_crack(tmp_path):
    source, infos = make(tmp_path, strip="own", sidecar=False)
    mesh = read_obj(source)
    node = infos["Blade1"]["node"]
    verts = np.vstack([mesh.verts, mesh.verts[0]])
    verts[node(0, M + 1)] = verts[node(0, M)]
    source.write_text(obj_text(ObjMesh(verts, mesh.families)))
    with pytest.warns(NotApplicableWarning):
        cfg = control(tmp_path, extra='base = ["Blade1Base"]')
        assert invoke(source, cfg, tmp_path / "sharp.obj") == 0


def test_mesh_collapse_te_a_family_the_merge_empties_loses_its_base_region(tmp_path):
    verts, families, info = blade_parts("Blade1", 0, strip="own", cap="fan")
    node = info["node"]
    tip = [f for f in families["Blade1"] if {node(STATIONS, M), node(STATIONS, M + 1)} <= set(f)]
    assert len(tip) == 1
    families["Blade1"] = [f for f in families["Blade1"] if f not in tip]
    families = {"TipSliver": tip, **families}
    source = tmp_path / "sector.obj"
    source.write_text(obj_text(ObjMesh(verts, families)))
    pts(tmp_path / "U.te.txt", info["upper"])
    pts(tmp_path / "L.te.txt", info["lower"])
    Path(str(source) + ".boundaries.toml").write_text(
        f'fs_build = "26.125"\nboundaries = {json.dumps(list(families))}\n'
        '[import]\nunits = "METER"\n'
        '[trailing_edges.U]\nfile = "U.te.txt"\ntype = "STANDARD"\n'
        '[trailing_edges.L]\nfile = "L.te.txt"\ntype = "STANDARD"\n'
        '[base_regions.TipSliver]\ntrailing_edges = "auto"\n'
    )
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    written = Path(str(out) + ".boundaries.toml")
    assert "TipSliver" not in read_obj(out).families
    assert not read_boundary_declarations(written).base_regions
    assert json.loads(Path(str(out) + ".collapse.json").read_text())["checks"][
        "boundaries_file_reads_back"
    ]


def test_mesh_collapse_te_a_family_in_two_blocks_is_refused_for_the_face_order(tmp_path, capsys):
    source, _ = make(tmp_path, strip="own", wake=False)
    lines = source.read_text().splitlines()
    # One blade face moves between two hub faces: the same faces, in an order of two hub blocks.
    moved = lines.pop(lines.index("g Blade1Base") - 1)
    hub = lines.index("g Hub1")
    lines[hub + 2 : hub + 2] = ["g Blade1", moved, "g Hub1"]
    source.write_text("\n".join(lines) + "\n")
    refused(tmp_path, capsys, source, control(tmp_path), "returns after another family")


def test_mesh_collapse_te_a_repeated_group_line_with_no_face_between_is_one_block(tmp_path):
    source, _ = make(tmp_path, strip="own", wake=False)
    text = source.read_text().replace("g Hub1\n", "g Hub1\ng Hub1\n")
    source.write_text(text)
    assert invoke(source, control(tmp_path), tmp_path / "sharp.obj") == 0


def split_blade(folder):
    """One blunt blade whose upper and lower skins are two families around its strip group."""
    verts, families, info = blade_parts("Blade1", 0, strip="own")
    count = info["count"]
    upper, lower = [], []
    for face in families["Blade1"]:
        ring = [n % count for n in face]
        on_upper = len(face) == 4 and max(ring) <= M and not (0 in ring and M + 1 in ring)
        (upper if on_upper else lower).append(face)
    groups = {"BUp": upper, "BLo": lower, "Base": families["Blade1Base"], "Hub1": families["Hub1"]}
    source = folder / "sector.obj"
    source.write_text(obj_text(ObjMesh(verts, groups)))
    pts(folder / "U.te.txt", info["upper"])
    pts(folder / "L.te.txt", info["lower"])
    Path(str(source) + ".boundaries.toml").write_text(
        f'fs_build = "26.125"\nboundaries = {json.dumps(list(groups))}\n'
        '[import]\nunits = "METER"\n'
        '[trailing_edges.U]\nfile = "U.te.txt"\ntype = "STANDARD"\n'
        '[trailing_edges.L]\nfile = "L.te.txt"\ntype = "STANDARD"\n'
        '[base_regions.Base]\ntrailing_edges = "auto"\n'
    )
    return source


@pytest.mark.parametrize("blades", ['"all"', '["BUp", "BLo"]'])
def test_mesh_collapse_te_a_blade_split_across_families_is_not_called_sharp_in_silence(
    tmp_path, capsys, blades
):
    source = split_blade(tmp_path)
    with pytest.warns(NotApplicableWarning, match="named by the group holding its strip"):
        refused(tmp_path, capsys, source, control(tmp_path, blades), "taken as already sharp")


def test_mesh_collapse_te_a_blade_split_across_families_collapses_by_its_strip_group(tmp_path):
    source = split_blade(tmp_path)
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path, '["Base"]'), out) == 0
    assert len(read_obj(out).verts) == len(read_obj(source).verts) - 5
    assert "Base" not in read_obj(out).families


def test_mesh_collapse_te_all_names_the_blade_not_a_strip_group_written_first(tmp_path):
    verts, families, info = blade_parts("Blade1", 0, strip="own")
    groups = {"Blade1Base": families["Blade1Base"], **families}
    source = tmp_path / "sector.obj"
    source.write_text(obj_text(ObjMesh(verts, groups)))
    pts(tmp_path / "U.te.txt", info["upper"])
    pts(tmp_path / "L.te.txt", info["lower"])
    Path(str(source) + ".boundaries.toml").write_text(
        f'fs_build = "26.125"\nboundaries = {json.dumps(list(groups))}\n'
        '[import]\nunits = "METER"\n'
        '[trailing_edges.U]\nfile = "U.te.txt"\ntype = "STANDARD"\n'
        '[trailing_edges.L]\nfile = "L.te.txt"\ntype = "STANDARD"\n'
    )
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path, '"all"'), out) == 0
    assert list(json.loads(Path(str(out) + ".collapse.json").read_text())["blades"]) == ["Blade1"]


def test_mesh_collapse_te_a_rotor_reference_with_no_axis_is_refused_in_words(tmp_path, capsys):
    source, _ = make(tmp_path, strip="own")
    (tmp_path / "rotor.toml").write_text('[R]\nkind = "rotor"\nfamilies_blades = ["A", "B"]\n')
    cfg = tmp_path / "morphing.toml"
    cfg.write_text('[collapse_te]\nblades = ["Blade1"]\n[repitch]\nreference = "rotor.toml"\n')
    refused(tmp_path, capsys, source, cfg, "states no 'axis'")


def test_mesh_collapse_te_a_detection_entry_loses_only_the_family_the_collapse_removes(
    tmp_path, capsys
):
    source, _ = make(tmp_path, strip="own", wake=False)
    side = Path(str(source) + ".boundaries.toml")
    text = side.read_text()
    side.write_text(
        text + '[trailing_edges.Det]\ntype = "STANDARD"\n'
        'detect = { surfaces = ["Blade1Base", "Hub1"] }\n'
    )
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    tables = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())["trailing_edges"]
    assert tables["Det"]["detect"]["surfaces"] == ["Hub1"]
    side.write_text(
        text + '[trailing_edges.Det]\ntype = "STANDARD"\ndetect = { surfaces = ["Blade1Base"] }\n'
    )
    refused(tmp_path, capsys, source, control(tmp_path), "detects only on families")


def test_mesh_collapse_te_a_rotor_table_refusal_names_collapse_te(tmp_path, capsys):
    source, _ = make(tmp_path, strip="own")
    (tmp_path / "rotor.toml").write_text('[A]\nkind = "rotor"\naxis = "Z"\n')
    cfg = tmp_path / "morphing.toml"
    cfg.write_text(
        '[collapse_te]\nblades = ["Blade1"]\n[repitch]\nreference = "rotor.toml"\nrotor = "B"\n'
    )
    refused(tmp_path, capsys, source, cfg, "collapse_te: reference has no rotor 'B'")


def test_mesh_collapse_te_an_unstated_unit_warns_only_of_a_file_it_labels(tmp_path, recwarn):
    source, _ = make(tmp_path, strip="own", wake=False)
    side = Path(str(source) + ".boundaries.toml")
    side.write_text(side.read_text().replace('[import]\nunits = "METER"\n', ""))
    rows = read_te(tmp_path / "Blade1U.te.txt")[1]
    (tmp_path / "Blade1U.te.txt").write_text(
        "MILLIMETER\n" + "\n".join(",".join(format(float(v), ".17g") for v in r) for r in rows)
    )
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    assert "unit is unstated" not in " ".join(str(w.message) for w in recwarn)
    entry = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())["trailing_edges"]
    assert read_te(tmp_path / entry["Blade1U"]["file"])[0] == "MILLIMETER"


def detect_blade(source, blade):
    """Replace the two rim entries of ``blade`` with one detection entry."""
    side = Path(str(source) + ".boundaries.toml")
    data = side.read_text().split("[")
    kept = [
        t
        for t in data
        if not t.startswith((f"trailing_edges.{blade}U", f"trailing_edges.{blade}L"))
    ]
    side.write_text(
        "[".join(kept) + f'[trailing_edges.{blade}]\ndetect = "auto"\ntype = "STANDARD"\n'
    )


def test_mesh_collapse_te_all_says_it_skipped_a_blade_marked_by_detection(tmp_path):
    source, _ = make(
        tmp_path, [("Blade1", {}), ("Blade2", {"move": AWAY})], strip="own", wake=False
    )
    detect_blade(source, "Blade2")
    out = tmp_path / "sharp.obj"
    with pytest.warns(NotApplicableWarning, match="'Blade2' is marked by detection"):
        assert invoke(source, control(tmp_path, '"all"'), out) == 0
    report = json.loads(Path(str(out) + ".collapse.json").read_text())
    assert set(report["blades"]) == {"Blade1"}
    assert "detection" in report["skipped"]["Blade2"]


def test_mesh_collapse_te_all_with_no_points_entry_says_so(tmp_path, capsys):
    source, _ = make(tmp_path, strip="own", wake=False)
    detect_blade(source, "Blade1")
    with pytest.warns(NotApplicableWarning, match="marked by detection"):
        refused(tmp_path, capsys, source, control(tmp_path, '"all"'), "no trailing-edge entry")


def far_hub_face_edit(source, edit):
    """Rewrite the source through ``edit`` of its verts and hub faces, away from the edge."""
    mesh = read_obj(source)
    hub = mesh.families["Hub1"]
    verts = mesh.verts.copy()
    edit(verts, hub)
    source.write_text(obj_text(ObjMesh(verts, mesh.families)))


def test_mesh_collapse_te_a_face_away_from_the_edge_is_written_as_the_source_holds_it(tmp_path):
    source, _ = make(tmp_path, strip="own", sidecar=False)

    def repeat_a_node(verts, hub):
        hub[-1][3] = hub[-1][2]

    far_hub_face_edit(source, repeat_a_node)
    out = tmp_path / "sharp.obj"
    with pytest.warns(NotApplicableWarning):
        assert invoke(source, control(tmp_path, extra='base = ["Blade1Base"]'), out) == 0
    old, new = read_obj(source), read_obj(out)
    place = by_place(new)
    assert new.families["Hub1"][-1] == [place[key(old.verts[n])] for n in old.families["Hub1"][-1]]
    assert len(new.families["Hub1"][-1]) == 4


@pytest.mark.parametrize("collinear", [False, True], ids=["coincident", "zero-area"])
def test_mesh_collapse_te_a_degenerate_face_away_from_the_edge_is_no_refusal(tmp_path, collinear):
    source, _ = make(tmp_path, strip="own", sidecar=False)

    def flatten(verts, hub):
        a, b, c, d = hub[0]
        verts[d] = verts[a] if collinear else verts[c]
        if collinear:
            verts[c] = verts[b]

    far_hub_face_edit(source, flatten)
    with pytest.warns(NotApplicableWarning):
        cfg = control(tmp_path, extra='base = ["Blade1Base"]')
        assert invoke(source, cfg, tmp_path / "sharp.obj") == 0


def test_mesh_collapse_te_a_near_face_with_a_repeated_node_or_no_area_is_refused(tmp_path):
    result, plans = collapsed(tmp_path, strip="own")
    node = plans[0].rim_a[0]
    face = next(f for f in result.source.families["Hub1"] if node in f)
    at = result.source.families["Hub1"].index(face)
    written = result.mesh.families["Hub1"][at]
    keep = list(written)
    written[1] = written[0]
    with pytest.raises(InputValueError, match="repeats a node"):
        cte._verify(result)
    written[:] = keep
    # A third corner on the first: the nodes stay distinct and the area is exactly zero.
    result.mesh.verts[written[2]] = result.mesh.verts[written[0]]
    with pytest.raises(InputValueError, match="zero area"):
        cte._verify(result)


@pytest.mark.parametrize(
    "text",
    [
        '"broken\n',
        'fs_build = "26.125"\nboundaries = ["A"]\n[base_regions.Gone]\nmerge_into = "A"\n',
    ],
    ids=["toml", "stale-family"],
)
def test_mesh_collapse_te_a_boundaries_file_that_would_not_read_back_is_refused(tmp_path, text):
    target = tmp_path / "sharp.obj"
    written = Path(str(target) + ".boundaries.toml")
    with pytest.raises(InputValueError, match="would not read back"):
        cte._sidecar_reads_back({written: text.encode()}, target)


def test_mesh_collapse_te_an_import_operation_on_a_family_the_collapse_removes_is_skipped(
    tmp_path, capsys
):
    source, _ = make(tmp_path, strip="own", wake=False)
    side = Path(str(source) + ".boundaries.toml")
    operations = (
        '[[import.operations]]\nop = "rename"\nsurface = "Blade1Base"\nto = "Base"\n'
        '[[import.operations]]\nop = "scale"\nsurface = "Base"\nfactors = [1.0, 1.0, 1.0]\n'
        '[[import.operations]]\nop = "scale"\nsurface = "Hub1"\nfactors = [2.0, 2.0, 2.0]\n'
    )
    side.write_text(side.read_text().replace('units = "METER"\n', 'units = "METER"\n' + operations))
    out = tmp_path / "sharp.obj"
    with pytest.warns(NotApplicableWarning, match="'Blade1Base', which the collapse removes"):
        assert invoke(source, control(tmp_path), out) == 0
    kept = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())["import"]["operations"]
    assert kept == [{"op": "scale", "surface": "Hub1", "factors": [2.0, 2.0, 2.0]}]


@pytest.mark.parametrize("route", ["points", "base"])
@pytest.mark.parametrize("swap", [False, True], ids=["upper-first", "lower-first"])
def test_mesh_collapse_te_the_first_declared_rim_keeps_its_place_on_either_route(
    tmp_path, route, swap
):
    source, infos = make(tmp_path, strip="own", wake=False)
    node = infos["Blade1"]["node"]
    mesh = read_obj(source)
    # The lower rim's nodes move to the end of the node list, so the two rims are told apart.
    lower = [node(s, M + 1) for s in range(STATIONS + 1)]
    order = [i for i in range(len(mesh.verts)) if i not in lower] + lower
    at = {old: new for new, old in enumerate(order)}
    families = {n: [[at[v] for v in f] for f in fs] for n, fs in mesh.families.items()}
    source.write_text(obj_text(ObjMesh(mesh.verts[order], families)))
    side = Path(str(source) + ".boundaries.toml")
    if swap:
        text = side.read_text()
        text = text.replace("Blade1U.te.txt", "@").replace("Blade1L.te.txt", "Blade1U.te.txt")
        side.write_text(text.replace("@", "Blade1L.te.txt"))
    out = tmp_path / "sharp.obj"
    extra = 'base = ["Blade1Base"]' if route == "base" else ""
    assert invoke(source, control(tmp_path, extra=extra), out) == 0
    new = read_obj(out).verts
    up, low = at[node(0, M)], at[node(0, M + 1)]
    midpoint = 0.5 * (mesh.verts[node(0, M)] + mesh.verts[node(0, M + 1)])
    # Declared first, the lower rim keeps its places, less the upper nodes removed before them.
    place = low - len(lower) if swap else up
    assert np.array_equal(new[place], midpoint)


@pytest.mark.parametrize("strip", ["same", "own"])
def test_mesh_collapse_te_004_the_points_route_keeps_another_marking_of_the_blade(tmp_path, strip):
    """CTE2-004: a flap line on the blade beside the declared rims is carried, not a third rim."""
    source, infos = make(tmp_path, strip=strip)
    old = read_obj(source)
    node = infos["Blade1"]["node"]
    flap = [0.5 * (old.verts[node(s, 1)] + old.verts[node(s + 1, 1)]) for s in range(2)]
    pts(tmp_path / "flap.te.txt", flap)
    side = Path(str(source) + ".boundaries.toml")
    side.write_text(
        side.read_text() + '[trailing_edges.Flap]\nfile = "flap.te.txt"\ntype = "RELAXED"\n'
    )
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    tables = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())["trailing_edges"]
    assert list(tables) == ["Blade1U", "Flap"]
    assert tables["Flap"]["type"] == "RELAXED"
    np.testing.assert_allclose(read_te(tmp_path / tables["Flap"]["file"])[1], flap, atol=1e-15)
    assert len(read_te(tmp_path / tables["Blade1U"]["file"])[1]) == STATIONS
    report = json.loads(Path(str(out) + ".collapse.json").read_text())
    assert report["blades"]["Blade1"]["route"] == "boundaries"
    assert report["blades"]["Blade1"]["nodes_merged"] == STATIONS + 1


def test_mesh_collapse_te_the_verb_prints_its_record_and_writes_a_provenance_sidecar(
    tmp_path, capsys
):
    source, _ = make(tmp_path, strip="own", wake=False)
    out = tmp_path / "sharp.obj"
    capsys.readouterr()
    assert invoke(source, control(tmp_path), out) == 0
    record = json.loads(capsys.readouterr().out)
    assert Path(record["provenance"]).name == "sharp.obj.provenance.json"
    provenance = json.loads(Path(record["provenance"]).read_text(encoding="utf-8"))
    assert provenance["command"] == "cfd-mesh collapse-te"
    assert out.is_file()

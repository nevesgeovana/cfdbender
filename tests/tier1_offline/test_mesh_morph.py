"""Hub rings and the pitch axis rule on the OBJ, with the fixture the morph tests share.

Usage:
    cfd-mesh hub-rings sector.obj --morphing morphing.toml --out rings.obj
    cfd-mesh morph sector.obj --morphing morphing.toml --out morphed.obj

The fixture is a quad-dominant cylinder hub along x (some cells split into two triangles), with
radial blades whose root is a one-cell-thick slot of the hub grid and whose trailing edge is a
one-cell base strip declared by two rim entries; as one sector of three or as the full wheel,
optionally installed beside a nacelle that shares the hub's aft ring and a wing that touches
nothing.
"""

from __future__ import annotations

import json
import shutil
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest

from cfdbender._errors import GeometryWarning, InputValueError, NotApplicableWarning
from cfdbender.mesh._refine._obj import ObjMesh, obj_text, read_obj
from cfdbender.mesh._repitch_grid import incidence, rings, root_loop
from cfdbender.mesh._wheel import Faces, rotate, wheel_symmetry
from cfdbender.mesh.cli import main

R, X0, NX, DX = 1.0, -1.0, 80, 0.05
COPIES, PER = 3, 24
DT = 2 * np.pi / (COPIES * PER)
FOOT_I, FOOT_J = (36, 44), 12
STATIONS = np.linspace(1.0, 3.0, 9)
STATION = 1.0123  # a plane along x through every root, between the hub's grid lines
CENTRE = (FOOT_J + 0.5) * DT  # the azimuth of the first blade


class _Nodes:
    def __init__(self, closed: bool, nt: int) -> None:
        self.closed, self.nt, self.key, self.verts = closed, nt, {}, []

    def at(self, tag, i, j, x):
        j = j % self.nt if self.closed else j
        if (tag, i, j) not in self.key:
            self.key[(tag, i, j)] = len(self.verts)
            self.verts.append([x, R * np.cos(j * DT), R * np.sin(j * DT)])
        return self.key[(tag, i, j)]

    def hub(self, i, j):
        return self.at("h", i, j, X0 + i * DX)

    def add(self, point):
        self.verts.append(list(point))
        return len(self.verts) - 1


def _blade(nodes, j0):
    loop = [(i, j0) for i in range(FOOT_I[0], FOOT_I[1] + 1)]
    loop += [(i, j0 + 1) for i in range(FOOT_I[1], FOOT_I[0] - 1, -1)]
    root = [nodes.hub(i, j) for i, j in loop]
    radial = np.array([0.0, np.cos((j0 + 0.5) * DT), np.sin((j0 + 0.5) * DT)])
    rows = [root] + [
        [nodes.add(np.asarray(nodes.verts[v]) + (r - 1.0) * radial) for v in root]
        for r in STATIONS[1:]
    ]
    n = len(root)
    faces = [
        [rows[s][q], rows[s + 1][q], rows[s + 1][(q + 1) % n], rows[s][(q + 1) % n]]
        for s in range(len(rows) - 1)
        for q in range(n)
    ]
    faces.append(rows[-1][::-1])
    te = loop.index((FOOT_I[1], j0))
    return faces, ([row[te] for row in rows], [row[te + 1] for row in rows])


def build(folder, *, wheel=False, nacelle=False, wing=False, name="hub.obj", columns=None):
    """Write the mesh and its boundaries file, two rim entries per blade; return the path.

    ``columns`` places the blades' roots at those hub columns instead of one per sector.
    """
    sectors = COPIES if wheel else 1
    nt = PER * sectors
    nodes = _Nodes(wheel, nt)
    families, rims = {}, {}
    columns = columns or [k * PER + FOOT_J for k in range(sectors)]
    for k, j0 in enumerate(columns):
        families[f"Blade{k + 1}"], rims[f"Blade{k + 1}"] = _blade(nodes, j0)
    feet = {(i, j0) for j0 in columns for i in range(*FOOT_I)}
    hub = []
    for i in range(NX):
        for j in range(nt):
            if (i, j) in feet:
                continue
            a, b = nodes.hub(i, j), nodes.hub(i + 1, j)
            c, d = nodes.hub(i + 1, j + 1), nodes.hub(i, j + 1)
            hub += [[a, d, c], [a, c, b]] if (i + (j % PER)) % 5 == 0 else [[a, d, c, b]]
    families["Spinner"] = hub
    if nacelle:
        far = lambda j: nodes.at("n", 0, j, X0 + NX * DX + 0.5)  # noqa: E731
        families["Nacelle"] = [
            [nodes.hub(NX, j), nodes.hub(NX, j + 1), far(j + 1), far(j)] for j in range(nt)
        ]
    if wing:
        base = len(nodes.verts)
        nodes.verts += [[0.0, 6.0 + a, b] for a in range(3) for b in range(3)]
        families["Wing"] = [
            [base + 3 * a + b, base + 3 * a + b + 1, base + 3 * a + b + 4, base + 3 * a + b + 3]
            for a in range(2)
            for b in range(2)
        ]
    verts = np.array(nodes.verts, dtype=float)
    path = Path(folder) / name
    path.write_text(obj_text(ObjMesh(verts, families)))
    lines = ['fs_build = "26.125"', f"boundaries = {json.dumps(list(families))}"]
    lines += ["[import]", 'units = "METER"']
    for blade, pair in rims.items():
        for tag, rim in zip("UL", pair, strict=True):
            mids = 0.5 * (verts[rim[:-1]] + verts[rim[1:]])
            rows = "\n".join(",".join(format(float(v), ".17g") for v in p) for p in mids)
            (Path(folder) / f"{blade}{tag}.te.txt").write_text(f"METER\n{rows}\n")
            lines += [f"[trailing_edges.{blade}{tag}]", f'file = "{blade}{tag}.te.txt"']
            lines.append('type = "STANDARD"')
    Path(str(path) + ".boundaries.toml").write_text("\n".join(lines) + "\n")
    return path


def blades_of(wheel):
    return json.dumps([f"Blade{k + 1}" for k in range(COPIES if wheel else 1)])


def control(folder, text, *, wheel=False, name="morphing.toml"):
    path = Path(folder) / name
    path.write_text(f'blades = {blades_of(wheel)}\nhub = ["Spinner"]\n{text}')
    return path


def invoke(verb, source, config, out):
    try:
        return main([verb, str(source), "--morphing", str(config), "--out", str(out)])
    except SystemExit as error:
        return error.code


def record(out, kind):
    return json.loads(Path(f"{out}.{kind}.json").read_text())


def places(mesh, faces):
    return [[tuple(mesh.verts[v].tolist()) for v in f] for f in faces]


def open_nodes(mesh):
    owners = incidence([f for fs in mesh.families.values() for f in fs])
    return {tuple(mesh.verts[v].tolist()) for e, own in owners.items() if len(own) == 1 for v in e}


@pytest.fixture(scope="module")
def ringed(tmp_path_factory):
    """The sector and the full wheel through hub-rings once, for the tests that read them."""
    made = {}
    for wheel in (False, True):
        folder = tmp_path_factory.mktemp("wheel" if wheel else "sector")
        source = build(folder, wheel=wheel, nacelle=True, wing=wheel)
        cfg = control(folder, "[hub_rings]\nrings = 4\n", wheel=wheel)
        out = folder / "rings.obj"
        assert invoke("hub-rings", source, cfg, out) == 0
        made[wheel] = (folder, source, out)
    return made


@pytest.mark.parametrize("wheel", [False, True], ids=["sector", "wheel"])
def test_mesh_morph_hub_rings_are_complete_and_everything_else_stays(ringed, wheel):
    folder, source, out = ringed[wheel]
    old, new = read_obj(source), read_obj(out)
    report = record(out, "hub_rings")
    assert report["checks"]["ring_counts"] == [4] * (COPIES if wheel else 1)
    for name in [n for n in old.families if n.startswith("Blade")]:
        # blade nodes to the bit, faces in order; the root loop gives exactly four rings
        assert places(new, new.families[name]) == places(old, old.families[name])
        loop = root_loop(new.families[name], new.families["Spinner"], name)
        assert len(rings(loop, new.families["Spinner"])) - 1 == 4
    for name in ("Nacelle", "Wing") if wheel else ("Nacelle",):
        assert places(new, new.families[name]) == places(old, old.families[name])
    assert open_nodes(new) == open_nodes(old)  # seams, the hub's ends and the nacelle's end
    added = sum(b["faces_added"] for b in report["blades"].values())
    taken = sum(b["faces_taken"] for b in report["blades"].values())
    hub = new.families["Spinner"]
    assert len(hub) == len(old.families["Spinner"]) - taken + added
    quads = sum(len(f) == 4 for f in hub)
    assert quads / len(hub) > 0.5  # quad-dominant
    assert (
        report["checks"]["largest_distance_to_source_hub"]
        <= (report["checks"]["source_chord_deviation"])
    )
    sidecar = tomllib.loads(Path(f"{out}.boundaries.toml").read_text())
    assert sidecar["boundary_faces"] == [len(f) for f in new.families.values()]


def test_mesh_morph_kept_hub_faces_keep_their_order_and_new_faces_take_the_first_place(ringed):
    _, source, out = ringed[False]
    old, new = read_obj(source), read_obj(out)
    report = record(out, "hub_rings")["blades"]["Blade1"]
    old_hub, new_hub = places(old, old.families["Spinner"]), places(new, new.families["Spinner"])
    kept = [f for f in old_hub if f in new_hub]
    assert len(kept) == len(old_hub) - report["faces_taken"]
    first = next(i for i, f in enumerate(old_hub) if f not in new_hub)
    assert new_hub[:first] == old_hub[:first]
    assert new_hub[first + report["faces_added"] :] == kept[first:]


def test_mesh_morph_rings_that_do_not_fit_are_refused_naming_a_width_that_fits(tmp_path, capsys):
    source = build(tmp_path)
    cfg = control(tmp_path, "[hub_rings]\nrings = 4\nband_width = 1.5\n")
    out = tmp_path / "wide.obj"
    assert invoke("hub-rings", source, cfg, out) == 2
    err = capsys.readouterr().err
    assert "do not fit" in err and "a width of" in err
    assert not out.exists() and not Path(f"{out}.hub_rings.json").exists()
    width = float(err.split("a width of ")[1].split(" fits")[0])
    cfg.write_text(cfg.read_text().replace("band_width = 1.5", f"band_width = {width}"))
    assert invoke("hub-rings", source, cfg, tmp_path / "fits.obj") == 0


def test_mor1_5_two_holes_that_touch_keep_the_nodes_they_share(tmp_path):
    """Two roots five cells apart, three narrow rings each: the holes meet along a line of hub
    nodes whose every face one hole or the other takes. Those nodes were removed while both
    annuli still named them, so their faces took the last node and were refused as folds."""
    source = build(tmp_path, columns=[6, 11])
    cfg = control(tmp_path, "[hub_rings]\nrings = 3\nband_width = 0.04\n")
    cfg.write_text(cfg.read_text().replace('["Blade1"]', '["Blade1", "Blade2"]'))
    out = tmp_path / "touch.obj"
    assert invoke("hub-rings", source, cfg, out) == 0
    mesh = read_obj(out)
    assert {v for fs in mesh.families.values() for f in fs for v in f} == set(
        range(len(mesh.verts))
    )
    assert record(out, "hub_rings")["checks"]["ring_counts"] == [3, 3]
    assert open_nodes(mesh) == open_nodes(read_obj(source))


def test_mor1_3_the_width_a_refusal_names_fits_every_blade(tmp_path, capsys):
    """Two roots three cells apart: the refusal of the first named a width judged on it alone,
    and at that width the second blade's rings found no room."""
    source = build(tmp_path, columns=[8, 12])
    cfg = control(tmp_path, "[hub_rings]\nrings = 4\n")
    cfg.write_text(cfg.read_text().replace('["Blade1"]', '["Blade1", "Blade2"]'))
    assert invoke("hub-rings", source, cfg, tmp_path / "wide.obj") == 2
    err = capsys.readouterr().err
    assert "do not fit" in err
    if "a width of" in err:
        width = float(err.split("a width of ")[1].split(" fits")[0])
        cfg.write_text(cfg.read_text() + f"band_width = {width}\n")
        assert invoke("hub-rings", source, cfg, tmp_path / "fits.obj") == 0, capsys.readouterr()


@pytest.mark.parametrize(
    ("table", "text"),
    [
        ("[hub_rings]\nrings = 0\n", "whole number of rings"),
        ("[hub_rings]\nrings = 4\nband_width = -1\n", "positive length"),
        ('[hub_rings]\nrings = 4\nblades = ["Absent"]\n', "absent families"),
    ],
)
def test_mesh_morph_hub_rings_refuses_a_table_it_cannot_read(tmp_path, capsys, table, text):
    source = build(tmp_path)
    out = tmp_path / "bad.obj"
    assert invoke("hub-rings", source, control(tmp_path, table), out) == 2
    assert text in capsys.readouterr().err and not out.exists()


def _hr():
    from cfdbender.mesh import hub_rings

    return hub_rings


def _built(folder):
    hr = _hr()
    source = build(folder)
    mesh = read_obj(source)
    cfg = hr.read_config(
        tomllib.loads(control(folder, "[hub_rings]\nrings = 4\n").read_text()), mesh
    )
    return hr.build_rings(mesh, cfg)


def test_mesh_morph_the_check_arms_of_hub_rings_pass_unmutated(tmp_path):
    result, surface = _built(tmp_path)
    checks = _hr().verify_rings(result, surface, ["Spinner"])
    assert checks["ring_counts"] == [4] and checks["new_cell_quality"]["folded_cells"] == 0


def _mutants():
    def kept_node(r):
        node = r.mesh.families["Spinner"][-1][0]
        r.mesh.verts[node, 0] += 1e-12

    def kept_face(r):
        face = r.mesh.families["Spinner"][-1]
        r.mesh.families["Spinner"][-1] = face[1:] + face[:1]

    def lifted(r):
        node = r.added["Blade1"][-1][0]
        if node in {v for f in r.added["Blade1"][:72] for v in f}:
            node = r.added["Blade1"][-1][1]
        r.mesh.verts[node] *= 1.01

    def moved_block(r):
        hub = r.mesh.families["Spinner"]
        block = [f for f in hub if f in r.added["Blade1"]]
        r.mesh.families["Spinner"] = [f for f in hub if f not in r.added["Blade1"]] + block

    def ring_lost(r):
        quad = r.added["Blade1"][0]
        r.mesh.families["Spinner"].remove(quad)

    def turned(r):
        hub = r.mesh.families["Spinner"]
        i = hub.index(r.added["Blade1"][-1])
        hub[i] = hub[i][::-1]
        r.added["Blade1"][-1] = hub[i]

    return [
        (kept_node, "a kept node moved"),
        (kept_face, "differs from the source in place"),
        (lifted, "WARNS: chord deviation"),
        (moved_block, "not where its first taken face was"),
        (ring_lost, "complete rings, not 4"),
        (turned, "turned against the source hub|runs an edge the way"),
    ]


@pytest.mark.parametrize(("mutate", "text"), _mutants(), ids=lambda m: getattr(m, "__name__", ""))
def test_mesh_morph_a_hub_rings_check_arm_refuses_its_mutant(tmp_path, mutate, text):
    # HUBQ (): a new node off the hub (lifted) is quality, warned with the worst
    # node and its distance and listed in the checks; the other arms are broken invariants.
    result, surface = _built(tmp_path)
    mutate(result)
    if text.startswith("WARNS: "):
        with pytest.warns(GeometryWarning, match=text[7:]):
            checks = _hr().verify_rings(result, surface, ["Spinner"])
        assert any(text[7:] in w for w in checks["warnings"])
        return
    with pytest.raises(InputValueError, match=text):
        _hr().verify_rings(result, surface, ["Spinner"])


class _TurnedHub:
    """A source hub's surface mesh with its face normals turned over; all else delegated."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def face_normals(self):
        return -np.asarray(self._real.face_normals)


def test_mesh_morph_hub_rings_warns_new_cells_turned_against_the_hub(tmp_path, monkeypatch):
    """a new cell turned against the source hub is warned and listed, and
    the mesh is written; the check that measures it is quality, not a refusal."""
    hr = _hr()
    built = hr.build_rings

    def turned(source, cfg):
        result, surface = built(source, cfg)
        surface.mesh = _TurnedHub(surface.mesh)
        return result, surface

    monkeypatch.setattr(hr, "build_rings", turned)
    source = build(tmp_path)
    out = tmp_path / "rings.obj"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GeometryWarning)
        assert invoke("hub-rings", source, control(tmp_path, "[hub_rings]\nrings = 4\n"), out) == 0
    assert out.is_file()
    checks = record(out, "hub_rings")["checks"]
    assert checks["turned_cells"] >= 1 and checks["passed"] is False
    assert any("turned against the source hub" in w for w in checks["warnings"])
    assert any("turned against the source hub" in str(w.message) for w in caught)


def test_mor1_4_an_annulus_that_keeps_fewer_remesh_passes_is_warned(tmp_path, monkeypatch):
    """The fallback to fewer passes reached the record's pass count only; on the real wheel one
    annulus kept 2 passes, 18 percent quads and a scaled Jacobian of 0.0043, exit 0, silent."""
    hr = _hr()
    calls = []

    def turned_once(*args):
        calls.append(1)
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(hr, "_turned_cells", turned_once)
    with pytest.warns(GeometryWarning, match="Blade1.*kept 4.*quads.*scaled Jacobian"):
        result, _ = _built(tmp_path)
    assert result.annulus["Blade1"]["passes"] == 4


REPITCH = '[repitch]\nangle_deg = {angle}\nrotor_axis = "x"\nreference_point = {point}\n'


@pytest.mark.parametrize("wheel", [False, True], ids=["sector", "wheel"])
def test_mesh_morph_a_scalar_reference_point_turns_each_blade_about_its_radial_line(ringed, wheel):
    folder, _, rings_obj = ringed[wheel]
    cfg = control(folder, REPITCH.format(angle=2.0, point=STATION), wheel=wheel, name="r.toml")
    out = folder / f"pitched_{wheel}.obj"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        assert invoke("repitch", rings_obj, cfg, out) == 0
    report = record(out, "repitch")
    found = report["pitch_axes"]
    centre = np.array(found["hub_centre"]["point"])
    assert found["hub_centre"]["cut"] == ("closed loop" if wheel else "arc")
    assert centre[0] == pytest.approx(STATION, abs=1e-12)
    assert np.linalg.norm(centre[1:]) < (1e-4 if wheel else 2e-3)
    assert found["hub_centre"]["fit_residual"] < 1e-2
    old, new = read_obj(rings_obj), read_obj(out)
    for k, name in enumerate(json.loads(blades_of(wheel))):
        radial = np.array(
            [0.0, np.cos(CENTRE + k * 2 * np.pi / 3), np.sin(CENTRE + k * 2 * np.pi / 3)]
        )
        direction = np.array(found["blades"][name]["pitch_axis_direction"])
        assert direction == pytest.approx(radial, abs=1e-9)
        assert found["blades"][name]["pitch_axis_miss_of_root"] == 0.0
        ids = old.family_vertices(name)
        turned = rotate(old.verts[ids], direction, centre, 2.0)
        assert np.abs(new.verts[ids] - turned).max() < 1e-12
        assert report["blades"][name]["pitch_min_deg"] == pytest.approx(2.0, abs=1e-9)


def test_mesh_morph_one_blade_of_a_full_wheel_still_finds_the_closed_cut(ringed):
    """The cut is joined across every blade root, named or not, so it closes on a full wheel."""
    folder, _, rings_obj = ringed[True]
    cfg = control(folder, REPITCH.format(angle=1.0, point=STATION), wheel=True, name="one.toml")
    cfg.write_text(cfg.read_text() + 'blades = ["Blade2"]\n')
    out = folder / "one.obj"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        assert invoke("repitch", rings_obj, cfg, out) == 0
    found = record(out, "repitch")["pitch_axes"]
    assert found["hub_centre"]["cut"] == "closed loop" and list(found["blades"]) == ["Blade2"]
    assert np.linalg.norm(np.array(found["hub_centre"]["point"])[1:]) < 1e-4


def test_mesh_morph_a_point_reference_is_one_point_and_a_miss_of_the_root_is_warned(ringed):
    folder, _, rings_obj = ringed[False]
    cfg = control(folder, REPITCH.format(angle=1.0, point="[1.0, 0.0, 0.0]"), name="p.toml")
    out = folder / "point.obj"
    assert invoke("repitch", rings_obj, cfg, out) == 0
    report = record(out, "repitch")
    assert report["blades"]["Blade1"]["axis_origin"] == [1.0, 0.0, 0.0]
    cfg.write_text(cfg.read_text().replace("[1.0, 0.0, 0.0]", "[0.0, 0.0, 0.0]"))
    with pytest.warns(GeometryWarning, match="outside its root loop"):
        assert invoke("repitch", rings_obj, cfg, folder / "miss.obj") == 0


@pytest.mark.parametrize("radius", [2.0, 2.5], ids=["mid-span", "outboard"])
def test_mor1_2_a_point_on_the_pitch_line_turns_the_blade_as_one_at_its_root(ringed, radius):
    """A point anywhere on the blade's pitch line names the same line, so the same turn; the
    sense of the span came from the point, and a point outboard of the blade's middle turned
    the pitch axis towards the root and the blade by minus the angle."""
    folder, _, rings_obj = ringed[False]
    turned = {}
    for tag, r in (("root", 1.0), ("on", radius)):
        point = [1.0, r * np.cos(CENTRE), r * np.sin(CENTRE)]
        name = f"sense_{tag}_{radius}"
        cfg = control(folder, REPITCH.format(angle=2.0, point=json.dumps(point)), name=name)
        assert invoke("repitch", rings_obj, cfg, folder / f"{name}.obj") == 0
        turned[tag] = read_obj(folder / f"{name}.obj")
        direction = record(folder / f"{name}.obj", "repitch")["blades"]["Blade1"]
        assert direction["axis_direction"][1:] == pytest.approx(
            [np.cos(CENTRE), np.sin(CENTRE)], abs=1e-9
        )
    ids = read_obj(rings_obj).family_vertices("Blade1")
    assert np.abs(turned["on"].verts[ids] - turned["root"].verts[ids]).max() < 1e-12


def test_mesh_morph_a_0551_repitch_table_reads_and_turns_about_its_own_axis(ringed):
    folder, _, rings_obj = ringed[False]
    (folder / "old.toml").write_text(
        '[repitch]\nangle_deg = 1.0\nblades = ["Blade1"]\nhub = ["Spinner"]\nmax_angle_deg = 5.0\n'
        "axis = { origin = [1.0, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }\n"
    )
    out = folder / "old.obj"
    assert invoke("repitch", rings_obj, folder / "old.toml", out) == 0
    report = record(out, "repitch")
    assert "pitch_axes" not in report
    assert report["blades"]["Blade1"]["axis_direction"] == [0.0, 0.0, 1.0]
    # a table's own key wins over the top of the file; the top serves a table that has none
    both = control(folder, REPITCH.format(angle=1.0, point=STATION), name="both.toml")
    both.write_text(both.read_text().replace('hub = ["Spinner"]', 'hub = ["Absent"]'))
    both.write_text(both.read_text() + 'hub = ["Spinner"]\n')
    assert invoke("repitch", rings_obj, both, folder / "both.obj") == 0


def test_mesh_morph_periodicity_reads_the_blades_and_hub_of_an_installed_wheel(tmp_path):
    source = build(tmp_path, nacelle=True)
    cfg = control(tmp_path, "[hub_rings]\nrings = 3\n")
    sector = tmp_path / "rings.obj"
    assert invoke("hub-rings", source, cfg, sector) == 0
    from cfdbender.mesh._wheel import sector_to_wheel

    wheel = tmp_path / "wheel.obj"
    # without its rim entries the wheel keeps each blade, base strip included, in one group
    Path(f"{sector}.boundaries.toml").unlink()
    with pytest.warns(GeometryWarning, match="no trailing edge"):
        sector_to_wheel(sector, copies=COPIES, blades=["Blade1"], axis="x", out=wheel)
    mesh = read_obj(wheel)
    base = len(mesh.verts)
    mesh.verts = np.vstack([mesh.verts, [[0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 7.0, 1.0]]])
    mesh.families["Wing"] = [[base, base + 1, base + 2]]
    mesh.vertex_text = []
    wheel.write_text(obj_text(mesh))
    Path(f"{wheel}.boundaries.toml").unlink(missing_ok=True)
    whole = Faces.of([f for fs in mesh.families.values() for f in fs])
    inv = wheel_symmetry(mesh.verts, whole, np.array([1.0, 0, 0]), np.zeros(3), copies=3, tol=1e-7)
    assert inv["invariance"]["nodes_mapped"] < inv["invariance"]["nodes"]  # the wing breaks it
    cfg = control(tmp_path, REPITCH.format(angle=2.0, point=STATION), wheel=True, name="w.toml")
    out = tmp_path / "pitched.obj"
    assert invoke("repitch", wheel, cfg, out) == 0
    periodic = record(out, "repitch")["checks"]["periodicity"]
    assert periodic["nodes_mapped"] == periodic["nodes"]
    assert periodic["faces_mapped"] == periodic["faces"]
    cfg.write_text(cfg.read_text().replace('"Blade2", ', ""))
    with pytest.warns(NotApplicableWarning, match="not periodic either"):
        assert invoke("repitch", wheel, cfg, tmp_path / "two.obj") == 0


def test_mesh_morph_hub_rings_bare_output_name_lands_in_the_workspace_mesh_folder(tmp_path, capsys):
    source = build(tmp_path)
    cfg = control(tmp_path, "[hub_rings]\nrings = 3\n")
    ws = tmp_path / "ws"
    code = main(
        ["--workspace", str(ws), "hub-rings", str(source), "--morphing", str(cfg), "--out", "r.obj"]
    )
    assert code == 0
    assert (ws / "inputs" / "mesh" / "r.obj").is_file()
    assert (ws / "inputs" / "mesh" / "r.obj.provenance.json").is_file()
    capsys.readouterr()


CHAIN = "[hub_rings]\nrings = 4\n" + REPITCH + "[collapse_te]\n"


@pytest.mark.parametrize("wheel", [False, True], ids=["sector", "wheel"])
def test_mesh_morph_runs_the_chain_and_writes_one_output(tmp_path, wheel):
    source = build(tmp_path, wheel=wheel, nacelle=True, wing=wheel)
    text = CHAIN.format(angle=2.0, point=STATION) + ("" if wheel else "[wheel]\n")
    cfg = control(tmp_path, text, wheel=wheel)
    out = tmp_path / "out" / "morphed.obj"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        assert invoke("morph", source, cfg, out) == 0
    report = json.loads(Path(f"{out}.morph.json").read_text())
    order = ["hub_rings", "repitch", "collapse_te"] + ([] if wheel else ["wheel"])
    assert report["order"] == order and list(report["steps"]) == order
    written = sorted(p.name for p in out.parent.iterdir())
    assert written == sorted(
        [*report["files"], "morphed.obj.morph.json", "morphed.obj.provenance.json"]
    )
    assert not [n for n in written if n.endswith((".repitch.json", ".collapse.json"))]
    mesh = read_obj(out)
    assert sorted(n for n in mesh.families if n.startswith("Blade")) == [
        "Blade1",
        "Blade2",
        "Blade3",
    ]
    whole = Faces.of([f for n, fs in mesh.families.items() if n != "Wing" for f in fs])
    inv = wheel_symmetry(mesh.verts, whole, np.array([1.0, 0, 0]), np.zeros(3), copies=3, tol=1e-6)
    if not wheel:
        assert inv["invariance"]["nodes_mapped"] == inv["invariance"]["nodes"]
        assert inv["invariance"]["faces_mapped"] == inv["invariance"]["faces"]
        assert report["steps"]["wheel"]["copies"] == COPIES
    else:
        old = read_obj(source)
        assert places(mesh, mesh.families["Wing"]) == places(old, old.families["Wing"])
    # each blade's two rim entries became one trailing-edge line
    sidecar = tomllib.loads(Path(f"{out}.boundaries.toml").read_text())
    assert len(sidecar["trailing_edges"]) == COPIES


def test_mesh_morph_api_f1_a_wheel_without_a_centre_says_it_turns_about_the_origin(tmp_path):
    """RELEASE review API F1: [wheel] takes its axis from [repitch] but its centre defaults to the
    origin, so a rotor off the origin would be copied wrongly with no word. The default is now
    warned; a stated centre is not."""
    source = build(tmp_path, wheel=False, nacelle=True, wing=False)
    out = tmp_path / "out" / "morphed.obj"
    for text, said in (
        (CHAIN.format(angle=2.0, point=STATION) + "[wheel]\n", True),
        (CHAIN.format(angle=2.0, point=STATION) + "[wheel]\ncenter = [0.0, 0.0, 0.0]\n", False),
    ):
        cfg = control(tmp_path, text, wheel=False)
        if out.parent.exists():
            shutil.rmtree(out.parent)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert invoke("morph", source, cfg, out) == 0
        hits = [w for w in caught if "wheel.center is not stated" in str(w.message)]
        assert bool(hits) is said, [str(w.message) for w in caught]


def test_mesh_morph_morph_writes_nothing_when_a_step_refuses(tmp_path, capsys):
    source = build(tmp_path)
    cfg = control(tmp_path, CHAIN.format(angle=9.0, point=STATION))
    out = tmp_path / "out" / "refused.obj"
    assert invoke("morph", source, cfg, out) == 2
    assert "max_angle_deg" in capsys.readouterr().err
    assert not out.parent.exists() or not list(out.parent.iterdir())
    fsm = tmp_path / "saved.fsm"
    fsm.write_text("not a mesh")
    assert invoke("morph", fsm, cfg, tmp_path / "x.obj") == 2
    assert "must be an .obj" in capsys.readouterr().err


def test_mesh_morph_wheel_reads_the_sector_angle_and_refuses_a_full_wheel(tmp_path, capsys):
    source = build(tmp_path, wheel=True)
    cfg = control(tmp_path, '[wheel]\naxis = "x"\n', wheel=True)
    assert invoke("morph", source, cfg, tmp_path / "again.obj") == 2
    assert "no 360/N" in capsys.readouterr().err
    sector = build(tmp_path, name="sector.obj")
    cfg = control(tmp_path, "[wheel]\n", name="noaxis.toml")
    assert invoke("morph", sector, cfg, tmp_path / "noaxis.obj") == 2
    assert "rotor axis" in capsys.readouterr().err
    cfg.write_text(cfg.read_text() + "center = [0.0, 0.0, 0.5]\n" + 'axis = "x"\n')
    assert invoke("morph", sector, cfg, tmp_path / "off.obj") == 2
    assert not (tmp_path / "off.obj").exists()


@pytest.mark.parametrize(
    "extra",
    ["center = [0.0, 0.0, 0.0005]\n", "copies = 2\n", "copies = 4\n"],
    ids=["MOR1-1-centre-within-the-snap", "MOR1-1-copies-short", "MOR1-1-copies-over"],
)
def test_mor1_1_a_wheel_whose_seams_stay_open_is_refused(tmp_path, capsys, extra):
    """A center off the axis by less than the angle snap, or copies that do not tile the turn,
    leave every seam open; the seam check judged its seams at the merge tolerance, about the
    same center, so it agreed with any wheel and the morph exited 0."""
    sector = build(tmp_path)
    cfg = control(tmp_path, '[wheel]\naxis = "x"\n' + extra)
    out = tmp_path / "open.obj"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        assert invoke("morph", sector, cfg, out) == 2
    assert "seam" in capsys.readouterr().err
    assert not out.exists() and not Path(f"{out}.morph.json").exists()


def test_mor2_1_a_seam_split_differently_on_its_two_sides_is_refused(tmp_path, capsys):
    """One hub face of a seam carries an extra node on its seam edge, so that side has two edges
    where the other has one: the wheel merges the nodes and leaves a crack. The seam check took
    only an edge whose image is an open edge as a seam, counted the crack as no seam, and the
    wheel's open edges agreed with it."""
    source = build(tmp_path)
    mesh = read_obj(source)
    seam = {i for i, p in enumerate(mesh.verts) if abs(p[2]) < 1e-9 and p[1] > 0.0}
    hub = mesh.families["Spinner"]
    for k, face in enumerate(hub):
        pairs = [(q, face[q], face[(q + 1) % len(face)]) for q in range(len(face))]
        hit = [q for q, a, b in pairs if a in seam and b in seam and 0.0 < mesh.verts[a][0] < 1.0]
        if len(face) == 4 and hit:
            q = hit[0]
            a, b = face[q], face[(q + 1) % 4]
            hub[k] = face[: q + 1] + [len(mesh.verts)] + face[q + 1 :]
            verts = np.vstack([mesh.verts, 0.5 * (mesh.verts[a] + mesh.verts[b])])
            break
    cracked = tmp_path / "cracked.obj"
    cracked.write_text(obj_text(ObjMesh(verts, mesh.families)))
    Path(f"{cracked}.boundaries.toml").write_text(Path(f"{source}.boundaries.toml").read_text())
    cfg = control(tmp_path, '[wheel]\naxis = "x"\n')
    out = tmp_path / "cracked_wheel.obj"
    assert invoke("morph", cracked, cfg, out) == 2
    assert "seam" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("value", [-1, 0, True, "wide", float("nan"), float("inf"), None])
def test_mesh_morph_band_width_invalid(tmp_path, value):
    from cfdbender.mesh.hub_rings import read_config

    cfg = {"rings": 4, "blades": ["Blade1"], "hub": ["Spinner"], "band_width": value}
    with pytest.raises(InputValueError, match="positive length"):
        read_config({"hub_rings": cfg}, read_obj(build(tmp_path)))


@pytest.mark.parametrize("point", [1.0123, [1.0123, 0.3, 0.4]])
def test_mesh_morph_pitch_datum_never_supplies_wheel_center(tmp_path, point):
    source = build(tmp_path)
    for center in (None, [0, 0, 0]):
        text = CHAIN.format(angle=0.5, point=point) + "[wheel]\n"
        if center is not None:
            text += f"center = {center}\n"
        cfg = control(tmp_path, text)
        out = tmp_path / ("implicit.obj" if center is None else "explicit.obj")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert invoke("morph", source, cfg, out) == 0
        assert any("wheel.center is not stated" in str(w.message) for w in caught) == (
            center is None
        )
        assert record(out, "morph")["steps"]["wheel"]["hub"] == [0, 0, 0]
    np.testing.assert_array_equal(
        read_obj(tmp_path / "implicit.obj").verts, read_obj(tmp_path / "explicit.obj").verts
    )


@pytest.mark.parametrize("center", ['"bad"', '[0, "bad", 0]', "[0, 0]", "[0, nan, 0]"])
def test_mesh_morph_invalid_toml_center_is_didactic(tmp_path, capsys, center):
    source = build(tmp_path)
    cfg = control(tmp_path, f'[wheel]\naxis = "x"\ncenter = {center}\n')
    out = tmp_path / "invalid.obj"
    assert invoke("morph", source, cfg, out) == 2
    assert "wheel.center must be a point" in capsys.readouterr().err
    assert not out.exists()

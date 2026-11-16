"""The remesh route of repitch on a quad-dominant hub, and quality warned, not refused.

Usage:
    cfd-mesh repitch sector.obj --morphing morphing.toml --out pitched.obj
    cfd-mesh morph sector.obj --morphing morphing.toml --out morphed.obj

``[repitch] hub_method`` is ``"rings"``, ``"remesh"`` or ``"auto"`` (the default). The remesh route
turns each blade rigidly and remeshes the hub band around its moved root quad-dominant, every free
node on the source hub, the kept faces in place and order. Mesh quality (folded, reversed or
turned cells, a node off the hub) is warned with its numbers and the mesh is written; only broken
input and broken invariants are refused.

The fixture is the quad-dominant cylinder hub of ``test_mesh_morph``: it has no complete quad
rings around its roots, so the ring route alone refuses it.
"""

import json
import re
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_mesh_morph import (
    COPIES,
    STATION,
    build,
    control,
    invoke,
    places,
    record,
)

from cfdbender._errors import GeometryWarning
from cfdbender.mesh._refine._obj import ObjMesh, obj_text, read_obj

REPITCH = '[repitch]\nangle_deg = {angle}\nrotor_axis = "x"\nreference_point = {point}\n'


def _hub_distance(source, points):
    """Largest distance of points to the source hub's triangles, by trimesh."""
    import trimesh

    tris = [
        [f[0], f[j], f[j + 1]] for f in source.families["Spinner"] for j in range(1, len(f) - 1)
    ]
    surface = trimesh.Trimesh(source.verts, tris, process=False)
    _, distance, _ = trimesh.proximity.closest_point(surface, points)
    return float(np.max(distance, initial=0.0))


def _quiet_invoke(verb, source, cfg, out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        return invoke(verb, source, cfg, out)


CHAIN = REPITCH + "[collapse_te]\n"


@pytest.mark.parametrize(
    ("wheel", "angle"),
    [(False, 0.0), (False, 2.0), (False, 5.0), (True, 5.0)],
    ids=["sector-0", "sector-2", "sector-5", "wheel-5"],
)

# quality is warned with its numbers and the mesh is written.


def _thin_ring_sector(folder: Path) -> Path:
    """An O-grid hub of two thin rings hugging the root, so a 5 degree turn folds its cells."""
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    root = np.c_[0.3 * np.cos(theta), 0.12 * np.sin(theta), np.ones(16)]
    nodes = [root, root + [0, 0, 1], root + [0, 0, 2]]
    for scale in (1.02, 1.04):
        nodes.append(root * [scale, scale, 1])
    blade = [
        [k * 16 + j, k * 16 + (j + 1) % 16, (k + 1) * 16 + (j + 1) % 16, (k + 1) * 16 + j]
        for k in range(2)
        for j in range(16)
    ]
    rings = [0, 3, 4]
    hub = [
        [a * 16 + j, b * 16 + j, b * 16 + (j + 1) % 16, a * 16 + (j + 1) % 16]
        for a, b in zip(rings[:-1], rings[1:], strict=True)
        for j in range(16)
    ]
    path = folder / "thin.obj"
    path.write_text(obj_text(ObjMesh(np.vstack(nodes), {"Blade1": blade, "Spinner": hub})))
    (folder / "thin.toml").write_text(
        '[repitch]\nangle_deg = 5.0\nblades = ["Blade1"]\nhub = ["Spinner"]\n'
        "axis = { origin = [0.0, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }\n"
    )
    return path


def test_mesh_hub_remesh_folded_and_reversed_ring_cells_are_warned_and_written(
    tmp_path,
):
    source = _thin_ring_sector(tmp_path)
    out = tmp_path / "folded.obj"
    with pytest.warns(GeometryWarning) as caught:
        assert invoke("repitch", source, tmp_path / "thin.toml", out) == 0
    said = [str(w.message) for w in caught]
    folded = r"hub 'Spinner' has \d+ folded cell\(s\); the least scaled Jacobian is -?\d"
    assert any(re.search(folded, s) for s in said), said
    assert any(re.search(r"hub 'Spinner' has \d+ reversed cell", s) for s in said), said
    assert out.exists()
    report = record(out, "repitch")
    checks = report["checks"]
    assert checks["passed"] is False
    assert any("folded" in w for w in checks["warnings"])
    assert any("reversed" in w for w in checks["warnings"])
    assert report["hub"]["Spinner"]["reversed_cells"] >= 1
    assert report["hub"]["Spinner"]["output_quality"]["folded_cells"] >= 1


def test_mesh_hub_remesh_hubq_hub_rings_warns_a_folded_new_cell(tmp_path):
    """A node only new faces hold, reflected through a neighbour: its cells fold over. The arm
    refused; it now warns with the count and the least scaled Jacobian and lists the warning."""
    from cfdbender.mesh import hub_rings

    source = build(tmp_path)
    mesh = read_obj(source)
    cfg = hub_rings.read_config(
        tomllib.loads(control(tmp_path, "[hub_rings]\nrings = 4\n").read_text()), mesh
    )
    result, surface = hub_rings.build_rings(mesh, cfg)
    added = {tuple(f) for f in result.added["Blade1"]}
    others = {
        v for fs in result.mesh.families.values() for f in fs if tuple(f) not in added for v in f
    }
    face = next(f for f in reversed(result.added["Blade1"]) if any(v not in others for v in f))
    k = next(k for k, v in enumerate(face) if v not in others)
    node, across = face[k], face[(k + 2) % len(face)]
    result.mesh.verts[node] = 2.0 * result.mesh.verts[across] - result.mesh.verts[node]
    with pytest.warns(GeometryWarning, match=r"new hub cell\(s\) fold; the least scaled"):
        checks = hub_rings.verify_rings(result, surface, ["Spinner"])
    assert checks["new_cell_quality"]["folded_cells"] >= 1
    assert any("fold" in w for w in checks["warnings"])


def test_mesh_hub_remesh_a_missed_band_target_is_listed_and_fails_passed(tmp_path, monkeypatch):
    """a target no band meets is in checks.warnings; passed is false."""
    from cfdbender.mesh import _hub_remesh

    def unreachable(source_band):
        return {"least_scaled_jacobian": 2.0, "largest_aspect": 0.5}

    monkeypatch.setattr(_hub_remesh, "target_of", unreachable)
    source = build(tmp_path)
    text = REPITCH.format(angle=2.0, point=STATION) + 'hub_method = "remesh"\n'
    cfg = control(tmp_path, text)
    out = tmp_path / "missed.obj"
    assert _quiet_invoke("repitch", source, cfg, out) == 0
    checks = record(out, "repitch")["checks"]
    assert any("misses its quality target" in w for w in checks["warnings"])
    assert checks["passed"] is False


def _lone_root(folder: Path, share: float):
    """The sector's one blade root, planned as remesh_bands plans it, with a share stated."""
    from cfdbender.mesh import _hub_remesh as hr
    from cfdbender.mesh._repitch_grid import root_loop

    mesh = read_obj(build(folder))
    hub = mesh.families["Spinner"]
    loop = root_loop(mesh.families["Blade1"], hub, "Blade1")
    banned = hr.forbidden(mesh, "Spinner", "Blade1", set())
    cell = hr.root_edge_median(mesh, "Spinner", loop, "repitch")
    foot = hr.footprint(mesh, hub, loop)
    return mesh, hr.Root("Blade1", "Spinner", loop, cell, banned, foot, share)


def test_mesh_hub_remesh_morph_009_a_band_width_lowered_to_the_share_is_warned(tmp_path):
    """a stated band_width above the share is lowered and warned."""
    from cfdbender.mesh._hub_remesh import bands_tried

    mesh, root = _lone_root(tmp_path, share=0.0)
    root.share = 6.0 * root.cell
    stated = 8.0 * root.cell
    with pytest.warns(GeometryWarning, match=r"band_width [\d.]+ is lowered to [\d.]+"):
        bands, info = bands_tried(mesh, mesh.verts, root, stated)
    assert info["asked"] == stated
    assert bands[0].width == pytest.approx(5.0 * root.cell)


def test_mesh_hub_remesh_morph_009_a_share_with_no_room_keeps_the_marched_rows_only(tmp_path):
    """a share at or below one cell is warned; no negative band."""
    from cfdbender.mesh._hub_remesh import MIN_ROWS, bands_tried

    mesh, root = _lone_root(tmp_path, share=0.0)
    root.share = 0.5 * root.cell
    with pytest.warns(GeometryWarning, match="leaves no room beyond one local cell"):
        bands, _ = bands_tried(mesh, mesh.verts, root, None)
    assert bands and all(b.width == pytest.approx(MIN_ROWS * root.cell) for b in bands)


@pytest.mark.parametrize(
    ("wheel", "angle"),
    [(False, 0.0), (False, 2.0), (False, 5.0), (True, 5.0)],
    ids=["sector-0", "sector-2", "sector-5", "wheel-5"],
)
def test_mesh_hub_remesh_hubq_morph_writes_without_a_hub_rings_table(tmp_path, wheel, angle):
    source = build(tmp_path, wheel=wheel, nacelle=True, wing=wheel)
    text = CHAIN.format(angle=angle, point=STATION) + ("" if wheel else "[wheel]\n")
    cfg = control(tmp_path, text, wheel=wheel)
    out = tmp_path / "out" / "morphed.obj"
    assert _quiet_invoke("morph", source, cfg, out) == 0
    report = json.loads(Path(f"{out}.morph.json").read_text())
    assert report["order"][0] == "repitch"
    steps = report["steps"]["repitch"]
    assert {b["hub_route"] for b in steps["blades"].values()} == {"remesh"}
    mesh = read_obj(out)
    assert sorted(n for n in mesh.families if n.startswith("Blade")) == [
        f"Blade{k + 1}" for k in range(COPIES)
    ]
    sidecar = tomllib.loads(Path(f"{out}.boundaries.toml").read_text())
    assert len(sidecar["trailing_edges"]) == COPIES


@pytest.mark.parametrize("rings", [False, True], ids=["remesh", "rings"])
def test_mesh_hub_remesh_hubq_an_installed_junction_away_from_the_hub_is_morphed_by_either_route(
    tmp_path, rings
):
    """An airframe imprinted on itself (a wing through a body: its edges held by four faces) is
    no broken input for a rotor's morph: both hub routes write it and keep it to the bit."""
    source = build(tmp_path, nacelle=True, wing=True)
    mesh = read_obj(source)
    families = {**mesh.families, "Body": [list(f) for f in mesh.families["Wing"]]}
    source.write_text(obj_text(ObjMesh(mesh.verts, families)))
    side = Path(f"{source}.boundaries.toml")
    text = side.read_text().replace('"Wing"]', '"Wing", "Body"]')
    assert '"Body"' in text
    side.write_text(text)
    table = "[hub_rings]\nrings = 4\n" if rings else ""
    method = 'hub_method = "rings"\n' if rings else ""
    cfg = control(
        tmp_path,
        table
        + CHAIN.format(angle=2.0, point=STATION).replace("[collapse_te]", method + "[collapse_te]"),
    )
    out = tmp_path / "out" / "morphed.obj"
    assert _quiet_invoke("morph", source, cfg, out) == 0
    report = json.loads(Path(f"{out}.morph.json").read_text())
    route = {b["hub_route"] for b in report["steps"]["repitch"]["blades"].values()}
    assert route == {"rings" if rings else "remesh"}
    written = read_obj(out)
    for name in ("Wing", "Body"):
        assert places(written, written.families[name]) == places(mesh, mesh.families["Wing"])

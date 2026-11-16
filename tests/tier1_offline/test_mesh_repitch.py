"""P0551-REPITCH: rigid blades and a rebuilt hub through the real CLI.

Usage: pyfs-workspace geometry repitch sector.obj --morphing morphing.toml --out pitched.obj
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cfdbender.mesh._refine._obj import ObjMesh, obj_text, read_obj
from cfdbender.mesh.cli import main


def source_mesh(folder: Path, *, sharp: bool = False) -> Path:
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    root = np.c_[0.3 * np.cos(theta), 0.12 * np.sin(theta), np.ones(16)]
    if sharp:
        root[0, 0] = 0.45
    nodes = [root, root + [0, 0, 1], root + [0, 0, 2]]
    for k in range(1, 5):
        nodes.append(root * [1 + k / 2, 1 + k / 2, 1])
    blade = [
        [k * 16 + j, k * 16 + (j + 1) % 16, (k + 1) * 16 + (j + 1) % 16, (k + 1) * 16 + j]
        for k in range(2)
        for j in range(16)
    ]
    rings = [0, 3, 4, 5, 6]
    hub = [
        [a * 16 + j, b * 16 + j, b * 16 + (j + 1) % 16, a * 16 + (j + 1) % 16]
        for a, b in zip(rings[:-1], rings[1:], strict=True)
        for j in range(16)
    ]
    path = folder / "sector.obj"
    path.write_text(obj_text(ObjMesh(np.vstack(nodes), {"Blade1": blade, "Spinner": hub})))
    return path


def control(folder: Path, angle=1.0, extra="", hub="Spinner") -> Path:
    path = folder / "morphing.toml"
    path.write_text(f'''[repitch]
angle_deg = {angle}
blades = ["Blade1"]
hub = ["{hub}"]
axis = {{ origin = [0.0, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }}
max_angle_deg = 5.0
{extra}
''')
    return path


def invoke(source, config, out):
    try:
        return main(["repitch", str(source), "--morphing", str(config), "--out", str(out)])
    except SystemExit as error:
        return error.code


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.0, -1.0])
def test_mesh_repitch_cli_rigid_blade_fixed_outer_boundary_and_quad_counts(tmp_path, angle):
    source = source_mesh(tmp_path)
    out = tmp_path / "pitched.obj"
    assert invoke(source, control(tmp_path, angle), out) == 0
    old, new = read_obj(source), read_obj(out)
    assert new.families == old.families
    a = np.deg2rad(angle)
    matrix = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    np.testing.assert_allclose(new.verts[:48], old.verts[:48] @ matrix.T, atol=2e-14)
    np.testing.assert_array_equal(new.verts[96:], old.verts[96:])
    if angle:
        assert np.max(np.abs(new.verts[48:96] - old.verts[48:96])) > 1e-5
    else:
        np.testing.assert_allclose(new.verts, old.verts, atol=2e-14)
    report = json.loads(Path(str(out) + ".repitch.json").read_text())
    assert report["checks"]["passed"]
    checks = report["checks"]
    assert checks.get("non_root_max_distance_to_source_hub_over_local_edge", float("inf")) < 1e-8
    assert checks["root_max_distance_to_source_hub_m"] >= 0
    assert checks["root_max_distance_to_source_hub_over_local_edge"] >= 0
    assert len(checks["moved_hub_nodes"]) == (64 if angle else 0)


def test_mesh_repitch_inverse_returns_blade_to_rounding(tmp_path):
    source = source_mesh(tmp_path)
    plus, back = tmp_path / "plus.obj", tmp_path / "back.obj"
    assert invoke(source, control(tmp_path), plus) == 0
    assert invoke(plus, control(tmp_path, -1), back) == 0
    np.testing.assert_allclose(read_obj(back).verts[:48], read_obj(source).verts[:48], atol=2e-14)


def test_mesh_repitch_non_root_surface_escape_names_node():
    # P0560-HUBQ (): a node off the hub is quality, warned with its numbers, not refused.
    from cfdbender._errors import GeometryWarning
    from cfdbender.mesh._repitch import HubSurface, surface_checks

    old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    new = old.copy()
    new[2, 2] = 1e-7
    with pytest.warns(GeometryWarning, match="non-root hub node 2 .*1e-07 m"):
        result = surface_checks(HubSurface(old, [[0, 1, 2, 3]]), old, new, {0, 1}, 1.0)
    assert result["non_root_max_distance_to_source_hub_m"] == pytest.approx(1e-7)
    assert result["warnings"] and "non-root hub node 2" in result["warnings"][0]


@pytest.mark.parametrize("height,warning_count", [(0.5, 0), (0.5001, 1), (2.0, 1)])
def test_mesh_repitch_root_standoff_warns_once_and_converts_metres(height, warning_count, recwarn):
    from cfdbender.mesh._repitch import HubSurface, surface_checks

    # Analytic unit square: exact perpendicular distances, two exempt shared roots.
    old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    new = old.copy()
    new[:2, 2] = height
    result = surface_checks(HubSurface(old, [[0, 1, 2, 3]]), old, new, {0, 1}, 0.001)
    assert result["root_max_distance_to_source_hub_m"] == pytest.approx(height * 0.001)
    assert result["root_max_distance_to_source_hub_over_local_edge"] == pytest.approx(height)
    assert result["non_root_max_distance_to_source_hub_m"] == 0
    assert len(recwarn) == warning_count
    if warning_count:
        assert "hub shape near the root changes by this amount" in str(recwarn[0].message)


@pytest.mark.parametrize("angle,hub", [(6, "Spinner"), (1, "absent")])
def test_mesh_repitch_refuses_bad_request_without_outputs(tmp_path, angle, hub, capsys):
    source = source_mesh(tmp_path)
    out = tmp_path / "bad.obj"
    assert invoke(source, control(tmp_path, angle, hub=hub), out) == 2
    message = capsys.readouterr().err
    assert ("max_angle_deg" if angle == 6 else "absent") in message
    assert not out.exists() and not Path(str(out) + ".repitch.json").exists()


def test_mesh_repitch_missing_axis_names_blade_and_remedy(tmp_path, capsys):
    source = source_mesh(tmp_path)
    config = control(tmp_path)
    config.write_text(
        config.read_text().replace(
            "axis = { origin = [0.0, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }", 'axis = "auto"'
        )
    )
    assert invoke(source, config, tmp_path / "absent.obj") == 2
    assert "repitch.axes.Blade1.origin" in capsys.readouterr().err


def test_mesh_repitch_obj_markings_follow_their_nodes(tmp_path):
    import tomllib

    source = source_mesh(tmp_path)
    old = read_obj(source)
    point = old.verts[[16, 32]].mean(axis=0)
    (tmp_path / "rim.txt").write_text("METER\n" + ",".join(map(str, point)) + "\n")
    Path(str(source) + ".boundaries.toml").write_text("""[import]
units = "METER"
[trailing_edges.Blade1]
file = "rim.txt"
type = "STANDARD"
""")
    out = tmp_path / "marked.obj"
    assert invoke(source, control(tmp_path), out) == 0
    sidecar = tomllib.loads(Path(str(out) + ".boundaries.toml").read_text())
    rim = sidecar["trailing_edges"]["Blade1"]
    new = read_obj(out)
    np.testing.assert_allclose(
        np.loadtxt(tmp_path / rim["file"], delimiter=",", skiprows=1),
        new.verts[[16, 32]].mean(axis=0),
        atol=1e-14,
    )
    assert rim["type"] == "STANDARD"


def test_mesh_repitch_folded_hub_refuses_before_writing(tmp_path):
    source = source_mesh(tmp_path)
    mesh = read_obj(source)
    mesh.verts[96] = mesh.verts[97]
    mesh.vertex_text = []
    source.write_text(obj_text(mesh))
    out = tmp_path / "folded.obj"
    assert invoke(source, control(tmp_path), out) == 2
    assert not out.exists() and not Path(str(out) + ".repitch.json").exists()


def test_mesh_repitch_sharp_trailing_edge_keeps_its_connectivity(tmp_path):
    source = source_mesh(tmp_path, sharp=True)
    out = tmp_path / "sharp.obj"
    assert invoke(source, control(tmp_path), out) == 0
    old, new = read_obj(source), read_obj(out)
    assert old.families == new.families
    for a, b in ((0, 16), (16, 32)):
        assert np.linalg.norm(new.verts[a] - new.verts[b]) == pytest.approx(
            np.linalg.norm(old.verts[a] - old.verts[b]), abs=1e-14
        )


def test_mesh_repitch_unknown_key_and_absent_keep_warn_but_do_not_block(tmp_path, recwarn):
    source = source_mesh(tmp_path)
    out = tmp_path / "skip.obj"
    cfg = control(tmp_path, extra='unused = 9\nkeep = ["absent"]')
    assert invoke(source, cfg, out) == 0
    message = " ".join(str(w.message) for w in recwarn)
    assert "unused" in message and "absent" in message and "skipped" in message


@pytest.mark.parametrize("sense", [-1, 1])
@pytest.mark.parametrize("tilt", [0.0, 0.15])
@pytest.mark.parametrize(
    "scale", [1.0, 1e-200, 1e200], ids=["P0551-unit", "P0551-tiny", "P0551-large"]
)
def test_mesh_repitch_auto_positive_pitch_uses_rotor_sense(tmp_path, sense, tilt, scale):
    from cfdbender.mesh._wheel import rotate

    source = source_mesh(tmp_path)
    installation = -float(np.degrees(np.arctan(tilt)))
    mesh = read_obj(source)
    mesh.verts = rotate(mesh.verts, np.array([0.0, 1.0, 0.0]), np.zeros(3), installation)
    mesh.vertex_text = []
    source.write_text(obj_text(mesh))
    axis = (
        f'{{ shaft = "X", vector_wrt_global = [{scale}, 0.0, {tilt * scale}] }}'
        if tilt or scale != 1.0
        else '"X"'
    )
    (tmp_path / "rotor.toml").write_text(f"""[ROTOR]
kind = "rotor"
axis = {axis}
diameter_m = 6.0
rpm_sign = {sense}
families_blades = ["Blade1"]
blade1 = {{ zero = "Z", azimuth_deg = 0.0 }}
""")
    cfg = control(tmp_path)
    cfg.write_text(
        cfg.read_text().replace(
            "axis = { origin = [0.0, 0.0, 0.0], direction = [0.0, 0.0, 1.0] }",
            'axis = "auto"\nreference = "rotor.toml"',
        )
    )
    out = tmp_path / "positive.obj"
    assert invoke(source, cfg, out) == 0
    q = rotate(read_obj(out).verts[:1], np.array([0.0, 1.0, 0.0]), np.zeros(3), -installation)[0]
    angle = np.degrees(np.arctan2(q[1], q[0]))
    assert angle == pytest.approx(-sense, abs=1e-12)


def test_mesh_repitch_a_bare_output_name_lands_in_the_workspace_mesh_folder(tmp_path, capsys):
    source = source_mesh(tmp_path)
    ws = tmp_path / "ws"
    argv = ["--workspace", str(ws), "repitch", str(source)]
    assert main([*argv, "--morphing", str(control(tmp_path)), "--out", "p.obj"]) == 0
    assert (ws / "inputs" / "mesh" / "p.obj").is_file()
    assert (ws / "inputs" / "mesh" / "p.obj.provenance.json").is_file()
    capsys.readouterr()


def test_mesh_repitch_a_saved_simulation_is_refused_naming_the_obj(tmp_path, capsys):
    sim = tmp_path / "s.fsm"
    sim.write_bytes(b"x")
    config = control(tmp_path)
    assert invoke(sim, config, tmp_path / "o.fsm") == 2
    assert "must be an .obj" in capsys.readouterr().err

"""Several named volume cuts in one PPROC file (``[[volume_sections]]``, provisional name).

The single ``[volume_section]`` stays valid and unchanged. An annulus at a propeller disc is a
circle with an inner radius, in the REFERENCE frame, with its centre in the plane's own
coordinates. A finished point gets the cuts from its saved volume files by a new PPROC file.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pytest
import test_runtypes_b1_workflows as rt
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from wiring_support import block_volume, box_surface

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import new_post_file, workspace_path
from cfdbender.core.pproc import PprocError, read_post
from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import CONTEXT, _cell
from cfdbender.post.volume_layouts import LayoutError, volume_section_points
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

DISC = {
    "name": "disc",
    "shape": "circle",
    "frame": "REFERENCE",
    "plane": "YZ",
    "offset_m": 0.5,
    "center_m": [0.5, 0.0],
    "radii_m": [0.2, 0.8],
    "points": [3, 8],
}
PLANE = {
    "name": "plane",
    "shape": "rectangle",
    "frame": "REFERENCE",
    "plane": "XZ",
    "offset_m": 0.5,
    "corners_m": [0.0, -0.5, 1.0, 0.5],
    "points": [3, 3],
}


def read(post):
    return read_post(post, warn=lambda key, text: None)


# ---- the file


def test_an_array_of_cuts_is_read_with_names_and_centres():
    found = read({"volume_sections": [PLANE, DISC]})
    assert [c.name for c in found.volume_sections] == ["plane", "disc"]
    assert found.volume_sections[1].center_m == (0.5, 0.0)
    assert found.volume_sections[0].center_m == (0.0, 0.0)
    assert found.volume_section is None
    assert found.products == ("volume_section",)


def test_the_single_table_stays_valid_and_unnamed():
    single = {k: v for k, v in PLANE.items() if k != "name"}
    found = read({"volume_section": single})
    assert found.volume_section is not None and found.volume_section.name == ""
    assert found.volume_sections == ()


def test_both_forms_in_one_file_are_refused():
    single = {k: v for k, v in PLANE.items() if k != "name"}
    with pytest.raises(
        PprocError, match=r"volume_section.*volume_sections|volume_sections.*volume_section"
    ):
        read({"volume_section": single, "volume_sections": [DISC]})


@pytest.mark.parametrize(
    "entries,message",
    [
        ([{k: v for k, v in DISC.items() if k != "name"}], "name"),
        ([DISC, {**PLANE, "name": "disc"}], "unique"),
        ([{**DISC, "name": "a b"}], "name"),
        ([{**PLANE, "center_m": [0.0, 0.0]}], "center_m"),
        ([], "volume_sections"),
        ([{**DISC, "radii_m": [0.0, 0.0]}], "radii_m"),
    ],
)
def test_a_malformed_cut_is_refused_with_its_reason(entries, message):
    with pytest.raises(PprocError, match=message):
        read({"volume_sections": entries})


def test_an_adapter_without_volume_files_skips_the_array_with_one_warning():
    said = []
    found = read_post(
        {"volume_sections": [PLANE, DISC]},
        supported=("sections",),
        warn=lambda key, text: said.append(text),
    )
    assert found.volume_sections == () and found.products == ()
    assert len(said) == 1 and "volume_section" in said[0]


# ---- the annulus at a disc


def test_an_annulus_has_no_point_inside_its_inner_radius_and_follows_its_centre():
    cloud = volume_section_points(
        "circle",
        "YZ",
        radii_m=(0.2, 0.8),
        points=(3, 8),
        offset_m=0.5,
        center_m=(0.5, -0.25),
    )
    assert cloud.shape == (24, 3)
    assert np.allclose(cloud[:, 0], 0.5)
    radius = np.hypot(cloud[:, 1] - 0.5, cloud[:, 2] + 0.25)
    assert radius.min() == pytest.approx(0.2) and radius.max() == pytest.approx(0.8)
    assert np.all(radius >= 0.2 - 1e-12)


def test_a_rectangle_takes_no_centre():
    with pytest.raises(LayoutError, match="center_m"):
        volume_section_points("rectangle", "YZ", corners_m=(0, 0, 1, 1), center_m=(1.0, 1.0))


# ---- the points of a steady SU2 point, and the time-accurate zones

SPOST = "history = true\n"


def _steady(root, post):
    rt.workspace(root, setup="", condition="MACH:0.0529, ALPHA:0, BETA:0")
    (root / "inputs/pproc/p001.toml").write_text(post, encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    adapter = load_solver("su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    p_inf = adapter.product_inputs(point).free_stream.p_inf
    block_volume(folder / "flow.vtu", p_inf)
    return point, adapter


def _products(root, point, adapter):
    found = point_products(
        root,
        "7001",
        "DP",
        point,
        adapter,
        lead=dict.fromkeys(CONTEXT, "NA"),
        moment_point=(0.0, 0.0, 0.0),
        cell=_cell,
        target=lambda relative: root / "post/rt" / relative,
    )
    return {
        path.relative_to(root / "post/rt").as_posix(): facts for path, facts in found.files.items()
    }, found


def _vtk_points(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("POINTS"))
    count = int(lines[start].split()[1])
    return np.array(
        [[float(x) for x in row.split()] for row in lines[start + 1 : start + 1 + count]]
    )


ARRAY = (
    "history = true\n[[volume_sections]]\nname = 'plane'\nshape = 'rectangle'\nplane = 'XZ'\n"
    "frame = 'REFERENCE'\ncorners_m = [0.0, -0.5, 1.0, 0.5]\npoints = [3, 3]\noffset_m = 0.5\n"
    "[[volume_sections]]\nname = 'disc'\nshape = 'circle'\nplane = 'YZ'\nframe = 'REFERENCE'\n"
    "offset_m = 0.5\ncenter_m = [0.5, 0.0]\nradii_m = [0.2, 0.8]\npoints = [3, 8]\n"
)


def test_the_annulus_in_the_reference_frame_is_written_by_name_on_a_synthetic_volume(tmp_path):
    point, adapter = _steady(tmp_path, ARRAY)
    files, found = _products(tmp_path, point, adapter)
    assert not [s for s in found.skipped if "vsec" in s[0]], found.skipped
    assert sorted(f for f in files if "vsec" in f and not f.endswith(".json")) == [
        "fields/P7001/DP_vsec_disc.vtk",
        "fields/P7001/DP_vsec_plane.vtk",
    ]
    disc = _vtk_points(files["fields/P7001/DP_vsec_disc.vtk"]["text"])
    assert disc.shape == (24, 3) and np.allclose(disc[:, 0], 0.5)
    radius = np.hypot(disc[:, 1] - 0.5, disc[:, 2])
    assert radius.min() == pytest.approx(0.2) and radius.max() == pytest.approx(0.8)
    provenance = json.loads(files["fields/P7001/DP_vsec_disc.vtk.provenance.json"]["text"])
    assert provenance["sample_count"] == 24 and provenance["frame"] == "REFERENCE"
    assert files["fields/P7001/DP_vsec_disc.vtk"]["cut"] == "disc"
    assert files["fields/P7001/DP_vsec_plane.vtk"]["cut"] == "plane"


def test_the_single_table_keeps_its_file_name(tmp_path):
    single = (
        "history = true\n[volume_section]\nshape = 'rectangle'\nplane = 'XZ'\n"
        "frame = 'REFERENCE'\ncorners_m = [0.0, -0.5, 1.0, 0.5]\npoints = [3, 3]\n"
        "offset_m = 0.5\n"
    )
    point, adapter = _steady(tmp_path, single)
    files, _ = _products(tmp_path, point, adapter)
    assert "fields/P7001/DP_vsec.vtk" in files


# ---- a finished point gets the cuts afterwards


def test_a_finished_steady_point_gets_two_cuts_from_a_new_post_file_and_nothing_re_runs(tmp_path):
    first = (
        "history = true\n[[probes]]\nframe = 'REFERENCE'\nparameters = ['VX']\npoints = 3\n"
        "[[probes.lines]]\nstart = [0.0, 0.5, 0.0]\nend = [1.0, 0.5, 0.0]\n"
    )
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(first, encoding="ascii")
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    name = plan.points[0].point_id.split("/", 1)[1]
    post = tmp_path / "post/study"
    assert not list(post.glob("fields/*/*_vsec*"))
    folder = workspace_path(tmp_path, plan.points[0].directory)
    kept = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    runs = (tmp_path / "runs.json").read_bytes()
    assert "volume.vtu" in kept
    edited = first + ARRAY.split("history = true\n", 1)[1]
    (tmp_path / "inputs/pproc/p001.toml").write_text(edited, encoding="ascii")
    assert cli(["post", "study", "--workspace", str(tmp_path), "--new-pproc"]) == 0
    for cut in ("plane", "disc"):
        assert (post / f"fields/P1001/{name}_vsec_{cut}.vtk").is_file()
        assert (post / f"fields/P1001/{name}_vsec_{cut}.vtk.provenance.json").is_file()
    assert (tmp_path / "runs.json").read_bytes() == runs
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == kept
    written = (tmp_path / "inputs/pproc/p001.toml").read_bytes()
    digest = hashlib.sha256(written).hexdigest()
    summary = json.loads((post / "campaign_summary.json").read_bytes())
    assert summary["points"][0]["inputs_sha256"]["inputs/pproc/p001.toml"] == digest
    assert (post / f"pproc/{digest}.toml").read_bytes() == written
    manifest = json.loads((post / "products.json").read_bytes())
    listed = {e["path"]: e for e in manifest["products"] if e.get("product") == "volume_section"}
    assert sorted(e["cut"] for e in listed.values()) == ["disc", "plane"]
    assert all(path.endswith(".vtk") for path in listed)


SLIDING = '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_OUT", "X_IN"]\n'
MESH = (
    "NZONE= 2\nIZONE= 1\nNDIME= 3\nNMARK= 2\nMARKER_TAG= wing\nMARKER_ELEMS= 1\n"
    "MARKER_TAG= X_OUT\nMARKER_ELEMS= 1\nIZONE= 2\nNDIME= 3\nNMARK= 2\nMARKER_TAG= B1\n"
    "MARKER_ELEMS= 1\nMARKER_TAG= X_IN\nMARKER_ELEMS= 1\n"
)


def test_a_finished_time_accurate_point_gets_cuts_from_its_last_step_zone_files(tmp_path):
    root = tmp_path
    rt.workspace(
        root,
        workflow="unsteady_rotor",
        var="DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_ITERS_AVG: 2 / EXPORT_UNSTEADY_LAST_ITER: 2",
        setup='MARKER_PLOTTING = ["wing", "B1"]\n' + SLIDING,
    )
    (root / "inputs/pproc/p001.toml").write_text("history = true\n", encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    (folder / "mesh.su2").write_text(MESH, encoding="ascii")
    adapter = load_solver("su2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = load_point(root, plan.points[0], solver="su2")
    p_inf = adapter.product_inputs(before).free_stream.p_inf
    for step in (35, 36):
        for zone in (0, 1):
            box_surface(
                folder / f"surface_flow_{zone}_{step - 1:05d}.vtu", p_inf, shift_x=4.0 * zone
            )
            block_volume(folder / f"flow_{zone}_{step - 1:05d}.vtu", p_inf, shift_x=4.0 * zone)
    kept = {p.name: p.read_bytes() for p in folder.iterdir()}
    cuts = (
        "history = true\n[[volume_sections]]\nname = 'wide'\nshape = 'rectangle'\nplane = 'XZ'\n"
        "frame = 'REFERENCE'\ncorners_m = [-0.5, -0.5, 5.0, 0.5]\npoints = [4, 2]\n"
        "offset_m = 0.5\nkind = 'default'\n"
        "[[volume_sections]]\nname = 'ring'\nshape = 'circle'\nplane = 'YZ'\n"
        "frame = 'REFERENCE'\noffset_m = 0.5\ncenter_m = [0.5, 0.0]\nradii_m = [0.2, 0.8]\n"
        "points = [3, 8]\nkind = 'default'\n"
    )
    (root / "inputs/pproc/p001.toml").write_text(cuts, encoding="ascii")
    with new_post_file(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = load_point(root, plan.points[0], solver="su2")
    files, found = _products(root, point, adapter)
    assert not [s for s in found.skipped if "vsec" in s[0]], found.skipped
    wide = json.loads(files["fields/P7001/DP_vsec_wide_step_36.vtk.provenance.json"]["text"])
    assert (wide["declared_points"], wide["sample_count"]) == (8, 8)
    assert wide["source_file"] == "flow_0_00035.vtu, flow_1_00035.vtu"
    ring = json.loads(files["fields/P7001/DP_vsec_ring_step_36.vtk.provenance.json"]["text"])
    assert ring["sample_count"] == 24
    assert not [f for f in files if "_step_35" in f and "vsec" in f]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == kept

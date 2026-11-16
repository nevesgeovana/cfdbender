"""P0600 airfoil-2d: the structured C-mesh, its .su2 and its one-cell-thick gmsh 2.2 form.

The mesh is four transfinite gmsh blocks (lower wake, lower surface, upper surface, upper wake)
sharing the wake cut; a blunt trailing edge adds a fifth block behind its base. Counts follow
the level (wall cells W per side, normal cells N, wake cells K per side, base cells B):

* sharp: (2K + 2W) N quadrilaterals, (2K + 2W + 1)(N + 1) - (K + 1) nodes (the cut is shared),
  2W edges in ``airfoil`` and 2K + 2W + 2N in ``farfield``;
* blunt: B K more quadrilaterals, (B - 1)(K + 1) more nodes, the B base edges in
  ``airfoil_base`` and B more outflow edges in ``farfield``.

The far field is a semicircle of radius D about the trailing edge and straight lines to the
outflow at D downstream of it. The .su2 has markers ``airfoil`` and ``farfield``; the .msh is
gmsh 2.2 ASCII extruded one cell along z with ``airfoil``, ``farfield``, ``front``, ``back`` and
``fluid``, which the SU2 and OpenFOAM workspaces accept as they are.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from airfoil_mesh_support import edge_use, off_wall_lengths, read_msh, read_su2, signed_areas
from kernel_support import kernel_or_skip

from cfdbender.mesh import _airfoil2d as af
from cfdbender.mesh._gmsh_worker import ISOLATE_ENV

RE = 6e6


def _build(folder, *, level="coarse", trailing_edge="sharp", name="naca0012", section=None):
    kernel_or_skip()
    folder.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(ISOLATE_ENV, "0")
        record = af.build_mesh(
            folder / name,
            section if section is not None else af.naca_section("0012"),
            reynolds=RE,
            level=level,
            trailing_edge=trailing_edge,
        )
    return record, read_su2(folder / f"{name}.su2"), read_msh(folder / f"{name}.msh")


@pytest.fixture(scope="module")
def sharp(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("sharp"))


@pytest.fixture(scope="module")
def blunt(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("blunt"), trailing_edge="blunt")


def _expected(level, blunt=False):
    counts = af.LEVELS[level]
    w, n, k, b = counts.wall_cells, counts.normal_cells, counts.wake_cells, af.BASE_CELLS
    cells = (2 * k + 2 * w) * n
    nodes = (2 * k + 2 * w + 1) * (n + 1) - (k + 1)
    airfoil, farfield = 2 * w, 2 * k + 2 * w + 2 * n
    if blunt:
        cells, nodes = cells + b * k, nodes + (k + 1) + (b - 1) * (k + 1)
        airfoil, farfield = airfoil + b, farfield + b
    return cells, nodes, airfoil, farfield


def test_p0600_sharp_counts_and_markers(sharp):
    record, su2, _ = sharp
    cells, nodes, airfoil, farfield = _expected("coarse")
    assert su2["dimension"] == 2
    assert len(su2["cells"]) == cells == record["cells"]
    assert len(su2["points"]) == nodes == record["nodes"]
    assert list(su2["markers"]) == ["airfoil", "farfield"]
    assert len(su2["markers"]["airfoil"]) == airfoil
    assert len(su2["markers"]["farfield"]) == farfield
    assert record["markers"] == {"airfoil": airfoil, "farfield": farfield}


@pytest.mark.parametrize("which", ["sharp", "blunt"])
def test_p0600_topology_is_closed_oriented_and_without_duplicates(which, request):
    _, su2, _ = request.getfixturevalue(which)
    cells, points = su2["cells"], su2["points"]
    used = edge_use(cells)
    assert set(used.values()) <= {1, 2}
    boundary = {edge for edge, count in used.items() if count == 1}
    marked = [tuple(sorted(e)) for edges in su2["markers"].values() for e in edges]
    assert len(marked) == len(set(marked)), "an edge is in two markers"
    assert set(marked) == boundary, "every boundary edge is in exactly one marker"
    assert (signed_areas(cells, points) > 0).all(), "every cell is counter-clockwise"
    rounded = {tuple(np.round(p, 12)) for p in points}
    assert len(rounded) == len(points), "no duplicate node, the cut included"


def test_p0600_blunt_counts_and_base_has_its_own_marker(blunt):
    record, su2, _ = blunt
    cells, nodes, airfoil, farfield = _expected("coarse", blunt=True)
    assert len(su2["cells"]) == cells
    assert len(su2["points"]) == nodes
    assert len(su2["markers"]["airfoil"]) == airfoil - af.BASE_CELLS
    assert len(su2["markers"]["farfield"]) == farfield
    assert len(su2["markers"]["airfoil_base"]) == af.BASE_CELLS
    base = [
        e
        for e in su2["markers"]["airfoil_base"]
        if abs(su2["points"][list(e)][:, 0] - 1.0).max() < 1e-12
    ]
    assert len(base) == af.BASE_CELLS
    assert record["parameters"]["trailing_edge"] == "blunt"
    assert record["parameters"]["base_cells"] == af.BASE_CELLS


def test_p0600_first_cell_matches_the_request(sharp):
    record, su2, _ = sharp
    lengths = off_wall_lengths(su2["cells"], su2["points"], su2["markers"]["airfoil"])
    target = af.first_cell(1.0, RE)
    assert record["parameters"]["first_cell"] == pytest.approx(target, rel=1e-12)
    assert lengths.min() == pytest.approx(target, rel=0.01)
    assert lengths.max() == pytest.approx(target, rel=0.01)
    assert record["first_cell"]["mean"] == pytest.approx(target, rel=0.01)


def test_p0600_blunt_surface_first_cell_matches_the_request(blunt):
    record, _, _ = blunt
    assert record["first_cell"]["min"] == pytest.approx(af.first_cell(1.0, RE), rel=0.01)
    assert record["first_cell"]["max"] == pytest.approx(af.first_cell(1.0, RE), rel=0.01)
    assert record["base_first_cell"] > record["first_cell"]["max"]


def test_p0600_far_field_distance_and_outflow(sharp):
    record, su2, _ = sharp
    far = sorted({i for edge in su2["markers"]["farfield"] for i in edge})
    points = su2["points"][far]
    distance = np.linalg.norm(points - [1.0, 0.0], axis=1)
    assert distance.min() == pytest.approx(500.0, rel=0.01)
    assert points[:, 0].max() == pytest.approx(501.0, rel=1e-9)
    assert record["farfield"]["min"] == pytest.approx(500.0, rel=0.01)
    assert record["parameters"]["farfield"] == 500.0


def test_p0600_growth_measured_matches_the_solved_ratio(sharp):
    record, _, _ = sharp
    solved = record["parameters"]["normal_growth"]
    assert solved <= af.GROWTH_CAP
    assert record["largest_growth"] == pytest.approx(solved, rel=0.01)


def test_p0600_quality_record_has_no_defect(sharp):
    record, _, _ = sharp
    assert record["duplicate_nodes"] == 0
    assert record["unmarked_boundary_edges"] == 0
    assert record["smallest_area"] > 0
    assert record["largest_aspect_ratio"] > 1
    json.dumps(record, allow_nan=False)


def test_p0600_openfoam_form_is_gmsh_22_extruded_one_cell(sharp):
    record, su2, msh = sharp
    assert msh["head"] == ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]
    assert set(msh["names"]) == {"airfoil", "farfield", "front", "back", "fluid"}
    assert msh["names"]["fluid"][0] == 3
    assert all(msh["names"][n][0] == 2 for n in ("airfoil", "farfield", "front", "back"))
    assert msh["z"] == {0.0, af.SPAN}
    assert msh["nodes"] == msh["distinct"] == 2 * len(su2["points"])
    cells, _, airfoil, farfield = _expected("coarse")
    assert msh["types"][5] == cells
    assert msh["types"][3] == 2 * cells + airfoil + farfield
    assert record["parameters"]["span"] == af.SPAN == 1.0


@pytest.mark.parametrize("level", ["medium", "fine"])
def test_p0600_finer_levels_carry_their_counts(level, tmp_path):
    record, su2, msh = _build(tmp_path, level=level)
    cells, nodes, airfoil, farfield = _expected(level)
    assert len(su2["cells"]) == cells
    assert len(su2["points"]) == nodes
    assert len(su2["markers"]["airfoil"]) == airfoil
    assert len(su2["markers"]["farfield"]) == farfield
    assert msh["types"][5] == cells
    assert record["parameters"]["level"] == level
    assert record["first_cell"]["min"] == pytest.approx(af.first_cell(1.0, RE), rel=0.01)


def test_p0600_two_runs_write_the_same_bytes(sharp, tmp_path):
    _build(tmp_path)
    first = sharp[0]["files"]
    for kind in ("su2", "msh"):
        assert (tmp_path / f"naca0012.{kind}").read_bytes() == open(first[kind], "rb").read()


def test_p0600_file_and_designation_give_the_same_mesh(sharp, tmp_path):
    from cfdbender.geometry._airfoils import normalized_naca

    upper, lower, _ = normalized_naca("0012")
    path = tmp_path / "n0012.dat"
    rows = [*upper[::-1], *lower[1:]]
    path.write_text("".join(f"{x:.17g} {y:.17g}\n" for x, y in rows), encoding="utf-8")
    _, su2, _ = _build(tmp_path / "file", section=af.read_coordinates(path))
    np.testing.assert_allclose(su2["points"], sharp[1]["points"], atol=1e-9)
    assert (su2["cells"] == sharp[1]["cells"]).all()


def test_p0600_build_through_the_worker_publishes_both_files(tmp_path):
    kernel_or_skip()
    record = af.build_mesh(tmp_path / "w0012", af.naca_section("0012"), reynolds=RE)
    assert (tmp_path / "w0012.su2").is_file() and (tmp_path / "w0012.msh").is_file()
    assert record["files"]["su2"] == str(tmp_path / "w0012.su2")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w0012.msh", "w0012.su2"]


def test_p0600_su2_workspace_plans_the_generated_mesh(sharp, tmp_path):
    import test_p0000_item3d_config as su2_case

    from cfdbender.solvers.su2._history import dimensions

    su2_case.workspace(
        tmp_path,
        extra='MARKER_HEATFLUX = ["airfoil"]\nMARKER_FAR = ["farfield"]',
    )
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(
        setup.read_text(encoding="utf-8").replace(
            "inputs/meshes/tiny.su2", "inputs/mesh/naca0012.su2"
        ),
        encoding="utf-8",
    )
    (tmp_path / "inputs/mesh").mkdir()
    shutil.copyfile(sharp[0]["files"]["su2"], tmp_path / "inputs/mesh/naca0012.su2")
    point = su2_case.prepared(tmp_path)
    lines = su2_case.api().translate(point).splitlines()
    assert "MESH_FILENAME= mesh.su2" in lines
    assert "MARKER_FAR= ( farfield )" in lines
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copyfile(sharp[0]["files"]["su2"], staged / "mesh.su2")
    assert dimensions(staged) == 2


def test_p0600_openfoam_workspace_accepts_the_generated_mesh(sharp, tmp_path):
    import test_p0000_openfoam_adapter as openfoam_case

    from cfdbender.workspace.cli import main as workspace
    from cfdbender.workspace.planning import plan_workspace

    assert workspace(["init", "--solver", "openfoam", "--workspace", str(tmp_path)]) == 0
    files = {
        "inputs/setups/s001.toml": "\n".join(
            [
                'mesh = "inputs/mesh/naca0012.msh"',
                'application = "simpleFoam"',
                "speed = 10.0",
                "nu = 1.5e-5",
                'turbulence = "SpalartAllmaras"',
                "iterations = 4",
                "",
            ]
        ),
        "inputs/references/r001.toml": (
            "area = 1.0\nlength = 1.0\nmoment_origin = [0.25, 0.0, 0.0]\n"
        ),
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": (
            "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n1001|1|001|001|001|ALPHA:sweep|0,4\n"
        ),
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(text.encode("ascii"))
    (tmp_path / "inputs/mesh").mkdir(exist_ok=True)
    shutil.copyfile(sharp[0]["files"]["msh"], tmp_path / "inputs/mesh/naca0012.msh")
    plan = plan_workspace(tmp_path, "study")
    assert len(plan.points) == 2
    text = openfoam_case.api().translate(openfoam_case.point(plan))
    assert "gmshToFoam" in openfoam_case.split(text)["Allrun"]


def test_ra0_blunt_decks_cover_both_walls(blunt, tmp_path):
    import test_p0000_item3d_config as su2_case

    su2_case.workspace(
        tmp_path, extra='MARKER_HEATFLUX = ["airfoil", "airfoil_base"]\nMARKER_FAR = ["farfield"]'
    )
    lines = su2_case.api().translate(su2_case.prepared(tmp_path))
    assert "MARKER_HEATFLUX= ( airfoil, 0.0, airfoil_base, 0.0 )" in lines
    assert "MARKER_MONITORING= ( airfoil, airfoil_base )" in lines
    assert "MARKER_PLOTTING= ( airfoil, airfoil_base )" in lines
    from cfdbender.solvers.openfoam._config import config, files

    case = config(
        {
            "mesh": "inputs/mesh/a.msh",
            "application": "simpleFoam",
            "speed": 10,
            "nu": 1e-5,
            "turbulence": "SpalartAllmaras",
            "iterations": 4,
        },
        {"area": 1, "length": 1, "moment_origin": [0, 0, 0]},
        {"history": True},
        {},
    )
    deck = files(case, {})
    assert 'patches ("airfoil(_base)?");' in deck["system/controlDict"]
    for name in ("0/U", "0/p", "0/nut", "0/nuTilda"):
        assert '"airfoil(_base)?"' in deck[name]
    assert "entry0/airfoil_base/type -set wall" in deck["Allrun"]
    assert "airfoil_base" in blunt[2]["names"]


def test_ra0_force_and_moment_sum_is_preserved_by_partition(blunt):
    _, mesh, _ = blunt
    points = mesh["points"]

    def integrate(edges):
        total = np.zeros(3)
        for a, b in edges:
            p, q = points[a], points[b]
            force = np.array([q[1] - p[1], p[0] - q[0]]) * (1 + (p[0] + q[0]) / 2)
            mid = (p + q) / 2
            total += [force[0], force[1], mid[0] * force[1] - mid[1] * force[0]]
        return total

    wall, base = mesh["markers"]["airfoil"], mesh["markers"]["airfoil_base"]
    np.testing.assert_allclose(
        integrate(wall + base), integrate(wall) + integrate(base), atol=1e-12
    )
    assert np.linalg.norm(integrate(base)) > 0

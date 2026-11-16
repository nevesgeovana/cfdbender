"""geometry: generating a named IGES file from a configuration (uses the gmsh kernel).

Ported from the research tests of the bodies, fairings and assembly. The tests that need the
geometry kernel skip where the optional extra is absent; the refusal and the import isolation
run everywhere.
"""

import builtins
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from geom_inputs import body_input
from kernel_support import kernel_or_skip

from cfdbender._errors import InputValueError
from cfdbender.geometry import resolve_config
from cfdbender.geometry._bodies import body
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry._caps import wing_contour_normal
from cfdbender.geometry._fairings import wing_fuselage_fairing
from cfdbender.geometry._lifting import lifting_surface, section_models, thickness_station
from cfdbender.mesh._extras import MissingExtraError

WB = {
    "components": {
        "FUS": {"type": "body"},
        "WING": {
            "type": "lifting_surface",
            "attach": {"parent": "FUS", "station": 0.4},
            "stations": [{"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}],
        },
        "FAIRING": {
            "type": "wing_fuselage_fairing",
            "hosts": ["FUS", "WING"],
            "attach": {"parent": "FUS", "station": 0.0},
        },
    },
    "configurations": {"WB": ["FUS", "WING", "FAIRING"], "FUS_ONLY": ["FUS"]},
}


def gmsh_or_skip():
    return kernel_or_skip()


@pytest.fixture
def occ():
    from cfdbender.mesh._gmsh import session

    gmsh_or_skip()
    with session("a test") as gmsh:
        gmsh.model.add("t")
        yield gmsh


def cylinder(rear="pole"):
    raw = {
        "type": "body",
        "length": 20.0,
        "rear": rear,
        "sections": [{"at": "middle", "shape": "circle", "radius": 1.2}],
    }
    return unpack(resolve_config(body_input(raw)))["components"]["COMPONENT"]


def add_all(gmsh, patches):
    from cfdbender.geometry._geometry_generate import add_occ_patch

    return [add_occ_patch(gmsh.model.occ, p) for p in patches]


def free_edges_off_symmetry(model, tolerance=1e-7):
    free = []
    for _, tag in model.getEntities(1):
        parents, _ = model.getAdjacencies(1, tag)
        if len(parents) == 1:
            bounds = model.getParametrizationBounds(1, tag)
            xyz = np.array(
                model.getValue(1, tag, np.linspace(bounds[0][0], bounds[1][0], 31))
            ).reshape(-1, 3)
            if np.max(abs(xyz[:, 1])) > tolerance:
                free.append(tag)
    return free


def test_geom_iges_generate_body_patches_are_four_sided_with_positive_edges_and_area(occ):
    patches = body(cylinder())
    tags = add_all(occ, patches)
    occ.model.occ.synchronize()
    for tag in tags:
        edges = occ.model.getBoundary([(2, tag)], oriented=False)
        assert len(edges) == 4
        assert all(occ.model.occ.getMass(1, edge) > 1e-9 for _, edge in edges)
        assert occ.model.occ.getMass(2, tag) > 1e-9


@pytest.mark.parametrize("rear", ["pole", "base"])
def test_geom_iges_generate_body_free_edges_only_on_symmetry(occ, rear):
    add_all(occ, body(cylinder(rear)))
    occ.model.occ.synchronize()
    occ.model.occ.removeAllDuplicates()
    occ.model.occ.synchronize()
    assert free_edges_off_symmetry(occ.model) == []


def test_geom_iges_generate_body_station_tangency_below_a_tenth_of_a_degree(occ):
    patches = body(cylinder())
    tags = add_all(occ, patches)
    occ.model.occ.synchronize()
    nose = [(p, t) for p, t in zip(patches, tags, strict=True) if p.family == "nose"]
    angles = []
    for family, adjacent, endpoint, cells in [
        ("nose", "front", 0.0, nose),
        (
            "rear",
            "middle",
            1.0,
            [(p, t) for p, t in zip(patches, tags, strict=True) if p.family == "rear"],
        ),
    ]:
        caps = [tag for k, (_, tag) in enumerate(cells) if k % 3 in (1, 2)]
        bands = [t for p, t in zip(patches, tags, strict=True) if p.family == adjacent]
        assert family and len(caps) == len(bands)
        for cap, band in zip(caps, bands, strict=True):
            for u in [0.13, 0.37, 0.63, 0.87]:
                a = np.asarray(occ.model.getNormal(cap, [1.0, u]))
                b = np.asarray(occ.model.getNormal(band, [u, endpoint]))
                cosine = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
                angles.append(float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))))
    assert max(angles) < 0.1


@pytest.fixture
def fairing_model(occ):
    data = unpack(resolve_config({"components": {
        "FUS": {"type": "body"},
        "WING": {"type": "lifting_surface", "stations": [
            {"eta": 0.0, "section": {"kind": "naca", "code": "0012"}}
        ],
                 "distributions": {"leading_edge_x": {"stations": [0, 1.0], "values": [0.4, 0.4]}}},
        "FAIRING": {"type": "wing_fuselage_fairing", "hosts": ["FUS", "WING"]},
    }}))["components"]  # fmt: skip
    add_all(occ, body(data["FUS"]) + lifting_surface(data["WING"]))
    patches = wing_fuselage_fairing(data["FAIRING"], data["FUS"], data["WING"])
    tags = add_all(occ, patches)
    occ.model.occ.synchronize()
    return occ, data, patches, tags


def test_geom_iges_generate_fairing_patches_are_four_sided(fairing_model):
    gmsh, _, _, tags = fairing_model
    for tag in tags:
        edges = gmsh.model.getBoundary([(2, tag)], oriented=False)
        assert len(edges) == 4
        assert all(gmsh.model.occ.getMass(1, edge) > 1e-8 for _, edge in edges)
        assert gmsh.model.occ.getMass(2, tag) > 1e-10


def test_geom_iges_generate_fairing_tangency_and_host_contact(fairing_model):
    gmsh, data, patches, tags = fairing_model
    models = section_models(data["WING"]["sections"])
    maximum = thickness_station(data["WING"], models, 0.0)
    angles, gaps = [], []
    for k, (patch, tag) in enumerate(zip(patches, tags, strict=True)):
        for u in [0.25, 0.5, 0.75]:
            endpoint = 0.0 if patch.family == "fairing_body" else 1.0
            p = np.asarray(gmsh.model.getValue(2, tag, [u, endpoint]))
            actual = np.asarray(gmsh.model.getNormal(tag, [u, endpoint]))
            if endpoint == 0:
                expected = np.array([0.0, p[1], p[2]]) / np.hypot(p[1], p[2])
                gaps.append(abs(np.hypot(p[1], p[2]) - 0.1))
            else:
                expected = wing_contour_normal(data["WING"], models, maximum, k // 2, u, 2 * p[1])
            cosine = (actual @ expected) / np.linalg.norm(actual)
            angles.append(float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))))
    assert max(gaps) < 1e-7
    assert max(angles) < 0.1


def test_geom_iges_generate_iges_names_and_metres(occ, tmp_path):
    from cfdbender.geometry._iges_write import read_iges, write_iges

    surface = occ.model.occ.addRectangle(0, 0, 0, 2, 3)
    occ.model.occ.synchronize()
    path = tmp_path / "named.igs"
    write_iges(occ, {"TEST": [surface]}, path)
    document = read_iges(path)
    names = [e for e in document["entities"] if e["type"] == 406]
    assert len(names) == 1
    assert int(names[0]["d2"][4]) == 15
    assert names[0]["parameters"] == ["406", "1", "4HTEST"]
    assert document["global"][13:15] == ["6", "1HM"]
    occ.model.add("import")
    occ.option.setString("Geometry.OCCTargetUnit", "M")
    imported = occ.model.occ.importShapes(str(path))
    occ.model.occ.synchronize()
    area = sum(occ.model.occ.getMass(d, t) for d, t in imported if d == 2)
    assert area == pytest.approx(6.0, abs=1e-10)


def test_geom_iges_generate_iges_refuses_a_name_the_format_cannot_carry(occ, tmp_path):
    from cfdbender.geometry._iges_write import write_iges

    surface = occ.model.occ.addRectangle(0, 0, 0, 2, 3)
    occ.model.occ.synchronize()
    with pytest.raises(InputValueError, match="ASCII component identifiers"):
        write_iges(occ, {"A,B": [surface]}, tmp_path / "bad.igs")


def test_geom_iges_generate_generate_iges_writes_one_name_per_component_and_a_clean_skin(tmp_path):
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry._iges_write import read_iges

    path = tmp_path / "WB.igs"
    result = generate_iges(WB, path, configuration="WB")
    audit = result["audit"]
    assert audit["free_edges_off_symmetry"] == []
    assert audit["nonmanifold_edges"] == []
    assert audit["non_ring_nonquadrilateral_faces"] == []
    assert audit["intersection_curves"]
    assert audit["unshared_intersection_curves"] == []
    assert set(result["faces"]) == {"FUS", "WING", "FAIRING"}
    assert all(count > 0 for count in result["faces"].values())
    properties = [e for e in read_iges(path)["entities"] if e["type"] == 406]
    assert sorted(row["parameters"][2] for row in properties) == sorted(
        f"{len(n)}H{n}" for n in ("FUS", "WING", "FAIRING")
    )
    assert all(int(row["d2"][4]) == 15 for row in properties)


def test_geom_iges_generate_generate_iges_names_read_back_through_the_library_reader(tmp_path):
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry._iges_names import read_iges_names

    path = tmp_path / "WB.igs"
    result = generate_iges(WB, path)
    names = read_iges_names(path)
    assert names.unit == "METER"
    assert dict(names.components) == result["faces"]


def test_geom_iges_generate_generate_iges_reads_a_toml_file_and_selects_a_configuration(tmp_path):
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.geometry.stages import write_config

    toml = tmp_path / "wb.toml"
    write_config(resolve_config(WB), toml)
    result = generate_iges(toml, tmp_path / "fus.igs", configuration="FUS_ONLY")
    assert set(result["faces"]) == {"FUS"}
    with pytest.raises(InputValueError, match="available"):
        generate_iges(toml, tmp_path / "x.igs", configuration="NOPE")
    assert not (tmp_path / "x.igs").exists()


def test_geom_iges_generate_the_generate_and_check_modules_refuse_without_gmsh(
    monkeypatch, tmp_path
):
    from cfdbender.geometry import _geometry_check, _geometry_generate

    real = builtins.__import__

    def no_gmsh(name, *args, **kwargs):
        if name == "gmsh" or name.startswith("gmsh."):
            raise ImportError("No module named 'gmsh'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_gmsh)
    with pytest.raises(MissingExtraError, match=r"gmsh>=4\.15\.1"):
        _geometry_generate.generate_iges(WB, tmp_path / "x.igs")
    with pytest.raises(MissingExtraError, match=r"gmsh>=4\.15\.1"):
        _geometry_check.check_against_original([1], [2], {"chord_m": 1.0})
    assert not (tmp_path / "x.igs").exists()


#: The modules of cfdbender.geometry that reach the geometry kernel; every other module of the
#: package is the pure parametric layer, which needs numpy and scipy only.
KERNEL_MODULES = {
    "__init__.py",  # the package face re-exports the runner
    "_cli_verbs.py",
    "_geom_area.py",
    "_geom_area_brep.py",
    "_geom_area_domain.py",
    "_geom_area_integral.py",
    "_geom_gate.py",
    "_geom_reader.py",
    "_geom_assembly.py",
    "_geom_export.py",
    "_geom_parts.py",
    "_geom_propeller.py",
    "_geom_removal.py",
    "_geom_roundtrip.py",
    "_geom_variant.py",
    "_geometry_check.py",
    "_geometry_generate.py",
    "_geometry_mesh.py",
    "_geometry_mesh_body.py",
    "_geometry_mesh_junction.py",
    "_geometry_mesh_occ.py",
    "_geometry_mesh_s1.py",
    "_geometry_mesh_s2.py",
    "_geometry_reverse.py",
    "_identify.py",
    "cli.py",
    "doe.py",
    "iges_convert.py",
    "pipeline.py",
}


def test_geom_iges_generate_the_pure_geometry_layer_imports_no_kernel_and_no_upper_layer():
    import ast

    import cfdbender.geometry

    folder = Path(cfdbender.geometry.__file__).parent
    forbidden = ("gmsh", "cfdbender.mesh", "cfdbender.matrix", "cfdbender.workspace")
    kernel = tuple(f"cfdbender.geometry.{name[:-3]}" for name in KERNEL_MODULES)
    found = []
    for path in sorted(folder.glob("*.py")):
        if path.name in KERNEL_MODULES:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            found += [
                (path.name, n)
                for n in names
                if n.startswith(forbidden + kernel) and n != "cfdbender.mesh.objio"
            ]
    assert found == []
    assert len(list(folder.glob("*.py"))) >= 13


def test_geom_iges_generate_importing_the_geometry_math_does_not_import_gmsh():
    import cfdbender

    code = "import sys, cfdbender.geometry._generation; print('gmsh' in sys.modules)"
    source = str(Path(cfdbender.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([source, os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "False"


def test_geom_iges_generate_iges_records_break_between_parameters_never_inside_one():
    from cfdbender.geometry._iges_write import pack_records, tokens

    values = [repr(-123.456789012345678 * k) for k in range(1, 40)] + ["4HNAME", "7"]
    for width in (64, 72):
        records = pack_records(values, width)
        assert all(len(record) <= width for record in records)
        assert all(record.endswith(",") or record.endswith(";") for record in records)
        assert tokens("".join(records)) == values
    long_name = ["406", "1", "100H" + "a" * 100]
    assert tokens("".join(pack_records(long_name, 64))) == long_name


def test_geom_iges_generate_a_generated_body_reimports_with_every_face(tmp_path):
    gmsh_or_skip()
    from cfdbender.geometry._geometry_generate import generate_iges
    from cfdbender.mesh._gmsh import session

    path = tmp_path / "FUS.igs"
    written = generate_iges(WB, path, configuration="FUS_ONLY")["faces"]["FUS"]
    assert written > 1
    with session("a test") as gmsh:
        gmsh.model.add("again")
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        imported = gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        assert sum(1 for dim, _ in imported if dim == 2) == written

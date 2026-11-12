"""P0000: initialize SU2, plan s/r/p inputs and translate a deterministic deck."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import RowRequirements, plan_matrix
from cfdbender.solvers import available_solvers, load_solver


def api():
    try:
        return load_solver("su2")
    except ValueError:
        return SimpleNamespace(
            translate=lambda point: "",
            validate_row=lambda row, inputs: RowRequirements(
                ("solve",), ("sims/sim_{pol}/datapoints/{point}/history.csv",)
            ),
        )


def workspace(root: Path, *, extra="", condition="MACH:0.3,ALPHA:2.0"):
    files = {
        "init.toml": "schema_version = 1" + chr(10) + 'solver = "su2"',
        "inputs/setups/s001.toml": chr(10).join(
            [
                'mesh = "inputs/meshes/tiny.su2"',
                "iterations = 40",
                'convergence_field = "RMS_DENSITY"',
                "freestream_temperature = 288.15",
                "reynolds = 1000000",
                'SOLVER = "EULER"',
                "CFL_NUMBER = 2.0",
                extra,
            ]
        ),
        "inputs/references/r001.toml": chr(10).join(
            ["area = 2.0", "length = 3.0", "moment_origin = [1.0, 2.0, 3.0]"]
        ),
        "inputs/pproc/p001.toml": "history = true" + chr(10) + "output_frequency = 5",
        "inputs/matrices/study.fs": "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION"
        + chr(10)
        + "1001|1|001|001|001|"
        + condition,
        "inputs/meshes/tiny.su2": "NDIME= 2",
    }
    for name, value in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((value + chr(10)).encode("ascii"))
    return root


def plan(root):
    return plan_matrix(root, "study", validators={"su2": api().validate_row})


def prepared(root):
    return load_point(root, plan(root).points[0], solver="su2")


def test_p0000_item3d_registered_and_init(tmp_path):
    from cfdbender.workspace.cli import main

    assert "su2" in available_solvers(), "SU2 must be a known solver"
    assert main(["init", "--solver", "su2", "--workspace", str(tmp_path)]) == 0
    assert 'solver = "su2"' in (tmp_path / "init.toml").read_text()


def test_p0000_item3d_translation_and_mesh_provenance(tmp_path):
    workspace(tmp_path, extra="SIDESLIP_ANGLE = 1.0")
    point = prepared(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    text = api().translate(point)
    expected = {
        "MACH_NUMBER= 0.3",
        "AOA= 2",
        "SIDESLIP_ANGLE= 1",
        "FREESTREAM_TEMPERATURE= 288.15",
        "REYNOLDS_NUMBER= 1000000",
        "REF_AREA= 2",
        "REF_LENGTH= 3",
        "REF_ORIGIN_MOMENT_X= 1",
        "REF_ORIGIN_MOMENT_Y= 2",
        "REF_ORIGIN_MOMENT_Z= 3",
        "ITER= 40",
        "CONV_FIELD= RMS_DENSITY",
        "OUTPUT_WRT_FREQ= 5",
        "CFL_NUMBER= 2",
        "SOLVER= EULER",
        "MESH_FILENAME= mesh.su2",
    }
    assert expected <= set(text.splitlines()), "neutral fields must map to SU2 keys"
    assert text == api().translate(point) and text.isascii()
    mesh = "inputs/meshes/tiny.su2"
    assert (
        point.point.inputs_sha256.get(mesh) == hashlib.sha256(before[tmp_path / mesh]).hexdigest()
    )
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    (tmp_path / mesh).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        api().translate(point)


@pytest.mark.parametrize(
    "extra,problem",
    [
        ("MACH_NUMBER = 0.9", "conflict"),
        ("ITER = 12", "conflict"),
        ("[solver]" + chr(10) + 'route = "bad"', "route"),
        ("[solver]" + chr(10) + "threads = 0", "threads"),
        ('BAD = "line%comment"', "BAD"),
    ],
)
def test_p0000_item3d_refuses_bad_rows(tmp_path, extra, problem):
    workspace(tmp_path, extra=extra)
    with pytest.raises(ValueError, match=problem):
        plan(tmp_path)


@pytest.mark.parametrize(
    "route,available,expected,warning,refuse",
    [
        (None, False, "executable", True, False),
        (None, True, "pysu2", False, False),
        ("pysu2", False, None, False, True),
        ("pysu2", True, "pysu2", False, False),
        ("executable", False, "executable", False, False),
        ("executable", True, "executable", False, False),
    ],
)
def test_p0000_item3d_route_choice(route, available, expected, warning, refuse):
    calls = []

    def importer(name):
        calls.append(name)
        if not available:
            raise ImportError("wrapper absent")
        return SimpleNamespace()

    try:
        routes = importlib.import_module("cfdbender.solvers.su2.routes")
    except ModuleNotFoundError:
        routes = SimpleNamespace(select_route=lambda *args, **kwargs: "unavailable")
    if refuse:
        with pytest.raises(ValueError, match="pysu2.*unavailable"):
            routes.select_route(route, importer=importer)
    elif warning:
        with pytest.warns(RuntimeWarning, match="pysu2.*executable"):
            assert routes.select_route(route, importer=importer) == expected
    else:
        assert routes.select_route(route, importer=importer) == expected
    assert calls == ([] if route == "executable" else ["pysu2"])


def test_review6_workspace_solver_options_and_row_override(tmp_path):
    from cfdbender.core.settings import read_init
    from cfdbender.solvers.su2._config import options

    workspace(tmp_path, extra='[solver]\nroute = "pysu2"')
    (tmp_path / "init.toml").write_text(
        'schema_version = 1\n[solver]\nname = "su2"\nroute = "executable"\nexecutable = "native"\n',
        encoding="utf-8",
    )
    cfg = read_init(tmp_path, available_solvers=("su2",))
    assert cfg.solver == "su2"
    assert cfg.solver_options == {"route": "executable", "executable": "native"}
    point = prepared(tmp_path)
    merged = options(point.setup, cfg.solver_options)
    assert merged["route"] == "pysu2"
    assert merged["executable"] == "native"


@pytest.mark.parametrize("value", ["[1, 2]", "{nested = 1}", "1979-05-27", "nan"])
def test_review6_workspace_options_refuse_non_scalar_or_nonfinite(tmp_path, value):
    from cfdbender.core.settings import read_init

    (tmp_path / "init.toml").write_text(
        'schema_version = 1\n[solver]\nname = "su2"\noption = ' + value, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="option"):
        read_init(tmp_path, available_solvers=("su2",))


def test_review6_invalid_workspace_option_refused_during_planning(tmp_path):
    workspace(tmp_path)
    (tmp_path / "init.toml").write_text(
        'schema_version = 1\n[solver]\nname = "su2"\nroute = "wrong"', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="route"):
        plan(tmp_path)


def test_review6_unicode_mesh_path_is_staged(tmp_path):
    workspace(tmp_path)
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace("tiny.su2", "mesh caf\u00e9.su2"), encoding="utf-8")
    (tmp_path / "inputs/meshes/mesh caf\u00e9.su2").write_text("NDIME= 2", encoding="ascii")
    point = prepared(tmp_path)
    assert "MESH_FILENAME= mesh.su2" in api().translate(point)
    assert "inputs/meshes/mesh caf\u00e9.su2" in point.point.inputs_sha256


def test_review6_passthrough_keys_in_reference_and_post(tmp_path):
    workspace(tmp_path)
    reference = tmp_path / "inputs/references/r001.toml"
    reference.write_text(
        reference.read_text() + '\nREF_DIMENSIONALIZATION = "DIMENSIONAL"', encoding="utf-8"
    )
    post = tmp_path / "inputs/pproc/p001.toml"
    post.write_text(post.read_text() + '\nHISTORY_OUTPUT = ["ITER", "RMS_RES"]', encoding="utf-8")
    text = api().translate(prepared(tmp_path))
    assert "REF_DIMENSIONALIZATION= DIMENSIONAL" in text
    assert "HISTORY_OUTPUT= ( ITER, RMS_RES )" in text

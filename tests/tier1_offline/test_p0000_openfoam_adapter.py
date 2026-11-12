"""P0000: init, plan, run and collect an ESI OpenFOAM alpha sweep offline.

Usage: cfd-workspace init --solver openfoam; cfd-matrix plan study;
cfd-matrix run study; cfd-matrix collect study. Collection writes post/study.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.matrix.cli import main as matrix
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.cli import main as workspace
from cfdbender.workspace.planning import plan_workspace

FAKE = Path(__file__).resolve().parents[1] / "fixtures/openfoam/fake_openfoam.py"
CASE_FILES = (
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "0/U",
    "0/p",
    "0/nut",
    "0/nuTilda",
    "Allrun",
)


def api():
    assert "openfoam" in available_solvers(), "OpenFOAM workspace support is required"
    return load_solver("openfoam")


def case(root, *, sweep="0,4", turbulence="SpalartAllmaras", extra=""):
    api()
    assert workspace(["init", "--solver", "openfoam", "--workspace", str(root)]) == 0
    files = {
        "inputs/setups/s001.toml": (
            'mesh = "inputs/meshes/tiny.msh"\napplication = "simpleFoam"\n'
            f'speed = 10.0\nnu = 1.5e-5\nturbulence = "{turbulence}"\niterations = 4\n'
            "[solver]\nexecutable = " + json.dumps(FAKE.as_posix()) + "\n" + extra
        ),
        "inputs/references/r001.toml": (
            "area = 0.1\nlength = 1.0\nmoment_origin = [0.25, 0.0, 0.0]\n"
        ),
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": (
            "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
            f"1001|1|001|001|001|ALPHA:sweep|{sweep}\n"
        ),
        "inputs/meshes/tiny.msh": "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    return plan_workspace(root, "study")


def point(plan, index=0):
    return load_point(plan.root, plan.points[index], solver="openfoam")


def split(text):
    result = {}
    for section in text.split("// FILE: ")[1:]:
        name, body = section.split("\n", 1)
        result[name] = body
    return result


def test_p0000_openfoam_init_plan_deterministic_alpha(tmp_path):
    a = case(tmp_path / "a")
    b = case(tmp_path / "b")
    assert matrix(["plan", "study", "--workspace", str(a.root)]) == 0
    assert len(a.points) == len(b.points) == 2
    for i in range(2):
        text = api().translate(point(a, i))
        assert text == api().translate(point(b, i)) and text.isascii()
        files = split(text)
        assert tuple(files) == CASE_FILES
        assert all(f"/{name}" in "\n".join(a.points[i].outputs) for name in CASE_FILES)
        mesh = "inputs/meshes/tiny.msh"
        assert (
            a.points[i].inputs_sha256[mesh]
            == hashlib.sha256((a.root / mesh).read_bytes()).hexdigest()
        )
    files = split(api().translate(point(a, 1)))
    angle = math.radians(4)
    control = files["system/controlDict"]
    assert f"dragDir ({math.cos(angle):.15g} {math.sin(angle):.15g} 0);" in control
    assert f"liftDir ({-math.sin(angle):.15g} {math.cos(angle):.15g} 0);" in control
    assert "CofR (0.25 0 0);" in control and "Aref 0.1;" in control and "lRef 1;" in control
    assert "residualControl" in files["system/fvSolution"]
    assert "gmshToFoam mesh.msh > log.gmshToFoam 2>&1" in files["Allrun"]
    assert "foamDictionary" in files["Allrun"] and "entry0/front/type" in files["Allrun"]
    for name in ("0/U", "0/p", "0/nut", "0/nuTilda"):
        assert "front { type empty; }" in files[name]
        assert "back { type empty; }" in files[name]
    (a.root / "inputs/meshes/tiny.msh").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        api().translate(point(a))


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("converged", "converged"),
        ("residual", "converged"),
        ("iteration_limit", "iteration_limit"),
        ("fatal", "failed"),
        ("missing", "failed"),
        ("nonzero", "failed"),
        ("exit_only", "failed"),
    ],
)
def test_p0000_openfoam_run_collect_post(tmp_path, monkeypatch, mode, expected):
    monkeypatch.setenv("FAKE_OPENFOAM_MODE", mode)
    plan = case(tmp_path, sweep="4")
    assert matrix(["run", "study", "--workspace", str(tmp_path)]) == (
        2 if expected == "failed" else 0
    )
    result = api().collect(point(plan))
    assert result.classification == expected
    if mode != "missing":
        assert result.coefficients["Cd"].value == 0.031
        assert result.coefficients["Cl"].value == 0.47
        assert result.coefficients["CmPitch"].value == -0.025
        assert result.coefficients["Cl"].axis == "y"
        assert result.coefficients["CmPitch"].axis == "z"
    assert result.references.area == 0.1
    assert result.references.length == 1
    assert result.references.moment_origin == (0.25, 0, 0)
    assert result.solver_version == "v2412"
    folder = tmp_path / plan.points[0].directory
    assert all((folder / name).is_file() for name in CASE_FILES)
    assert (folder / "mesh.msh").read_bytes() == (tmp_path / "inputs/meshes/tiny.msh").read_bytes()
    assert matrix(["collect", "study", "--workspace", str(tmp_path)]) == (
        1 if expected == "failed" else 0
    )
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert summary["solver"] == "openfoam"
    assert summary["points"][0]["result"] == json.loads(json.dumps(result.to_dict()))
    with pytest.raises(ValueError, match="existing"):
        api().execute(point(plan), check_ram=lambda: None)


def test_p0000_openfoam_reader_column_mutant(tmp_path):
    plan = case(tmp_path)
    adapter = api()
    adapter.execute(point(plan), check_ram=lambda: None)
    reader = __import__("cfdbender.solvers.openfoam._history", fromlist=["read"])
    source = Path(reader.__file__).read_text(encoding="utf-8")
    assert 'row["Cl"]' in source, "mutation must touch the coefficient reader"
    scratch = tmp_path / "mutant_reader.py"
    scratch.write_text(source.replace('row["Cl"]', 'row["Cd"]'), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("mutant_openfoam_reader", scratch)
    mutant = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mutant
    spec.loader.exec_module(mutant)
    args = (tmp_path / plan.points[0].directory, point(plan))

    def check(module):
        _, coeffs, _ = module.read(*args)
        assert coeffs["Cl"].value == 0.47

    check(reader)
    with pytest.raises(AssertionError):
        check(mutant)


def test_p0000_openfoam_launcher_and_environment(tmp_path, monkeypatch):
    api()
    from cfdbender.solvers.openfoam._execute import argv

    launcher = ["wsl", "-d", "Ubuntu", "--"]
    monkeypatch.delenv("CFDBENDER_OPENFOAM_EXE", raising=False)
    assert argv({"launcher": launcher}, tmp_path) == [*launcher, "bash", "Allrun"]
    monkeypatch.setenv("CFDBENDER_OPENFOAM_EXE", str(FAKE))
    assert argv({"launcher": launcher}, tmp_path) == [sys.executable, str(FAKE)]
    monkeypatch.setenv("CFDBENDER_OPENFOAM_BASHRC", "/opt/OpenFOAM/etc/bashrc")
    files = split(api().translate(point(case(tmp_path))))
    assert "source /opt/OpenFOAM/etc/bashrc" in files["Allrun"]
    # ESI's etc/bashrc returns nonzero from internal probes, so errexit starts after it.
    allrun = files["Allrun"]
    assert allrun.index("source /opt/OpenFOAM/etc/bashrc") < allrun.index("set -e")
    # ESI v2412 freestreamPressure reads freestreamValue (FOAM FATAL IO ERROR without it).
    assert "type freestreamPressure; freestreamValue uniform 0;" in files["0/p"]
    # The kit mesh puts the first cell near y+ 25, so nut uses Spalding's law at the wall.
    assert "type nutUSpaldingWallFunction; value uniform 0;" in files["0/nut"]


def test_p0000_openfoam_laminar_and_guards(tmp_path, monkeypatch):
    plan = case(tmp_path, turbulence="laminar")
    files = split(api().translate(point(plan)))
    assert "0/nuTilda" not in files and "0/nut" not in files
    assert "simulationType laminar;" in files["constant/turbulenceProperties"]
    from cfdbender.solvers.su2 import _ram

    monkeypatch.setattr(_ram, "free_ram_gib", lambda: 0.49)
    with pytest.raises(ValueError, match="RAM"):
        api().execute(point(plan), check_ram=lambda: None)
    assert not (tmp_path / plan.points[0].directory).exists()


def test_p0000_openfoam_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_OPENFOAM_MODE", "timeout")
    plan = case(tmp_path, extra="timeout_s = 0.3\n")
    result = api().execute(point(plan), check_ram=lambda: None)
    assert api().classify(result) == "failed"
    assert result.exit_code == 124
    metadata = json.loads((tmp_path / plan.points[0].directory / "execution.json").read_bytes())
    assert metadata["timed_out"] is True


def test_p0000_openfoam_force_output_is_tracked(tmp_path):
    plan = case(tmp_path, sweep="4")
    coefficient = plan.points[0].directory + "/postProcessing/forceCoeffs/0/coefficient.dat"
    assert coefficient in plan.points[0].outputs, "force coefficients need output hash provenance"
    assert matrix(["run", "study", "--workspace", str(tmp_path)]) == 0
    with (tmp_path / coefficient).open("ab") as stream:
        stream.write(b"3 1 0 0 2 0 0 3 0 0 0 0 0\n")
    assert matrix(["collect", "study", "--workspace", str(tmp_path)]) == 1
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert "output changed" in summary["points"][0]["error"]


def test_p0000_openfoam_header_reordering_and_initial_residuals(tmp_path):
    plan = case(tmp_path)
    api().execute(point(plan), check_ram=lambda: None)
    folder = tmp_path / plan.points[0].directory
    coefficient = folder / "postProcessing/forceCoeffs/0/coefficient.dat"
    coefficient.write_text(
        "# Time CmPitch Cl Cd\n1 -0.02 0.41 0.04\n2 -0.025 0.47 0.031\n", encoding="ascii"
    )
    log = folder / "log.simpleFoam"
    original = log.read_text()
    assert api().collect(point(plan)).coefficients["Cl"].value == 0.47
    log.write_text(
        original.replace("SIMPLE solution converged in 2 iterations", "End"), encoding="ascii"
    )
    assert api().collect(point(plan)).classification == "failed", (
        "small final residuals do not count"
    )
    low_initial = log.read_text().replace("Initial residual = 0.01", "Initial residual = 1e-8")
    log.write_text(low_initial, encoding="ascii")
    assert api().collect(point(plan)).classification == "converged"
    log.write_text(low_initial + "FOAM FATAL IO ERROR: planted\n", encoding="ascii")
    assert api().collect(point(plan)).classification == "failed"


def test_p0000_openfoam_esi_pitch_reference_direction(tmp_path):
    plan = case(tmp_path, sweep="4")
    assert "pitchAxis (0 0 -1);" in api().translate(point(plan))
    api().execute(point(plan), check_ram=lambda: None)
    result = api().collect(point(plan))
    assert result.references.axis_directions[2] == "body -z pitch (ESI liftDir cross dragDir)"
    assert result.coefficients["CmPitch"].value == -0.025, "preserve the solver coefficient"

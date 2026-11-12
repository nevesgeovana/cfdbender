"""P0000: initialize, plan, run and collect a PyFR workspace offline."""

import json
import math

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.cli import main as workspace
from cfdbender.workspace.planning import plan_workspace


def api():
    assert "pyfr" in available_solvers()
    return load_solver("pyfr")


def case(root):
    api()
    assert workspace(["init", "--solver", "pyfr", "--workspace", str(root)]) == 0
    standin = root / "standin.py"
    standin.write_text(
        "import pathlib,sys\n"
        "if 'import' in sys.argv: pathlib.Path('mesh.pyfrm').write_text('mesh')\n"
        "else: pathlib.Path('forces.csv').write_text('t,px,py,vx,vy\\n'"
        " + '0,0.1,0.3,0,0\\n1,0.1,0.3,0,0\\n2,0.1,0.3,0,0\\n3,0.1,0.3,0,0\\n')\n",
        encoding="ascii",
    )
    files = {
        "inputs/setups/s001.toml": (
            'mesh = "inputs/meshes/tiny.msh"\nmach = 0.2\nreynolds = 10000\n'
            "order = 3\ntend = 3\nwindow_start = 0\n[solver]\nexecutable = "
            + json.dumps(standin.as_posix())
            + "\n"
        ),
        "inputs/references/r001.toml": "",
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": (
            "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION\n1001|1|001|001|001|ALPHA:4\n"
        ),
        "inputs/meshes/tiny.msh": "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii")
    return plan_workspace(root, "study")


def test_p0000_pyfr_plan_twice(tmp_path):
    plan = case(tmp_path)
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    text = api().translate(point)
    assert text == api().translate(point)
    assert "[solver-elements-quad]" in text and "[solver-elements-tri]" in text
    assert "quad-deg = 6" in text and "mu = 0.0001" in text
    assert "pinf = 17.8571428571429" in text
    assert "case.ini" in " ".join(plan.points[0].outputs)
    assert not (tmp_path / plan.points[0].directory).exists()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("normal", "converged"),
        ("short", "iteration_limit"),
        ("nan", "failed"),
        ("nonzero", "failed"),
    ],
)
def test_p0000_pyfr_run_collect(tmp_path, monkeypatch, mode, expected):
    plan = case(tmp_path)
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    adapter = api()
    result = adapter.execute(point, check_ram=lambda: None)
    folder = tmp_path / plan.points[0].directory
    if mode == "short":
        (folder / "forces.csv").write_text("t,px,py,vx,vy\n0,0.1,0.3,0,0\n", encoding="ascii")
    elif mode == "nan":
        (folder / "forces.csv").write_text("t,px,py,vx,vy\n0,nan,0.3,0,0\n", encoding="ascii")
    elif mode == "nonzero":
        metadata = json.loads((folder / "execution.json").read_text())
        metadata["exit_code"] = 1
        (folder / "execution.json").write_text(json.dumps(metadata))
    result = adapter.collect(point)
    assert result.classification == expected
    if mode == "normal":
        alpha = math.radians(4)
        assert result.coefficients["Cl"].value == pytest.approx(
            2 * (0.3 * math.cos(alpha) - 0.1 * math.sin(alpha))
        )
        assert result.coefficients["Cd"].value == pytest.approx(
            2 * (0.1 * math.cos(alpha) + 0.3 * math.sin(alpha))
        )
        assert (folder / "mesh.msh").read_bytes() == (
            tmp_path / "inputs/meshes/tiny.msh"
        ).read_bytes()
        assert (folder / "case.ini").read_text() == adapter.translate(point)


@pytest.mark.parametrize(
    "second_py,expected",
    [("0.3", "converged"), ("0.30001", "converged"), ("0.4", "iteration_limit")],
)
def test_p0000_pyfr_window_halves_decide_stationarity(tmp_path, second_py, expected):
    """The two window halves must agree within the tolerance; written without a run."""
    plan = case(tmp_path)
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    folder = tmp_path / plan.points[0].directory
    folder.mkdir(parents=True)
    (folder / "execution.json").write_text(json.dumps({"exit_code": 0}), encoding="ascii")
    (folder / "forces.csv").write_text(
        "t,px,py,vx,vy\n0,0.1,0.3,0,0\n1,0.1,0.3,0,0\n"
        f"2,0.1,{second_py},0,0\n3,0.1,{second_py},0,0\n",
        encoding="ascii",
    )
    assert api().collect(point).classification == expected


def test_p0000_pyfr_mutant_axes(tmp_path):
    plan = case(tmp_path)
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    api().execute(point, check_ram=lambda: None)
    folder = tmp_path / plan.points[0].directory
    original = (folder / "forces.csv").read_text()
    good = api().collect(point).coefficients["Cl"].value
    (folder / "forces.csv").write_text(original.replace("t,px,py,vx,vy", "t,py,px,vx,vy"))
    with pytest.raises(AssertionError):
        assert api().collect(point).coefficients["Cl"].value == pytest.approx(good)

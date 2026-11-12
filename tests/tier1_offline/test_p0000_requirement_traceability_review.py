"""Check traceability for the OpenFOAM references and XFOIL classification."""

from pathlib import Path


def test_register_cites_reference_and_positive_classification_checks():
    register = Path(__file__).resolve().parents[2] / "docs/unpublished/requirements/register.md"
    rows = {
        line.split("|")[1].strip(): line
        for line in register.read_text(encoding="utf-8").splitlines()
        if line.startswith("| FR-")
    }
    assert "test_openfoam_launcher_executes_in_point_directory" in rows["FR-37"]
    assert "test_p0000_openfoam_run_collect_post" in rows["FR-39"]
    assert "test_p0000_xfoil_run_collect_post_sweep" in rows["FR-43"]
    assert "test_p0000_xfoil_classification_does_not_use_iteration_limit" in rows["FR-43"]


def test_openfoam_launcher_executes_in_point_directory(tmp_path, monkeypatch):
    import json
    import sys

    from test_p0000_openfoam_adapter import FAKE, api, case, point

    from cfdbender.workspace.planning import plan_workspace

    monkeypatch.delenv("CFDBENDER_OPENFOAM_EXE", raising=False)
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import pathlib, runpy, sys\n"
        "pathlib.Path('launcher.json').write_text(__import__('json').dumps(sys.argv[1:]))\n"
        f"runpy.run_path({str(FAKE)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    plan = case(tmp_path / "campaign", sweep="4")
    setup = plan.root / "inputs/setups/s001.toml"
    text = setup.read_text(encoding="utf-8")
    text = text.replace(
        "executable = " + json.dumps(FAKE.as_posix()),
        "launcher = " + json.dumps([sys.executable, str(launcher)]),
    )
    setup.write_text(text, encoding="utf-8")
    plan = plan_workspace(plan.root, "study")
    result = api().execute(point(plan), check_ram=lambda: None)
    assert result.exit_code == 0
    folder = plan.root / plan.points[0].directory
    assert json.loads((folder / "launcher.json").read_text()) == ["bash", "Allrun"]

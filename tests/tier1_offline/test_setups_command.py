"""cfd-workspace setups: ready s files from the templates a solver ships as data (S7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli_route import main as routed
from test_p0000_item3c_adapter import workspace as mock_workspace

from cfdbender.core import setup_templates
from cfdbender.workspace.cli import build_parser, main

SRC = Path(__file__).resolve().parents[2] / "src" / "cfdbender" / "solvers"
SHIPPED = sorted(
    (path.parent.parent.name, path.stem) for path in SRC.glob("*/setup_templates/*.toml")
)


def init(root: Path, solver: str = "mock") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "init.toml").write_text(f'schema_version = 1\nsolver = "{solver}"\n', encoding="ascii")
    return root


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def fake_solvers(tmp_path: Path, monkeypatch, files: dict[str, str]) -> None:
    """Point the registry at a folder of template files, as if a solver shipped them."""
    for name, text in files.items():
        path = tmp_path / "pkg" / "solvers" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    monkeypatch.setattr(setup_templates.resources, "files", lambda _package: tmp_path / "pkg")


def test_list_and_bare_setups_print_names_and_descriptions(tmp_path, capsys):
    init(tmp_path)
    listed = run(capsys, "setups", "--list", "--workspace", str(tmp_path))
    bare = run(capsys, "setups", "--workspace", str(tmp_path))
    assert listed[:2] == bare[:2]
    code, out, _ = listed
    assert code == 0
    description = setup_templates.describe("mock", "converged")
    assert out.splitlines() == [f"converged  {description}"]
    assert not (tmp_path / "inputs").exists()


def test_unknown_template_is_refused_listing_the_available_ones(tmp_path, capsys):
    init(tmp_path)
    code, out, err = run(capsys, "setups", "--template", "supersonic", "--workspace", str(tmp_path))
    assert code == 2
    assert "unknown setup template 'supersonic' for mock; available: converged" in err
    assert out == "" and not (tmp_path / "inputs").exists()


def test_template_writes_the_next_free_setup_and_names_the_next_step(tmp_path, capsys):
    init(tmp_path)
    code, out, _ = run(capsys, "setups", "--template", "converged", "--workspace", str(tmp_path))
    assert code == 0
    assert "inputs/setups/s001.toml" in out and "cfd-matrix plan" in out
    code, out, _ = run(capsys, "setups", "--template", "converged", "--workspace", str(tmp_path))
    assert code == 0 and "inputs/setups/s002.toml" in out
    (tmp_path / "inputs/setups/s007.toml").write_text("x = 1\n", encoding="ascii")
    run(capsys, "setups", "--template", "converged", "--workspace", str(tmp_path))
    assert (tmp_path / "inputs/setups/s008.toml").is_file()


def test_id_is_honoured_and_checked(tmp_path, capsys):
    init(tmp_path)
    args = ("setups", "--template", "converged", "--workspace", str(tmp_path))
    assert run(capsys, *args, "--id", "042")[0] == 0
    assert (tmp_path / "inputs/setups/s042.toml").is_file()
    code, _, err = run(capsys, *args, "--id", "42")
    assert code == 2 and "three digits".replace("three", "3") in err
    assert run(capsys, "setups", "--id", "005", "--workspace", str(tmp_path))[0] == 2


def test_an_existing_file_is_refused_and_left_unchanged(tmp_path, capsys):
    init(tmp_path)
    existing = tmp_path / "inputs/setups/s003.toml"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"mine = 1\r\n")
    code, _, err = run(
        capsys, "setups", "--template", "converged", "--id", "003", "--workspace", str(tmp_path)
    )
    assert code == 2 and "inputs/setups/s003.toml already exists" in err
    assert existing.read_bytes() == b"mine = 1\r\n"


def test_workspace_option_and_current_folder_agree(tmp_path, capsys, monkeypatch):
    init(tmp_path / "a")
    init(tmp_path / "b")
    run(capsys, "setups", "--template", "converged", "--workspace", str(tmp_path / "a"))
    monkeypatch.chdir(tmp_path / "b")
    assert run(capsys, "setups", "--template", "converged")[0] == 0
    first = (tmp_path / "a/inputs/setups/s001.toml").read_bytes()
    assert (tmp_path / "b/inputs/setups/s001.toml").read_bytes() == first


def test_a_folder_that_is_not_a_workspace_is_refused(tmp_path, capsys):
    code, _, err = run(capsys, "setups", "--workspace", str(tmp_path))
    assert code == 2 and "not a cfdbender workspace" in err


def test_a_solver_without_templates_answers_with_one_sentence(tmp_path, capsys):
    init(tmp_path, "su2")
    code, out, _ = run(capsys, "setups", "--list", "--workspace", str(tmp_path))
    assert (code, out) == (0, "no setup templates for su2\n")
    code, out, _ = run(capsys, "setups", "--workspace", str(tmp_path))
    assert (code, out) == (0, "no setup templates for su2\n")
    code, _, err = run(capsys, "setups", "--template", "any", "--workspace", str(tmp_path))
    assert code == 2 and "available: none" in err


def test_a_flightstream_workspace_is_answered_here_not_forwarded(tmp_path, capsys):
    init(tmp_path, "flightstream")
    code, out, _ = run(capsys, "setups", "--workspace", str(tmp_path))
    assert (code, out) == (0, "no setup templates for flightstream\n")


def test_a_template_file_added_to_the_folder_appears_with_no_code_change(
    tmp_path, capsys, monkeypatch
):
    fake_solvers(
        tmp_path,
        monkeypatch,
        {
            "mock/setup_templates/alpha.toml": "# First one\nx = 1\n",
            "mock/setup_templates/notes.txt": "not a template",
        },
    )
    root = init(tmp_path / "ws")
    assert run(capsys, "setups", "--workspace", str(root))[1] == "alpha  First one\n"
    fake_solvers(tmp_path, monkeypatch, {"mock/setup_templates/beta.toml": "# Second\ny = 2\n"})
    out = run(capsys, "setups", "--workspace", str(root))[1]
    assert out.splitlines() == ["alpha  First one", "beta   Second"]


def test_written_bytes_equal_the_data_file(tmp_path, capsys, monkeypatch):
    raw = "# Windows endings, kept\r\nx = 1\r\n"
    fake_solvers(tmp_path, monkeypatch, {"mock/setup_templates/crlf.toml": raw})
    root = init(tmp_path / "ws")
    assert run(capsys, "setups", "--template", "crlf", "--workspace", str(root))[0] == 0
    assert (root / "inputs/setups/s001.toml").read_bytes() == raw.encode()
    shipped = (SRC / "mock/setup_templates/converged.toml").read_bytes()
    monkeypatch.undo()
    init(tmp_path / "ws2")
    run(capsys, "setups", "--template", "converged", "--workspace", str(tmp_path / "ws2"))
    assert (tmp_path / "ws2/inputs/setups/s001.toml").read_bytes() == shipped


def test_a_template_without_a_description_line_is_refused(tmp_path, capsys, monkeypatch):
    fake_solvers(tmp_path, monkeypatch, {"mock/setup_templates/bare.toml": "x = 1\n"})
    root = init(tmp_path / "ws")
    code, _, err = run(capsys, "setups", "--workspace", str(root))
    assert code == 2 and "must start with a comment line" in err


def test_the_cfd_matrix_program_names_the_one_that_owns_setups(capsys):
    from cfdbender.matrix.cli import main as matrix_main

    assert matrix_main(["setups", "--list"]) == 2
    err = capsys.readouterr().err
    assert "cfd-matrix setups moved" in err and "cfd-workspace setups --list" in err
    assert routed(["setups", "--list", "--workspace", "no-such-folder"]) == 2


def test_the_verb_is_in_the_workspace_help(capsys):
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["--help"])
    assert stop.value.code == 0
    assert "setups" in capsys.readouterr().out


def planned_mock(root: Path, template: str, monkeypatch) -> None:
    mock_workspace(root)
    (root / "inputs/setups/s001.toml").unlink()
    assert main(["setups", "--template", template, "--id", "001", "--workspace", str(root)]) == 0


def planned_cfdpp(root: Path, template: str, monkeypatch) -> None:
    """A CFD++ workspace on the real database of the supported version, synthetic grid."""
    from cfdpp_workspace import matrix
    from cfdpp_workspace import workspace as cfdpp_workspace

    cfdpp_workspace(root, table=matrix(build="22.5.U3"), executables=False)
    (root / "inputs/setups/s001.toml").unlink()
    assert main(["setups", "--template", template, "--id", "001", "--workspace", str(root)]) == 0


#: How to build a synthetic workspace around a shipped template, by solver. A solver that ships
#: a template and has no builder fails the test below, so a template is never left unplanned.
BUILDERS = {"mock": planned_mock, "cfdpp": planned_cfdpp}


def test_every_shipped_template_is_found_by_the_registry():
    assert SHIPPED, "no template ships"
    for solver, name in SHIPPED:
        assert name in setup_templates.available(solver)
        assert setup_templates.describe(solver, name)


@pytest.mark.parametrize(("solver", "name"), SHIPPED)
def test_every_shipped_template_plans_without_refusal(solver, name, tmp_path, capsys, monkeypatch):
    assert solver in BUILDERS, f"add a synthetic-workspace builder for {solver} to BUILDERS"
    BUILDERS[solver](tmp_path, name, monkeypatch)
    capsys.readouterr()
    code = routed(["plan", "study", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.err

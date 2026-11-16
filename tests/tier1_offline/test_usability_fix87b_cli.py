"""U-26 to U-29: what init says, what --help says, archive and free-space --list."""

from __future__ import annotations

import pytest
from cli_route import main as matrix_main
from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, run

from cfdbender.workspace.cli import main as workspace_main


def init(capsys, *argv):
    code = workspace_main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_u26_init_on_a_new_folder_lists_what_it_wrote_and_the_next_step(tmp_path, capsys):
    code, out, _ = init(capsys, "init", tmp_path / "w", "--solver", "su2")
    assert code == 0
    assert "kept untouched" not in out
    assert "inputs/INPUTS.md" in out and "inputs/.gitignore" in out
    assert "next: write inputs/matrices/<name>.fs, then cfd-matrix plan <name>" in out


def test_u26_init_on_an_existing_folder_says_what_was_kept(tmp_path, capsys):
    (tmp_path / "w" / "inputs").mkdir(parents=True)
    code, out, _ = init(capsys, "init", tmp_path / "w", "--solver", "su2")
    assert code == 0 and "existing folders and files were kept untouched" in out


def test_u26_unknown_solver_names_the_choice_not_a_file(tmp_path, capsys):
    code, _, err = init(capsys, "init", tmp_path / "x1", "--solver", "fluent")
    assert code == 2
    assert "--solver fluent: unknown solver" in err and "init.toml" not in err
    assert "su2" in err


def test_u26_a_solver_name_in_the_wrong_case_is_accepted(tmp_path, capsys):
    code, out, _ = init(capsys, "init", tmp_path / "w", "--solver", "SU2")
    assert code == 0
    assert 'solver = "su2"' in (tmp_path / "w" / "init.toml").read_text(encoding="utf-8")


def test_u27_the_matrix_help_has_a_description_and_the_first_campaign(capsys):
    with pytest.raises(SystemExit):
        matrix_main(["--help"])
    out = capsys.readouterr().out
    assert "first campaign" in out and "cfd-matrix plan <matrix>" in out
    assert "cfd-matrix acts on a matrix" in out and "cfd-workspace acts on" in out


def test_u27_shared_options_read_the_same_everywhere(capsys):
    texts = {}
    for verb in ("plan", "run", "collect", "post", "status", "log", "show"):
        main = workspace_main if verb in ("log", "show") else matrix_main
        with pytest.raises(SystemExit):
            main([verb, "--help"])
        texts[verb] = " ".join(capsys.readouterr().out.split())
    for verb, text in texts.items():
        assert "campaign workspace (default: the current directory)" in text, verb
    assert "(default: .)" not in " ".join(texts.values())
    assert "keep work on THIS machine" not in texts["collect"]
    assert "--local" not in texts["collect"]
    for verb in ("plan", "run", "collect", "post", "status"):
        assert "only these simulations, by POL" in texts[verb], verb


def test_u28_archive_says_it_acts_at_once_and_what_it_moved(tmp_path, capsys):
    root = build(tmp_path / "mock", "mock")
    edit(root / "inputs/matrices/full.fs", "| 2h | 1.0 |", "| 2h | - |")
    run(capsys, "plan", "full", "--workspace", root)
    run(capsys, "run", "full", "--workspace", root, "--local")
    with pytest.raises(SystemExit):
        workspace_main(["archive", "--help"])
    assert "at once" in " ".join(capsys.readouterr().out.split())
    code, out, _ = init(capsys, "archive", "7001", "--workspace", root)
    assert code == 0 and "2 point(s)" in out and "archive/" in out
    assert str(root) not in out


def test_u29_free_space_list_alone_lists_the_recipes(tmp_path, capsys):
    root = build(tmp_path / "mock", "mock")
    code, out, _ = run(capsys, "free-space", "--list", "--workspace", root)
    assert code == 0 and "no recipe in inputs/management/" in out
    (root / "inputs/management").mkdir(exist_ok=True)
    (root / "inputs/management/clean.toml").write_text("", encoding="ascii")
    code, out, _ = run(capsys, "free-space", "--list", "--workspace", root)
    assert code == 0 and "clean" in out

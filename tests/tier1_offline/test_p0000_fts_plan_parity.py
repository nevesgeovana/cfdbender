"""P0000-FTS: init and plan delegate to the installed campaign engine.

Usage: cfd-workspace init --solver flightstream campaign
       cfd-matrix plan m --workspace campaign

Fixture inputs are adapted from pyflightstream's test_matrix_cli.py
make_planned_workspace and copied from its fixtures/workflow_rotor_matrix.fs.
Parser capture and recursive spelling inventory adapt scripts/check_parity.py.
"""

import argparse
import importlib
import io
import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import main as workspace_main


@pytest.fixture
def upstream():
    pytest.importorskip("pyflightstream", reason="install cfdbender[flightstream] for parity")
    return (
        importlib.import_module("pyflightstream.workspace.cli").main,
        importlib.import_module("pyflightstream.run.cli").main,
    )


def _call(main, argv):
    try:
        return main(argv)
    except SystemExit as error:
        return error.code


def _tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
        if path.relative_to(root).as_posix() != "init.toml"
    }


def _declare(root):
    (root / "init.toml").write_text('schema_version = 1\nsolver = "flightstream"\n')


def test_p0000_fts_init_parity(tmp_path, upstream):
    direct, delegated = tmp_path / "direct", tmp_path / "delegated"
    assert _call(upstream[0], ["init", str(direct)]) == 0
    assert _call(workspace_main, ["init", "--solver", "flightstream", str(delegated)]) == 0
    assert _tree(direct) == _tree(delegated)
    from cfdbender.core import init_workspace

    control = init_workspace(
        tmp_path / "control", solver="flightstream", available_solvers=("flightstream",)
    )
    assert (delegated / "init.toml").read_bytes() == (control / "init.toml").read_bytes()


def _fixture(root, workspace_cli):
    # The pyflightstream tier-1 matrix is bundled under tests/fixtures/flightstream, so the
    # test runs against an installed (wheel) pyflightstream, which ships no test fixtures.
    bundled = Path(__file__).parents[1] / "fixtures/flightstream"
    fixtures = Path(os.environ.get("PYFLIGHTSTREAM_TEST_FIXTURES", bundled))
    source = fixtures / "workflow_rotor_matrix.fs"
    assert source.is_file(), f"bundled FlightStream fixture missing: {source}"
    assert _call(workspace_cli, ["init", str(root)]) == 0
    inputs = root / "inputs"
    (inputs / "references/r003.toml").write_text("area_m2 = 10.0\nchord_m = 1.2\nspan_m = 8.0\n")
    (inputs / "setups/s002.toml").write_text("iterations = 800\n")
    (inputs / "pproc/p001.toml").write_text('[groups]\n"1" = "wing_left"\n')
    with (inputs / "executables.toml").open("a") as stream:
        stream.write('"26.120" = "unavailable-offline-executable.exe"\n')
    shutil.copyfile(source, inputs / "matrices/m.fs")
    return root


def test_p0000_fts_plan_parity(tmp_path, upstream, monkeypatch):
    from pyflightstream.script import Script

    source = _fixture(tmp_path / "source", upstream[0])
    direct = tmp_path / "direct/campaign"
    delegated = tmp_path / "delegated/campaign"
    shutil.copytree(source, direct)
    shutil.copytree(source, delegated)
    _declare(delegated)
    renders = []
    original = Script.render

    def capture(script, *args, **kwargs):
        result = original(script, *args, **kwargs)
        renders.append(result.encode("utf-8"))
        return result

    monkeypatch.setattr(Script, "render", capture)
    assert _call(upstream[1], ["plan", "m", "--workspace", str(direct)]) == 0
    expected = list(renders)
    renders.clear()
    assert _call(matrix_main, ["plan", "m", "--workspace", str(delegated)]) == 0
    assert expected, "plan must actually render scripts"
    assert renders == expected
    print("Parity exclusions: [] (no absolute paths or timestamps excluded)")
    assert (direct / "post/m/plan.json").read_bytes() == (
        delegated / "post/m/plan.json"
    ).read_bytes()
    print(f"Byte-identical dry-run scripts: {len(expected)}; plan.json: identical")


class _CapturedError(Exception):
    def __init__(self, parser):
        self.parser = parser


def _capture_parser(main, monkeypatch):
    def grab(parser, *args, **kwargs):
        raise _CapturedError(parser)

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", grab)
        patch.setattr(argparse.ArgumentParser, "parse_known_args", grab)
        try:
            main(["--help"])
        except _CapturedError as caught:
            return caught.parser
    pytest.fail("CLI returned without parsing --help")


def _inventory(parser, path=()):
    spellings = set()
    parsers = {path: parser}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                child = (*path, name)
                spellings.add(child)
                more, children = _inventory(sub, child)
                spellings.update(more)
                parsers.update(children)
        else:
            spellings.update((*path, option) for option in action.option_strings)
            if action.choices is not None and not isinstance(action.choices, dict):
                key = action.option_strings[0] if action.option_strings else f"<{action.dest}>"
                spellings.update((*path, f"{key}={choice}") for choice in action.choices)
    return spellings, parsers


@pytest.mark.parametrize("kind", [0, 1], ids=["workspace", "matrix"])
def test_p0000_fts_all_help_and_spellings(tmp_path, upstream, monkeypatch, kind):
    _declare(tmp_path)
    monkeypatch.chdir(tmp_path)
    direct = upstream[kind]
    forwarded = (workspace_main, matrix_main)[kind]
    expected, parsers = _inventory(_capture_parser(direct, monkeypatch))
    actual, _ = _inventory(_capture_parser(forwarded, monkeypatch))
    assert expected == actual
    for path in parsers:
        for help_flag in ("-h", "--help"):
            outputs = []
            for main in (direct, forwarded):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    assert _call(main, [*path, help_flag]) == 0
                # Upstream stderr chooses a random decorative help footer.
                # The parser's actual help, on stdout, must be identical.
                outputs.append(out.getvalue())
            assert outputs[0] == outputs[1], path
    print(f"{('workspace', 'matrix')[kind]}: {len(parsers)} parsers, {len(expected)} spellings")


@pytest.mark.parametrize("kind", [0, 1], ids=["workspace", "matrix"])
@pytest.mark.parametrize("root_form", ["cwd", "option", "equals"])
def test_p0000_fts_argv_and_exit_status(tmp_path, upstream, monkeypatch, kind, root_form):
    _declare(tmp_path)
    if root_form == "cwd":
        monkeypatch.chdir(tmp_path)
        selection = []
    elif root_form == "option":
        selection = ["--workspace", str(tmp_path)]
    else:
        selection = [f"--workspace={tmp_path}"]
    argv = ["future-verb", "--future-flag", "untouched value", *selection]
    module = importlib.import_module(
        ("pyflightstream.workspace.cli", "pyflightstream.run.cli")[kind]
    )
    seen = []

    def stand_in(arguments):
        seen.append(arguments)
        return 7

    monkeypatch.setattr(module, "main", stand_in)
    assert (workspace_main, matrix_main)[kind](argv) == 7
    assert seen == [argv]
    assert seen[0] is argv


def test_p0000_fts_global_flag_before_positional_root(tmp_path, upstream, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    _declare(root)
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("pyflightstream.workspace.cli")
    seen = []
    monkeypatch.setattr(module, "main", lambda args: seen.append(args) or 7)
    argv = ["--warning-refusals", "migrate", str(root)]
    assert workspace_main(argv) == 7
    assert seen == [argv]
    assert seen[0] is argv


def test_p0000_fts_last_workspace_option_wins(tmp_path, upstream, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    _declare(root)
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("pyflightstream.run.cli")
    seen = []
    monkeypatch.setattr(module, "main", lambda args: seen.append(args) or 7)
    argv = ["plan", "m", "--workspace", str(tmp_path), "--workspace", str(root)]
    assert matrix_main(argv) == 7
    assert seen == [argv]
    assert seen[0] is argv


def test_p0000_fts_abbreviated_workspace_option(tmp_path, upstream, monkeypatch):
    root = tmp_path / "campaign"
    root.mkdir()
    _declare(root)
    monkeypatch.chdir(tmp_path)
    outputs = []
    for main in (upstream[1], matrix_main):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert _call(main, ["plan", "--work", str(root), "--help"]) == 0
        outputs.append(out.getvalue())
    assert outputs[0] == outputs[1]

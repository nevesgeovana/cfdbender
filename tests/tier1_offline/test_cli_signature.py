"""Every console command ends with one signature box on stderr, as pyflightstream's do.

One box per invocation, drawn from the pool of the outcome, on standard error; help and
version keep one short line; the return value, the exceptions, standard output and the exit
code are unchanged.
"""

from __future__ import annotations

import importlib
import io
import random
import sys
import tomllib
import types
from pathlib import Path

import pytest
from test_matrix_columns_all_solvers import build
from test_signature_module import BORDER, KOALA, SEES_YOU, drawn, outcome_of

from cfdbender import _cli, _signature

PROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
SCRIPTS = tomllib.loads(PROJECT.read_text(encoding="utf-8"))["project"]["scripts"]


def _main_of(entry: str):
    module, function = entry.split(":", 1)
    return getattr(importlib.import_module(module), function)


def test_every_console_script_is_declared_and_wrapped():
    assert set(SCRIPTS) == {"cfd-workspace", "cfd-matrix", "cfd-mesh", "cfd-geom"}
    for entry in SCRIPTS.values():
        assert hasattr(_main_of(entry), "__wrapped__"), entry


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_help_of_every_console_script_has_one_short_line_and_keeps_stdout(script, capsys):
    main = _main_of(SCRIPTS[script])
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    streams = capsys.readouterr()
    assert "usage:" in streams.out
    assert SEES_YOU not in streams.out
    assert streams.err.count(SEES_YOU) == 1
    assert BORDER not in streams.err, "help stays compact: no box"
    assert len(streams.err.strip().splitlines()) == 1
    assert "help" in streams.err.lower()


def test_every_outcome_signs_once_on_stderr_with_the_fixed_phrase(capsys):
    def returns(code):
        return lambda argv: code

    def exits(code):
        def command(argv):
            raise SystemExit(code)

        return command

    def interrupted(argv):
        raise KeyboardInterrupt

    paths = {
        "success": (returns(0), [], None),
        "failed": (returns(1), [], None),
        "help": (exits(0), ["--help"], SystemExit),
        "version": (exits(0), ["--version"], SystemExit),
        "cancelled": (interrupted, [], KeyboardInterrupt),
    }
    for outcome, (body, argv, raised) in paths.items():
        command = _cli.cli_entrypoint(body)
        if raised is None:
            command(argv)
        else:
            with pytest.raises(raised):
                command(argv)
        streams = capsys.readouterr()
        assert streams.out == "", outcome
        assert streams.err.count(SEES_YOU) == 1, outcome
        if outcome in ("help", "version"):
            assert streams.err.endswith(f" {SEES_YOU}\n"), outcome
            assert len(streams.err.splitlines()) == 1, outcome
        else:
            assert streams.err.startswith("\n" + BORDER + "\n"), outcome
            assert streams.err.endswith("\n" + BORDER + "\n\n"), outcome
            assert outcome_of(streams.err) == outcome


def test_a_nested_entry_point_signs_only_once(capsys):
    inner = _cli.cli_entrypoint(lambda argv: 0)
    outer = _cli.cli_entrypoint(lambda argv: inner(argv))
    assert outer([]) == 0
    assert capsys.readouterr().err.count(SEES_YOU) == 1


def test_a_forwarded_invocation_is_signed_by_the_forwarded_program_only(capsys):
    def forwards(argv):
        _cli.note_forwarded()
        return 0

    assert _cli.cli_entrypoint(forwards)([]) == 0
    assert capsys.readouterr().err == ""


def test_every_drawing_and_every_phrase_is_reachable(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "_RNG", random.Random(20260928))
    reached = set()
    bodies = {
        "success": lambda argv: 0,
        "failed": lambda argv: 1,
        "cancelled": lambda argv: (_ for _ in ()).throw(KeyboardInterrupt),
    }
    for body in bodies.values():
        command = _cli.cli_entrypoint(body)
        for _ in range(400):
            try:
                command([])
            except KeyboardInterrupt:
                pass
            found = drawn(capsys.readouterr().err)
            assert found is not None
            reached.add(found)
    expected = {
        (name, phrase)
        for outcome in ("success", "failed", "cancelled")
        for name in _signature.OUTCOME_DRAWINGS[outcome]
        for phrase in _signature.PHRASES[name]
    }
    assert reached == expected, sorted(expected - reached)


def test_invalid_arguments_are_not_congratulated(capsys):
    from cfdbender.matrix.cli import main

    with pytest.raises(SystemExit) as exited:
        main(["--not-a-command"])
    assert exited.value.code == 2
    streams = capsys.readouterr()
    assert streams.err.count(SEES_YOU) == 1
    assert outcome_of(streams.err) == "failed"


def test_a_refusal_keeps_its_code_its_text_and_signs_failed(tmp_path, capsys):
    from cfdbender.matrix.cli import main

    assert main(["run", "full", "--workspace", str(tmp_path)]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "matrix not run" in streams.err
    assert streams.err.count(SEES_YOU) == 1
    assert outcome_of(streams.err) == "failed"


def test_a_closed_or_unencodable_stderr_never_changes_the_result(monkeypatch):
    closed = io.StringIO()
    closed.close()
    ascii_only = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
    for stream in (closed, ascii_only):
        monkeypatch.setattr(sys, "stderr", stream)
        assert _cli.cli_entrypoint(lambda argv: 4)([]) == 4
        assert _cli.cli_entrypoint(lambda argv: 0)([]) == 0
    monkeypatch.setattr(_signature, "SEES_YOU", "geoversegoddess sees you é")
    monkeypatch.setattr(sys, "stderr", ascii_only)
    with pytest.raises(SystemExit) as exited:
        _cli.cli_entrypoint(lambda argv: (_ for _ in ()).throw(SystemExit(0)))(["--help"])
    assert exited.value.code == 0
    monkeypatch.setattr(sys, "stderr", None)
    assert _cli.cli_entrypoint(lambda argv: 5)([]) == 5


def test_an_unencodable_box_is_written_with_replacement_marks(monkeypatch):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(raw, encoding="ascii", errors="strict"))
    monkeypatch.setattr(_signature, "SEES_YOU", "geoversegoddess sees you é")
    assert _cli.cli_entrypoint(lambda argv: 0)([]) == 0
    assert b"sees you ?" in raw.getvalue()


def _mock_workspace(tmp_path):
    root = build(tmp_path, "mock")
    (root / "runs.json").write_text('{"schema_version": 1, "records": []}', encoding="utf-8")
    return root


def test_post_success_is_signed_with_the_koala(monkeypatch, tmp_path, capsys):
    from cfdbender.matrix import cli

    root = _mock_workspace(tmp_path)
    report = types.SimpleNamespace(
        summaries=(), tables=(), collected=0, failed=0, outstanding=0, exit_code=0
    )
    monkeypatch.setattr(cli, "post_workspace", lambda *a, **k: report)
    assert cli.main(["post", "--workspace", str(root)]) == 0
    assert KOALA in capsys.readouterr().err
    report.outstanding = 2
    assert cli.main(["post", "--workspace", str(root)]) == 3
    err = capsys.readouterr().err
    assert KOALA not in err
    assert outcome_of(err) == "failed"


def test_the_post_a_collect_runs_is_signed_with_the_koala(monkeypatch, tmp_path, capsys):
    from cfdbender.matrix import cli

    root = _mock_workspace(tmp_path)
    report = types.SimpleNamespace(
        summaries=(),
        tables=(),
        strays={},
        collected=1,
        failed=0,
        outstanding=0,
        exit_code=0,
    )
    monkeypatch.setattr(cli, "collect_workspace", lambda *a, **k: report)
    assert cli.main(["collect", "--workspace", str(root)]) == 0
    assert KOALA in capsys.readouterr().err


def test_success_phrases_vary_without_changing_stdout_or_status(monkeypatch, tmp_path, capsys):
    from cfdbender.matrix import cli

    root = _mock_workspace(tmp_path)
    monkeypatch.setattr(_cli, "_RNG", random.Random(7))
    seen = set()
    outs = set()
    for _ in range(20):
        assert cli.main(["status", "--workspace", str(root)]) == 0
        streams = capsys.readouterr()
        outs.add(streams.out)
        assert streams.err.count(SEES_YOU) == 1
        assert outcome_of(streams.err) == "success"
        seen.add(drawn(streams.err))
    assert len(seen) >= 2
    assert len(outs) == 1

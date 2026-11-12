"""Exercise refusal cases against real inputs and a real Git history."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def guard():
    spec = importlib.util.spec_from_file_location(
        "readiness_review", ROOT / "tools/check_public_readiness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["license", "copyright", "en", "em", "output", "private"])
def test_refusal_arms(kind):
    files = {
        "LICENSE": b"Copyright (c) 2026 Jane Doe\n",
        "CITATION.cff": b"authors:\n  - given-names: Jane\n    family-names: Doe\n",
        ".github/workflows/ci.yml": b"name: CI\n",
    }
    if kind == "license":
        del files["LICENSE"]
    elif kind == "copyright":
        files["LICENSE"] = b"Copyright (c) 2026 Wrong Person\n"
    elif kind in {"en", "em"}:
        files["bad.txt"] = ("\u2013" if kind == "en" else "\u2014").encode()
    elif kind == "output":
        files["case.vtu"] = b"output"
    else:
        files["bad.txt"] = b"HIDDENWORD"
    findings = guard().inspect_files(files, {"hiddenword"})
    expected = {
        "license": "missing LICENSE",
        "copyright": "copyright",
        "en": "dash:",
        "em": "dash:",
        "output": "solver or mesh output:",
        "private": "private term",
    }
    assert any(expected[kind] in finding for finding in findings)


def test_vocabulary_refusals(tmp_path):
    module = guard()
    with pytest.raises(OSError):
        module.terms(tmp_path / "missing.json")
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        module.terms(empty)


def test_old_commit_refused(tmp_path, monkeypatch, capsys):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    for name, data in {
        "LICENSE": "Copyright (c) 2026 Jane Doe\n",
        "CITATION.cff": "authors:\n  - given-names: Jane\n    family-names: Doe\n",
        ".github/workflows/ci.yml": "name: CI\n",
        "old.txt": "HIDDENWORD",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
    git(
        "add",
        *(
            str(tmp_path / name)
            for name in ("LICENSE", "CITATION.cff", ".github/workflows/ci.yml", "old.txt")
        ),
    )
    git("commit", "-m", "old")
    (tmp_path / "old.txt").write_text("safe", encoding="utf-8")
    git("add", str(tmp_path / "old.txt"))
    git("commit", "-m", "clean")
    vocabulary = tmp_path / "vocabulary.json"
    vocabulary.write_text('["hiddenword"]', encoding="utf-8")
    module = guard()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check", "--vocabulary", str(vocabulary)])
    assert module.main() == 1
    assert "private term in history:" in capsys.readouterr().out

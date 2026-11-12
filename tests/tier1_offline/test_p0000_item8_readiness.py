"""P0000 item 8: numbered, private readiness findings."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_guard():
    spec = importlib.util.spec_from_file_location(
        "item8_guard", ROOT / "tools/check_public_readiness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0000_item8_order(tmp_path):
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"a": ["ZuluUnique", {"x": "BetaUnique"}], "b": "ZULUUNIQUE"}))
    assert list(load_guard().terms(path)) == ["zuluunique", "betaunique"]


def test_p0000_item8_order_documented():
    doc = load_guard().original_terms.__doc__
    assert all(phrase in doc for phrase in ("starts at 1", "object member order", "four"))


@pytest.mark.parametrize("allowed,expected", [([], 1), (["1", "2"], 0), (["1"], 1)])
def test_p0000_item8_output(tmp_path, monkeypatch, capsys, allowed, expected):
    guard = load_guard()
    path = tmp_path / "fake.json"
    words = ["ZuluUnique", "BetaUnique2"]
    path.write_text(json.dumps(words))
    files = {
        "LICENSE": b"Copyright (c) 2026 Jane Doe",
        "CITATION.cff": b"authors:\n  - given-names: Jane\n    family-names: Doe",
        ".github/workflows/ci.yml": b"CI",
        "test.txt": b"ZuluUnique BetaUnique2",
    }
    for name, data in files.items():
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def git(*args):
        return {
            "ls-files": b"\0".join(n.encode() for n in files) + b"\0",
            "rev-list": b"abc123\n",
            "show": b"ZuluUnique",
            "ls-tree": b"100644 blob deadbeef\told.txt\0",
            "cat-file": b"ZuluUnique BetaUnique2",
        }[args[0]]

    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "git", git)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", "--vocabulary", str(path)] + [x for n in allowed for x in ("--allow-term", n)],
    )
    assert guard.main() == expected
    output = capsys.readouterr()
    text = output.out + output.err
    assert all(word.casefold() not in text.casefold() for word in words)
    assert "term #1" in text and "term #2" in text
    assert "tracked files: 1" in text and "commits: 1" in text
    assert "metadata commits: 1" in text and "contains digits" in text
    assert "substring" in text.casefold() and "false positive" in text.casefold()
    assert "substring" in guard.__doc__ and "false positive" in guard.__doc__
    if allowed:
        assert "allowed" in text


def test_p0000_item8_allow_refusals(tmp_path, monkeypatch, capsys):
    guard = load_guard()
    monkeypatch.setattr(guard, "ROOT", tmp_path / "repo")
    vocab = tmp_path / "fake.json"
    vocab.write_text('["DistinctiveFake"]')
    allow = tmp_path / "allow.json"
    allow.write_text("[2]")
    monkeypatch.setattr(
        sys, "argv", ["check", "--vocabulary", str(vocab), "--allow-term-file", str(allow)]
    )
    assert guard.main() == 2
    assert "unknown" in capsys.readouterr().err
    vocab.write_text("[]")
    assert guard.main() == 2
    vocab.unlink()
    assert guard.main() == 2


def test_p0000_item8_shapes_and_allow_file(tmp_path, monkeypatch, capsys):
    guard = load_guard()
    repo = tmp_path / "repo"
    repo.mkdir()
    vocab = tmp_path / "vocab.json"
    words = ["Secret Path", "Folder/Unique", "Number987"]
    vocab.write_text(json.dumps(words))
    allow = tmp_path / "allow.json"
    allow.write_text("[1, 2, 3]")
    files = {
        "LICENSE": b"Copyright (c) 2026 Jane Doe",
        "CITATION.cff": b"authors:\n  - given-names: Jane\n    family-names: Doe",
        ".github/workflows/ci.yml": b"CI",
        "Secret Path": b"Number987",
    }
    for name, data in files.items():
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def git(*args):
        return {
            "ls-files": b"\0".join(n.encode() for n in files) + b"\0",
            "rev-list": b"abc123\n",
            "show": b"safe",
            "ls-tree": b"100644 blob deadbeef\tFolder/Unique\0",
            "cat-file": b"safe",
        }[args[0]]

    monkeypatch.setattr(guard, "ROOT", repo)
    monkeypatch.setattr(guard, "git", git)
    monkeypatch.setattr(
        sys, "argv", ["check", "--vocabulary", str(vocab), "--allow-term-file", str(allow)]
    )
    assert guard.main() == 0
    output = capsys.readouterr()
    text = output.out + output.err
    assert all(word.casefold() not in text.casefold() for word in words)
    assert "contains spaces" in text and "contains path separators" in text
    assert text.count("allowed") == 3

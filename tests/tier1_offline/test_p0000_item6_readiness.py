"""P0000: public readiness guard and its refusal mutations."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/check_public_readiness.py"


def load_guard():
    assert TOOL.is_file(), "readiness guard must exist"
    spec = importlib.util.spec_from_file_location("readiness", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0000_item6_nested_terms_and_empty_vocabulary(tmp_path):
    guard = load_guard()
    vocabulary = tmp_path / "fake.json"
    vocabulary.write_text('{"nested": ["AlPhAbEt", {"x": "beta"}, "abc"]}')
    assert guard.terms(vocabulary) == ["alphabet", "beta"]
    vocabulary.write_text('{"x": ["abc", 123]}')
    try:
        guard.terms(vocabulary)
    except ValueError:
        pass
    else:
        raise AssertionError("empty vocabulary must be refused")


def test_p0000_item6_mutations():
    guard = load_guard()
    clean = {
        "LICENSE": b"Copyright (c) 2026 Jane Doe\n",
        "CITATION.cff": b"authors:\n  - given-names: Jane\n    family-names: Doe\n",
        ".github/workflows/ci.yml": b"name: CI\n",
        "src/example.py": b"safe",
    }
    assert not guard.inspect_files(clean, {"hiddenword"})
    for name in ("LICENSE", "CITATION.cff", ".github/workflows/ci.yml"):
        mutant = dict(clean)
        del mutant[name]
        assert guard.inspect_files(mutant, {"hiddenword"}), name
    for name, content in (
        ("LICENSE", b"Copyright (c) 2026 Someone Else"),
        ("src/example.py", "bad\u2014dash".encode()),
        ("case.vtu", b"mesh"),
        ("src/example.py", b"HIDDENWORD"),
    ):
        mutant = dict(clean)
        mutant[name] = content
        assert guard.inspect_files(mutant, {"hiddenword"}), name
    assert guard.find_terms(b"previous HIDDENWORD", {"hiddenword"})


def test_p0000_item6_history_mutation(tmp_path, monkeypatch, capsys):
    guard = load_guard()
    vocabulary = tmp_path / "fake.json"
    vocabulary.write_text('["hiddenword"]')
    files = {
        "LICENSE": b"Copyright (c) 2026 Jane Doe\n",
        "CITATION.cff": b"authors:\n  - given-names: Jane\n    family-names: Doe\n",
        ".github/workflows/ci.yml": b"name: CI\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def fake_git(*args):
        if args[0] == "ls-files":
            return b"\0".join(name.encode() for name in files) + b"\0"
        if args[0] == "rev-list":
            return b"abc123\n"
        if args[0] == "show":
            return b"commit message"
        if args[0] == "ls-tree":
            return b"100644 blob deadbeef\told.txt\0"
        return b"old HIDDENWORD"

    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "git", fake_git)
    monkeypatch.setattr("sys.argv", ["check", "--vocabulary", str(vocabulary)])
    assert guard.main() == 1
    assert "private term in history" in capsys.readouterr().out

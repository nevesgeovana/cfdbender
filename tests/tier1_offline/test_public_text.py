"""Exercise the public-text guard against planted text and the repository."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools" / "check_public_text.py"

# Split the samples so the guard can inspect this test as ordinary text.
SAMPLES = [
    "h" + "er",
    "s" + "he",
    "h" + "ers",
    "h" + "erself",
    "the own" + "er",
    "own" + "er's",
    "the author" + "'s",
    "the user" + "'s request",
    "Cla" + "ude",
    "Co" + "dex",
    "GP" + "T-6",
    "Anth" + "ropic",
    "Open" + "AI",
    "agent " + "session",
    "GO" + "AL-123",
    "la" + "ne",
    "work" + "tree",
    "review " + "round",
    "R" + "084C-002",
    "planning " + "node",
    "Not" + "ion",
    "Geo" + "verse",
    "GEO" + "VERSE_HEADER",
    "artifact" + "_id:",
    "last_modified" + "_by:",
    "C:" + "/WORK",
    "C:" + chr(92) + "private",
    "Users/" + "geova",
    "One" + "Drive",
    "Geo" + "verseCloud",
    "fts" + "-files",
    "Share" + "Point",
    "XP" + "ROP",
    "S" + "MI",
    "PF" + "S-2094",
    "PF" + "S-2094.009",
    "DL" + "V-059",
    "the work" + "-PC kit",
    "a work" + " pc",
    "private-company-example",
]


@pytest.fixture(scope="module")
def tree_copy(tmp_path_factory):
    """Copy tracked working files into a temporary repository."""
    root = tmp_path_factory.mktemp("public_text")
    names = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True
    ).stdout.split(b"\0")
    paths = []
    for name in names:
        if name:
            relative = Path(os.fsdecode(name))
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            paths.append(str(target))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "--", *paths], check=True)
    return root


@pytest.mark.parametrize("sample", SAMPLES)
def test_p0000_public_text_rejects_mutants(tree_copy, sample):
    assert CHECKER.is_file(), "public-text checker must exist"
    mutant = tree_copy / "mutant.txt"
    mutant.write_text("safe first line\n" + sample + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tree_copy), "add", "--", str(mutant)], check=True)
    env = dict(os.environ, CFDB_PRIVATE_TERMS="private-company-example")
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(tree_copy)], capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert "mutant.txt:2" in result.stdout


def test_p0000_public_text_real_tree():
    assert CHECKER.is_file(), "public-text checker must exist"
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(ROOT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_p0000_public_text_does_not_exempt_prose_in_symbol(tree_copy):
    mutant = tree_copy / "mutant.txt"
    mutant.write_text("ordinary text\n", encoding="utf-8")
    path = tree_copy / "src/cfdbender/core/plan.py"
    original = path.read_text(encoding="utf-8")
    marker = '    """Refuse duplicate point ids, overlapping directories and output path claims.'
    altered = marker + "\n\n    " + "The own" + "er requested this."
    path.write_text(original.replace(marker, altered), encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(CHECKER), str(tree_copy)], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "src/cfdbender/core/plan.py:" in result.stdout
    finally:
        path.write_text(original, encoding="utf-8")


def test_p0000_local_assistant_state_stays_untracked():
    # The assistant's local state folder (an attestation file, for example) must never be
    # staged into the public tree by a broad add.
    name = "." + "cla" + "ude/.role_review_attestation.json"
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index", name],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, name + " is not ignored"

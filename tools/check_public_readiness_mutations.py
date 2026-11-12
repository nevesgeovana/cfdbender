"""Prove the review tests detect each requested readiness refusal bypass."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools/check_public_readiness.py"
TEST = ROOT / "tests/tier1_offline/test_review_readiness.py"
ITEM8_TEST = ROOT / "tests/tier1_offline/test_p0000_item8_readiness.py"
ITEM8B_TEST = ROOT / "tests/tier1_offline/test_p0000_item8b_readiness.py"

MUTANTS = {
    "license": ("if name not in files", 'if name not in files and name != "LICENSE"'),
    "copyright": ('findings.append("LICENSE copyright does not match CITATION author")', "pass"),
    "en": ('"\\u2013".encode()', 'b"no-en-dash"'),
    "em": ('"\\u2014".encode()', 'b"no-em-dash"'),
    "output": ('findings.append(f"solver or mesh output: {name}")', "pass"),
    "private": ('findings.append(f"private term in tracked file: term #{number} {name}")', "pass"),
    "history": (
        'f"private term in history: term #{index + 1} {commit} {safe_name}"',
        '"history omitted"',
    ),
    "empty": ('raise ValueError("vocabulary has no usable terms")', "return []"),
    "missing": (
        'path.read_text(encoding="utf-8")',
        'path.read_text(encoding="utf-8") if path.exists() else \'["safe"]\'',
    ),
}

ITEM8_MUTANTS = {
    "order": (
        "list(unique.values())",
        "sorted(unique.values())",
        "test_p0000_item8_order",
    ),
    "number": ("term #{index + 1}", "term", "test_p0000_item8_output"),
    "summary": ("len(metadata_hits[index])", "0", "test_p0000_item8_output"),
    "shape": ('classes.append("contains digits")', "pass", "test_p0000_item8_output"),
    "allow": ("i + 1 not in allowed", "True", "test_p0000_item8_output"),
    "allow_file": ("allowed.update(values)", "pass", "test_p0000_item8_shapes_and_allow_file"),
    "unknown": ("number > len(vocabulary)", "False", "test_p0000_item8_allow_refusals"),
    "privacy": (
        "print(finding)",
        "print(finding, vocabulary[0])",
        "test_p0000_item8_shapes_and_allow_file",
    ),
}

ITEM8B_MUTANTS = {
    "suffix": "next((v for k, v in OUTPUT_EXEMPTIONS.items() if name.endswith(k)), None)",
    "directory": (
        "next((v for k, v in OUTPUT_EXEMPTIONS.items() "
        'if name.rsplit("/", 1)[0] == k.rsplit("/", 1)[0]), None)'
    ),
}


def main() -> int:
    """Run each mutant in an isolated directory and require an assertion failure."""
    source = GUARD.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        for name, (old, new) in MUTANTS.items():
            root = Path(directory) / name
            (root / "tools").mkdir(parents=True)
            (root / "tests/tier1_offline").mkdir(parents=True)
            shutil.copyfile(TEST, root / "tests/tier1_offline/test_review_readiness.py")
            if source.count(old) != 1:
                raise ValueError(f"mutation anchor {name} is not unique")
            (root / "tools/check_public_readiness.py").write_text(
                source.replace(old, new), encoding="utf-8"
            )
            selector = (
                f"test_refusal_arms[{name}]"
                if name in {"license", "copyright", "en", "em", "output", "private"}
                else "test_old_commit_refused"
                if name == "history"
                else "test_vocabulary_refusals"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"{root / 'tests/tier1_offline/test_review_readiness.py'}::{selector}",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            failure = "DID NOT RAISE" if name in {"empty", "missing"} else "AssertionError"
            if (
                result.returncode != 1
                or "1 failed" not in result.stdout
                or failure not in result.stdout
            ):
                print(f"{name}: mutation not detected\n{result.stdout}\n{result.stderr}")
                return 1
            print(f"{name}: test failed as expected")
        for name, (old, new, selector) in ITEM8_MUTANTS.items():
            root = Path(directory) / ("item8_" + name)
            (root / "tools").mkdir(parents=True)
            (root / "tests/tier1_offline").mkdir(parents=True)
            shutil.copyfile(ITEM8_TEST, root / "tests/tier1_offline/test_p0000_item8_readiness.py")
            if old not in source:
                raise ValueError(f"mutation anchor {name} is missing")
            (root / "tools/check_public_readiness.py").write_text(
                source.replace(old, new), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"{root / 'tests/tier1_offline/test_p0000_item8_readiness.py'}::{selector}",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 1 or "AssertionError" not in result.stdout:
                print(f"{name}: mutation not detected\n{result.stdout}\n{result.stderr}")
                return 1
            print(f"{name}: test failed as expected")
        for name, replacement in ITEM8B_MUTANTS.items():
            root = Path(directory) / ("item8b_" + name)
            (root / "tools").mkdir(parents=True)
            (root / "tests/tier1_offline").mkdir(parents=True)
            test = root / "tests/tier1_offline/test_p0000_item8b_readiness.py"
            shutil.copyfile(ITEM8B_TEST, test)
            anchor = "OUTPUT_EXEMPTIONS.get(name)"
            if source.count(anchor) != 1:
                raise ValueError(f"mutation anchor {name} is not unique")
            (root / "tools/check_public_readiness.py").write_text(
                source.replace(anchor, replacement), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"{test}::test_p0000_item8b_exact_exception",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 1 or "AssertionError" not in result.stdout:
                print(f"{name}: mutation not detected\n{result.stdout}\n{result.stderr}")
                return 1
            print(f"{name}: test failed as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

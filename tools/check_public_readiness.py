"""Check the tracked tree and all reachable history before publication.

Matching is casefolded substring matching, which can produce a false positive
when a term occurs inside a longer identifier. Review each number before allowing it.

Output files are refused except for explicitly listed, size-limited reference
fixtures. Exemptions affect only the output rule, and are printed when applied.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {".su2", ".msh", ".vtk", ".vtu", ".cgns", ".dat", ".csv.gz"}
OUTPUT_EXEMPTIONS = {
    # Small public reference mesh enables the real solver test without gmsh.
    "tests/fixtures/su2/naca0012_small.su2": (300 * 1024, "reference mesh for solver test"),
}


def original_terms(path: Path) -> list[str]:
    """Collect string leaves in depth-first array and object member order.

    Ignore non-string leaves and strings shorter than four characters. For
    casefold-equivalent strings keep the first. Numbering starts at 1.
    """

    def leaves(value: object) -> list[str]:
        if isinstance(value, str):
            return [value] if len(value) >= 4 else []
        if isinstance(value, list):
            return [term for item in value for term in leaves(item)]
        if isinstance(value, dict):
            return [term for item in value.values() for term in leaves(item)]
        return []

    unique: dict[str, str] = {}
    for value in leaves(json.loads(path.read_text(encoding="utf-8"))):
        unique.setdefault(value.casefold(), value)
    result = list(unique.values())
    if not result:
        raise ValueError("vocabulary has no usable terms")
    return result


def terms(path: Path) -> list[str]:
    """Return matching terms casefolded in their first appearance order."""
    return [value.casefold() for value in original_terms(path)]


def find_terms(data: bytes, vocabulary: set[str] | list[str]) -> bool:
    """Check text as UTF-8 and UTF-16, including binary payloads."""
    texts = [
        data.decode(codec, errors="ignore").casefold()
        for codec in ("utf-8", "utf-16-le", "utf-16-be")
    ]
    return any(term in content for term in vocabulary for content in texts)


def inspect_files(
    files: dict[str, bytes], vocabulary: set[str] | list[str], exemptions: list[str] | None = None
) -> list[str]:
    """Return findings for a tracked inventory and its contents."""
    findings = [
        f"missing {name}"
        for name in ("LICENSE", "CITATION.cff", ".github/workflows/ci.yml")
        if name not in files
    ]
    citation = files.get("CITATION.cff", b"").decode("utf-8", errors="replace")
    given = re.search(r"(?m)^\s*-?\s*given-names:\s*['\"]?([^'\"\r\n]+)", citation)
    family = re.search(r"(?m)^\s*-?\s*family-names:\s*['\"]?([^'\"\r\n]+)", citation)
    copyright_lines = re.findall(
        r"(?im)^copyright[^\r\n]*", files.get("LICENSE", b"").decode("utf-8", errors="replace")
    )
    if (
        not given
        or not family
        or len(copyright_lines) != 1
        or not re.search(
            r"^Copyright\s+\(c\)\s+\d{4}\s+"
            + re.escape(given.group(1).strip() + " " + family.group(1).strip())
            + r"\s*$",
            copyright_lines[0],
            re.I,
        )
    ):
        findings.append("LICENSE copyright does not match CITATION author")
    for name, data in files.items():
        if any(name.casefold().endswith(ext) for ext in OUTPUTS):
            exception = OUTPUT_EXEMPTIONS.get(name)
            if exception is not None and len(data) < exception[0]:
                if exemptions is not None:
                    exemptions.append(f"exempt: {name} ({exception[1]})")
            else:
                findings.append(f"solver or mesh output: {name}")
        if any(mark in data for mark in ("\u2013".encode(), "\u2014".encode())):
            findings.append(f"dash: {name}")
        for number, term in enumerate(vocabulary, 1):
            if find_terms(data, [term]) or find_terms(name.encode(), [term]):
                findings.append(f"private term in tracked file: term #{number} {name}")
    return findings


def git(*args: str) -> bytes:
    """Run git at the repository root and refuse failures."""
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main() -> int:
    """Scan the tracked files and all commits, refusing incomplete scans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--allow-term", action="append", type=int, default=[])
    parser.add_argument("--allow-term-file", type=Path)
    args = parser.parse_args()
    try:
        originals = original_terms(args.vocabulary)
        vocabulary = [term.casefold() for term in originals]
        allowed = set(args.allow_term)
        if args.allow_term_file:
            if args.allow_term_file.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("allow-term file must be outside the repository")
            values = json.loads(args.allow_term_file.read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(type(value) is not int for value in values):
                raise ValueError("allow-term file must contain a list of integers")
            allowed.update(values)
        if any(number < 1 or number > len(vocabulary) for number in allowed):
            raise ValueError("unknown term number")
        names = git("ls-files", "-z").decode("utf-8").split("\0")[:-1]
        commits = git("rev-list", "--all").decode("ascii").splitlines()
        if not names or not commits:
            raise ValueError("empty tracked inventory or history")
        files = {name: (ROOT / name).read_bytes() for name in names}
        exemptions: list[str] = []
        findings = inspect_files(files, [], exemptions)
        tracked: list[set[str]] = [set() for _ in vocabulary]
        histories: list[set[str]] = [set() for _ in vocabulary]
        metadata_hits: list[set[str]] = [set() for _ in vocabulary]
        for name, data in files.items():
            for index, term in enumerate(vocabulary):
                if find_terms(data, [term]) or find_terms(name.encode(), [term]):
                    tracked[index].add(name)
                    findings.append(f"private term in tracked file: term #{index + 1} {name}")
        for commit in commits:
            metadata = git("show", "-s", "--format=fuller", commit)
            tree = git("ls-tree", "-r", "-z", "--full-tree", commit)
            for index, term in enumerate(vocabulary):
                if find_terms(metadata, [term]):
                    metadata_hits[index].add(commit)
                    findings.append(f"private term in history metadata: term #{index + 1} {commit}")
            for entry in tree.split(b"\0"):
                if not entry:
                    continue
                descriptor, path = entry.split(b"\t", 1)
                mode, kind, oid = descriptor.split()
                if kind != b"blob" or mode == b"160000":
                    raise ValueError("history contains an unsupported tree entry")
                data = git("cat-file", "blob", oid.decode("ascii"))
                name = path.decode("utf-8", errors="replace")
                safe_name = "[redacted path]" if find_terms(path, vocabulary) else name
                for index, term in enumerate(vocabulary):
                    if find_terms(data, [term]) or find_terms(path, [term]):
                        histories[index].add(commit)
                        findings.append(
                            f"private term in history: term #{index + 1} {commit} {safe_name}"
                        )
        print(
            f"Scanned {len(names)} tracked files, {len(commits)} commits, {len(vocabulary)} terms"
        )
        print(
            "Substring matches can be a false positive inside a longer identifier; "
            "review before allowing."
        )
        for finding in exemptions + findings:
            for name in names:
                if find_terms(name.encode(), vocabulary):
                    finding = finding.replace(name, "[redacted path]")
            for term in vocabulary:
                finding = re.sub(re.escape(term), "[redacted]", finding, flags=re.IGNORECASE)
            print(finding)
        for index, term in enumerate(originals):
            classes = []
            if term.isalpha():
                classes.append("letters only")
            if any(char.isdigit() for char in term):
                classes.append("contains digits")
            if any(char in term for char in ("/", chr(92))):
                classes.append("contains path separators")
            if any(char.isspace() for char in term):
                classes.append("contains spaces")
            shape = ", ".join(classes) or "other"
            status = "allowed" if index + 1 in allowed else "not allowed"
            print(
                f"term #{index + 1}: tracked files: {len(tracked[index])}, commits: "
                f"{len(histories[index])}, metadata commits: {len(metadata_hits[index])}, "
                f"length: {len(term)}, {shape}, {status}"
            )
        other = any(not finding.startswith("private term") for finding in findings)
        private = any(
            (tracked[i] or histories[i] or metadata_hits[i]) and i + 1 not in allowed
            for i in range(len(vocabulary))
        )
        return int(other or private)
    except (OSError, ValueError, UnicodeError, subprocess.CalledProcessError) as error:
        reason = (
            "unknown term number" if str(error) == "unknown term number" else type(error).__name__
        )
        print(f"readiness refused: {reason}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Check tracked text for private and process-specific language."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# Each exception is limited to a symbol containing a programming identifier or
# a mutex comment; it does not exempt other patterns or possessive wording.
ALLOW = {
    "src/cfdbender/core/plan.py": {"check_claims": "Identifier for a path claimant."},
    "tools/check_architecture.py": {"check": "Identifier for the solver package being checked."},
    "tests/tier1_offline/test_p0000_solver_host_architecture.py": {
        "test_p0000_solver_host_sibling_import_refused": "Parameter naming the solver package.",
        "test_p0000_solver_host_legal_imports": "Parameter naming the solver package.",
    },
    "src/cfdbender/solvers/_host/_single.py": {
        "process_lease": "Mutex terminology in the abandoned-lock comment."
    },
    "tests/tier1_offline/test_p0000_solver_host_lease.py": {
        "LEGACY": "Mutex terminology in the frozen lease fixture."
    },
}


def patterns() -> list[re.Pattern[str]]:
    """Compile prohibited expressions, including optional private terms."""
    words = [
        "h" + "er",
        "s" + "he",
        "h" + "ers",
        "h" + "erself",
        "own" + "er",
        "Cla" + "ude",
        "Co" + "dex",
        "Anth" + "ropic",
        "Open" + "AI",
        "agent " + "session",
        "la" + "ne",
        "work" + "tree",
        "review " + "round",
        "planning " + "node",
        "Not" + "ion",
        "Geo" + "verse",
        "One" + "Drive",
        "Geo" + "verseCloud",
        "fts" + "-files",
        "Share" + "Point",
        "XP" + "ROP",
        "S" + "MI",
        "the author" + "'s",
        "the user" + "'s request",
    ]
    words += [s.strip() for s in os.environ.get("CFDB_PRIVATE_TERMS", "").split(",") if s.strip()]
    result = [re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.I) for word in words]
    for expression in [
        r"\bGP" + r"T-",
        r"\bGO" + r"AL-\d+\b",
        r"\bPF" + r"S-\d+(?:\.\d+)?",
        r"\bDL" + r"V-\d+\b",
        r"\bwork" + r"[- ]?PC\b",
        r"\bGEO" + r"VERSE_HEADER",
        r"\bartifact" + r"_id:",
        r"\blast_modified" + r"_by:",
        r"C:" + r"/WORK\b",
        r"C:" + r"\\",
        r"Users/" + r"geova\b",
    ]:
        result.append(re.compile(expression, re.I))
    result.append(re.compile(r"\bR\d{3}[A-Z]?(?:-\d+)?\b"))
    return result


def allowed_lines(path: str, text: str) -> set[int]:
    """Locate explicitly listed Python symbols, including decorators."""
    symbols = ALLOW.get(path, {})
    if not symbols:
        return set()
    tree = ast.parse(text, filename=path)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            lines.update(range(start, (node.end_lineno or start) + 1))
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in symbols for target in node.targets
        ):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def allowed_identifier(line: str) -> bool:
    """Recognize the specific identifier uses and the frozen mutex comment."""
    name = "own" + "er"
    stripped = line.strip()
    if stripped.startswith("#"):
        return stripped == "# An abandoned mutex can be acquired after a crashed " + name + "."
    return any(
        fragment in line
        for fragment in (
            "for previous, " + name + ",",
            name + " == point.point_id",
            "{" + name + "}",
            "{" + name + "!r}",
            name + " = relative.parts[1]",
            "{" + name + ', "_host"}',
            '"' + name + ',sibling"',
            "def test_p0000_solver_host_sibling_import_refused(tmp_path, " + name + ",",
            " / " + name + " / ",
            'parametrize("' + name + '",',
            "def test_p0000_solver_host_legal_imports(tmp_path, " + name + ")",
        )
    )


def main(root: Path) -> int:
    """Print each finding with its tracked path and line number."""
    paths = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
    ).stdout.split(b"\0")
    rules = patterns()
    project = root / "pyproject.toml"
    authors = set()
    if project.is_file():
        with project.open("rb") as stream:
            authors = {
                author["name"].casefold()
                for author in tomllib.load(stream).get("project", {}).get("authors", [])
                if "name" in author
            }
    count = 0
    for raw in paths:
        if not raw:
            continue
        path = os.fsdecode(raw).replace("\\", "/")
        data = (root / path).read_bytes()
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
        elif b"\0" in data:
            continue
        else:
            text = data.decode("utf-8", errors="replace")
        exempt = allowed_lines(path, text)
        citation = False
        for number, line in enumerate(text.splitlines(), 1):
            if path == "README.md" and line.startswith("#"):
                citation = "citation" in line.casefold()
            for rule in rules:
                for match in rule.finditer(line):
                    if match.group().casefold() in authors and (
                        path in {"LICENSE", "CITATION.cff"}
                        or (path == "pyproject.toml" and line.lstrip().startswith("authors ="))
                        or (path == "README.md" and citation)
                    ):
                        continue
                    if (
                        number in exempt
                        and match.group().lower() == "own" + "er"
                        and allowed_identifier(line)
                        and not re.search(r"\bthe\s+own" + r"er\b|\bown" + r"er's\b", line, re.I)
                    ):
                        continue
                    print(f"{path}:{number}: {match.group()}")
                    count += 1
    print(f"{count} finding(s)")
    return int(count > 0)


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()))

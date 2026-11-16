"""The cheatsheet (guide 4) is bound to the parsers of cfd-matrix and cfd-workspace.

Page 1 must name every command and every option of both parsers in that
command's own entry, and may name none a parser lacks. The stage pages may name
only commands and options that exist, must carry every command, and every
command line they or the decks show must parse.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

from cfdbender.matrix import cli as matrix_cli
from cfdbender.workspace import cli as workspace_cli

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "guide" / "latex-sources"
PAGE_ONE = SOURCES / "cheatsheet" / "text" / "01-commands.tex"
STAGES = SOURCES / "cheatsheet" / "text" / "02-by-stage.tex"
PDFS = ROOT / "guide" / "pdf"
GUIDES = (
    "cfdb-guide-01-overview",
    "cfdb-guide-02-workspaces",
    "cfdb-guide-03-solvers",
    "cfdb-guide-04-cheatsheet",
)
MACRO = {"cfd-matrix": "sub", "cfd-workspace": "wsub"}
BS = chr(92)

_ENTRY = re.compile(r"\\(w?sub)\{([^{}]*)\}")
_ENTRY_END = re.compile(r"\\w?sub\{|\\stage\{|\\end\{multicols\}")
_FLAG = re.compile(r"\\fl\{(--[a-z][a-z-]*)")
_BLOCK = re.compile(
    r"\\begin\{cmdblock\}\{(cfd-matrix|cfd-workspace) ([a-z-]+)\}\{([^{}]*)\}"
    r"(.*?)\\end\{cmdblock\}",
    re.S,
)
_OPT = re.compile(r"\\opt\{(--[a-z][a-z-]*)")
_EXAMPLE = re.compile(r"\\ex\{((?:cfd-matrix|cfd-workspace) [^{}]*)\}")
_LISTING = re.compile(r"\\begin\{lstlisting\}.*?\n(.*?)\\end\{lstlisting\}", re.S)

Surface = dict[str, tuple[set[str], list[str]]]


def _parsers() -> dict[str, argparse.ArgumentParser]:
    return {
        "cfd-matrix": matrix_cli.build_parser()[0],
        "cfd-workspace": workspace_cli.build_parser(),
    }


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return the parser's own mapping of commands, so a test can plant one."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _options(parser: argparse.ArgumentParser) -> tuple[set[str], list[str]]:
    """Return the long options and the positional names of one parser."""
    options: set[str] = set()
    positionals: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction | argparse._SubParsersAction):
            continue
        if action.option_strings:
            options.update(name for name in action.option_strings if name.startswith("--"))
        else:
            positionals.append(action.dest)
    return options, positionals


def _surface(parser: argparse.ArgumentParser) -> tuple[set[str], Surface]:
    """Return the options written before the command, and each command's own."""
    return _options(parser)[0], {name: _options(sub) for name, sub in _subparsers(parser).items()}


def _source(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("%"))


def _entries(body: str) -> list[tuple[str, str, str, str]]:
    """Return (macro, command, header, text) for each entry of page 1."""
    found = []
    for match in _ENTRY.finditer(body):
        end = _ENTRY_END.search(body, match.end())
        text = body[match.start() : end.start() if end else len(body)]
        header = match.group(2).replace(BS, "")
        found.append((match.group(1), header.split()[0], header, text))
    return found


def _has_positional(name: str, header: str) -> bool:
    return f"<{name}>" in header or f"[{name}]" in header


def page_one_findings(parsers: dict[str, argparse.ArgumentParser], text: str) -> list[str]:
    """Return every disagreement between page 1 and the parsers."""
    head, _, body = text.partition("\\begin{multicols}")
    findings: list[str] = []
    entries = _entries(body)
    everything: set[str] = set()
    for tool, parser in parsers.items():
        before, commands = _surface(parser)
        everything |= before
        for name in sorted(before):
            if name not in _FLAG.findall(head):
                findings.append(f"{tool}: {name} is missing from the header box")
        mine = [entry for entry in entries if entry[0] == MACRO[tool]]
        for _, command, _, _ in mine:
            if command not in commands:
                findings.append(f"{tool} {command}: the sheet names a command the parser lacks")
        for command, (options, positionals) in sorted(commands.items()):
            everything |= options
            own = [entry for entry in mine if entry[1] == command]
            if len(own) != 1:
                findings.append(f"{tool} {command}: {len(own)} entries on page 1, expected 1")
                continue
            _, _, header, entry = own[0]
            named = set(_FLAG.findall(entry))
            for name in sorted(options - named):
                findings.append(f"{tool} {command}: {name} is missing from its entry")
            for name in sorted(named - options):
                findings.append(f"{tool} {command}: its entry names {name}, which it does not take")
            for name in positionals:
                if not _has_positional(name, header):
                    findings.append(f"{tool} {command}: positional {name} is missing")
    for name in sorted(set(_FLAG.findall(head)) - everything):
        findings.append(f"header box: {name} is an option of no command")
    return findings


def _parses(parsers: dict[str, argparse.ArgumentParser], line: str) -> bool:
    tool, *arguments = shlex.split(line)
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            parsers[tool].parse_args(arguments)
    except SystemExit:
        return False
    return True


def stage_findings(parsers: dict[str, argparse.ArgumentParser], text: str) -> list[str]:
    """Return every disagreement between the stage pages and the parsers."""
    findings: list[str] = []
    surfaces = {tool: _surface(parser)[1] for tool, parser in parsers.items()}
    everything: set[str] = set()
    for commands in surfaces.values():
        for options, _ in commands.values():
            everything |= options
    seen: set[tuple[str, str]] = set()
    for tool, command, arguments, block in _BLOCK.findall(text):
        if command not in surfaces[tool]:
            findings.append(f"{tool} {command}: a stage page names a command the parser lacks")
            continue
        seen.add((tool, command))
        options, positionals = surfaces[tool][command]
        for name in _OPT.findall(block):
            if name not in options:
                findings.append(
                    f"{tool} {command}: a stage page names {name}, which it does not take"
                )
        for name in positionals:
            if not _has_positional(name, arguments):
                findings.append(f"{tool} {command}: positional {name} is missing on its stage page")
    for tool, commands in surfaces.items():
        for command in sorted(commands):
            if (tool, command) not in seen:
                findings.append(f"{tool} {command}: no stage page carries it")
    for name in _OPT.findall(_BLOCK.sub("", text)):
        if name not in everything:
            findings.append(f"stage pages: {name} is an option of no command")
    for line in _EXAMPLE.findall(text):
        if not _parses(parsers, line.replace(BS, "")):
            findings.append(f"stage pages: the example does not parse: {line}")
    return findings


def listing_findings(parsers: dict[str, argparse.ArgumentParser], text: str) -> list[str]:
    """Return every command line of a listing that the parsers refuse."""
    findings = []
    for listing in _LISTING.findall(text):
        for line in listing.splitlines():
            if line.startswith(("cfd-matrix ", "cfd-workspace ")) and not _parses(parsers, line):
                findings.append(f"the command line does not parse: {line}")
    return findings


def test_page_one_names_every_command_and_option_and_no_other():
    assert page_one_findings(_parsers(), _source(PAGE_ONE)) == []


def test_the_stage_pages_name_only_what_exists_and_carry_every_command():
    assert stage_findings(_parsers(), _source(STAGES)) == []


def test_every_command_line_in_a_listing_of_the_decks_parses():
    lines = 0
    for path in sorted(SOURCES.glob("*/text/*.tex")):
        text = path.read_text(encoding="utf-8")
        lines += sum(
            line.startswith(("cfd-matrix ", "cfd-workspace "))
            for listing in _LISTING.findall(text)
            for line in listing.splitlines()
        )
        assert listing_findings(_parsers(), text) == [], path.name
    assert lines >= 10


def test_the_sheet_covers_both_tools_and_every_command():
    commands = {tool: _surface(parser)[1] for tool, parser in _parsers().items()}
    assert {"plan", "run", "collect", "post", "status"} <= set(commands["cfd-matrix"])
    assert len(commands["cfd-workspace"]) >= 15
    assert {"init", "archive"} <= set(commands["cfd-workspace"])
    both = (*commands["cfd-matrix"].values(), *commands["cfd-workspace"].values())
    assert sum(len(options) for options, _ in both) > 100


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "\\item \\fl{--resume} skip",
            "\\item skip",
            "cfd-matrix run: --resume is missing from its entry",
        ),
        (
            "\\item \\fl{--top N}",
            "\\item \\fl{--top N}, \\fl{--bogus}",
            "cfd-workspace space-in-use: its entry names --bogus, which it does not take",
        ),
        (
            "\\item \\fl{--top N}",
            "\\item \\fl{--top N}, \\fl{--apply}",
            "cfd-workspace space-in-use: its entry names --apply, which it does not take",
        ),
        ("\\wsub{trace <product>}", "\\wsub{track <product>}", "cfd-workspace trace: 0 entries"),
        (
            "\\wsub{trace <product>}",
            "\\wsub{track <product>}",
            "cfd-workspace track: the sheet names",
        ),
        ("\\sub{plan <matrix>}", "\\wsub{plan <matrix>}", "cfd-workspace plan: the sheet names"),
        ("\\sub{plan <matrix>}", "\\sub{plan}", "cfd-matrix plan: positional matrix is missing"),
        ("\\fl{--warning-off}", "the warning switch", "--warning-off is missing from the header"),
    ],
)
def test_a_planted_defect_on_page_one_is_named(old, new, expected):
    text = _source(PAGE_ONE)
    assert text.count(old) == 1
    findings = page_one_findings(_parsers(), text.replace(old, new))
    assert any(expected in finding for finding in findings), findings


def test_an_option_added_to_a_parser_is_named_until_the_sheet_carries_it():
    parsers = _parsers()
    _subparsers(parsers["cfd-matrix"])["plan"].add_argument("--planted", action="store_true")
    _subparsers(parsers["cfd-workspace"]).update({"planted": argparse.ArgumentParser()})
    findings = page_one_findings(parsers, _source(PAGE_ONE))
    assert "cfd-matrix plan: --planted is missing from its entry" in findings
    assert "cfd-workspace planted: 0 entries on page 1, expected 1" in findings
    assert "cfd-workspace planted: no stage page carries it" in stage_findings(
        parsers, _source(STAGES)
    )


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "\\opt{--hpc NAME}{the profile \\file{inputs/hpc/<NAME>.toml}",
            "\\opt{--apply}{the profile \\file{inputs/hpc/<NAME>.toml}",
            "cfd-matrix plan: a stage page names --apply",
        ),
        (
            "\\begin{cmdblock}{cfd-workspace history}",
            "\\begin{cmdblock}{cfd-workspace past}",
            "cfd-workspace history: no stage page carries it",
        ),
        (
            "\\ex{cfd-matrix plan polar --local --verbose}",
            "\\ex{cfd-matrix plan polar --local --loud}",
            "the example does not parse: cfd-matrix plan polar --local --loud",
        ),
        (
            "\\opt{--json}{the rows as one JSON document}",
            "\\opt{--yaml}{the rows as one JSON document}",
            "stage pages: --yaml is an option of no command",
        ),
    ],
)
def test_a_planted_defect_on_a_stage_page_is_named(old, new, expected):
    text = _source(STAGES)
    assert text.count(old) == 1
    findings = stage_findings(_parsers(), text.replace(old, new))
    assert any(expected in finding for finding in findings), findings


def test_a_listing_line_the_parser_refuses_is_named():
    text = "\\begin{lstlisting}\ncfd-matrix plan polar\ncfd-matrix plam polar\n\\end{lstlisting}"
    assert listing_findings(_parsers(), text) == [
        "the command line does not parse: cfd-matrix plam polar"
    ]


def test_no_guide_source_holds_an_em_or_en_dash():
    for path in sorted(SOURCES.rglob("*.tex")):
        if "build" in path.relative_to(SOURCES).parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, path.name


def test_the_four_guides_are_built_and_the_cheatsheet_has_nine_pages():
    for name in GUIDES:
        assert (PDFS / f"{name}.pdf").stat().st_size > 20_000, name
    data = (PDFS / "cfdb-guide-04-cheatsheet.pdf").read_bytes()
    assert len(re.findall(rb"/Type\s*/Page(?![s\w])", data)) == 9
    info = (SOURCES / "shared" / "info.tex").read_text(encoding="utf-8")
    version = re.search(r"\\newcommand\{\\pkgversion\}\{([^}]+)\}", info)
    assert version is not None
    keywords = re.search(rb"/Keywords\((.*?)\)", data)
    assert keywords is not None
    raw = re.sub(rb"\\(\d{3})", lambda octal: bytes([int(octal.group(1), 8)]), keywords.group(1))
    written = raw.decode("utf-16") if raw.startswith(b"\xfe\xff") else raw.decode("latin-1")
    assert written == f"cfdbender {version.group(1)}"


def test_every_documentation_page_a_guide_cites_exists():
    slugs: set[str] = set()
    for path in sorted(SOURCES.glob("*/*.tex")):
        slugs.update(re.findall(r"docref\{([a-z0-9-]+)\}", path.read_text(encoding="utf-8")))
    assert len(slugs) >= 8
    for slug in sorted(slugs):
        assert (ROOT / "docs" / f"{slug}.md").is_file(), slug

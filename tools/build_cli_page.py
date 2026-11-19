"""Write the command reference of docs/cli.md from the two argument parsers.

Usage: python tools/build_cli_page.py [--check]

The page keeps its hand-written introduction. Everything between the BEGIN and
END marker lines is replaced by one section per command, grouped by workflow
stage, with every positional argument and option the parser accepts and the
parser's own help text. With --check nothing is written and the exit code is 1
when the page differs from what the parsers give.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "cli.md"
BEGIN = "<!-- BEGIN GENERATED COMMAND REFERENCE -->"
END = "<!-- END GENERATED COMMAND REFERENCE -->"

#: Workflow stage of every command. A command the parsers hold and this table
#: lacks stops the build, so a new command cannot stay off the page.
STAGES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Create", (("cfd-workspace", "init"), ("cfd-workspace", "setups"))),
    ("Plan", (("cfd-matrix", "plan"),)),
    ("Run", (("cfd-matrix", "run"),)),
    (
        "Follow",
        (
            ("cfd-matrix", "status"),
            ("cfd-workspace", "show"),
            ("cfd-workspace", "history"),
            ("cfd-workspace", "log"),
            ("cfd-workspace", "trace"),
            ("cfd-workspace", "diff"),
        ),
    ),
    ("Collect and post", (("cfd-matrix", "collect"), ("cfd-matrix", "post"))),
    (
        "Correct the records",
        (
            ("cfd-workspace", "mark-failed"),
            ("cfd-workspace", "mark-converged"),
            ("cfd-workspace", "restore"),
            ("cfd-workspace", "rebuild"),
            ("cfd-workspace", "archive"),
        ),
    ),
    (
        "Manage storage",
        (
            ("cfd-workspace", "space-in-use"),
            ("cfd-workspace", "free-space"),
            ("cfd-workspace", "delete-sims"),
            ("cfd-workspace", "sync"),
        ),
    ),
)


def parsers() -> dict[str, argparse.ArgumentParser]:
    """Return the two top-level parsers by program name."""
    sys.path.insert(0, str(ROOT / "src"))
    from cfdbender.matrix.cli import build_parser as matrix_parser
    from cfdbender.workspace.cli import build_parser as workspace_parser

    return {"cfd-workspace": workspace_parser(), "cfd-matrix": matrix_parser()[0]}


def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return the subcommand parsers of a top-level parser, in declaration order."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def summaries(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Return the one-line help of each subcommand."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return {choice.dest: choice.help or "" for choice in action._choices_actions}
    return {}


def arguments(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Return the actions of a parser except its help option and subcommand slot."""
    return [
        action
        for action in parser._actions
        if not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))
    ]


def invocation(action: argparse.Action) -> str:
    """Return the way an argument is written on the command line."""
    if action.metavar is not None:
        name = action.metavar if isinstance(action.metavar, str) else " ".join(action.metavar)
    elif action.choices is not None:
        name = "{" + ",".join(str(choice) for choice in action.choices) + "}"
    else:
        name = action.dest.upper()
    if action.nargs == 0:
        value = ""
    elif action.nargs is None:
        value = name
    elif action.nargs == "?":
        value = f"[{name}]"
    elif action.nargs == "+":
        value = f"{name} [{name} ...]"
    elif action.nargs == "*":
        value = f"[{name} ...]"
    else:
        value = name if isinstance(action.metavar, tuple) else " ".join([name] * int(action.nargs))
    if not action.option_strings:
        return value
    return ", ".join(action.option_strings) + (f" {value}" if value else "")


def meaning(action: argparse.Action, prog: str) -> str:
    """Return the help text of an argument as one table cell."""
    text = action.help or ""
    if "%(" in text:
        text = text % {**vars(action), "prog": prog}
    return escape(text[:1].upper() + text[1:])


def escape(text: str) -> str:
    """Return prose for a table cell: one line, no column break, no HTML tag."""
    text = " ".join(text.split())
    return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def sentence(text: str) -> str:
    """Return a help line as a sentence."""
    text = escape(text[:1].upper() + text[1:])
    return text if text.endswith(".") else text + "."


def table(rows: list[tuple[str, str]], head: str) -> list[str]:
    """Return a two-column table."""
    lines = [f"| {head} | Meaning |", "|---|---|"]
    lines += [f"| `{name}` | {text} |" for name, text in rows]
    return lines


def reference() -> str:
    """Return the generated block, without its markers."""
    tops = parsers()
    subs = {prog: subcommands(parser) for prog, parser in tops.items()}
    helps = {prog: summaries(parser) for prog, parser in tops.items()}
    staged = {entry for _, entries in STAGES for entry in entries}
    held = {(prog, name) for prog, commands in subs.items() for name in commands}
    if staged != held:
        missing = ", ".join(" ".join(entry) for entry in sorted(held - staged))
        extra = ", ".join(" ".join(entry) for entry in sorted(staged - held))
        raise SystemExit(
            f"STAGES and the parsers disagree; without a stage: {missing or 'none'}; "
            f"no such command: {extra or 'none'}"
        )
    lines: list[str] = ["## Options before the command name", ""]
    lines += [
        "These go between the program name and the command, for example",
        "`cfd-matrix --warning-off run polar`; `--warning-refusals` works the same way.",
        "",
    ]
    rows = [
        (f"{prog} {invocation(a)}", meaning(a, prog))
        for prog, parser in tops.items()
        for a in arguments(parser)
    ]
    lines += table(rows, "Option")
    lines.append("")
    for stage, entries in STAGES:
        lines += [f"## {stage}", ""]
        for prog, name in entries:
            parser = subs[prog][name]
            actions = arguments(parser)
            positional = [a for a in actions if not a.option_strings]
            optional = [a for a in actions if a.option_strings]
            usage = " ".join([prog, name, *(invocation(a) for a in positional)])
            lines += [f"### `{prog} {name}`", ""]
            lines += [sentence(parser.description or helps[prog].get(name, "")), ""]
            lines += ["```text", usage + (" [options]" if optional else ""), "```", ""]
            rows = [(invocation(a), meaning(a, prog)) for a in (*positional, *optional)]
            if rows:
                lines += table(rows, "Argument or option")
                lines.append("")
    return "\n".join(lines).rstrip("\n")


def render(page: str) -> str:
    """Return the page with its generated block replaced."""
    if page.count(BEGIN) != 1 or page.count(END) != 1:
        raise SystemExit(f"{PAGE.name}: expected one {BEGIN} line and one {END} line")
    head, rest = page.split(BEGIN)
    tail = rest.split(END)[1]
    return f"{head}{BEGIN}\n\n{reference()}\n\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    """Write the page, or compare it under --check."""
    options = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    options.add_argument("--check", action="store_true", help="compare only; write nothing")
    args = options.parse_args(argv)
    current = PAGE.read_text(encoding="utf-8")
    wanted = render(current)
    if args.check:
        if current != wanted:
            print(f"{PAGE.name} is out of date; run python tools/build_cli_page.py")
            return 1
        return 0
    if current != wanted:
        PAGE.write_text(wanted, encoding="utf-8", newline="\n")
        print(f"wrote {PAGE.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

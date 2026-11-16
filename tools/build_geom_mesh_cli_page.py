"""Write the command reference of docs/cli-geometry-mesh.md from the cfd-geom and cfd-mesh parsers.

Usage: python tools/build_geom_mesh_cli_page.py [--check]

The page keeps its hand-written introduction. Everything between the BEGIN and END marker lines
is replaced by one section per command, in the order the parsers declare them: cfd-geom first,
then cfd-mesh by its verb groups, with every positional argument and option the parser accepts
and the parser's own help text. The rendering is the one of tools/build_cli_page.py. With
--check nothing is written and the exit code is 1 when the page differs from the parsers.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "cli-geometry-mesh.md"
BEGIN = "<!-- BEGIN GENERATED COMMAND REFERENCE -->"
END = "<!-- END GENERATED COMMAND REFERENCE -->"

#: The section title of each cfd-mesh verb group module.
GROUP_TITLES = {
    "_cli_surface": "cfd-mesh: mesh generated geometry",
    "_cli_refine": "cfd-mesh: refine, coarsen and audit",
    "_cli_edges": "cfd-mesh: trailing edges",
    "_cli_rotor": "cfd-mesh: rotors and whole meshes",
    "_cli_airfoil2d": "cfd-mesh: 2D airfoil volume meshes",
}


def _base() -> ModuleType:
    """Return tools/build_cli_page.py as a module, for its rendering helpers."""
    spec = importlib.util.spec_from_file_location(
        "build_cli_page", ROOT / "tools" / "build_cli_page.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sections() -> list[tuple[str, str, list[str]]]:
    """Return ``(title, program, verbs)`` per section, in the order the parsers declare them."""
    sys.path.insert(0, str(ROOT / "src"))
    from cfdbender.mesh import cli as mesh_cli

    geom = _parsers()["cfd-geom"]
    found = [("cfd-geom: parametric geometry", "cfd-geom", list(_base().subcommands(geom)))]
    for group in mesh_cli.GROUPS:
        probe = argparse.ArgumentParser(allow_abbrev=False)
        verbs = probe.add_subparsers()
        group.add_parsers(verbs)
        names = list(verbs.choices)
        if names:
            found.append((GROUP_TITLES[group.__name__.rsplit(".", 1)[1]], "cfd-mesh", names))
    return found


def _parsers() -> dict[str, argparse.ArgumentParser]:
    sys.path.insert(0, str(ROOT / "src"))
    from cfdbender.geometry.cli import _build_parser as geom_parser
    from cfdbender.mesh.cli import build_parser as mesh_parser

    return {"cfd-geom": geom_parser(), "cfd-mesh": mesh_parser()}


def reference() -> str:
    """Return the generated block, without its markers."""
    base = _base()
    tops = _parsers()
    subs = {prog: base.subcommands(parser) for prog, parser in tops.items()}
    helps = {prog: base.summaries(parser) for prog, parser in tops.items()}
    listed = {(prog, name) for _, prog, names in sections() for name in names}
    held = {(prog, name) for prog, commands in subs.items() for name in commands}
    if listed != held:
        raise SystemExit(f"sections and parsers disagree: {sorted(listed ^ held)}")
    lines: list[str] = ["## Options before the command name", ""]
    lines += [
        "These go between the program name and the command, for example",
        "`cfd-mesh --workspace campaign refine ...`.",
        "",
    ]
    rows = [
        (f"{prog} {base.invocation(a)}", base.meaning(a, prog))
        for prog, parser in tops.items()
        for a in base.arguments(parser)
    ]
    lines += base.table(rows, "Option")
    lines.append("")
    for title, prog, names in sections():
        lines += [f"## {title}", ""]
        for name in names:
            parser = subs[prog][name]
            actions = base.arguments(parser)
            positional = [a for a in actions if not a.option_strings]
            optional = [a for a in actions if a.option_strings]
            usage = " ".join([prog, name, *(base.invocation(a) for a in positional)])
            lines += [f"### `{prog} {name}`", ""]
            lines += [base.sentence(parser.description or helps[prog].get(name, "")), ""]
            lines += ["```text", usage + (" [options]" if optional else ""), "```", ""]
            rows = [(base.invocation(a), base.meaning(a, prog)) for a in (*positional, *optional)]
            if rows:
                lines += base.table(rows, "Argument or option")
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
            print(f"{PAGE.name} is out of date; run python tools/build_geom_mesh_cli_page.py")
            return 1
        return 0
    if current != wanted:
        PAGE.write_text(wanted, encoding="utf-8", newline="\n")
        print(f"wrote {PAGE.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

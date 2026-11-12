"""Check source imports without importing or running the product or any adapter.

Adapted from the source-reading and planted-mutant approach of the architecture
checker: inspect absolute, relative, deferred and literal dynamic imports.
Computed dynamic import targets are outside this static check's coverage.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path


def _callee(node: ast.expr, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_callee(node.value, aliases)}.{node.attr}"
    return ""


def _imports(tree: ast.AST, package: str) -> Iterator[tuple[int, str]]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package)
            yield node.lineno, base
            for alias in node.names:
                target = f"{base}.{alias.name}"
                aliases[alias.asname or alias.name] = target
                # These are registry functions, not sibling adapter modules.
                if base != "cfdbender.solvers" or alias.name not in {
                    "available_solvers",
                    "load_solver",
                }:
                    yield node.lineno, target
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if _callee(node.func, aliases) not in {
            "importlib.import_module",
            "__import__",
            "builtins.__import__",
        }:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            target = argument.value
            if target.startswith("."):
                context = (
                    node.args[1]
                    if len(node.args) > 1
                    else next(
                        (keyword.value for keyword in node.keywords if keyword.arg == "package"),
                        None,
                    )
                )
                if isinstance(context, ast.Constant) and isinstance(context.value, str):
                    target = resolve_name(target, context.value)
                elif isinstance(context, ast.Name) and context.id == "__package__":
                    target = resolve_name(target, package)
            yield node.lineno, target


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def check(root: Path) -> list[str]:
    """Return import violations in the package at root.

    Parameters
    ----------
    root : Path
        Directory containing the cfdbender source package.

    Returns
    -------
    list[str]
        Sorted diagnostics naming the file, line and violated boundary.

    Raises
    ------
    ValueError
        If the core source package is absent or empty.
    SyntaxError
        If any source cannot be parsed; unmeasured source never passes.
    """
    paths = sorted(root.rglob("*.py"))
    if not any("core" == path.relative_to(root).parts[0] for path in paths):
        raise ValueError(f"{root}: no core source files found")
    findings: set[str] = set()
    for path in paths:
        relative = path.relative_to(root)
        package = ".".join(("cfdbender", *relative.parts[:-1]))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, target in _imports(tree, package):
            prefix = f"{relative.as_posix()}:{line}: "
            if relative.parts[0] == "solvers" and len(relative.parts) > 2:
                owner = relative.parts[1]
                parts = target.split(".")
                if (
                    parts[:2] == ["cfdbender", "solvers"]
                    and len(parts) > 2
                    and parts[2] not in {owner, "_host"}
                ):
                    findings.add(prefix + f"{owner} must not import sibling solver: {target}")
            if relative.parts[0] in {"core", "hpc"} and _within(target, "cfdbender.solvers"):
                findings.add(prefix + f"{relative.parts[0]} must not import solvers: {target}")
            if relative.parts[0] in {"core", "hpc"} and _within(target, "cfdbender._pyfs"):
                findings.add(prefix + f"{relative.parts[0]} must not import cfdbender._pyfs")
            if relative.as_posix() != "_pyfs.py" and _within(target, "pyflightstream"):
                findings.add(prefix + "use cfdbender._pyfs for upstream imports")
    return sorted(findings)


def main(argv: list[str] | None = None) -> int:
    """Print architecture findings and return 0, 1 for violations, or 2 if unmeasured.

    Parameters
    ----------
    argv : list[str], optional
        Command arguments; defaults to process arguments.

    Returns
    -------
    int
        The check's exit code, with refusals printed to stderr.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "src/cfdbender"
    )
    args = parser.parse_args(argv)
    try:
        findings = check(args.root)
    except (OSError, UnicodeError, SyntaxError, ValueError, ImportError) as error:
        print(f"architecture refused: {error}", file=sys.stderr)
        return 2
    if findings:
        print("\n".join(findings))
        return 1
    print("architecture: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

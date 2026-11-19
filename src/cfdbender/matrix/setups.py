"""The setups verb: write a ready s file from a template the workspace's solver ships."""

from __future__ import annotations

import argparse
from typing import Any

from cfdbender.core import setup_templates
from cfdbender.core.settings import read_init, require_workspace
from cfdbender.matrix._console import WORKSPACE_HELP, Console, level_of
from cfdbender.solvers import available_solvers


def run_setups(args: argparse.Namespace) -> int:
    """List the solver's templates, or write the next free s file from one."""
    root = require_workspace(args.workspace)
    solver = read_init(root, available_solvers=available_solvers()).solver
    names = setup_templates.available(solver)
    console = Console(level_of(args))
    if args.id is not None and args.template is None:
        raise ValueError("--id needs --template NAME")
    if args.template is None:
        if not names:
            print(f"no setup templates for {solver}")
            return 0
        width = max(len(name) for name in names)
        for name in names:
            print(f"{name:<{width}}  {setup_templates.describe(solver, name)}")
        return 0
    target = setup_templates.write_setup(root, solver, args.template, args.id)
    shown = target.relative_to(root).as_posix()
    console.final(f"wrote {shown} from the {args.template} template of {solver}")
    console.say("next: edit it to match your case, then run cfd-matrix plan")
    return 0


def register_workspace(commands: Any) -> dict[str, Any]:
    """Add the setups verb to the cfd-workspace subparsers and return its handler."""
    setups = commands.add_parser(
        "setups",
        help="write a ready setup file from a template of the workspace's solver",
        description=(
            "Writes the next free inputs/setups/s<id>.toml from a named template of the "
            "workspace's solver and overwrites nothing. Without --template, or with --list, "
            "it lists the templates."
        ),
        allow_abbrev=False,
    )
    choice = setups.add_mutually_exclusive_group()
    choice.add_argument(
        "--template",
        metavar="NAME",
        help="write inputs/setups/s<ID>.toml from this template of the workspace's solver",
    )
    choice.add_argument(
        "--list",
        dest="list_templates",
        action="store_true",
        help="print the templates of the workspace's solver with their descriptions "
        "(the same as no option)",
    )
    setups.add_argument(
        "--id",
        metavar="ID",
        help="the setup id to write, three digits (default: the next free one); "
        "an existing file is refused",
    )
    setups.add_argument("--workspace", default=".", help=WORKSPACE_HELP)
    return {"setups": run_setups}

"""The cfd-workspace command line: initialize a single-solver campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfdbender.core import WorkspaceError, init_workspace, input_kinds
from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.matrix._console import (
    WORKSPACE_HELP,
    Console,
    add_console_options,
    add_warning_option,
    held_warnings,
    level_of,
)
from cfdbender.matrix.records import archive_sim
from cfdbender.solvers import available_solvers
from cfdbender.solvers.flightstream import dispatch
from cfdbender.solvers.flightstream import init_workspace as init_flightstream


def build_parser() -> argparse.ArgumentParser:
    """Return the cfd-workspace argument parser."""
    parser = argparse.ArgumentParser(
        prog="cfd-workspace",
        description="Managed campaign workspace tooling for CFD studies.",
        allow_abbrev=False,
    )
    add_warning_option(parser)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    init = subparsers.add_parser(
        "init",
        help="create the full campaign tree (inputs library, sims, post, archive)",
        allow_abbrev=False,
    )
    init.add_argument(
        "root",
        nargs="?",
        default=".",
        help="campaign root to create or complete (default: the current directory)",
    )
    init.add_argument(
        "--workspace",
        help=WORKSPACE_HELP + "; the positional root names the same folder",
    )
    init.add_argument(
        "--solver",
        required=True,
        metavar="NAME",
        help=f"solver for this workspace, one of: {', '.join(available_solvers())}",
    )
    archive = subparsers.add_parser(
        "archive",
        help="move one recorded simulation's folder into archive/<stamp> and record it",
        description=(
            "Moves sims/sim_<sim_id>/ of a recorded simulation into archive/<stamp>/ and "
            "moves its records from runs.json to the manifest history, so the simulation "
            "can be run again into a clean folder while its evidence stays. Refuses a "
            "simulation the manifest does not record. It does not read the status of the "
            "runs: do not archive a simulation whose run is still going. It acts at once, with "
            "no preview, as pyflightstream's archive does; look first with cfd-matrix status "
            "--points --sims <POL>."
        ),
        allow_abbrev=False,
    )
    archive.add_argument(
        "root",
        nargs="?",
        help="the campaign root, an alias of --workspace",
    )
    archive.add_argument(
        "sim_id", help="the simulation to archive, as the manifest records it (sims/sim_<sim_id>/)"
    )
    archive.add_argument(
        "--workspace",
        help=WORKSPACE_HELP,
    )
    for command in (init, archive):
        add_console_options(command)
    return parser


def _campaign_root(args: argparse.Namespace) -> str:
    root = "." if args.root is None else str(args.root)
    if args.workspace is None:
        return root
    named = str(args.workspace)
    if root != "." and Path(root).resolve() != Path(named).resolve():
        raise WorkspaceError(
            f"the campaign is named twice: --workspace {named!r} and the positional root "
            f"{root!r}. Name it once."
        )
    return named


def main(argv: list[str] | None = None) -> int:
    """Initialize a workspace from command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Arguments excluding the program name; defaults to the process arguments.

    Returns
    -------
    int
        Zero for success, 2 for a workspace or filesystem refusal printed to stderr.
        Argument parsing exits with 0 for help and 2 for invalid arguments.
    """
    argv = sys.argv[1:] if argv is None else argv
    if (forwarded := dispatch(argv, workspace=True)) is not None:
        return forwarded
    args = build_parser().parse_args(argv)
    try:
        console = Console(level_of(args))
        root = Path(_campaign_root(args)).resolve()
    except WorkspaceError as error:
        print(str(error), file=sys.stderr)
        return 2
    with held_warnings(root, show=args.console_warnings):
        return _run(args, console)


def _run(args: argparse.Namespace, console: Console) -> int:
    if args.subcommand == "archive":
        try:
            root = Path(_campaign_root(args)).resolve()
            count = sum(1 for r in RunRegistry(root).read() if r.point.pol == args.sim_id)
            target = archive_sim(root, args.sim_id)
        except (OSError, WorkspaceError, RegistryError) as error:
            print(str(error), file=sys.stderr)
            return 2
        console.final(
            f"archived sim {args.sim_id} ({count} point(s)) to "
            f"{target.relative_to(root).as_posix()}; sims/sim_{args.sim_id}/ is removed"
        )
        return 0
    solver = str(args.solver)
    if solver not in available_solvers():
        if solver.lower() not in available_solvers():
            print(
                f"--solver {solver}: unknown solver; choose one of "
                f"{', '.join(available_solvers())}",
                file=sys.stderr,
            )
            return 2
        solver = solver.lower()
    target = Path(_campaign_root(args)).resolve()
    existed = target.is_dir() and any(target.iterdir())
    try:
        root = (
            init_flightstream(_campaign_root(args))
            if solver == "flightstream"
            else init_workspace(
                _campaign_root(args), solver=solver, available_solvers=available_solvers()
            )
        )
    except (OSError, WorkspaceError) as error:
        print(str(error), file=sys.stderr)
        return 2
    console.final(f"campaign workspace ready at {root}")
    console.say(f"  init.toml (solver: {solver})")
    for kind in input_kinds(solver):
        console.say(f"  inputs/{kind}/")
    if solver != "flightstream":
        console.say("  inputs/INPUTS.md (what each input file holds)")
        console.say("  inputs/.gitignore (keeps this machine's executable paths out of git)")
    for name in ("sims", "post", "archive"):
        console.say(f"  {name}/")
    if existed:
        console.say("existing folders and files were kept untouched")
    console.say("next: write inputs/matrices/<name>.fs, then cfd-matrix plan <name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

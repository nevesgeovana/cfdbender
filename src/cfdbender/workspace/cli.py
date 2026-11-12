"""The cfd-workspace command line: initialize a single-solver campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfdbender.core import WorkspaceError, init_workspace, input_kinds
from cfdbender.solvers import available_solvers
from cfdbender.solvers.flightstream import dispatch
from cfdbender.solvers.flightstream import init_workspace as init_flightstream


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfd-workspace",
        description="Managed campaign workspace tooling for CFD studies.",
        allow_abbrev=False,
    )
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
        help="campaign workspace (default: the current directory); the positional "
        "root names the same folder",
    )
    init.add_argument(
        "--solver",
        required=True,
        help=f"solver for this workspace; available solvers: {', '.join(available_solvers())}",
    )
    return parser


def _campaign_root(args: argparse.Namespace) -> str:
    root = str(args.root)
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
    args = _build_parser().parse_args(argv)
    try:
        root = (
            init_flightstream(_campaign_root(args))
            if args.solver == "flightstream"
            else init_workspace(
                _campaign_root(args), solver=args.solver, available_solvers=available_solvers()
            )
        )
    except (OSError, WorkspaceError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"campaign workspace ready at {root}")
    print(f"  init.toml (solver: {args.solver})")
    for kind in input_kinds(args.solver):
        print(f"  inputs/{kind}/")
    for name in ("sims", "post", "archive"):
        print(f"  {name}/")
    print("existing folders and files were kept untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

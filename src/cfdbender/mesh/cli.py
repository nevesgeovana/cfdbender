"""The cfd-mesh program: operations on surface meshes and the meshing of generated geometry.

Every verb reads its inputs, writes a new file (never the input) and prints its record as JSON.
A mesh named without a folder lands in the workspace's ``inputs/mesh/`` with a
``<file name>.provenance.json`` sidecar recording how it was made. Exit code 0 means done, 1
means a verb ran but its result failed its acceptance (``mesh``, ``audit``, ``merge``) and 2
means refused; a refusal names what is at fault and writes nothing.

The verbs are declared by group modules, each owning its parsers and handlers.
"""

from __future__ import annotations

import argparse
import sys
from types import ModuleType

from cfdbender.mesh import _cli_airfoil2d, _cli_edges, _cli_refine, _cli_rotor, _cli_surface

__all__ = ["build_parser", "main"]

#: The modules that declare the verbs, in the order the help lists them.
GROUPS: tuple[ModuleType, ...] = (_cli_surface, _cli_refine, _cli_edges, _cli_rotor, _cli_airfoil2d)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of cfd-mesh with every verb of its groups.

    Returns
    -------
    argparse.ArgumentParser
        The top-level parser; each verb sets ``handler`` to the function that runs it.
    """
    parser = argparse.ArgumentParser(
        prog="cfd-mesh",
        description=(
            "Operations on surface meshes: mesh generated geometry, refine or coarsen, audit, "
            "edit trailing edges, scale, morph, repitch and build rotor wheels. Each verb writes "
            "a new file and prints its record as JSON."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--workspace",
        default=".",
        metavar="PATH",
        help="the workspace a bare output name is written into (default: the current folder)",
    )
    verbs = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for group in GROUPS:
        group.add_parsers(verbs)
    for sub in verbs.choices.values():
        sub.allow_abbrev = False
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one cfd-mesh verb from command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Arguments excluding the program name; defaults to the process arguments.

    Returns
    -------
    int
        0 when done, 1 when a verb's result failed its acceptance, 2 when refused (the
        reason is printed to the error stream).
    """
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    code: int = args.handler(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

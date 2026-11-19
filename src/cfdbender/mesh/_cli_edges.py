"""The cfd-mesh verbs of trailing edges.

Verbs: trailing-edge, identify-te, convert2midnodes and collapse-te. Each prints its record as
JSON and records its main output in a ``<file name>.provenance.json``. Exit code 0 means done
and 2 means refused; a refusal names what is at fault and writes nothing. With
``--accept-empty`` a verb that finds nothing to write warns, writes nothing and exits 0.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cfdbender._errors import InputValueError
from cfdbender.geometry.trailing_edges import DEFAULT_SECTIONS, write_trailing_edge_node_file
from cfdbender.mesh._cli_support import mesh_target, print_checks, write_provenance
from cfdbender.mesh._te_nodes import (
    write_trailing_edge_midnodes,
    write_trailing_edge_midnodes_per_family,
    write_trailing_edge_points_from_nodes,
    write_trailing_edges_identified,
)
from cfdbender.mesh.collapse_te import collapse_trailing_edge

__all__ = ["add_parsers"]


def _run(work: Callable[[], dict[str, Any]]) -> int:
    return print_checks(work)


def _options(args: argparse.Namespace, *names: str) -> dict[str, Any]:
    return {name: getattr(args, name) for name in names}


# --- trailing-edge -----------------------------------------------------------------


def _add_trailing_edge(verbs: Any) -> None:
    edge = verbs.add_parser(
        "trailing-edge",
        help="extract trailing-edge midpoint coordinates",
        description=(
            "Writes a trailing-edge points file, the mid-point of every mesh edge on the "
            "trailing edge under a unit line. Either from a blade mesh (MESH OUTPUT with "
            "--axis, --hub and --unit: the aftmost vertex of each chordwise section, joined "
            "along the aft ridge) or from an ordered node-window export (--from-nodes, no "
            "mesh needed, --out names the file)."
        ),
        allow_abbrev=False,
    )
    edge.add_argument("mesh", nargs="?", help="the blade surface OBJ")
    edge.add_argument("output", nargs="?", help="the points file to write")
    edge.add_argument("--from-nodes", help="ordered node-window export; no mesh needed")
    edge.add_argument("--out", help="destination for --from-nodes; default: <input stem>.te.txt")
    edge.add_argument("--axis", nargs=3, type=float, metavar=("X", "Y", "Z"))
    edge.add_argument("--hub", nargs=3, type=float, metavar=("X", "Y", "Z"))
    edge.add_argument("--unit", help="coordinate unit; METER by default for --from-nodes")
    edge.add_argument("--sections", type=int)
    edge.add_argument("--overwrite", action="store_true")
    edge.set_defaults(handler=_run_trailing_edge)


def _run_trailing_edge(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        if args.from_nodes:
            if any(
                value is not None
                for value in (args.mesh, args.output, args.axis, args.hub, args.sections)
            ):
                raise InputValueError(
                    "--from-nodes takes no mesh, output positional, --axis, --hub or --sections; "
                    "use --out"
                )
            named = args.out or Path(args.from_nodes).with_suffix(".te.txt")
            target = mesh_target(args.workspace, named)
            path = write_trailing_edge_points_from_nodes(
                args.from_nodes, target, unit=args.unit or "METER", overwrite=args.overwrite
            )
            sources = [args.from_nodes]
        else:
            if args.out is not None or any(
                value is None for value in (args.mesh, args.output, args.axis, args.hub, args.unit)
            ):
                raise InputValueError(
                    "mesh extraction needs mesh output --axis --hub --unit; "
                    "--out belongs to --from-nodes"
                )
            path = write_trailing_edge_node_file(
                args.mesh,
                mesh_target(args.workspace, args.output),
                axis=args.axis,
                hub=args.hub,
                unit=args.unit,
                sections=args.sections if args.sections is not None else DEFAULT_SECTIONS,
                overwrite=args.overwrite,
            )
            sources = [args.mesh]
        provenance = write_provenance(
            path,
            command="trailing-edge",
            sources=sources,
            options=_options(args, "axis", "hub", "unit", "sections", "overwrite"),
            workspace=args.workspace,
        )
        return {"points": str(path), "provenance": str(provenance)}

    return _run(work)


# --- convert2midnodes --------------------------------------------------------------


def _add_midnodes(verbs: Any) -> None:
    midnodes = verbs.add_parser(
        "convert2midnodes",
        help="write a .te.txt from a node-set list, in any node order; no mesh needed",
        description=(
            "Reads the node list a mesh-preparation tool saves from a node set (Id, Name, X, Y, "
            "Z), puts the nodes in order along their lines (a long jump starts a new line, so "
            "one list may hold every trailing edge) and writes one mid-point per neighboring "
            "pair."
        ),
        allow_abbrev=False,
    )
    midnodes.add_argument("nodes", metavar="NODES", help="the node list saved from the node set")
    midnodes.add_argument("output", metavar="OUTPUT", help="the .te.txt to write")
    midnodes.add_argument("--unit", default="METER", help="the coordinates' unit (default: METER)")
    midnodes.add_argument("--overwrite", action="store_true", help="replace an existing OUTPUT")
    midnodes.add_argument(
        "--accept-empty",
        action="store_true",
        help="a list with fewer than two nodes, no usable node or no edge of MESH is warned and "
        "nothing is written, exit 0, in place of refusing it",
    )
    midnodes.add_argument(
        "--per-family",
        metavar="MESH",
        help="the OBJ the nodes belong to: write <OUTPUT stem>.<family>.te.txt per family from "
        "its edges joining two listed nodes, in place of OUTPUT",
    )
    midnodes.set_defaults(handler=_run_midnodes)


def _run_midnodes(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        output = mesh_target(args.workspace, args.output)
        if args.per_family:
            paths = write_trailing_edge_midnodes_per_family(
                args.nodes,
                args.per_family,
                output,
                unit=args.unit,
                overwrite=args.overwrite,
                accept_empty=args.accept_empty,
            )
            sources = [args.nodes, args.per_family]
        else:
            written = write_trailing_edge_midnodes(
                args.nodes,
                output,
                unit=args.unit,
                overwrite=args.overwrite,
                accept_empty=args.accept_empty,
            )
            paths = [] if written is None else [written]
            sources = [args.nodes]
        provenance = [
            str(
                write_provenance(
                    path,
                    command="convert2midnodes",
                    sources=sources,
                    options=_options(args, "unit", "overwrite", "accept_empty", "per_family"),
                    workspace=args.workspace,
                )
            )
            for path in paths
        ]
        return {"points": [str(p) for p in paths], "provenance": provenance}

    return _run(work)


# --- identify-te -------------------------------------------------------------------


def _add_identify(verbs: Any) -> None:
    identify = verbs.add_parser(
        "identify-te",
        help="find the trailing edges of an OBJ and write their points files",
        description=(
            "A trailing edge is a family's rearmost line: the edges its surface lies forward "
            "of. Without NODES they are found from the mesh alone (each family cut at its "
            "sharp edges, an edge kept when it lies within --max-angle of the plane across "
            "--aft, the yz plane by default; a sharp trailing edge is followed to a twisted "
            "tip and a closed tip cap's contour is left out; a perimeter line found for more "
            "than half its edges is taken whole, to its corners) and the nodes are also "
            "written as <mesh stem>.nodes.txt, the node list a refine file reads. With NODES "
            "(a node-set list, every perimeter in one file, any order) they are taken among "
            "its lines, on the OBJ or a refined level of it. Written beside the mesh: "
            "<mesh stem>.te.txt, the points file a boundaries file names, for every mapped "
            "family; or <mesh stem>.<family>.te.txt per family with --families. Without "
            "--families only quad families are mapped (quadrilaterals at least 80 percent of "
            "the faces); --all-families maps the others too."
        ),
        allow_abbrev=False,
    )
    identify.add_argument("mesh", metavar="OBJ", help="the mesh, with its families")
    identify.add_argument(
        "nodes",
        metavar="NODES",
        nargs="?",
        help="the node list saved from the node set (default: the edges are found from the mesh)",
    )
    identify.add_argument(
        "--families",
        metavar="A,B",
        help="the families, a points file each (default: every quad family, one file)",
    )
    identify.add_argument(
        "--all-families",
        action="store_true",
        help="without --families, map every family, not only the quad families",
    )
    identify.add_argument(
        "--max-angle",
        type=float,
        default=45.0,
        metavar="DEG",
        help="without NODES, the largest angle of an edge with the plane across --aft (default 45)",
    )
    identify.add_argument(
        "--aft", nargs=3, type=float, metavar=("X", "Y", "Z"), help="the aft direction (default +x)"
    )
    identify.add_argument("--unit", help="the coordinates' unit (default: the mesh's, else METER)")
    identify.add_argument("--overwrite", action="store_true", help="replace existing files")
    identify.add_argument(
        "--accept-empty",
        action="store_true",
        help="a NODES list with fewer than two nodes, or no trailing edge found, is warned and "
        "nothing is written, exit 0, in place of refusing it",
    )
    identify.set_defaults(handler=_run_identify)


def _run_identify(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        paths = write_trailing_edges_identified(
            args.mesh,
            args.nodes,
            None
            if args.families is None
            else [n.strip() for n in args.families.split(",") if n.strip()],
            aft=args.aft or (1.0, 0.0, 0.0),
            max_angle=args.max_angle,
            unit=args.unit,
            overwrite=args.overwrite,
            accept_empty=args.accept_empty,
            all_families=args.all_families,
        )
        provenance = []
        if paths:
            provenance.append(
                str(
                    write_provenance(
                        paths[0],
                        command="identify-te",
                        sources=[args.mesh] + ([args.nodes] if args.nodes else []),
                        options=_options(
                            args,
                            "families",
                            "all_families",
                            "max_angle",
                            "aft",
                            "unit",
                            "overwrite",
                        ),
                        workspace=args.workspace,
                        outputs={"files": [str(p) for p in paths]},
                    )
                )
            )
        return {"files": [str(p) for p in paths], "provenance": provenance}

    return _run(work)


# --- collapse-te -------------------------------------------------------------------


def _add_collapse(verbs: Any) -> None:
    parser = verbs.add_parser(
        "collapse-te",
        help="collapse the blunt trailing edge of named blades of an OBJ",
        description=(
            "MESH is an OBJ. The [collapse_te] table of the morphing file names the blades; "
            "each pair of rim nodes across a blade's one-cell base strip merges at its "
            "midpoint and the strip leaves. Every other node and face keeps its place; the "
            "boundaries file gets one trailing-edge line per blade and the checks are written "
            "to <out>.collapse.json. Quality findings (a face the merge turns over) are warned "
            "with their numbers and the mesh is written; a broken input or invariant refuses "
            "and nothing is written."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("mesh", help="source .obj, read only")
    parser.add_argument("--morphing", required=True, help="TOML with a [collapse_te] table")
    parser.add_argument(
        "--out", required=True, help="new .obj; a bare name lands in the workspace's inputs/mesh/"
    )
    parser.set_defaults(handler=_run_collapse)


def _run_collapse(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        target = mesh_target(args.workspace, args.out)
        record = collapse_trailing_edge(args.mesh, args.morphing, target)
        provenance = write_provenance(
            target,
            command="collapse-te",
            sources=[args.mesh, args.morphing],
            options={},
            workspace=args.workspace,
        )
        return {**record, "provenance": str(provenance)}

    return _run(work)


def add_parsers(verbs: Any) -> None:
    """Add this group's verbs to the cfd-mesh subparsers.

    Parameters
    ----------
    verbs : argparse._SubParsersAction
        The subparsers of :func:`cfdbender.mesh.cli.build_parser`.
    """
    _add_trailing_edge(verbs)
    _add_midnodes(verbs)
    _add_identify(verbs)
    _add_collapse(verbs)

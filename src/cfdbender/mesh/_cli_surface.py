"""The cfd-mesh verbs that mesh generated geometry with the kernel: mesh, mesh-generated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cfdbender._errors import GeometryError
from cfdbender._textio import write_json
from cfdbender.core.mesh import MESH_FOLDER
from cfdbender.mesh._cli_support import mesh_target, print_checks, write_provenance

__all__ = ["add_parsers"]


def add_parsers(verbs: Any) -> None:
    """Add this group's verbs to the cfd-mesh subparsers.

    Parameters
    ----------
    verbs : argparse._SubParsersAction
        The subparsers of :func:`cfdbender.mesh.cli.build_parser`.
    """
    mesh = verbs.add_parser(
        "mesh",
        help="mesh a parametric configuration from its mesh controls (strategy 1 or 2, gmsh)",
        description=(
            "Validates the mesh controls against the configuration (errors E01 to E16 refuse "
            "by name, W01 warns), plans every component's counts, meshes the configuration "
            "with strategy 1, or strategy 2 (full grid) when mesh.strategy = 2, and measures the "
            "acceptance table of that strategy on the realized mesh. With --dry-run it prints "
            "the plan and the warnings and writes nothing. Otherwise it writes "
            "<descriptor>.obj and <descriptor>.mesh.json into --out (default: inputs/mesh/ of "
            "the workspace), with the provenance sidecar of the OBJ."
        ),
        allow_abbrev=False,
    )
    mesh.add_argument("config", metavar="CONFIG", help="the configuration (TOML)")
    mesh.add_argument("--controls", required=True, metavar="TOML", help="the mesh controls")
    mesh.add_argument(
        "--dry-run", action="store_true", help="validate and print the planned counts only"
    )
    mesh.add_argument(
        "--out",
        metavar="DIR",
        help="folder of the written mesh and record (default: inputs/mesh/ of the workspace)",
    )
    mesh.add_argument("--descriptor", metavar="NAME", help="file stem (mesh.output.descriptor)")
    mesh.set_defaults(handler=run_mesh)

    generated = verbs.add_parser(
        "mesh-generated",
        help="mesh a parametric configuration into an all-quad OBJ (uses the gmsh kernel)",
        description=(
            "Resolves the configuration and meshes its patches from their role: constant "
            "spanwise width with a refined tip, chordwise growth from the leading edge, "
            "constant azimuth and axial step on a body, a planar all-quad flat tip cap and, "
            "for a wing on a body, the fairing as one quad-dominant transition ring. The "
            "orientation is outward. The record (counts, the acceptance checks, the spacing "
            "rules) is printed; the OBJ holds one group per component. A bare --obj name is "
            "written into inputs/mesh/ of the workspace, with its provenance sidecar."
        ),
        allow_abbrev=False,
    )
    generated.add_argument("config", metavar="CONFIG", help="the configuration (TOML)")
    generated.add_argument("--obj", required=True, help="new .obj")
    generated.add_argument(
        "--component",
        metavar="NAME",
        help="mesh this body or lifting surface alone, in its own frame (default: the assembly)",
    )
    generated.add_argument(
        "--controls",
        metavar="TOML",
        help="mesh controls overriding the defaults, by table: lifting_surface, body, checks, "
        "assembly",
    )
    generated.add_argument(
        "--out",
        metavar="JSON",
        help="also write the record to this file (a file, unlike mesh --out, which is a folder)",
    )
    generated.set_defaults(handler=run_mesh_generated)


def run_mesh_generated(args: argparse.Namespace) -> int:
    """Run mesh-generated: write the OBJ and print the record of its checks.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line.

    Returns
    -------
    int
        0 when written, 2 when refused.
    """
    from cfdbender.geometry import load_config
    from cfdbender.geometry._identify import mesh_generated

    def work() -> dict[str, Any]:
        obj = mesh_target(args.workspace, args.obj)
        resolved = load_config(args.config)
        result = mesh_generated(
            resolved, component=args.component, controls=args.controls, obj=str(obj)
        )
        record: dict[str, Any] = {"obj": str(obj), **result.record}
        if args.out is not None:
            write_json(args.out, record, default=str)
        sources = [args.config, *([args.controls] if args.controls else [])]
        options = {"component": args.component, "controls": args.controls}
        write_provenance(
            obj,
            command="mesh-generated",
            sources=sources,
            options=options,
            workspace=args.workspace,
        )
        return record

    return print_checks(work)


def run_mesh(args: argparse.Namespace) -> int:
    """Plan or mesh a configuration; exit 0 accepted or planned, 1 not accepted, 2 refused.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line.

    Returns
    -------
    int
        0 accepted or planned, 1 meshed and not accepted, 2 refused.
    """
    if args.descriptor is not None and args.dry_run:
        print(
            "mesh: --descriptor names written files, so it needs no --dry-run. Nothing was meshed.",
            file=sys.stderr,
        )
        return 2
    from cfdbender.geometry._identify import mesh_configuration

    out = None if args.dry_run else Path(args.out or Path(args.workspace) / MESH_FOLDER)
    try:
        result = mesh_configuration(
            args.config,
            args.controls,
            dry_run=args.dry_run,
            out=out,
            descriptor=args.descriptor,
        )
        record = result.record
        obj = result.written.get("obj")
        if obj is not None:
            write_provenance(
                obj,
                command="mesh",
                sources=[args.config, args.controls],
                options={"descriptor": args.descriptor, "strategy": record.get("strategy")},
                workspace=args.workspace,
                outputs={"record": Path(result.written["record"]).name},
            )
    except (GeometryError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    for warning in record["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    shown = {k: record[k] for k in ("strategy", "variant", "members", "components")}
    if not args.dry_run:
        shown.update(counts=record["counts"], accepted=record["accepted"])
        shown["failed"] = [c["id"] for c in record["checks"] if c["status"] == "failed"]
    print(json.dumps(shown, indent=2, default=str))
    for path in result.written.values():
        print(path)
    return 0 if args.dry_run or record["accepted"] else 1

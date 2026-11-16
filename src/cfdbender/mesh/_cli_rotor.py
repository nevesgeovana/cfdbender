"""The cfd-mesh verbs of rotors and whole meshes.

Verbs: scale, morph, repitch, hub-rings, sector-to-wheel,
prop-slice, iges-grid and visual-inspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cfdbender.mesh._cli_support import (
    mesh_target,
    print_checks,
    provenance_path,
    write_provenance,
)

__all__ = ["add_parsers"]


def _scale_parser(verbs: Any) -> None:
    parser = verbs.add_parser(
        "scale",
        help="scale every vertex of an OBJ by a factor about a point",
        description=(
            "FACTOR is a positive number (0.5 halves every length). Every vertex of the OBJ "
            "given by --mesh is scaled about the origin, or about --about x,y,z; face order, "
            "groups, normals and comments stay byte for byte. --out names the scaled OBJ "
            "(default <stem>_scaled<factor>.obj in the workspace's inputs/mesh); "
            "<name>.scale.json beside it holds the input digest, factor, centre, counts and "
            "the bounding boxes before and after. The input is never overwritten; an existing "
            "output is refused unless --overwrite."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("factor", help="positive uniform scale factor, for example 0.5")
    parser.add_argument("--mesh", required=True, help="source OBJ, read only")
    parser.add_argument("--out", help="the scaled OBJ; default <stem>_scaled<factor>.obj")
    parser.add_argument("--about", help="scale centre x,y,z (default 0,0,0)")
    parser.add_argument("--overwrite", action="store_true", help="replace the output if it exists")
    parser.set_defaults(handler=_run_scale)


def _run_scale(args: argparse.Namespace) -> int:
    from cfdbender.mesh.scale import parse_factor, scale_mesh

    def work() -> dict[str, Any]:
        value = parse_factor(args.factor)
        source = Path(args.mesh)
        named = args.out or f"{source.stem}_scaled{value:g}{source.suffix or '.obj'}"
        target = mesh_target(args.workspace, named)
        sidecar = provenance_path(target)
        if not args.overwrite and (sidecar.exists() or sidecar.is_symlink()):
            raise ValueError(f"{sidecar}: existing provenance; refusing overwrite")
        record = scale_mesh(
            source,
            value,
            out=target.parent,
            about=args.about,
            overwrite=args.overwrite,
            name=target.name,
        )
        sidecar = write_provenance(
            target,
            command="scale",
            sources=[source],
            options={"factor": value, "about": args.about, "overwrite": args.overwrite},
            workspace=args.workspace,
            outputs={"vertices": record["vertices"], "faces": record["faces"]},
        )
        record["provenance"] = str(sidecar)
        return record

    return print_checks(work)


def _prop_slice_parser(verbs: Any) -> None:
    parser = verbs.add_parser(
        "prop-slice",
        help="cut a propeller OBJ into blade sections and a propeller definition",
        description=(
            "Isolates one blade about the axis, cuts it at radial stations by the planes the "
            "propeller builder places its sections in, and writes to --out the normalized "
            "profile of each station (sections/sNN.dat), the table over r/R of chord, pitch "
            "angle, thickness, camber, skew and rake (table.csv), the propeller definition "
            "that rebuilds the propeller (propeller.toml) and the record prop-slice.json."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--geometry", required=True, help="the propeller, an .obj")
    parser.add_argument("--blades", required=True, type=int, help="the number of blades")
    parser.add_argument(
        "--axis",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="thrust direction, the builder's frame.thrust_dir (default -1 0 0)",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="a point of the shaft in mm (default 0 0 0)",
    )
    parser.add_argument(
        "--blade-dir",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="reference blade direction; blade 0 is the nearest (default 0 0 1)",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        type=float,
        metavar="R_R",
        help="r/R of the sections (default 21 steps from the root to the tip, and 0.7)",
    )
    parser.add_argument(
        "--unit",
        default="mm",
        help="OBJ length unit: mm (default), cm, m, in, or MILLIMETER, CENTIMETER, METER, INCH",
    )
    parser.add_argument("--out", required=True, help="the folder to write")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the outputs --out already holds (refused without it)",
    )
    parser.set_defaults(handler=_run_prop_slice)


def _run_prop_slice(args: argparse.Namespace) -> int:
    from cfdbender.mesh.prop_slice import slice_propeller

    def work() -> dict[str, Any]:
        folder = Path(args.out)
        if not folder.is_absolute():
            folder = Path(args.workspace) / folder
        record = slice_propeller(
            args.geometry,
            blades=args.blades,
            out=folder,
            axis=args.axis,
            origin=args.origin,
            blade_dir=args.blade_dir,
            stations=args.stations,
            unit=args.unit,
            overwrite=args.overwrite,
        )
        sidecar = write_provenance(
            folder / "propeller.toml",
            command="prop-slice",
            sources=[args.geometry],
            options={
                "blades": args.blades,
                "axis": args.axis,
                "origin": args.origin,
                "blade_dir": args.blade_dir,
                "stations": args.stations,
                "unit": args.unit,
                "overwrite": args.overwrite,
            },
            workspace=args.workspace,
            outputs={"folder": str(folder), "sections": len(record["files"]["sections"])},
        )
        record["provenance"] = str(sidecar)
        return record

    return print_checks(work)


def _sector_to_wheel_parser(verbs: Any) -> None:
    wheel = verbs.add_parser(
        "sector-to-wheel",
        help="copy one sector OBJ about an axis into a full wheel OBJ",
        description=(
            "SECTOR is an OBJ (with its .obj.boundaries.toml and trailing-edge points files "
            "beside it). The sector is copied --copies times about --axis through --center, "
            "the nodes of the seams are merged within --tolerance, the surfaces named in "
            "--blades are numbered one per copy and every other surface is one group. The "
            "wheel is an OBJ with its boundaries file in the named form, one points file per "
            "rim per blade and <out stem>.wheel.json. --out names the wheel (default "
            "<sector stem>_wheel.obj in the workspace's inputs/mesh). A saved simulation is "
            "not accepted."
        ),
        allow_abbrev=False,
    )
    wheel.add_argument("sector", metavar="SECTOR", help="the sector OBJ")
    wheel.add_argument("--copies", type=int, required=True, help="the number of copies, 2 or more")
    wheel.add_argument(
        "--blades",
        nargs="+",
        required=True,
        metavar="FAMILY",
        help="the OBJ group(s) that are the blade",
    )
    wheel.add_argument(
        "--axis",
        default="x",
        help="x, y, z, a signed one of them written --axis=-x, or three comma-separated numbers "
        "(default x)",
    )
    wheel.add_argument(
        "--center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="a point of the axis, in the mesh's unit (default the origin)",
    )
    wheel.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        metavar="METRES",
        help="nodes within this distance, in metres, are one node (default 1e-4)",
    )
    wheel.add_argument("--out", metavar="PATH", help="the wheel (default <sector stem>_wheel.obj)")
    wheel.add_argument("--overwrite", action="store_true", help="replace an existing wheel")
    wheel.set_defaults(handler=_run_sector_to_wheel)


def _run_sector_to_wheel(args: argparse.Namespace) -> int:
    from cfdbender.mesh._wheel import sector_to_wheel

    def work() -> dict[str, Any]:
        source = Path(args.sector)
        named = args.out or f"{source.stem}_wheel.obj"
        target = mesh_target(args.workspace, named)
        result = sector_to_wheel(
            source,
            copies=args.copies,
            blades=args.blades,
            axis=args.axis,
            center=args.center,
            tolerance=args.tolerance,
            out=target,
            overwrite=args.overwrite,
        )
        if result.output is None:
            raise ValueError("sector-to-wheel wrote nothing")
        sidecar = write_provenance(
            result.output,
            command="sector-to-wheel",
            sources=[source],
            options={
                "copies": args.copies,
                "blades": args.blades,
                "axis": args.axis,
                "center": args.center,
                "tolerance": args.tolerance,
                "overwrite": args.overwrite,
            },
            workspace=args.workspace,
            outputs={"files": [str(path) for path in result.files]},
        )
        return {
            "output": str(result.output),
            "files": [str(path) for path in result.files],
            "summary": result.lines(),
            "report": result.report,
            "provenance": str(sidecar),
        }

    return print_checks(work)


def _morphing_verb(
    verbs: Any, name: str, help_text: str, description: str, module: str, function: str
) -> None:
    """Add one verb of the morphing-file family: MESH, --morphing, --out."""
    parser = verbs.add_parser(name, help=help_text, description=description, allow_abbrev=False)
    parser.add_argument("mesh", help="source .obj, read only")
    parser.add_argument("--morphing", required=True, help="TOML control file of the verb")
    parser.add_argument(
        "--out", required=True, help="new .obj (a bare name lands in the workspace's inputs/mesh)"
    )
    parser.set_defaults(handler=lambda args: _run_morphing(args, name, module, function))


def _run_morphing(args: argparse.Namespace, name: str, module: str, function: str) -> int:
    from importlib import import_module

    run = getattr(import_module(module), function)

    def work() -> dict[str, Any]:
        source, control = Path(args.mesh), Path(args.morphing)
        target = mesh_target(args.workspace, args.out)
        record = run(source, control, target)
        record["provenance"] = str(
            write_provenance(
                target,
                command=name,
                sources=[source, control],
                options={"morphing": str(control)},
                workspace=args.workspace,
            )
        )
        return dict(record)

    return print_checks(work)


def _repitch_parser(verbs: Any) -> None:
    _morphing_verb(
        verbs,
        "repitch",
        "rigidly repitch blades and rebuild the named hub around their roots",
        (
            "MESH is an OBJ. The [repitch] table of the morphing file names the blades, the hub "
            "and the angle; each blade turns rigidly about its pitch axis; [repitch] hub_method "
            "chooses how the hub follows: its complete quad rings move (rings), the band around "
            "each turned root is remeshed quad-dominant (remesh), or auto picks one. The checks "
            "are written to <out>.repitch.json. Quality findings are warned with their numbers "
            "and the mesh is written; a broken input or invariant refuses and nothing is "
            "written."
        ),
        "cfdbender.mesh.repitch",
        "repitch_mesh",
    )


def _hub_rings_parser(verbs: Any) -> None:
    _morphing_verb(
        verbs,
        "hub-rings",
        "rebuild the named hub around blade roots as rings of quads, on an OBJ",
        (
            "MESH is an OBJ. The [hub_rings] table of the morphing file states rings (and "
            "optionally band_width); blades and hub sit at the top or in the table. Per "
            "blade, rings rows of quads are marched outward from the root on the hub, the hub "
            "faces they reach plus one layer leave, and the band between them is remeshed "
            "quad-dominant. Blade nodes, other families and the hub's seams stay to the bit; the "
            "checks are written to <out>.hub_rings.json. Quality findings are warned with their "
            "numbers and the mesh is written; a broken input or invariant refuses and nothing "
            "is written."
        ),
        "cfdbender.mesh.hub_rings",
        "hub_rings_mesh",
    )


def _iges_grid_parser(verbs: Any) -> None:
    parser = verbs.add_parser(
        "iges-grid",
        help="grid the named components of IGES files into one OBJ (uses the gmsh kernel)",
        description=(
            "Each component the IGES files name (the NAME of a level) becomes one family of the "
            "OBJ. Faces are joined so shared edges are conformal; four-sided patches follow the "
            "laws of the grid control file and become quads, other patches are quad-dominant at "
            "the component's size. Writes the OBJ, <out>.boundaries.toml, one points file per "
            "trailing edge found and the record <out>.json. Each junction of two components is "
            "measured against the original faces of both. Quality is warned, never refused."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("iges", nargs="+", help="IGES files, read only")
    parser.add_argument("--grid", required=True, help="grid control file (TOML)")
    parser.add_argument(
        "--out", required=True, help="new .obj (a bare name lands in the workspace's inputs/mesh)"
    )
    parser.add_argument(
        "--component",
        dest="components",
        action="append",
        metavar="NAME",
        help=(
            "grid only this component, named as the files name it (repeatable); the others "
            "leave the model before the join. An unknown or ambiguous name is refused with the "
            "names found. Default: every component"
        ),
    )
    parser.add_argument(
        "--name-encoding",
        default="latin-1",
        help=(
            "text encoding of the NAME bytes in the IGES files (default latin-1, one "
            "character per byte; use utf-8 for UTF-8 names). Never guessed: a name invalid "
            "in the encoding is refused"
        ),
    )
    parser.set_defaults(handler=_run_iges_grid)


def _run_iges_grid(args: argparse.Namespace) -> int:
    from cfdbender.mesh.iges_grid import grid_iges

    def work() -> dict[str, Any]:
        target = mesh_target(args.workspace, args.out)
        record = grid_iges(
            args.iges,
            args.grid,
            target,
            name_encoding=args.name_encoding,
            components=args.components,
        )
        sidecar = write_provenance(
            target,
            command="iges-grid",
            sources=[*args.iges, args.grid],
            options={"components": args.components, "name_encoding": args.name_encoding},
            workspace=args.workspace,
            outputs={"nodes": record["nodes"], "faces": record["faces"]},
        )
        record["provenance"] = str(sidecar)
        return dict(record)

    return print_checks(work)


def _visual_inspect_parser(verbs: Any) -> None:
    inspect = verbs.add_parser(
        "visual-inspect",
        help="draw a mesh in three views and an isometric view, with marked points, into a PDF",
        description=(
            "Writes <mesh>.inspect.pdf: the top, front, side and isometric views as a technical "
            "drawing (one scale, aligned, a scale bar), one color per patch from --cmap, each "
            "patch's perimeter in a thick line, the node lists and trailing-edge points files "
            "over the surface, and a table of each patch's faces and marked points. MESH is an "
            "OBJ or a NASTRAN export."
        ),
        allow_abbrev=False,
    )
    inspect.add_argument("mesh", metavar="MESH", help="the OBJ or NASTRAN mesh")
    inspect.add_argument(
        "--nodes",
        action="extend",
        nargs="+",
        default=[],
        metavar="FILE",
        help="node lists saved from an ANSA SET; repeat the flag or put MESH first",
    )
    inspect.add_argument(
        "--te",
        action="extend",
        nargs="+",
        default=[],
        metavar="FILE",
        help=(
            "trailing-edge points files (.te.txt, .nodes.te.txt), one color each; "
            "repeat the flag or put MESH first"
        ),
    )
    inspect.add_argument(
        "--cmap", help="the matplotlib colormap of the patches (default: matplotlib's default)"
    )
    inspect.add_argument("--families", metavar="A,B", help="draw only these patches")
    inspect.add_argument(
        "--box",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="draw only the faces and points inside this box",
    )
    inspect.add_argument(
        "--mesh-unit",
        help=(
            "the drawing's length unit: an OBJ's own unit, or the unit a NASTRAN mesh is "
            "converted to (default METER); the scale bar names it"
        ),
    )
    inspect.add_argument(
        "--points-unit", help="the unit of the node lists and points files that state none"
    )
    inspect.add_argument(
        "--unit",
        default="MILLIMETER",
        help="a NASTRAN mesh's length unit (default MILLIMETER); not read for an OBJ",
    )
    inspect.add_argument(
        "--iso-from",
        nargs=3,
        type=float,
        default=(1.0, -1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="the direction the isometric view is seen from (default 1 -1 1)",
    )
    inspect.add_argument("--labels", action="store_true", help="write each patch's name on it")
    inspect.add_argument("--no-edges", action="store_true", help="draw the faces without edges")
    inspect.add_argument(
        "--perimeter-width",
        type=float,
        default=0.9,
        metavar="PT",
        help="width of the patch perimeters in points (default 0.9; 0 draws none)",
    )
    inspect.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="resolution of the rasterized surface in dots per inch (default 300)",
    )
    inspect.add_argument("--out", help="the PDF (default: <mesh>.inspect.pdf beside the mesh)")
    inspect.add_argument("--overwrite", action="store_true", help="replace an existing PDF")
    inspect.add_argument(
        "--png", action="store_true", help="also write each page of views as a PNG beside the PDF"
    )
    inspect.set_defaults(handler=_run_visual_inspect)


def _run_visual_inspect(args: argparse.Namespace) -> int:
    from cfdbender.mesh._visual_inspect import visual_inspect

    def work() -> dict[str, Any]:
        out = None
        if args.out is not None:
            out = Path(args.out)
            if not out.is_absolute():
                out = Path(args.workspace) / out
            out.parent.mkdir(parents=True, exist_ok=True)
        path = visual_inspect(
            args.mesh,
            nodes=args.nodes,
            te=args.te,
            out=out,
            cmap=args.cmap,
            families=args.families.split(",") if args.families else None,
            box=args.box,
            mesh_unit=args.mesh_unit,
            points_unit=args.points_unit,
            nastran_unit=args.unit,
            iso_from=args.iso_from,
            labels=args.labels,
            edges=not args.no_edges,
            perimeter_width=args.perimeter_width,
            dpi=args.dpi,
            overwrite=args.overwrite,
            png=args.png,
        )
        sidecar = write_provenance(
            path,
            command="visual-inspect",
            sources=[args.mesh, *args.nodes, *args.te],
            options={
                "cmap": args.cmap,
                "families": args.families,
                "box": args.box,
                "mesh_unit": args.mesh_unit,
                "points_unit": args.points_unit,
                "unit": args.unit,
                "iso_from": list(args.iso_from),
                "labels": args.labels,
                "edges": not args.no_edges,
                "perimeter_width": args.perimeter_width,
                "dpi": args.dpi,
                "png": args.png,
                "overwrite": args.overwrite,
            },
            workspace=args.workspace,
        )
        return {"pdf": str(path), "provenance": str(sidecar)}

    return print_checks(work)


def _morph_parser(verbs: Any) -> None:
    _morphing_verb(
        verbs,
        "morph",
        "apply the morphing file's tables in order: hub rings, repitch, collapse, wheel",
        (
            "MESH is an OBJ, a sector or a full wheel. The tables the morphing file holds are "
            "applied in one fixed order: [hub_rings], [repitch], [collapse_te], [wheel] (a "
            "sector only), each with its own checks; blades and hub sit at the top of the file. "
            "One OBJ is written with its boundaries file, its points files and "
            "<out>.morph.json. Each step warns of quality with its numbers and goes on; "
            "nothing is written when a step refuses a broken input or invariant."
        ),
        "cfdbender.mesh.morph",
        "morph_mesh",
    )


def add_parsers(verbs: Any) -> None:
    """Add this group's verbs to the cfd-mesh subparsers.

    Parameters
    ----------
    verbs : argparse._SubParsersAction
        The subparsers of :func:`cfdbender.mesh.cli.build_parser`.
    """
    _scale_parser(verbs)
    _sector_to_wheel_parser(verbs)
    _repitch_parser(verbs)
    _hub_rings_parser(verbs)
    _morph_parser(verbs)
    _iges_grid_parser(verbs)
    _visual_inspect_parser(verbs)
    _prop_slice_parser(verbs)

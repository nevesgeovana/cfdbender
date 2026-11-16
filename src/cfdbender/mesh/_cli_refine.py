"""The cfd-mesh verbs that refine, coarsen and audit a mesh.

Verbs: refine, sector-remesh, prepare-to-refine, audit, inventory, degenerate and merge.
New meshes default to the workspace's inputs/mesh/; ``--out`` names another location.
Audit and inventory records stay beside their mesh. None modifies its input. Every verb
prints its record as JSON and records its output in a ``<file name>.provenance.json``.

Exit codes: 0 when the work was done, 2 when the input was refused (the reason is printed to
the error stream and nothing is written). ``audit`` exits 1 when a gate or a check fails: the
audit is still written, and each failure is a warning.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cfdbender._errors import InputArtifactError
from cfdbender.core.mesh import MESH_FOLDER
from cfdbender.mesh._cli_support import mesh_target, print_checks, write_provenance
from cfdbender.mesh._degenerate import derive_thin_blade
from cfdbender.mesh._merge import merge_objs
from cfdbender.mesh._refine import MeshAudit, audit_mesh, prepare_to_refine, refine_mesh
from cfdbender.mesh._sidecars import write_inventory

__all__ = ["add_parsers"]

_AUDIT_DESCRIPTION = (
    "Audits an OBJ and writes <stem>.audit.json beside it. The gates G1 to G6 judge the "
    "topology, the orientation, the open boundary, the trailing edge and the nodes families "
    "share; the relative checks compare the 95th percentiles of the skewness, the warp and the "
    "size growth with the source's; the other figures are reported and never judged. Without "
    "--against only G1, G2 and G4 are judged. The trailing-edge points file is the one "
    "<stem>.boundaries.toml names, or else <stem>.te.txt. A gate or check that fails is a "
    "warning, and the audit is still written. Exits 0 when every gate and check passes, 1 when "
    "one fails, 2 when the input is refused or when the --out CSV cannot be written after the "
    "audit was."
)

_REFINE_DESCRIPTION = (
    "Refines or coarsens a panel mesh from its OBJ into a new geometry folder "
    "<out>/<stem>_<tag>/ holding the OBJ, the trailing-edge points file, the boundaries "
    "file, <stem>_<tag>.refine.json and the audit against the source. FACTOR multiplies the "
    "intervals of each index direction of a grid family and divides the target edge length of "
    "a remeshed family; a FACTOR below 1 coarsens. --parallel and --normal replace it along "
    "and across the reference perimeter of the grid families ([[perimeter]] node lists of the "
    "refinement file, else the family's longest boundary side). Without FACTOR, --parallel or "
    "--normal, the factors come from the refinement file (--config, or <stem>.refine.toml "
    "beside the mesh), whose [components], [refine] tag and elements (triangles or "
    "quad-dominant) are read whenever it exists. Faces are written in the source's order. The "
    "source and its folder are never modified. Exits 0 when the level is written, even when "
    "its audit fails: each failed gate or check is a warning and the audit is written beside "
    "the level, while audit exits 1 on the same audit. Exits 2 when the request is refused, "
    "and then writes nothing."
)


def _number(text: str) -> float | str:
    """Return a factor as typed: a float when it reads as one, else the text.

    The refusal of a text that is not a number is the function's own, so the verb and
    :func:`~cfdbender.mesh._refine.refine_mesh` refuse it with the same words.
    """
    try:
        return float(text)
    except ValueError:
        return text


def _names(text: str | None) -> list[str] | None:
    """Return the family names of a comma separated option, as ``--families`` takes them."""
    return None if text is None else [n.strip() for n in text.split(",") if n.strip()]


def _progress_printer(stage: str) -> Callable[[str], None]:
    """Return a printer of a long step's progress, with clock time and elapsed seconds."""
    start = time.monotonic()

    def say(line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stage} {stamp} +{time.monotonic() - start:.1f} s] {line}", file=sys.stderr)

    return say


def _options(args: argparse.Namespace, *names: str) -> dict[str, Any]:
    """Return the parsed options named, as JSON values."""
    return {name: getattr(args, name) for name in names}


def _run(work: Callable[[], dict[str, Any]]) -> int:
    """Run one verb's work and return its exit code; a ``passed`` entry of False gives 1."""
    record: dict[str, Any] = {}

    def wrapped() -> dict[str, Any]:
        record.update(work())
        return record

    code = print_checks(wrapped)
    if code == 0 and record.get("passed") is False:
        return 1
    return code


# --- refine ------------------------------------------------------------------------


def _add_refine(verbs: Any, *, remesh: bool = False) -> None:
    name = "sector-remesh" if remesh else "refine"
    refine = verbs.add_parser(
        name,
        help=(
            "remesh an OBJ sector without a solver, including its structured families"
            if remesh
            else "refine or coarsen a panel mesh from its OBJ into a new geometry folder"
        ),
        description=(
            "Remeshes the selected families of a sector OBJ using the generic surface "
            "remesher, including grids and families at factor 1. FACTOR divides the local "
            "target edge length; below 1 coarsens. The refinement file's element modes "
            "choose triangles or quad-dominant cells. Free nodes are projected onto the "
            "source surface; boundaries and trailing edges follow their source curves, "
            "and interfaces with unchanged families are frozen. Boundary nodes can be "
            "resampled, so check periodic seams before sector-to-wheel. No saved simulation "
            "or solver is needed. Writes a level folder under inputs/mesh/ by default, "
            "with boundary and point files, a refinement record, audit and provenance. "
            "--out chooses its parent folder. Other options follow refine."
            if remesh
            else _REFINE_DESCRIPTION
        ),
        allow_abbrev=False,
    )
    refine.add_argument("mesh", metavar="MESH", help="the source OBJ")
    refine.add_argument(
        "factor",
        metavar="FACTOR",
        nargs="?",
        type=_number,
        help="the factor of every selected family; below 1 coarsens",
    )
    refine.add_argument("--families", help="the families to change, comma separated (default: all)")
    refine.add_argument(
        "--parallel",
        type=_number,
        help="the factor of the grid families along their reference perimeter",
    )
    refine.add_argument(
        "--normal",
        type=_number,
        help="the factor of the grid families across their reference perimeter",
    )
    refine.add_argument(
        "--config", metavar="FILE", help="the refinement file (default: <stem>.refine.toml)"
    )
    refine.add_argument(
        "--accept-remesh",
        action="store_true",
        help="remesh with FACTOR a family given --parallel or --normal, or method = grid, "
        "whose grid is not recovered, with a warning, in place of refusing it",
    )
    refine.add_argument(
        "--accept-unknown",
        action="store_true",
        help="warn and skip, in place of refusing, an unknown table or key of the refinement "
        "file, a value it cannot read (the key takes its default), an implicit refinement "
        "file that cannot be read, a family table or component member the mesh does not "
        "hold, and a --families name it does not hold",
    )
    refine.add_argument(
        "--strict-factors",
        dest="force_factors",
        action="store_false",
        help="refuse a grid factor below the least its grid allows (1/m for its m intervals, "
        "the least a tube's caps carry); by default it is raised to that least with a warning",
    )
    refine.add_argument(
        "--out",
        metavar="DIR",
        help="where the level folder goes (default: inputs/mesh/ of the workspace)",
    )
    refine.add_argument("--overwrite", action="store_true", help="replace an existing level folder")
    refine.add_argument(
        "--visual-inspect",
        action="store_true",
        help="write <level>.refine-governors.pdf beside its OBJ: the governing families "
        "per direction, then the patches and directions pages",
    )
    refine.add_argument(
        "--keep-within-component",
        action="store_true",
        default=None,
        help="share directional governors across each declared component's grid patches",
    )
    refine.add_argument(
        "--quiet", action="store_true", help="do not print the progress lines on the error stream"
    )
    refine.add_argument(
        "--audit-off",
        action="store_true",
        help="skip the level audit and print how to run it later",
    )
    refine.set_defaults(handler=_run_refine, force_factors=True, remesh=remesh)


def _run_refine(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        out_dir = (
            Path(args.workspace) / MESH_FOLDER
            if args.out is None
            else mesh_target(args.workspace, args.out)
        )
        level = refine_mesh(
            args.mesh,
            args.factor,
            families=_names(args.families),
            parallel=args.parallel,
            normal=args.normal,
            config=args.config,
            out_dir=out_dir,
            overwrite=args.overwrite,
            progress=None if args.quiet else _progress_printer("refine"),
            accept_remesh=args.accept_remesh,
            accept_unknown=args.accept_unknown,
            force_factors=args.force_factors,
            visual_inspect=args.visual_inspect,
            keep_within_component=args.keep_within_component,
            audit=not args.audit_off,
            remesh=args.remesh,
        )
        if level is None:
            print("no family selected: no level written", file=sys.stderr)
            return {"level": None, "files": []}
        config = Path(args.config) if args.config else Path(args.mesh).with_suffix(".refine.toml")
        sources = [args.mesh] + ([config] if config.is_file() else [])
        provenance = write_provenance(
            level.obj,
            command="sector-remesh" if args.remesh else "refine",
            sources=sources,
            options=_options(
                args,
                "factor",
                "families",
                "parallel",
                "normal",
                "config",
                "accept_remesh",
                "accept_unknown",
                "force_factors",
                "keep_within_component",
                "audit_off",
                "remesh",
                "overwrite",
            ),
            workspace=args.workspace,
            outputs={"folder": str(level.folder), "families": level.report},
        )
        return {
            "level": str(level.folder),
            "obj": str(level.obj),
            "families": {
                name: entry.get("method", "copied") for name, entry in level.report.items()
            },
            "audit": None if level.audit is None else level.audit.summary(),
            "files": [str(path) for path in level.files],
            "provenance": str(provenance),
        }

    return _run(work)


# --- prepare-to-refine -------------------------------------------------------------


def _add_prepare(verbs: Any) -> None:
    prepare = verbs.add_parser(
        "prepare-to-refine",
        help="split a mesh's families into grid and mixed patches and write a refine file for them",
        description=(
            "Optional step before refine. MESH is an OBJ, whose families are cut at their sharp "
            "edges, or an ANSA NASTRAN export (.nas, .bdf, .dat) written with the FE perimeters "
            "on, whose families are cut along those perimeters. Writes into inputs/mesh/ "
            "of the workspace, or --out, "
            "<stem>_patches.obj (each family cut into <family>_<NNN>_grid or _mixed patches, "
            "numbered within the family from the largest extent along y), "
            "<stem>_patches.refine.toml (each patch at factor 1, grid or remesh, [components] "
            "putting each family back together) and <stem>_patches.nodes.txt (the rearmost "
            "line of each grid patch, its edges within --max-angle of the plane across --aft, "
            "as a node list)."
        ),
        allow_abbrev=False,
    )
    prepare.add_argument(
        "mesh", metavar="MESH", help="the source OBJ or NASTRAN export; it is not modified"
    )
    prepare.add_argument(
        "--max-angle",
        type=float,
        default=45.0,
        metavar="DEG",
        help="the largest angle of a listed edge with the plane across --aft (default 45)",
    )
    prepare.add_argument(
        "--aft", nargs=3, type=float, metavar=("X", "Y", "Z"), help="the aft direction (default +x)"
    )
    prepare.add_argument("--overwrite", action="store_true", help="replace the prepared files")
    prepare.add_argument(
        "--out", metavar="DIR", help="output folder (default: inputs/mesh/ of the workspace)"
    )
    prepare.add_argument(
        "--unit",
        default="MILLIMETER",
        help="a NASTRAN mesh's length unit (default MILLIMETER); an OBJ is written as read",
    )
    prepare.add_argument(
        "--obj-unit",
        default="METER",
        help="the unit a NASTRAN mesh is written in to the patches OBJ (default METER)",
    )
    prepare.add_argument(
        "--visual-inspect",
        action="store_true",
        help="also write <stem>_patches.inspect.pdf: one page per original family, each patch "
        "colored and named, the governing patches labelled GOVERNING",
    )
    prepare.add_argument(
        "--keep-within-component",
        action="store_true",
        help="share directional governors across each original family's grid patches",
    )
    prepare.set_defaults(handler=_run_prepare)


def _run_prepare(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        x, y, z = args.aft or (1.0, 0.0, 0.0)
        prepared = prepare_to_refine(
            args.mesh,
            max_angle=args.max_angle,
            aft=(x, y, z),
            overwrite=args.overwrite,
            unit=args.unit,
            obj_unit=args.obj_unit,
            visual_inspect=args.visual_inspect,
            keep_within_component=args.keep_within_component,
            out_dir=(
                Path(args.workspace) / MESH_FOLDER
                if args.out is None
                else mesh_target(args.workspace, args.out)
            ),
        )
        files = [prepared.obj, prepared.config, prepared.nodes]
        if prepared.inspect is not None:
            files.append(prepared.inspect)
        provenance = write_provenance(
            prepared.obj,
            command="prepare-to-refine",
            sources=[args.mesh],
            options=_options(
                args, "max_angle", "aft", "unit", "obj_unit", "keep_within_component", "overwrite"
            ),
            workspace=args.workspace,
            outputs={"patches": prepared.patches, "governing": list(prepared.governing)},
        )
        return {
            "patches": prepared.patches,
            "files": [str(path) for path in files],
            "provenance": str(provenance),
        }

    return _run(work)


# --- audit -------------------------------------------------------------------------


def _reason(error: OSError) -> str:
    """Return what the system said of a failed file operation, without the path it repeats."""
    return str(error.strerror or error)


def _same_file(a: Path, b: Path) -> bool:
    """Return whether two paths name one file: one spelling, or one file on disk."""
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
        return os.path.normcase(a.resolve()) == os.path.normcase(b.resolve())
    except OSError:
        return False


def _csv_path(workspace: str | Path, out: str) -> Path:
    """Return where the CSV goes, as :func:`mesh_target` places a file but creating no folder.

    A folder that does not exist is refused by the pre-flight check, never created.
    """
    named = Path(out)
    if named.is_absolute():
        return named
    if named.parent == Path("."):
        return mesh_target(workspace, named)
    return Path(workspace) / named


def _refused_csv(target: Path) -> str | None:
    """Return the refusal of a CSV file that cannot be written, before anything is written.

    A folder that does not exist, a path that is a folder, and a file that cannot be opened
    for writing are refused before the audit runs, so a refusal leaves no audit behind. The
    file is opened for appending, which changes nothing in it, and removed again when the
    probe created it.
    """
    if target.is_dir() or not target.parent.is_dir():
        return (
            f"{target}: --out names a folder, or a file in a folder that does not exist. "
            "Name a file in an existing folder and run again. Nothing was written."
        )
    created = False
    try:
        if not target.exists():
            os.close(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL))
            created = True
        else:
            target.open("a", encoding="utf-8").close()
    except FileExistsError:
        pass
    except OSError as error:
        return (
            f"{target}: --out names a file that cannot be written ({_reason(error)}). "
            "Close it or make it writable, or name another file, and run again. "
            "Nothing was written."
        )
    if created:
        target.unlink(missing_ok=True)
    return None


def _csv_collision(target: Path, mesh: str, against: str | None) -> str | None:
    """Return the refusal of a CSV file the audit reads or writes, before any is written.

    The CSV must not be the audited OBJ, the source of ``--against``, a boundaries or
    trailing-edge points file either is read with, the OBJ's refinement record, or the audit
    JSON written beside the OBJ, under any spelling.
    """
    audit = Path(mesh).with_name(f"{Path(mesh).stem}.audit.json")
    taken = [(f"mesh {what}", path) for what, path in MeshAudit.files_read(mesh)]
    if against is not None:
        taken += [
            (f"--against source {what}", path) for what, path in MeshAudit.files_read(against)
        ]
    taken.append(("the audit JSON the command writes", audit))
    for what, path in taken:
        if _same_file(target, path):
            return (
                f"{target}: --out names {what}, {path}, which audit reads or writes. "
                "Name another file and run again. Nothing was written."
            )
    return None


def _add_audit(verbs: Any) -> None:
    audit = verbs.add_parser(
        "audit",
        help=(
            "audit a panel mesh's topology, trailing edge and panel quality, alone or against "
            "the source it was made from"
        ),
        description=_AUDIT_DESCRIPTION,
        allow_abbrev=False,
    )
    audit.add_argument("mesh", metavar="OBJ", help="the OBJ to audit")
    audit.add_argument(
        "--against",
        metavar="SOURCE",
        help="the OBJ the mesh was refined or coarsened from; the gates and checks that compare "
        "with a source are judged against it",
    )
    audit.add_argument(
        "--out",
        metavar="FILE",
        help="write the figures as CSV, one row per family and figure; "
        "the audit JSON is always written beside the OBJ",
    )
    audit.set_defaults(handler=_run_audit)


def _run_audit(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        target = None if args.out is None else _csv_path(args.workspace, args.out)
        if target is not None:
            refused = _csv_collision(target, args.mesh, args.against) or _refused_csv(target)
            if refused is not None:
                raise InputArtifactError(refused)
        result = audit_mesh(args.mesh, against=args.against)
        files = [result.path]
        if target is not None:
            try:
                files.append(result.write_csv(target))
            except OSError as error:
                raise OSError(
                    f"{target}: the CSV could not be written ({error.strerror or error}). "
                    f"The audit {result.path} was written."
                ) from error
        sources = [args.mesh] + ([args.against] if args.against else [])
        provenance = write_provenance(
            result.path,
            command="audit",
            sources=sources,
            options=_options(args, "against"),
            workspace=args.workspace,
            outputs={"files": [str(path) for path in files], "passed": result.passed},
        )
        return {
            "passed": result.passed,
            "summary": result.summary(),
            "files": [str(path) for path in files],
            "provenance": str(provenance),
        }

    return _run(work)


# --- inventory ---------------------------------------------------------------------


def _add_inventory(verbs: Any) -> None:
    inventory = verbs.add_parser(
        "inventory",
        help="write <stem>.obj.boundaries.toml beside an OBJ, from its groups",
        description=(
            "Writes the boundary inventory of an OBJ beside it: its groups that hold a face, "
            "in the order of the file, with the face counts and the digest of the file. An "
            "OBJ's existing sidecar is never rewritten, because beside a raw mesh it also "
            "carries the [import] table and the trailing edges."
        ),
        allow_abbrev=False,
    )
    inventory.add_argument("geometry", help="an OBJ mesh")
    inventory.add_argument(
        "--overwrite",
        action="store_true",
        help="accepted for the other verbs' sake; an OBJ's existing sidecar is refused either way",
    )
    inventory.set_defaults(handler=_run_inventory)


def _run_inventory(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        sidecar = write_inventory(args.geometry, overwrite=args.overwrite)
        provenance = write_provenance(
            sidecar,
            command="inventory",
            sources=[args.geometry],
            options=_options(args, "overwrite"),
            workspace=args.workspace,
        )
        return {"inventory": str(sidecar), "provenance": str(provenance)}

    return _run(work)


# --- degenerate --------------------------------------------------------------------


def _add_degenerate(verbs: Any) -> None:
    degenerate = verbs.add_parser(
        "degenerate",
        help=(
            "derive a degenerate geometry, the thin blade first, from a blade mesh and write "
            "it into the workspace's inputs/mesh/"
        ),
        description=(
            "Derives the thin blade of a blade mesh, the sheet midway between its two sides "
            "from the root to the tip, with the root moved outward along the span by "
            "--root-offset so it does not cross the spinner, and writes into inputs/mesh/ "
            "of the workspace, or --out, "
            "as <stem>_thin_blade.obj with its boundary inventory "
            "<stem>_thin_blade.boundaries.toml. The root is the end of the blade nearer the "
            "origin of the mesh's own coordinates. The source is read only; a file holding "
            "more than one boundary needs --boundary. The derived geometry is a modelling "
            "choice: a solver run confirms what it makes of it."
        ),
        allow_abbrev=False,
    )
    degenerate.add_argument("geometry", help="a blade mesh: an OBJ")
    degenerate.add_argument(
        "--out", metavar="DIR", help="output folder (default: inputs/mesh/ of the workspace)"
    )
    degenerate.add_argument(
        "--kind",
        choices=["thin-blade"],
        default="thin-blade",
        help="the degenerate geometry to derive; thin-blade is the supported kind "
        "(default: %(default)s)",
    )
    degenerate.add_argument(
        "--root-offset",
        type=float,
        required=True,
        metavar="LENGTH",
        help="how far the root of the thin blade is moved outward along the span, a positive "
        "length in the mesh's own unit, shorter than the blade",
    )
    degenerate.add_argument(
        "--boundary",
        metavar="NAME",
        help="the boundary (an OBJ's group) that is the blade, for a file that also holds a "
        "spinner, a nacelle or other bodies; only its faces are read and the output is "
        "named <stem>_<NAME>_thin_blade; without it a file holding more than one "
        "boundary is refused",
    )
    degenerate.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite a thin blade and its inventory that already exist; without it an "
        "existing one is refused",
    )
    degenerate.set_defaults(handler=_run_degenerate)


def _run_degenerate(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        made = derive_thin_blade(
            args.geometry,
            root_offset=args.root_offset,
            overwrite=args.overwrite,
            boundary=args.boundary,
            out_dir=(
                Path(args.workspace) / MESH_FOLDER
                if args.out is None
                else mesh_target(args.workspace, args.out)
            ),
        )
        unit = made.unit or "a length unit the source does not state"
        print(f"the root moved outward by {made.root_offset!r} in {unit}", file=sys.stderr)
        provenance = write_provenance(
            made.mesh,
            command="degenerate",
            sources=[args.geometry],
            options=_options(args, "kind", "root_offset", "boundary", "overwrite"),
            workspace=args.workspace,
            outputs={"inventory": str(made.sidecar), "unit": made.unit},
        )
        return {
            "mesh": str(made.mesh),
            "inventory": str(made.sidecar),
            "root_offset": made.root_offset,
            "unit": made.unit,
            "provenance": str(provenance),
        }

    return _run(work)


# --- merge -------------------------------------------------------------------------


def _add_merge(verbs: Any) -> None:
    merge = verbs.add_parser(
        "merge",
        help="join two trimmed OBJ along their open boundaries within a threshold",
        description=(
            "Joins two trimmed OBJ along the open boundaries that lie within --threshold of "
            "each other. The part --adapt names takes the other's nodes along that interface; "
            "the other part is kept unchanged. Writes the OBJ, <out stem>.merge.json, its audit "
            "and its inventory. The boundary declarations of the two sources are not merged. "
            "A threshold that finds no interface warns, writes nothing and exits 1."
        ),
        allow_abbrev=False,
    )
    merge.add_argument("parts", nargs=2, metavar="OBJ", help="the two trimmed parts")
    merge.add_argument(
        "--threshold",
        type=float,
        required=True,
        metavar="LENGTH",
        help="the largest distance across the interface, in the OBJ's own length unit",
    )
    merge.add_argument(
        "--adapt",
        required=True,
        metavar="OBJ",
        help="which of the two parts is adapted, by its path or its file name; the other keeps "
        "its nodes and faces",
    )
    merge.add_argument(
        "--out",
        required=True,
        metavar="OBJ",
        help="the merged OBJ to write; a name of another kind is warned and takes .obj; a "
        "bare name lands in the workspace's inputs/mesh/",
    )
    merge.add_argument("--overwrite", action="store_true", help="replace an existing output")
    merge.set_defaults(handler=_run_merge)


def _run_merge(args: argparse.Namespace) -> int:
    def work() -> dict[str, Any]:
        first, second = args.parts
        target = mesh_target(args.workspace, args.out)
        merged = merge_objs(
            first,
            second,
            threshold=args.threshold,
            adapt=args.adapt,
            out=target,
            overwrite=args.overwrite,
            say=lambda line: print(line, file=sys.stderr),
        )
        if merged.obj is None:
            return {"passed": False, "merged": None, "files": []}
        files = [merged.obj, merged.record, merged.boundaries]
        if merged.audit is not None:
            files.append(merged.audit.path)
        provenance = write_provenance(
            merged.obj,
            command="merge",
            sources=[first, second],
            options=_options(args, "threshold", "adapt", "overwrite"),
            workspace=args.workspace,
            outputs={
                "interface_edges": merged.interface_edges,
                "moved_nodes": merged.moved_nodes,
                "band_faces": merged.band_faces,
                "triangles_created": merged.triangles_created,
                "open_edges_left": merged.open_edges_left,
            },
        )
        return {
            "merged": str(merged.obj),
            "interface_edges": merged.interface_edges,
            "moved_nodes": merged.moved_nodes,
            "band_faces": merged.band_faces,
            "triangles_created": merged.triangles_created,
            "open_edges_left": merged.open_edges_left,
            "facing_edges_left": merged.facing_edges_left,
            "files": [str(path) for path in files],
            "provenance": str(provenance),
        }

    return _run(work)


def add_parsers(verbs: Any) -> None:
    """Add this group's verbs to the cfd-mesh subparsers.

    Parameters
    ----------
    verbs : argparse._SubParsersAction
        The subparsers of :func:`cfdbender.mesh.cli.build_parser`.
    """
    _add_refine(verbs)
    _add_refine(verbs, remesh=True)
    _add_prepare(verbs)
    _add_audit(verbs)
    _add_inventory(verbs)
    _add_degenerate(verbs)
    _add_merge(verbs)

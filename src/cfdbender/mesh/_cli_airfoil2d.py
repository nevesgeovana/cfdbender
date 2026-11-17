"""The cfd-mesh verb airfoil-2d: the 2D airfoil volume mesh for SU2 and OpenFOAM.

``cfd-mesh airfoil-2d --naca 0012 --reynolds 6e6`` writes ``inputs/mesh/naca0012.su2`` and
``inputs/mesh/naca0012.msh`` in the workspace, each with a ``<file name>.provenance.json``
sidecar that records how it was made: the command, the source file with its SHA-256, every
option after defaults, the measured quality and the SHA-256 of each written file. The mesh is
made by :func:`cfdbender.mesh._airfoil2d.build_mesh`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cfdbender._digest import file_sha256
from cfdbender._errors import InputValueError
from cfdbender.mesh import _airfoil2d as af
from cfdbender.mesh._cli_support import (
    mesh_target,
    print_checks,
    provenance_path,
    write_provenance,
)

__all__ = ["add_parsers"]

#: The verb, without the program name.
VERB = "airfoil-2d"


def add_parsers(verbs: Any) -> None:
    """Add the airfoil-2d verb to the cfd-mesh subparsers.

    Parameters
    ----------
    verbs : argparse._SubParsersAction
        The subparsers of the cfd-mesh parser, whose top level holds ``--workspace``.
    """
    parser = verbs.add_parser(
        VERB,
        help="mesh a 2D airfoil section as a structured C-mesh for SU2 and OpenFOAM",
        description=(
            "Mesh a 2D airfoil section as a structured C-mesh of quadrilaterals and write "
            "<name>.su2 (markers airfoil and farfield) and <name>.msh (gmsh 2.2, one cell "
            "thick, patches airfoil, farfield, front and back, volume fluid) into the "
            "workspace's inputs/mesh/, each with a provenance sidecar. The first cell is sized "
            "for a target y+ at the given Reynolds number. A blunt edge adds airfoil_base."
        ),
        allow_abbrev=False,
    )
    section = parser.add_mutually_exclusive_group(required=True)
    section.add_argument(
        "--naca", metavar="DIGITS", help="a NACA 4 or 5 digit designation, for example 0012"
    )
    section.add_argument(
        "--coordinates",
        metavar="FILE",
        help="a coordinates file in Selig order (a relative path is read from the workspace)",
    )
    parser.add_argument(
        "--reynolds",
        type=float,
        required=True,
        metavar="RE",
        help="the chord Reynolds number the first cell is sized for",
    )
    parser.add_argument(
        "--yplus",
        type=float,
        default=af.DEFAULT_YPLUS,
        metavar="Y+",
        help=f"the target y+ of the first cell (default {af.DEFAULT_YPLUS:g})",
    )
    parser.add_argument(
        "--level",
        choices=tuple(af.LEVELS),
        default=af.DEFAULT_LEVEL,
        help=f"the cell counts (default {af.DEFAULT_LEVEL})",
    )
    parser.add_argument(
        "--farfield",
        type=float,
        default=af.DEFAULT_FARFIELD,
        metavar="CHORDS",
        help=f"the far-field distance in chords (default {af.DEFAULT_FARFIELD:g})",
    )
    parser.add_argument(
        "--trailing-edge",
        choices=af.TRAILING_EDGES,
        default=af.DEFAULT_TRAILING_EDGE,
        help=f"collapse the trailing-edge gap or keep it as a base (default "
        f"{af.DEFAULT_TRAILING_EDGE})",
    )
    parser.add_argument(
        "--out",
        metavar="NAME",
        help="the mesh name without a suffix (default: naca<digits> or the file's stem); a "
        "name with a folder is written there",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing mesh of the same name and its provenance sidecars",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    return print_checks(lambda: run(args))


def _shown(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _stem(out: str | None, section: af.Section) -> str:
    if out is None:
        if "naca" in section.source:
            return "naca" + section.source["naca"]
        return Path(section.source["coordinates"]).stem
    named = Path(out)
    if named.suffix.lower() in (".su2", ".msh"):
        return str(named.with_suffix(""))
    if named.suffix:
        raise InputValueError(
            f"--out {out!r} has the suffix {named.suffix!r}; give the name without a suffix "
            "(both .su2 and .msh are written). Nothing was written."
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run airfoil-2d from parsed arguments and return its record.

    Every input is checked before anything is written; a refusal writes nothing.
    """
    workspace = Path(args.workspace)
    sources: list[Path] = []
    if args.naca is not None:
        section = af.naca_section(args.naca)
    else:
        path = Path(args.coordinates)
        if not path.is_absolute():
            path = workspace / path
        section = af.read_coordinates(path)
        sources.append(path)
    stem = _stem(args.out, section)
    af.shape_trailing_edge(section, args.trailing_edge)
    af.plan_spacing(
        reynolds=args.reynolds, yplus=args.yplus, level=args.level, farfield=args.farfield
    )
    target = mesh_target(workspace, stem)
    clash = [
        p
        for suffix in (".su2", ".msh")
        for p in (Path(f"{target}{suffix}"), provenance_path(Path(f"{target}{suffix}")))
        if p.exists()
    ]
    if clash and not args.overwrite:
        raise InputValueError(
            f"{clash[0]}: exists; pass --overwrite to replace it or give another --out. "
            "Nothing was written."
        )
    record = af.build_mesh(
        target,
        section,
        reynolds=args.reynolds,
        yplus=args.yplus,
        level=args.level,
        farfield=args.farfield,
        trailing_edge=args.trailing_edge,
    )
    parameters = dict(record.pop("parameters"))
    source = parameters.pop("source")
    files = {kind: Path(path) for kind, path in record.pop("files").items()}
    options = {
        **({"naca": source["naca"]} if "naca" in source else {}),
        **({"coordinates": _shown(sources[0], workspace)} if sources else {}),
        **parameters,
        "gmsh": record.pop("gmsh"),
        **({"overwrite": True} if args.overwrite else {}),
    }
    outputs = {
        **record,
        "files": {kind: _shown(path, workspace) for kind, path in files.items()},
        "sha256": {kind: file_sha256(path) for kind, path in files.items()},
    }
    sidecars = [
        write_provenance(
            path,
            command=VERB,
            sources=sources,
            options=options,
            workspace=workspace,
            outputs=outputs,
        )
        for path in files.values()
    ]
    return {
        "command": f"cfd-mesh {VERB}",
        "meshes": outputs["files"],
        "provenance": [_shown(p, workspace) for p in sidecars],
        "options": options,
        "quality": record,
    }

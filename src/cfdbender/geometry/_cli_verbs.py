"""The ``identify``, ``check`` and ``schema`` verbs of ``cfd-geom``.

Pipeline role: the run layer's parsers and handlers for the verbs that read an original
IGES or a component record with their options
unchanged. The work is the geometry package's (:mod:`cfdbender.geometry`) and the
workspace's (:mod:`cfdbender.geometry._identify`, on the gmsh kernel); this module
only reads the command line, calls it, prints the record it returns as JSON and gives the exit
code: 0 when the work was done (and, for ``check`` and ``schema --validate``, accepted), 1 when
the work measured the input and rejected it, 2 when an input is refused or the gmsh library
cannot load, with the message on standard error and nothing written. Each parser names its
handler as the ``mesh_command`` default, which :mod:`cfdbender.geometry.cli` calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cfdbender._textio as _textio
from cfdbender._errors import GeometryError, NotApplicableWarning, warn

__all__ = ["MOVED_VERBS", "add_moved_verbs", "report_of", "write_json"]

_PROFILES_HELP = "named deviation profile (default: study_default, 0.5 mm or 0.1 percent of chord)"


def report_of(work: Callable[[], dict[str, Any]], *, rejects: bool = False) -> int:
    """Run one verb and print its record; 2 when refused, 1 when ``rejects`` and not accepted."""
    try:
        record = work()
    except (OSError, GeometryError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, default=str))
    return 1 if rejects and not record.get("accepted", True) else 0


def write_json(path: str | None, record: dict[str, Any]) -> None:
    """Write ``record`` as JSON to ``path`` when one is given."""
    if path is not None:
        _textio.write_text(path, json.dumps(record, indent=2, default=str))


def add_identify_parser(subparsers: Any) -> None:
    """Add the verb that identifies a lifting surface of an original IGES."""
    parser = subparsers.add_parser(
        "identify",
        help="identify a lifting surface of an original IGES: planform and sections "
        "(uses the gmsh kernel)",
        description=(
            "Cuts the trimmed faces of one component of an IGES file (named on its level) by "
            "planes normal to its span axis, finds the clean band of sections, builds the "
            "reference planform between the root and the outermost clean cut and fits a CST "
            "section to the root, the middle and the tip. The root is the symmetry plane, the "
            "host surface or the span extent of the original's own boundary curves. A coarse "
            "and a fine pass must agree for the result to be accepted. Prints the result; "
            "--config-out writes a configuration that rebuilds the planform."
        ),
    )
    parser.add_argument("iges", metavar="IGES", help="the original IGES, read only")
    parser.add_argument("component", metavar="COMPONENT", help="the component to identify")
    parser.add_argument(
        "--host",
        dest="hosts",
        action="append",
        metavar="NAME",
        help="a component the surface sits on, NAME or FILE#NAME (repeatable); the root is "
        "where the extended leading edge meets it",
    )
    parser.add_argument(
        "--root",
        dest="root_from",
        choices=("auto", "plane", "host", "boundary"),
        default="auto",
        help="where the planform starts: the symmetry plane, a host, or the span extent of the "
        "face boundary curves (default: host when --host is given, else plane)",
    )
    parser.add_argument(
        "--mirrored",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="whether the surface has a mirror image across its root, doubling span and area "
        "(default: yes for a plane root, no otherwise)",
    )
    parser.add_argument(
        "--cut",
        choices=("plain", "bridge_te", "mirror"),
        default="plain",
        help="bridge_te closes an open blunt trailing-edge slit; mirror adds the reflection of "
        "a half-body original (default: plain)",
    )
    parser.add_argument(
        "--seed",
        default="NACA0012",
        metavar="NACA",
        help="the NACA designation the section fit starts from (default: NACA0012)",
    )
    parser.add_argument(
        "--span-axis",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="first estimate of the span axis (default: y or z, the larger extent)",
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=161,
        metavar="N",
        help="stations of the fine pass (default: 161)",
    )
    parser.add_argument(
        "--grid",
        dest="grid_points",
        type=int,
        default=257,
        metavar="N",
        help="cut-grid points per side of each face (default: 257)",
    )
    parser.add_argument(
        "--section-tolerance-mm",
        type=float,
        metavar="MM",
        help="section residual tolerance in mm (default: 0.5 mm or 0.1 percent of the chord)",
    )
    parser.add_argument(
        "--section-degree",
        type=int,
        default=5,
        metavar="N",
        help="CST degree a section fit starts at (default: 5)",
    )
    parser.add_argument(
        "--section-max-degree",
        type=int,
        default=20,
        metavar="N",
        help="highest CST degree a section fit may reach (default: 20)",
    )
    parser.add_argument(
        "--host-tolerance",
        type=float,
        default=5e-5,
        metavar="M",
        help="largest gap in m at which the extended leading edge meets a host (default: 5e-5)",
    )
    parser.add_argument(
        "--no-converge",
        dest="converge",
        action="store_false",
        help="skip the coarse pass that must agree with the fine one",
    )
    parser.add_argument("--out", metavar="JSON", help="also write the result to this file")
    parser.add_argument(
        "--config-out",
        metavar="TOML",
        help="write a configuration (TOML) that rebuilds the identified surface",
    )
    parser.set_defaults(mesh_command=run_identify)


def run_identify(args: argparse.Namespace) -> int:
    """Run identify: print the result, optionally write it and the configuration it implies."""
    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._identify import identify_lifting_surface
    from cfdbender.geometry.stages import write_config

    def work() -> dict[str, Any]:
        found = identify_lifting_surface(
            args.iges,
            args.component,
            hosts=args.hosts or (),
            root=args.root_from,
            mirrored=args.mirrored,
            cut=args.cut,
            seed=args.seed,
            span_axis=args.span_axis,
            stations=args.stations,
            grid=args.grid_points,
            section_tolerance_mm=args.section_tolerance_mm,
            section_degree=args.section_degree,
            section_max_degree=args.section_max_degree,
            host_tolerance=args.host_tolerance,
            converge=args.converge,
        )
        record = found.to_dict()
        if args.config_out is not None:
            entry = found.to_component()
            write_config(resolve_config({"components": {args.component: entry}}), args.config_out)
            record["config_out"] = args.config_out
        write_json(args.out, record)
        return record

    return report_of(work)


def add_check_parser(subparsers: Any) -> None:
    """Add the verb that measures a rebuilt surface against an original."""
    parser = subparsers.add_parser(
        "check",
        help="measure a rebuilt IGES against the original: forward and restricted reverse "
        "distance (uses the gmsh kernel)",
        description=(
            "Samples the original and the rebuilt faces and measures the distance of each to "
            "the other: forward (the original to the rebuilt, everywhere) and reverse (the "
            "rebuilt to the original, inside its trimmed boundary). The limit is the named "
            "profile applied to the local chord. The distances are sampled upper bounds, not "
            "continuous Hausdorff distances, as the record's method says. Exit 0 accepted, 1 "
            "rejected, 2 refused. Give the local chord as --chord-m, or as --stations-m with "
            "--chords-m."
        ),
    )
    parser.add_argument("rebuilt", metavar="REBUILT", help="the rebuilt IGES, FILE or FILE#NAME")
    parser.add_argument("original", metavar="ORIGINAL", help="the original IGES, FILE or FILE#NAME")
    parser.add_argument("--chord-m", type=float, metavar="M", help="one local chord in m")
    parser.add_argument(
        "--stations-m",
        type=float,
        nargs="+",
        metavar="S",
        help="increasing stations in m along --axis, with --chords-m",
    )
    parser.add_argument(
        "--chords-m", type=float, nargs="+", metavar="C", help="the chord in m at each station"
    )
    parser.add_argument(
        "--axis",
        dest="chord_axis",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="the axis of --stations-m, 0 x, 1 y, 2 z (default: 1)",
    )
    parser.add_argument(
        "--component", metavar="NAME", help="the component of a file both surfaces are named in"
    )
    parser.add_argument(
        "--exclude-face",
        dest="exclude_faces",
        type=int,
        action="append",
        metavar="INDEX",
        help="leave this face of the original out, by index (repeatable)",
    )
    parser.add_argument(
        "--forward-count",
        type=int,
        default=45,
        metavar="N",
        help="samples per side on each original face (default: 45)",
    )
    parser.add_argument(
        "--reverse-count",
        type=int,
        default=31,
        metavar="N",
        help="samples per side on each rebuilt face (default: 31)",
    )
    parser.add_argument(
        "--edge-count",
        type=int,
        default=101,
        metavar="N",
        help="samples on each trim edge (default: 101)",
    )
    parser.add_argument(
        "--fold-y",
        action="store_true",
        help="the original is a half body: fold the rebuilt mirror half onto the modeled side",
    )
    parser.add_argument("--profile", default="study_default", help=_PROFILES_HELP)
    parser.add_argument("--out", metavar="JSON", help="also write the record to this file")
    parser.set_defaults(mesh_command=run_check)


def run_check(args: argparse.Namespace) -> int:
    """Run check: print the record; exit 1 when the rebuilt surface is rejected."""
    from cfdbender._errors import InputValueError
    from cfdbender.geometry._identify import check_against_original

    def work() -> dict[str, Any]:
        limits: dict[str, Any] = {
            "axis": args.chord_axis,
            "forward_count": args.forward_count,
            "reverse_count": args.reverse_count,
            "edge_count": args.edge_count,
        }
        if args.stations_m is not None or args.chords_m is not None:
            if args.stations_m is None or args.chords_m is None:
                raise InputValueError(
                    "stations_m (CLI: --stations-m) and chords_m (CLI: --chords-m): give both"
                )
            limits.update(stations_m=args.stations_m, chords_m=args.chords_m)
            if args.chord_m is not None:
                warn(
                    "chord_m (CLI: --chord-m): not used, "
                    "stations_m (CLI: --stations-m) and chords_m (CLI: --chords-m) give",
                    NotApplicableWarning,
                )
        elif args.chord_m is not None:
            limits["chord_m"] = args.chord_m
        else:
            raise InputValueError(
                "give chord_m (CLI: --chord-m), or stations_m (CLI: --stations-m) "
                "with chords_m (CLI: --chords-m)"
            )
        if args.component is not None:
            limits["component"] = args.component
        if args.exclude_faces:
            limits["exclude_faces"] = args.exclude_faces
        if args.fold_y:
            limits["fold_y"] = True
        record = check_against_original(args.rebuilt, args.original, limits, profile=args.profile)
        write_json(args.out, record)
        return record

    return report_of(work, rejects=True)


def add_schema_parser(subparsers: Any) -> None:
    """Add the verb that prints the component record schema or validates a record."""
    parser = subparsers.add_parser(
        "schema",
        help="print the JSON Schema of the component record, or validate a record against it",
        description=(
            "Prints the draft 2020-12 JSON Schema of the component parameter record (schema "
            "0.2.0): the contract between a reconstruction and this library. With --validate "
            "the record file is parsed and its structural rules checked instead, and with "
            "--asset-root every file it pins is re-read against its SHA-256. Needs no gmsh."
        ),
    )
    parser.add_argument("--out", metavar="JSON", help="write the schema to this file")
    parser.add_argument("--validate", metavar="RECORD", help="validate this record (JSON)")
    parser.add_argument(
        "--asset-root",
        metavar="DIR",
        help="with --validate: the folder the record's file paths resolve against, to check "
        "every pin",
    )
    parser.set_defaults(mesh_command=run_schema)


def run_schema(args: argparse.Namespace) -> int:
    """Run schema: print the schema, or validate a record and its pins."""
    from cfdbender._errors import InputValueError
    from cfdbender.geometry import ComponentRecord
    from cfdbender.geometry.stages import export_schema, verify_pins

    def work() -> dict[str, Any]:
        if args.validate is None:
            if args.asset_root is not None:
                raise InputValueError(
                    "asset_root (CLI: --asset-root): only with validate (CLI: --validate)"
                )
            schema = export_schema()
            write_json(args.out, schema)
            return schema
        if args.out is not None:
            warn("--out: not used with --validate, which writes nothing", NotApplicableWarning)
        record = ComponentRecord.from_json(Path(args.validate).read_text(encoding="utf-8"))
        problems = verify_pins(record, args.asset_root) if args.asset_root is not None else []
        return {
            "record": args.validate,
            "component": record.component_code,
            "kind": record.kind,
            "stations": len(record.stations),
            "pins_checked": args.asset_root is not None,
            "pin_problems": problems,
            "accepted": not problems,
        }

    return report_of(work, rejects=True)


#: The parser of each verb this module adds, by its name.
_PARSERS: dict[str, Callable[[Any], None]] = {
    "identify": add_identify_parser,
    "check": add_check_parser,
    "schema": add_schema_parser,
}

#: The verbs this module adds.
MOVED_VERBS = tuple(_PARSERS)


def add_moved_verbs(verbs: Any) -> None:
    """Add this module's verbs to the ``cfd-geom`` subparsers."""
    for add in _PARSERS.values():
        add(verbs)

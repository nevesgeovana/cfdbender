"""The ``cfd-geom`` command line: run, loop, inspect and convert the parametric geometry.

Pipeline role: thin. Each verb parses its arguments, calls the entry of
:mod:`cfdbender.geometry._identify` (or reads the pure configuration boundary of
:mod:`cfdbender.geometry`), prints the record it returns as JSON and exits 0, 1, 2 or 3.

``run CONFIG`` runs the stages S0 to S13 (``--until`` ends earlier, ``--set
component.key=value`` overrides a value, ``--configuration`` and ``--variant`` choose what is
built) and writes into ``inputs/geometry/<name>/`` of the workspace, with the sidecar
``geometry.json`` that marks the folder as generated geometry; ``--out`` names another folder.
``doe CONFIG SETS`` runs one geometry per parameter set of ``SETS``, a TOML file with one
``[[sets]]`` table per set whose keys are the override paths and whose ``name`` names the set's
folder ``inputs/geometry/<name>/`` (the matrix row the geometry feeds), and continues after a
failed set. ``validate`` resolves the configuration and prints every value with its source;
``configurations`` and ``stages`` list what a configuration and the pipeline hold; ``schema``,
``identify`` and ``check`` read a component record or an original IGES; ``iges-convert``
rewrites named IGES files as one IGES of B-spline faces.

Exit codes: 0 accepted, 1 a run failed at a stage (the report names the stage and the check),
2 an input is refused or the gmsh library cannot load, with the message on standard error,
3 incomplete: no check failed but a check of the stage table was not measured, and the report
lists each one in ``not_run``. For ``doe`` the code is the worst of its sets, refused over
failed over incomplete over accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from cfdbender._errors import GeometryError, InputValueError
from cfdbender.core.mesh import GEOMETRY_FOLDER, GEOMETRY_SIDECAR
from cfdbender.geometry._cli_verbs import add_moved_verbs, report_of
from cfdbender.geometry.stages import STAGE_ROWS, exit_code
from cfdbender.mesh._cli_support import print_checks

__all__ = ["main"]


def _value(text: str) -> Any:
    """Return a TOML value read from ``text``, or the text itself when it is not TOML."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _overrides(items: list[str] | None) -> dict[str, Any]:
    """Return ``{path: value}`` from ``--set path=value`` items; a repeat or a bare path refuses."""
    found: dict[str, Any] = {}
    for item in items or []:
        key, separator, text = item.partition("=")
        if not separator or not key:
            raise InputValueError(f"--set {item!r}: expected component.key=value")
        if key in found:
            raise InputValueError(f"--set {key}: given twice")
        found[key] = _value(text)
    return found


def _read_sets(path: str) -> list[dict[str, Any]]:
    """Read the parameter sets of a ``[[sets]]`` file; a misspelt table or key is refused."""
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    extra = sorted(set(data) - {"sets"})
    if extra or not isinstance(data.get("sets"), list) or not data["sets"]:
        raise InputValueError(
            f"{path}: expected one or more [[sets]] tables and nothing else "
            f"(other keys: {extra or 'none'})"
        )
    return [_flat(row) for row in data["sets"]]


def _flat(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return the overrides of one set with a nested table read as its dotted paths.

    ``COMPONENT.length = 2.5`` written unquoted is the TOML table
    ``{"COMPONENT": {"length": 2.5}}`` and means the same override as
    ``"COMPONENT.length" = 2.5``.
    """
    found: dict[str, Any] = {}
    for key, value in row.items():
        path = f"{prefix}{key}"
        pairs = _flat(value, f"{path}.") if isinstance(value, dict) else {path: value}
        for leaf, item in pairs.items():
            if leaf in found:
                raise InputValueError(f"[[sets]]: {leaf} given twice")
            found[leaf] = item
    return found


#: The schema of the sidecar that marks inputs/geometry/<name>/ as generated geometry.
GEOMETRY_SIDECAR_SCHEMA = "cfdbender-geometry/1"


def _geometry_folder(args: argparse.Namespace) -> Path:
    """Return the run's folder: ``--out``, else ``inputs/geometry/<name>/`` of the workspace."""
    if args.out:
        return Path(args.out)
    name = args.name or args.configuration or Path(str(args.config)).stem
    return Path(args.workspace) / GEOMETRY_FOLDER / name


def write_geometry_sidecar(
    folder: Path, args: argparse.Namespace, report: dict[str, Any], overrides: dict[str, Any]
) -> Path:
    """Write ``geometry.json`` into ``folder``: how the geometry was made, and its status.

    Parameters
    ----------
    folder : Path
        The run's folder.
    args : argparse.Namespace
        The parsed command line.
    report : dict
        The run's report.
    overrides : dict
        The values the run overrode, by dotted path.

    Returns
    -------
    Path
        The sidecar path.
    """
    from cfdbender._textio import write_json

    record = {
        "schema": GEOMETRY_SIDECAR_SCHEMA,
        "command": f"cfd-geom {args.verb}",
        "config": str(args.config),
        "configuration": args.configuration,
        "variant": args.variant,
        "until": args.until,
        "overrides": overrides,
        "status": report.get("status"),
        "parameter_id": report.get("parameter_id"),
        "files": sorted(p.name for p in folder.glob("*.igs")),
    }
    path = folder / GEOMETRY_SIDECAR
    folder.mkdir(parents=True, exist_ok=True)
    write_json(path, record, default=str)
    return path


def _iges_convert(args: argparse.Namespace) -> int:
    from cfdbender.geometry.iges_convert import convert_iges

    return print_checks(lambda: convert_iges(args.iges, args.out, name_encoding=args.name_encoding))


#: The order of the exit codes from best to worst, for the code of a loop of sets.
_RANK = (0, 3, 1, 2)


def _print_report(report: dict[str, Any]) -> int:
    print(json.dumps(report, indent=2, default=str))
    return exit_code(report)


def _run(args: argparse.Namespace) -> int:
    from cfdbender.geometry._identify import run_geometry

    try:
        overrides = _overrides(args.set)
    except InputValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    out = _geometry_folder(args)
    report = run_geometry(
        args.config,
        out=out,
        configuration=args.configuration,
        variant=args.variant,
        overrides=overrides,
        until=args.until,
    )
    if report["status"] == "refused":
        print(report["message"], file=sys.stderr)
    else:
        write_geometry_sidecar(out, args, report, overrides)
    return _print_report(report)


def _doe(args: argparse.Namespace) -> int:
    from cfdbender.geometry.doe import run_doe, set_names

    def work() -> dict[str, Any]:
        sets = _read_sets(args.sets)
        names = set_names(sets)
        out = Path(args.out) if args.out else Path(args.workspace) / GEOMETRY_FOLDER
        reports = list(
            run_doe(
                args.config,
                sets,
                out=out,
                configuration=args.configuration,
                variant=args.variant,
                until=args.until,
                workers=args.workers,
            )
        )
        for name, row, report in zip(names, sets, reports, strict=True):
            if report["status"] != "refused":
                overrides = {k: v for k, v in row.items() if k != "name"}
                write_geometry_sidecar(out / name, args, report, overrides)
        worst = max((exit_code(r) for r in reports), key=_RANK.index)
        return {
            "sets": [{"name": n, "report": r} for n, r in zip(names, reports, strict=True)],
            "accepted": worst == 0,
            "exit": worst,
        }

    try:
        record = work()
    except (OSError, GeometryError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, default=str))
    return int(record["exit"])


def _validate(args: argparse.Namespace) -> int:
    from cfdbender.geometry._identify import describe_config

    return report_of(lambda: describe_config(args.config))


def _configurations(args: argparse.Namespace) -> int:
    from cfdbender.geometry._identify import list_configurations

    return report_of(lambda: list_configurations(args.config))


def _stages(args: argparse.Namespace) -> int:
    from cfdbender.geometry._identify import built_stages

    built = built_stages()
    rows = [
        {
            "stage": row.stage,
            "name": row.name,
            "kernel": row.kernel,
            "built": row.stage in built,
            "checks": [{"id": cid, "limit": limit} for cid, limit in row.checks],
        }
        for row in STAGE_ROWS
    ]
    return report_of(lambda: {"stages": rows})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfd-geom",
        description=(
            "Parametric geometry: validate a configuration, run its stages into "
            "inputs/geometry/<name>/, loop parameter sets, identify an original IGES and check "
            "a rebuilt one."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--workspace",
        default=".",
        metavar="PATH",
        help="the workspace whose inputs/geometry/ receives the output (default: the current "
        "folder)",
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="COMMAND")

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--configuration", metavar="NAME", help="entry of the configuration registry"
        )
        sub.add_argument("--variant", default="clean", metavar="V", help="variant (default: clean)")
        sub.add_argument(
            "--until",
            default="S13",
            metavar="Sxx",
            help="last stage to run, S0 to S13 (default S13; S10 and earlier need no kernel)",
        )

    run = verbs.add_parser("run", help="run the stages S0 to S13 and write the report")
    run.add_argument("config", metavar="CONFIG", help="the configuration (TOML)")
    run.add_argument(
        "--out",
        metavar="DIR",
        help="folder of the records and files (default: inputs/geometry/<name>/ of the workspace)",
    )
    run.add_argument(
        "--name",
        metavar="NAME",
        help="the geometry's folder name (default: the configuration, else the file's stem)",
    )
    run.add_argument(
        "--set",
        action="append",
        metavar="K=V",
        help="override one value, component.key=value (repeatable; the value is TOML)",
    )
    common(run)
    run.set_defaults(handler=_run)

    doe = verbs.add_parser("doe", help="one report per parameter set; continues after a failure")
    doe.add_argument("config", metavar="CONFIG", help="the configuration (TOML)")
    doe.add_argument("sets", metavar="SETS", help="TOML with one [[sets]] table per parameter set")
    doe.add_argument(
        "--out",
        metavar="DIR",
        help="folder of one sub-folder per set, named by the set's name or set-0001, set-0002, "
        "... (default: inputs/geometry/ of the workspace)",
    )
    doe.add_argument("--workers", type=int, default=1, metavar="N", help="sets in parallel")
    common(doe)
    doe.set_defaults(handler=_doe)

    for name, handler, text in (
        ("validate", _validate, "resolve the configuration and print every value with its source"),
        ("configurations", _configurations, "list the configurations, variants and members"),
    ):
        sub = verbs.add_parser(name, help=text)
        sub.add_argument("config", metavar="CONFIG", help="the configuration (TOML)")
        sub.set_defaults(handler=handler)

    stages = verbs.add_parser("stages", help="list the stages, their checks and limits")
    stages.set_defaults(handler=_stages)

    add_moved_verbs(verbs)

    convert = verbs.add_parser(
        "iges-convert",
        help="rewrite IGES files as one IGES of B-spline faces, names kept (uses the gmsh kernel)",
        description=(
            "Every face of the IGES files is converted to a B-spline surface and written to one "
            "IGES, each component on its own level with one NAME, so a CAD import and this "
            "package read the same components back. The first file's length unit is kept. "
            "Writes the IGES and the record <out>.json."
        ),
    )
    convert.add_argument("iges", nargs="+", help="IGES files, read only")
    convert.add_argument("--out", required=True, help="new .igs")
    convert.add_argument(
        "--name-encoding",
        default="latin-1",
        help=(
            "text encoding of the NAME bytes in the IGES files (default latin-1, one "
            "character per byte; use utf-8 for UTF-8 names). Never guessed: a name invalid "
            "in the encoding is refused"
        ),
    )
    convert.set_defaults(handler=_iges_convert)
    for sub in verbs.choices.values():
        sub.allow_abbrev = False

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``cfd-geom``; returns the exit code (0 accepted, 1 failed, 2 refused, 3 incomplete).

    Parameters
    ----------
    argv : list of str, optional
        The command line without the program name; None reads ``sys.argv``.

    Returns
    -------
    int
        The exit code: 0 accepted, 1 failed, 2 refused, 3 incomplete.

    Examples
    --------
    >>> from cfdbender.geometry.cli import main
    >>> main(["validate", "no-such-file.toml"])  # doctest: +ELLIPSIS
    ...
    2
    """
    args = _build_parser().parse_args(argv)
    handler = getattr(args, "handler", None) or args.mesh_command
    try:
        return int(handler(args))
    except (GeometryError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

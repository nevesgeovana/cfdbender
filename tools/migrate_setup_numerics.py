"""Migrate a workspace to the SETUP, NUMERICS and MESH matrix columns.

Usage::

    python migrate_setup_numerics.py WORKSPACE [--apply] [--solver su2]

The module uses only the standard library (Python 3.11 or later) and imports nothing
from cfdbender, so the same file also runs beside an older cfdbender; the copy in
``tools/`` of the repository is identical. The default is a dry run that prints what
would change. With ``--apply`` the script

1. copies every file it will change to ``archive/migration-setup-numerics-<n>/`` first,
   keeping the layout under the workspace (nothing is deleted);
2. renames the header ``SET`` of every root or ``inputs/matrices/*.fs`` matrix to ``SETUP``
   and inserts a ``NUMERICS`` column right after it (every other line stays byte-identical);
3. splits the top-level keys of every ``inputs/setups/s<id>.toml`` into numerical keys,
   moved to ``inputs/numerics/n<id>.toml`` with the comment lines directly above each key,
   and the other keys, which stay; tables such as ``[solver]`` are never touched;
4. gives setups whose numerical keys and values are identical ONE numerics file (numbered
   in the order of the sorted setup names) and writes its id in the NUMERICS cell of every
   row that names such a setup; a setup without numerical keys gets an empty cell;
5. renames the header ``GEOMETRY`` of those matrices to ``MESH`` in the
   same pass (the column keeps its width, every cell stays as it is).

Only SU2 has a list of numerical keys; for any other solver the script renames the column
and adds an empty NUMERICS column. Running it on a migrated workspace changes nothing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: Numerical native SU2 keys: the "numerical" rows of the key classification of this
#: package (schemes, limiters, gradient method, time discretisation, CFL, linear solver,
#: multigrid, iteration counts, convergence criteria).
SU2_NUMERICAL_KEYS = frozenset(
    {
        "CFL_ADAPT",
        "CFL_ADAPT_PARAM",
        "CFL_NUMBER",
        "CFL_REDUCTION_TURB",
        "CONV_CAUCHY_ELEMS",
        "CONV_CAUCHY_EPS",
        "CONV_FIELD",
        "CONV_NUM_METHOD_FLOW",
        "CONV_NUM_METHOD_TURB",
        "CONV_RESIDUAL_MINVAL",
        "CONV_STARTITER",
        "INNER_ITER",
        "ITER",
        "JST_SENSOR_COEFF",
        "LINEAR_SOLVER",
        "LINEAR_SOLVER_ERROR",
        "LINEAR_SOLVER_ITER",
        "LINEAR_SOLVER_PREC",
        "MGCYCLE",
        "MGLEVEL",
        "MG_CORRECTION_SMOOTH",
        "MG_DAMP_PROLONGATION",
        "MG_DAMP_RESTRICTION",
        "MG_POST_SMOOTH",
        "MG_PRE_SMOOTH",
        "MUSCL_FLOW",
        "MUSCL_TURB",
        "NEWTON_KRYLOV",
        "NEWTON_KRYLOV_DPARAM",
        "NEWTON_KRYLOV_IPARAM",
        "NUM_METHOD_GRAD",
        "OUTER_ITER",
        "REF_DIMENSIONALIZATION",
        "RK_ALPHA_COEFF",
        "SLOPE_LIMITER_FLOW",
        "SLOPE_LIMITER_TURB",
        "TIME_DISCRE_FLOW",
        "TIME_DISCRE_TURB",
        "TIME_MARCHING",
        "VENKAT_LIMITER_COEFF",
        # Neutral keys of the SU2 adapter that mean the same as a native key above.
        "iterations",
        "cfl",
        "convergence_field",
        "convergence_residual",
        # Added beyond the classification: the CFL ramp keys and the gradient method.
        "cfl_max",
        "cfl_growth",
        "NUM_METHOD_GRAD_RECON",
    }
)
NUMERICAL_KEYS = {"su2": SU2_NUMERICAL_KEYS}
_RULE = frozenset("-:+| \t")
_KEY = re.compile(r"""^\s*("[^"]*"|'[^']*'|[A-Za-z0-9_-]+)\s*(=|\.)""")
_BASE = "migration-setup-numerics-"


@dataclass
class Edit:
    """One file the script changes: its workspace-relative name, new bytes and a note."""

    name: str
    data: bytes
    note: str


@dataclass
class Result:
    """What a migration found: the edits, the new files and the summary rows."""

    edits: list[Edit] = field(default_factory=list)
    created: dict[str, bytes] = field(default_factory=dict)
    summary: list[tuple[str, str, list[str]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _is_content(line: str) -> bool:
    text = line.strip()
    return bool(text) and not line.lstrip().startswith("#") and bool(set(text) - _RULE)


def _is_rule(line: str) -> bool:
    text = line.strip()
    return bool(text) and not line.lstrip().startswith("#") and not (set(text) - _RULE)


def _split_eol(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body) :]


def _pieces(body: str) -> tuple[str, list[str], str]:
    core = body.strip()
    start = body.index(core) if core else 0
    return body[:start], core.split("|"), body[start + len(core) :]


def _artifact_id(cell: str, prefix: str) -> str:
    name = cell.strip().removesuffix(".toml")
    if name.isascii() and name.isdecimal():
        return prefix + name
    return name


def rename_mesh_header(text: str) -> tuple[str, str]:
    """Return a matrix text with the header GEOMETRY renamed MESH, and a note.

    The note is empty when nothing changes (no GEOMETRY header, or a header that
    already holds MESH too, which is left unchanged).
    """
    lines = text.splitlines(keepends=True)
    bom = "\ufeff" if text.startswith("\ufeff") else ""
    if bom:
        lines[0] = lines[0][1:]
    head = next((i for i, line in enumerate(lines) if _is_content(line)), None)
    if head is None:
        return text, ""
    body, eol = _split_eol(lines[head])
    lead, cells, tail = _pieces(body)
    names = [c.strip() for c in cells]
    if "GEOMETRY" not in names:
        return text, ""
    if "MESH" in names:
        return text, "both GEOMETRY and MESH in the header; left unchanged"
    column = names.index("GEOMETRY")
    pad = column < len(cells) - 1
    cells[column] = cells[column].replace("GEOMETRY", "MESH" + "    " * pad, 1)
    lines[head] = lead + "|".join(cells) + tail + eol
    return bom + "".join(lines), "renamed GEOMETRY to MESH"


def migrate_matrix(text: str, numerics_of: dict[str, str]) -> tuple[str, str]:
    """Return a matrix text with SET, GEOMETRY and the NUMERICS column migrated, and a note.

    Parameters
    ----------
    text : str
        The matrix file text, with its line endings.
    numerics_of : dict of str to str
        Setup id to the NUMERICS cell its rows get; a setup absent from it gets an empty cell.

    Returns
    -------
    tuple of str
        The new text (equal to ``text`` when nothing changes) and a one-line note.
    """
    renamed, mesh_note = rename_mesh_header(text)
    new, note = _migrate_columns(renamed, numerics_of)
    if not mesh_note:
        return new, note
    if new == renamed:
        return new, mesh_note
    return new, f"{mesh_note} and {note}"


def _migrate_columns(text: str, numerics_of: dict[str, str]) -> tuple[str, str]:
    """Rename SET to SETUP and add the NUMERICS column; see :func:`migrate_matrix`."""
    lines = text.splitlines(keepends=True)
    bom = "﻿" if text.startswith("﻿") else ""
    if bom:
        lines[0] = lines[0][1:]
    head = next((i for i, line in enumerate(lines) if _is_content(line)), None)
    if head is None:
        return text, "no header"
    body, eol = _split_eol(lines[head])
    lead, cells, tail = _pieces(body)
    names = [c.strip() for c in cells]
    if "NUMERICS" in names:
        return text, "already migrated" if "SETUP" in names else "has NUMERICS but no SETUP; left"
    if "SET" in names and "SETUP" in names:
        return text, "both SET and SETUP in the header; left unchanged"
    column = (
        names.index("SET") if "SET" in names else names.index("SETUP") if "SETUP" in names else -1
    )
    if column < 0:
        return text, "no SET or SETUP column; left unchanged"

    def inserted(piece: str, value: str) -> str:
        pre = piece[: len(piece) - len(piece.lstrip())]
        post = piece[len(piece.rstrip()) :]
        return pre + value + post if value else pre + post

    out = list(lines)
    renamed = "SET" in names
    new_cells = list(cells)
    if renamed:
        new_cells[column] = cells[column].replace("SET", "SETUP", 1)
    new_cells.insert(column + 1, inserted(cells[column], "NUMERICS"))
    out[head] = lead + "|".join(new_cells) + tail + eol
    width = len(cells)
    for index in range(head + 1, len(lines)):
        body, eol = _split_eol(lines[index])
        if _is_content(lines[index]):
            lead, pieces, tail = _pieces(body)
            if len(pieces) != width:
                continue
            ident = _artifact_id(pieces[column], "s")
            pieces.insert(column + 1, inserted(pieces[column], numerics_of.get(ident, "")))
            out[index] = lead + "|".join(pieces) + tail + eol
        elif _is_rule(lines[index]):
            lead, pieces, tail = _pieces(body)
            if len(pieces) != width:
                continue
            fill = "-" * max(1, len(pieces[column].strip()))
            pieces.insert(
                column + 1, inserted(pieces[column], fill) if pieces[column].strip() else ""
            )
            out[index] = lead + "|".join(pieces) + tail + eol
    done = ("renamed SET to SETUP and " if renamed else "") + "added NUMERICS"
    return bom + "".join(out), done


def split_setup(text: str, numerical: frozenset[str]) -> tuple[str, str, list[str]]:
    """Split a setup file into its kept text, the moved text and the moved key names.

    Parameters
    ----------
    text : str
        The setup file text.
    numerical : frozenset of str
        The top-level keys that are numerical.

    Returns
    -------
    tuple
        The text that stays, the text moved to the numerics file (empty when none moves) and
        the moved keys in file order. Only top-level keys before the first table move, each
        with the comment lines directly above it.
    """
    lines = text.splitlines(keepends=True)
    moved: set[int] = set()
    blocks: list[str] = []
    keys: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if stripped.startswith("["):
            break
        found = _KEY.match(lines[index])
        end = index + 1
        while found and end <= len(lines):
            try:
                tomllib.loads("".join(lines[index:end]))
                break
            except tomllib.TOMLDecodeError:
                end += 1
        if not found or end > len(lines):
            index += 1
            continue
        name = found.group(1).strip("\"'")
        if name in numerical:
            start = index
            while (
                start > 0 and lines[start - 1].lstrip().startswith("#") and (start - 1) not in moved
            ):
                start -= 1
            span = range(start, end)
            moved.update(span)
            blocks.append("".join(lines[start:end]))
            keys.append(name)
        index = end
    kept = "".join(line for i, line in enumerate(lines) if i not in moved)
    return kept, "".join(blocks), keys


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def migrate(root: Path, solver: str | None = None) -> Result:
    """Work out the migration of a workspace without writing anything.

    Parameters
    ----------
    root : Path
        The workspace root.
    solver : str, optional
        The solver whose numerical-key list applies; read from ``init.toml`` when omitted.

    Returns
    -------
    Result
        The files to change, the numerics files to create and the summary table.
    """
    result = Result()
    if solver is None:
        try:
            solver = tomllib.loads(_read(root / "init.toml")).get("solver")
        except (OSError, tomllib.TOMLDecodeError):
            solver = None
    keys = NUMERICAL_KEYS.get(str(solver), frozenset())
    if not keys:
        result.notes.append(
            f"solver {solver}: no numerical-key list; only the columns are migrated "
            "(SET renamed SETUP, GEOMETRY renamed MESH, an empty NUMERICS column added)"
        )
    numerics_dir = root / "inputs" / "numerics"
    if keys and numerics_dir.is_dir() and any(numerics_dir.glob("n*.toml")):
        result.notes.append("inputs/numerics already holds numerics files; setups left unchanged")
        keys = frozenset()
    numerics_of: dict[str, str] = {}
    seen: list[tuple[dict[str, object], str]] = []
    for path in sorted((root / "inputs" / "setups").glob("*.toml")):
        text = _read(path)
        kept, moved, names = split_setup(text, keys) if keys else (text, "", [])
        ident = path.stem
        if not names:
            result.summary.append((path.name, "-", []))
            continue
        values = tomllib.loads(moved)
        for known, file in seen:
            if known == values:
                number = file
                break
        else:
            number = f"{len(seen) + 1:03d}"
            seen.append((values, number))
            header = "# Numerical keys moved from the setup files by migrate_setup_numerics.\n"
            result.created[f"inputs/numerics/n{number}.toml"] = (header + moved).encode("utf-8")
        numerics_of[ident] = number
        result.edits.append(
            Edit(f"inputs/setups/{path.name}", kept.encode("utf-8"), f"{len(names)} key(s) moved")
        )
        result.summary.append((path.name, f"n{number}.toml", names))
    matrices = [*root.glob("*.fs"), *(root / "inputs" / "matrices").glob("*.fs")]
    for path in sorted(matrices, key=lambda p: p.relative_to(root).as_posix()):
        text = path.read_bytes().decode("utf-8")
        new, note = migrate_matrix(text, numerics_of)
        relative = path.relative_to(root).as_posix()
        if new != text:
            result.edits.append(Edit(relative, new.encode("utf-8"), note))
        else:
            result.notes.append(f"{relative}: {note}")
    return result


def apply(root: Path, result: Result) -> Path | None:
    """Archive the files to change, then write the edits and the new numerics files.

    Returns
    -------
    Path or None
        The archive folder, or None when nothing changes.
    """
    if not result.edits and not result.created:
        return None
    number = 1
    while (root / "archive" / f"{_BASE}{number}").exists():
        number += 1
    archive = root / "archive" / f"{_BASE}{number}"
    for edit in result.edits:
        target = archive / edit.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / edit.name, target)
    for edit in result.edits:
        (root / edit.name).write_bytes(edit.data)
    for name, data in result.created.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return archive


def report(root: Path, result: Result, *, applied: bool, archive: Path | None) -> str:
    """Return the text the command prints: the actions, the summary table and the checks."""
    lines = [
        f"workspace: {root}",
        "mode: " + ("applied" if applied else "dry run (no file changed)"),
    ]
    lines += result.notes
    if not result.edits and not result.created:
        lines.append("nothing to change: the workspace is already migrated")
        return "\n".join(lines) + "\n"
    verb = "changed" if applied else "would change"
    lines += [f"{verb}: {edit.name} ({edit.note})" for edit in result.edits]
    verb = "created" if applied else "would create"
    lines += [f"{verb}: {name}" for name in result.created]
    if archive is not None:
        lines.append(f"archive of the originals: {archive.relative_to(root).as_posix()}/")
    lines += ["", "setup | numerics file | keys moved", "---|---|---"]
    lines += [f"{s} | {n} | {', '.join(k) if k else '-'}" for s, n, k in result.summary]
    lines += [
        "",
        "verify: cfd-matrix plan <matrix> --workspace " + str(root) + " --verbose",
        "        the generated solver input of every point must equal the one planned before",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the migration; return 0, or 2 on a usage error."""
    parser = argparse.ArgumentParser(
        prog="migrate_setup_numerics",
        description=(
            "Rename the matrix columns SET to SETUP and GEOMETRY to MESH, add NUMERICS and "
            "split the setups."
        ),
    )
    parser.add_argument("workspace", help="the workspace root (the folder with init.toml)")
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument("--solver", help="solver whose numerical keys apply (default: init.toml)")
    args = parser.parse_args(argv)
    root = Path(args.workspace).resolve()
    if not (root / "init.toml").is_file():
        print(f"{root} is not a workspace (no init.toml)", file=sys.stderr)
        return 2
    result = migrate(root, args.solver)
    archive = apply(root, result) if args.apply else None
    sys.stdout.write(report(root, result, applied=args.apply, archive=archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

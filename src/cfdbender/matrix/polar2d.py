"""Judge each point of an SU2 or OpenFOAM matrix and write its 2D polars.

The rule (:mod:`cfdbender.core.convergence`) reads the files a point already wrote,
SU2's ``history.csv`` or OpenFOAM's solver log and ``forceCoeffs``, and decides R, C or
neither. Its verdict is written beside the adapter's classification, which it never
changes, in three columns of the matrix's CSV tables: ``rule_verdict`` (R, C or none),
``rule_iteration`` (where the deciding criterion first held) and ``rule_drop_orders``
(the residual drop below its first value at the last row, in orders of magnitude).

A matrix row that sweeps ALPHA on a two-dimensional case (an SU2 mesh stating two
dimensions; every OpenFOAM case, which is one extruded x-y mesh) also gets
``post/<matrix>/polars/P<POL>.txt``: a polar in the fixed-width layout of an XFOIL
polar file, with the seven columns alpha, CL, CD, CDp, CM, Top_Xtr and Bot_Xtr, one row
per point converged by the rule, in ascending alpha. CL, CD and CM are the neutral
coefficients of the CSV tables (CM nose-up positive about the reference origin). The
transition columns are 0.0000, a fully turbulent flow; CDp is 0.00000 and the header
says it is not measured, because neither adapter reads a pressure-drag split. A point
not converged by the rule, or without a readable result, is left out with one warning
naming it (a refusal under warning_refusals) and keeps its CSV row. Other rows get no
polar and no warning.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from cfdbender import __version__
from cfdbender.core.convergence import SOLVERS, Verdict, judge_point
from cfdbender.core.matrix import MatrixError, read_matrix
from cfdbender.core.neutral import neutral_values
from cfdbender.core.plan import shown_digest, workspace_path
from cfdbender.core.refusals import warn_or_refuse

#: The CSV columns of the verdict, after the fixed block of every table.
VERDICT_COLUMNS = ("rule_verdict", "rule_iteration", "rule_drop_orders")


def point_folder(root: Path, row: Mapping[str, object]) -> Path:
    """Return the datapoint folder of a summary row (``sims/sim_<POL>/datapoints/<DP>``)."""
    tag = str(row["point_id"]).split("/", 1)[1]
    return workspace_path(root, f"sims/sim_{row['pol']}/datapoints/{tag}")


def judge_rows(
    root: Path, rows: Sequence[Mapping[str, object]], solver: str | None
) -> dict[str, Verdict | str]:
    """Return the rule's verdict on each point that has a result.

    Parameters
    ----------
    root : Path
        Workspace root.
    rows : sequence of mapping
        The campaign summary's point rows.
    solver : str, optional
        The workspace's solver; a solver the rule does not read gives no verdicts.

    Returns
    -------
    dict
        Point id to its verdict, or to the reason the rule could not read its files.
        A point without a result has no entry.
    """
    if solver not in SOLVERS:
        return {}
    verdicts: dict[str, Verdict | str] = {}
    for row in rows:
        if row.get("result") is None:
            continue
        try:
            verdicts[str(row["point_id"])] = judge_point(point_folder(root, row), solver)
        except (OSError, ValueError) as error:
            verdicts[str(row["point_id"])] = str(error)
    return verdicts


def verdict_cells(verdict: Verdict | str | None) -> dict[str, object]:
    """Return the three verdict cells of one point (None where the point has none)."""
    if not isinstance(verdict, Verdict):
        return dict.fromkeys(VERDICT_COLUMNS)
    return {
        "rule_verdict": verdict.criterion or "none",
        "rule_iteration": verdict.iteration,
        "rule_drop_orders": verdict.drop_orders_last,
    }


#: The column line and dashed rule of the XFOIL polar layout, seven columns.
_COLUMN_LINE = "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr"
_RULE_LINE = "  ------ -------- --------- --------- -------- -------- --------"
_NAMES = {"su2": "SU2", "openfoam": "OpenFOAM"}
#: QPROP fits a section polar from at least three points between these angles.
_FIT_RANGE = (-4.0, 6.0)


def _alpha_rows(root: Path, matrix: str) -> dict[str, str | None]:
    """Return POL to its GEOMETRY cell for every matrix row that sweeps ALPHA."""
    try:
        rows = read_matrix(workspace_path(root, matrix), active_only=False)
    except (MatrixError, OSError, ValueError):
        return {}
    return {
        row.pol: row.geometry
        for row in rows
        if row.sweep_key is not None and row.sweep_key.casefold() == "alpha"
    }


def _planar(solver: str, members: Sequence[Mapping[str, object]]) -> bool:
    if solver == "openfoam":
        return True
    stated = {
        cast(Mapping[str, object], row["result"]).get("dimensions")
        for row in members
        if row.get("result") is not None
    }
    return stated == {2}


def _alpha(row: Mapping[str, object]) -> float:
    condition = cast(Mapping[str, object], row["condition"])
    value = next(v for k, v in condition.items() if str(k).casefold() == "alpha")
    return float(cast(float, value))


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _reason(
    row: Mapping[str, object], verdict: Verdict | str | None, cells: Mapping[str, object]
) -> str | None:
    if row.get("result") is None:
        return f"no readable result (status {row['status']})"
    result = cast(Mapping[str, object], row["result"])
    if result.get("classification") == "failed" or row.get("classification") == "failed":
        return "solver execution or collected evidence failed"
    if not isinstance(verdict, Verdict):
        return f"the convergence rule cannot read its files: {verdict}"
    if not verdict.converged:
        return f"not converged by the rule: {verdict.describe()}"
    if not all(_finite(cells.get(name)) for name in ("CL", "CD", "CM")):
        return "the solver reported no finite CL, CD and CM"
    return None


def _turbulence(folder: Path, solver: str) -> str:
    sources = (
        [(folder / "su2.cfg", r"^KIND_TURB_MODEL[ ]*=[ ]*([A-Za-z0-9_]+)")]
        if solver == "su2"
        else [
            (folder / "constant" / name, r"(?:RASModel|model)[ ]+([A-Za-z0-9_]+);")
            for name in ("momentumTransport", "turbulenceProperties")
        ]
    )
    for path, pattern in sources:
        if path.is_file():
            found = re.search(pattern, path.read_text(encoding="utf-8", errors="replace"), re.M)
            if found:
                return found[1]
    return "not stated"


def _mesh(geometry: str | None, members: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    name = Path(geometry).name if geometry else "not stated"
    for row in members:
        for path, digest in cast(Mapping[str, str], row.get("inputs_sha256") or {}).items():
            if Path(path).name == name:
                return name, str(shown_digest(digest))
    return name, "not recorded"


def _number(value: object) -> float:
    return float(cast(float, value)) if _finite(value) else 0.0


def _cells(row: Mapping[str, object], solver: str | None) -> dict[str, object]:
    """Resolve polar context and neutral coefficients through the family-table point."""
    from cfdbender.matrix.exports import summary_point

    point = summary_point(row, solver)
    result = cast(Mapping[str, object], row.get("result") or {})
    return {
        **point.condition,
        **neutral_values(
            solver, point.native, dimensions=result.get("dimensions"), condition=point.condition
        ),
    }


def _polar_text(
    root: Path,
    solver: str,
    geometry: str | None,
    members: Sequence[Mapping[str, object]],
    kept: Sequence[tuple[float, Mapping[str, object]]],
    left: Sequence[float],
    context: Mapping[str, object],
) -> str:
    known = [row for row in members if row.get("result") is not None]
    result = cast(Mapping[str, object], known[0]["result"]) if known else {}
    version = str(result.get("solver_version") or "unknown")
    first = kept[0][1] if kept else context
    mesh, digest = _mesh(geometry, members)
    model = _turbulence(point_folder(root, known[0]), solver) if known else "not stated"
    lines = [
        " ",
        f"       cfdbender {__version__} polar from {_NAMES[solver]} {version}",
        " ",
        f" Calculated polar for: {mesh}",
        " ",
        " 1 1 Reynolds number fixed          Mach number fixed",
        " ",
        " xtrf =   0.000 (top)        0.000 (bottom)   fully turbulent",
        f" Mach = {_number(first.get('MACH')):7.3f}     Re = {_number(first.get('RE')):9.3f} e 6"
        f"     turbulence model {model}",
        " ",
        f" mesh sha256 {digest}",
        " CDp not measured: the column holds 0.00000",
        " rows: the points converged by the rule (R, residual target; C, Cauchy on the forces)",
    ]
    if solver == "openfoam":
        lines.append(" incompressible solver: Mach sets only the speed")
    if left:
        lines.append(" left out, not converged: " + " ".join(f"{a:.3f}" for a in left))
    lines += [" ", _COLUMN_LINE, _RULE_LINE]
    for alpha, cells in kept:
        lines.append(
            f"{alpha:8.3f}{_number(cells['CL']):9.4f}{_number(cells['CD']):10.5f}"
            f"{0.0:10.5f}{_number(cells['CM']):9.4f}{0.0:9.4f}{0.0:9.4f}"
        )
    return chr(10).join(lines) + chr(10)


def write_polars(
    root: Path,
    matrix: str,
    rows: Sequence[Mapping[str, object]],
    solver: str | None,
    verdicts: Mapping[str, Verdict | str],
) -> tuple[Path, ...]:
    """Write post/<matrix>/polars/P<POL>.txt for every 2D ALPHA sweep of the matrix.

    Parameters
    ----------
    root : Path
        Workspace root.
    matrix : str
        Recorded matrix path, inputs/matrices/<name>.fs; its rows give the swept key
        and the mesh.
    rows : sequence of mapping
        The campaign summary's point rows.
    solver : str, optional
        The workspace's solver; only SU2 and OpenFOAM get a polar.
    verdicts : mapping
        The rule's verdict per point id, from :func:`judge_rows`.

    Returns
    -------
    tuple of Path
        The polars written.

    Raises
    ------
    WorkspaceError
        Under warning_refusals, for the first point left out; no polar is written then.
    """
    from cfdbender.matrix.exports import _write

    if solver not in SOLVERS:
        return ()
    sweeps = _alpha_rows(root, matrix)
    by_pol: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_pol.setdefault(str(row["pol"]), []).append(row)
    texts: dict[str, str] = {}
    for pol, members in by_pol.items():
        if pol not in sweeps or not _planar(solver, members):
            continue
        kept: list[tuple[float, Mapping[str, object]]] = []
        left: list[float] = []
        for row in sorted(members, key=_alpha):
            alpha = _alpha(row)
            cells = _cells(row, solver)
            reason = _reason(row, verdicts.get(str(row["point_id"])), cells)
            if reason is None:
                kept.append((alpha, cells))
                continue
            left.append(alpha)
            tag = str(row["point_id"]).split("/", 1)[1]
            warn_or_refuse(
                f"POL {pol} point {tag} at ALPHA {alpha:g} left out of polars/P{pol}.txt: {reason}"
            )
        fitted = [a for a, _ in kept if _FIT_RANGE[0] <= a <= _FIT_RANGE[1]]
        if len(fitted) < 3:
            warnings.warn(
                f"POL {pol}: polars/P{pol}.txt has fewer than three converged points between "
                f"{_FIT_RANGE[0]:g} and {_FIT_RANGE[1]:g} degrees, so QPROP cannot fit it",
                UserWarning,
                stacklevel=2,
            )
        texts[pol] = _polar_text(
            root, solver, sweeps[pol], members, kept, left, _cells(members[0], solver)
        )
    folder = workspace_path(root, f"post/{Path(matrix).stem}/polars")
    return tuple(_write(folder / f"P{pol}.txt", text) for pol, text in texts.items())

"""Read SU2 CSV and affirmative solver log evidence into the neutral contract.

SU2 residual columns are log10 values and steady Inner_Iter starts at zero.
The neutral Iteration carries linear residual magnitudes and one-based counts.
Coefficient names retain SU2 spelling: CL/CD/CSF use wind z/x/y; CFx/CFy/CFz
and CMx/CMy/CMz use body x/y/z. No wind-to-body rotation is inferred.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from cfdbender.core.results import Coefficient, ExecutionResult, Iteration, References


@dataclass(frozen=True)
class _FramedCoefficient(Coefficient):
    """Retain a per-column frame when the shared dataclass serializes coefficients."""

    axis_system: Literal["wind", "body"]


_AXES: dict[str, tuple[Literal["x", "y", "z"], Literal["force", "moment"]]] = {
    "CL": ("z", "force"),
    "CD": ("x", "force"),
    "CSF": ("y", "force"),
    "CFx": ("x", "force"),
    "CFy": ("y", "force"),
    "CFz": ("z", "force"),
    "CMx": ("x", "moment"),
    "CMy": ("y", "moment"),
    "CMz": ("z", "moment"),
}


def references(cfg: dict[str, str]) -> References:
    return References(
        float(cfg["REF_AREA"]),
        float(cfg["REF_LENGTH"]),
        tuple(float(cfg[f"REF_ORIGIN_MOMENT_{axis}"]) for axis in "XYZ"),  # type: ignore[arg-type]
        "SU2 wind (CD, CSF, CL); SU2 body (CFx, CFy, CFz, CMx, CMy, CMz)",
        ("wind drag / body +x", "wind side force / body +y", "wind lift / body +z"),
    )


def dimensions(folder: Path, *, mesh: bool = True) -> int | None:
    """Return 2 or 3, the dimension of the problem SU2 solved, or None when unstated.

    SU2 prints "Two dimensional problem." or "Three dimensional problem." while it
    reads the mesh; a native mesh states NDIME near its top. The dimension decides
    which body moment is the pitching moment: CMz in two dimensions (the flow is in
    the x-y plane), CMy in three.
    """
    for name in ("stdout.log", "stderr.log"):
        path = folder / name
        if path.is_file():
            found = re.search(
                "(Two|Three) dimensional problem[.]",
                path.read_text(encoding="utf-8", errors="replace"),
            )
            if found:
                return 2 if found[1] == "Two" else 3
    if not mesh:
        return None
    staged = folder / "mesh.su2"
    try:
        with staged.open(encoding="ascii", errors="replace") as stream:
            for _, line in zip(range(200), stream, strict=False):
                stated = re.fullmatch("NDIME[ ]*=[ ]*([23])[ ]*", line.strip())
                if stated:
                    return int(stated[1])
    except OSError:
        return None
    return None


#: SU2's CONV_FIELD names and the history.csv columns they converge on.
CONV_COLUMNS = {
    "RMS_DENSITY": "rms[Rho]",
    "RMS_MOMENTUM-X": "rms[RhoU]",
    "RMS_MOMENTUM-Y": "rms[RhoV]",
    "RMS_MOMENTUM-Z": "rms[RhoW]",
    "RMS_ENERGY": "rms[RhoE]",
    "RMS_NU_TILDE": "rms[nu]",
    "RMS_TKE": "rms[k]",
    "RMS_DISSIPATION": "rms[w]",
    "RMS_PRESSURE": "rms[P]",
    "RMS_VELOCITY-X": "rms[U]",
    "RMS_VELOCITY-Y": "rms[V]",
    "RMS_VELOCITY-Z": "rms[W]",
    "RMS_TEMPERATURE": "rms[T]",
}
#: The name recorded when CONV_FIELD names no residual column (a Cauchy coefficient).
LARGEST_RMS = "largest rms"


def _segment_configs(folder: Path, *, required: bool = True) -> list[dict[str, str]]:
    """Return the keys of each configuration the point ran, in order: su2.cfg, then segment two.

    A folder without ``su2.cfg`` is refused, or, with ``required=False``, read as one cold
    configuration.
    """
    if not required and not (folder / "su2.cfg").is_file():
        return [{}]
    found: list[dict[str, str]] = []
    for name in ("su2.cfg", "su2_segment2.cfg"):
        path = folder / name
        if name == "su2.cfg" or path.is_file():
            lines = path.read_text().splitlines()
            found.append(
                {
                    key.strip(): value.strip()
                    for key, value in (line.split("=", 1) for line in lines)
                }
            )
    return found


def _cfg(folder: Path) -> dict[str, str]:
    """Return the keys of the point's configuration: its last segment's where it has two."""
    found: dict[str, str] = {}
    for keys in _segment_configs(folder):
        found.update(keys)
    return found


def _yes(keys: Mapping[str, str], key: str) -> bool:
    return keys.get(key, "").strip().upper() in ("YES", "TRUE")


#: The stem of the master configuration file, which names a multizone run's history.
MULTIZONE_HISTORY_STEM = "su2"


def history_name(keys: Mapping[str, str]) -> str:
    """Return the history file SU2 8.6.0 writes for one configuration.

    A restarted time-accurate run (``TIME_DOMAIN`` and ``RESTART_SOL``) appends its
    ``RESTART_ITER`` as five digits, ``history_00060.csv``, and writes no ``history.csv``; every
    other single-zone run writes ``history.csv`` (``CONV_FILENAME= history`` is the adapter's).
    A multizone run (``MULTIZONE``) writes its driver history under the stem of the master
    configuration file, ``su2.csv``, whatever ``CONV_FILENAME`` says.
    """
    if _yes(keys, "MULTIZONE"):
        return f"{MULTIZONE_HISTORY_STEM}.csv"
    if _yes(keys, "TIME_DOMAIN") and _yes(keys, "RESTART_SOL"):
        return f"history_{int(keys.get('RESTART_ITER', '0')):05d}.csv"
    return "history.csv"


def history_outputs(values: Mapping[str, object]) -> list[str]:
    """Return the history files a point's configurations write, from its SU2 keys, in order."""
    from cfdbender.solvers.su2._run_types import SEGMENT_TWO

    def rendered(prefix: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for key in ("MULTIZONE", "TIME_DOMAIN", "RESTART_SOL", "RESTART_ITER"):
            value = values.get(prefix + key, values.get(key))
            if value is not None:
                found[key] = ("YES" if value else "NO") if isinstance(value, bool) else str(value)
        return found

    names = [history_name(rendered(""))]
    if any(key.startswith(SEGMENT_TWO + "/") for key in values):
        names.append(history_name(rendered(SEGMENT_TWO + "/")))
    return list(dict.fromkeys(names))


def step_offset(folder: Path) -> int:
    """Return the SU2 time iterations before the row's first step: RESTART_ITER of su2.cfg.

    A time-accurate point started from a steady solution begins at ``RESTART_ITER`` (the
    steady solution is staged as the iterations before it), so its step ``s`` is SU2's time
    iteration ``s - 1 + offset``; a cold start has offset 0.
    """
    first = _segment_configs(folder, required=False)[0]
    if _yes(first, "TIME_DOMAIN") and _yes(first, "RESTART_SOL"):
        return int(first.get("RESTART_ITER", "0"))
    return 0


def history_names(folder: Path) -> list[str]:
    """Return the history files of a point's configurations, in the order they ran."""
    configs = _segment_configs(folder, required=False)
    return list(dict.fromkeys(history_name(keys) for keys in configs))


def history_table(folder: Path, counter: str = "Time_Iter") -> tuple[list[str], list[list[str]]]:
    """Return the joined time history of a point: one header, the rows of every segment in order.

    Each configuration's history (:func:`history_name`) is read; the segments must share their
    columns and continue each other: the first ``counter`` of a segment is the last of the one
    before plus one. A gap or an overlap is refused naming both files, and a missing history
    names every file looked for.
    """
    names = history_names(folder)
    missing = [name for name in names if not (folder / name).is_file()]
    if missing:
        raise ValueError(
            f"{folder}: no SU2 history {', '.join(missing)} (looked for {', '.join(names)}; "
            "SU2 writes history_<RESTART_ITER>.csv for a restarted time-accurate run and "
            "history.csv otherwise)"
        )
    headers: list[str] = []
    rows: list[list[str]] = []
    previous: tuple[str, int] | None = None
    for name in names:
        path = folder / name
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, skipinitialspace=True)
            raw = next(reader, None)
            if raw is None:
                raise ValueError(f"{path}: empty history")
            own = history_headers(raw, path)
            own_rows = [cells for cells in reader if cells]
        if not own_rows:
            raise ValueError(f"{path}: history has no data rows")
        if headers and own != headers:
            raise ValueError(f"{path}: its columns differ from those of {names[0]}")
        headers = own
        if len(names) > 1 and counter in headers and own_rows:
            index = headers.index(counter)
            try:
                first, last = int(float(own_rows[0][index])), int(float(own_rows[-1][index]))
            except (ValueError, IndexError, OverflowError) as error:
                raise ValueError(f"{path}: {counter} is not a whole number") from error
            if previous is not None and first != previous[1] + 1:
                word = "overlap" if first <= previous[1] else "gap"
                raise ValueError(
                    f"{folder}: {word} between the segments: {previous[0]} ends at {counter} "
                    f"{previous[1]} and {name} begins at {first}"
                )
            previous = (name, last)
        rows.extend(own_rows)
    return headers, rows


def _convergence_columns(cfg: dict[str, str], residuals: list[str]) -> tuple[list[str], str]:
    """Return the residual columns SU2 converges on and the name they are recorded under.

    The columns are those of CONV_FIELD (SU2's default RMS_DENSITY). A CONV_FIELD that
    names no residual column, such as a Cauchy criterion on a coefficient, falls back to
    the largest of all rms columns.
    """
    fields = re.split(r"[\s,()]+", cfg.get("CONV_FIELD", "RMS_DENSITY").upper())
    # A multizone history names a column ``rms[Rho][<zone>]``: every zone's one is converged on.
    wanted = [CONV_COLUMNS[name] for name in fields if name in CONV_COLUMNS]
    columns = [key for key in residuals if re.sub(r"\[\d+\]$", "", key) in wanted]
    if not columns:
        return residuals, LARGEST_RMS
    return columns, ", ".join(columns)


def _zone_headers(folder: Path, headers: list[str]) -> list[str]:
    """Name the loads of a multizone history as a single-zone one: those of the turning zone.

    SU2's multizone output writes ``CD[0]``, ``CFx[1](blade)``: the field, the zone, then the
    marker. The loads are read from the zone whose file states a grid motion, as the time
    average reads them (``sliding.rotating_zone``); the residual columns keep their zones. A
    history without zone loads is returned as it is.
    """
    keys = _segment_configs(folder, required=False)[0]
    if not _yes(keys, "MULTIZONE"):
        return headers
    zone = 0
    for index in range(len(re.findall(r"zone_\d+[.]cfg", keys.get("CONFIG_LIST", "")))):
        path = folder / f"zone_{index}.cfg"
        if path.is_file() and re.search(
            r"^GRID_MOVEMENT\s*=\s*(?!NONE\b)\S", path.read_text(), re.MULTILINE
        ):
            zone = index
            break
    pattern = re.compile(rf"([A-Za-z]+)\[{zone}\](.*)")
    renamed: list[str] = []
    for key in headers:
        found = pattern.fullmatch(key)
        renamed.append(found[1] + found[2] if found and found[1] in _AXES else key)
    return renamed


#: The counters of a history row the adapter reads.
_COUNTERS = ("Inner_Iter", "Outer_Iter", "Time_Iter")


def _used(name: str) -> bool:
    """Whether the adapter reads a history column: a counter, a residual or a coefficient."""
    return (
        name in _COUNTERS
        or name.startswith(("rms[", "TAVG_"))
        or name.split("(", 1)[0].split("[", 1)[0] in _AXES
    )


def history_headers(raw: list[str], path: Path) -> list[str]:
    """Return the history's column names, a duplicated one kept apart by position.

    SU2 8.6.0 writes ``Time(sec)`` twice when HISTORY_OUTPUT holds WALL_TIME. The first
    occurrence keeps its name and a later one gets ``#2``, ``#3``; a duplicated column the
    adapter reads is refused, naming the column and the file.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for key in (cell.strip() for cell in raw):
        count = seen.get(key, 0) + 1
        seen[key] = count
        if count > 1 and _used(key):
            raise ValueError(f"{path}: column {key} is duplicated, so its value is ambiguous")
        names.append(key if count == 1 else f"{key}#{count}")
    return names


def residual_field(folder: Path) -> str | None:
    """Return the name of the residual the history records, or None without a history."""
    try:
        if _yes(_segment_configs(folder, required=False)[0], "TIME_DOMAIN"):
            headers = _zone_headers(folder, history_table(folder)[0])
        else:
            with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
                headers = history_headers(
                    next(csv.reader(stream, skipinitialspace=True)), folder / "history.csv"
                )
        residuals = [key for key in headers if key.startswith("rms[")]
        return _convergence_columns(_cfg(folder), residuals)[1] if residuals else None
    except (OSError, ValueError, StopIteration):
        return None


def read(folder: Path) -> tuple[ExecutionResult, dict[str, Coefficient], References]:
    cfg = _cfg(folder)
    refs = references(cfg)
    # A cluster job leaves no execution.json: its exit code is not seen here and
    # classification rests on the history and the solver log alone.
    execution = folder / "execution.json"
    metadata = json.loads(execution.read_bytes()) if execution.is_file() else {"exit_code": 0}
    log = chr(10).join(
        path.read_text(encoding="utf-8", errors="replace")
        for name in ("stdout.log", "stderr.log")
        if (path := folder / name).is_file()
    )
    version = re.search(
        "(?:SU2.*?(?:version|Version|v)|Version|Release)[ ]*([0-9]+[.][0-9]+[.][0-9]+)", log
    )
    # A time-accurate deck marches TIME_ITER steps, each one Time_Iter of the history; its
    # iterations are the steps and the last row of a step is that step's state.
    timed = "TIME_ITER" in cfg
    counter = "Time_Iter" if timed else "Inner_Iter"
    # A time-accurate point's iterations are the row's steps, 1-based, whatever SU2 numbered.
    offset = step_offset(folder) if timed else 0
    limit = int(cfg["TIME_ITER"] if timed else cfg["ITER"]) - offset
    result = ExecutionResult(
        int(metadata["exit_code"]), (), limit, version[1] if version else "unknown"
    )
    # Affirmative lines only. 'Exit Success' says nothing about numerical convergence.
    # The message counts only when no iteration line follows it: a banner printed before
    # later iterations belongs to an earlier state of the log, not to the last iteration.
    lines = log.splitlines()
    last_iteration = max(
        (n for n, line in enumerate(lines) if re.match(r"\s*[|]\s*\d+\s*[|]", line)),
        default=-1,
    )
    converged = any(
        re.fullmatch(
            "(?:[|* ]*)(?:(?:All )?Convergence criteria (?:were )?satisfied[.!]?|"
            "SU2 has successfully converged[.!]?)(?:[|* ]*)",
            line.strip(),
            re.IGNORECASE,
        )
        is not None
        for line in lines[last_iteration + 1 :]
    )
    history: list[Iteration] = []
    coefficients: dict[str, Coefficient] = {}
    try:
        if timed:
            # Every segment's history, joined in step order (history_<RESTART_ITER>.csv).
            path = folder / history_name(_segment_configs(folder)[0])
            headers, table = history_table(folder, counter)
            headers = _zone_headers(folder, headers)
        else:
            path = folder / "history.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, skipinitialspace=True)
                headers = history_headers(next(reader), path)
                table = list(reader)
        if counter not in headers:
            raise ValueError(f"{path}: no {counter} column")
        residuals = [key for key in headers if key.startswith("rms[")]
        if not residuals or not any(key in _AXES for key in headers):
            raise ValueError(f"{path}: no residual or no coefficient column")
        # The residual recorded is the one SU2 converges on, so it shows the criterion.
        residuals = _convergence_columns(cfg, residuals)[0]
        for cells in table:
            row = dict(zip(headers, cells, strict=True))
            # Reject non-finite numerical history, including uncollected coefficient fields.
            if not all(math.isfinite(float(value)) for value in cells):
                raise ValueError("history.csv contains non-finite values")
            iteration = int(row[counter]) + 1 - offset
            residual = max(10.0 ** float(row[key]) for key in residuals)
            if timed and history and iteration == history[-1].iteration:
                history[-1] = Iteration(iteration, residual)
            else:
                if history and iteration <= history[-1].iteration:
                    raise ValueError("history.csv iterations must increase")
                history.append(Iteration(iteration, residual))
            coefficients = {
                key: _FramedCoefficient(
                    float(row[key]),
                    axis,
                    kind,
                    "wind" if key in ("CL", "CD", "CSF") else "body",
                )
                for key, (axis, kind) in _AXES.items()
                if key in row
            }
        if history and converged:
            history[-1] = replace(history[-1], converged=True)
    except (OSError, ValueError, StopIteration, OverflowError) as error:
        reason = str(error) or type(error).__name__
        named = "history" if timed else "history.csv"
        if isinstance(error, StopIteration):
            reason = f"{folder / 'history.csv'}: empty"
        elif named not in reason:
            reason = f"{folder / 'history.csv'}: {reason}"
        return replace(result, evidence_error=reason), {}, refs
    return replace(result, history=tuple(history)), coefficients, refs


def force_groups(
    folder: Path, requested: Mapping[str, tuple[str, ...]], total: Mapping[str, Coefficient]
) -> Mapping[str, Mapping[str, Coefficient]]:
    """Sum native per-marker coefficients about the same axes and reference origin."""
    if not requested:
        return {}
    from cfdbender.solvers.su2._time_history import zone_column

    path = folder / "history.csv"
    if _yes(_segment_configs(folder, required=False)[0], "TIME_DOMAIN"):
        headers, rows = history_table(folder)
        headers = _zone_headers(folder, headers)
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, skipinitialspace=True)
            headers = history_headers(next(reader, []), path)
            rows = list(reader)
    if not rows or not total:
        raise ValueError("history.csv: missing boundary force results")
    row = dict(zip(headers, rows[-1], strict=False))
    result: dict[str, Mapping[str, Coefficient]] = {"TOTAL": dict(total)}
    for name, markers in requested.items():
        coefficients = {}
        for key, coefficient in total.items():
            columns = [zone_column(key, None, marker) for marker in markers]
            if missing := [column for column in columns if column not in row]:
                raise ValueError(
                    f"history.csv: group {name} lacks {', '.join(missing)}; keep AERO_COEFF_SURF"
                )
            values = [float(row[column]) for column in columns]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"history.csv: group {name} contains non-finite coefficients")
            coefficients[key] = replace(coefficient, value=sum(values))
        result[name] = coefficients
    return result

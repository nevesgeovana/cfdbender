"""Continue a finished or killed time-accurate SU2 point in place (``cfd-matrix run --continue``).

SU2 8.6.0 continues a dual-time run as a restart at ``RESTART_ITER = R``: it reads the restart
files of the iterations before ``R`` (``R - 2`` and ``R - 1`` under second-order dual time
stepping, ``R - 1`` otherwise), named ``<RESTART_FILENAME stem>_<k:05d>.dat`` for one zone and
``<stem>_<zone>_<k:05d>.dat`` for each zone of a multizone run. A finished run keeps its last
two (a real two-zone run, 9081, kept ``restart_flow_0_00012.dat``, ``_00013.dat`` and the same
of zone 1 with ``TIME_ITER= 14``); a run with ``RESTART_EVERY_REV`` keeps a pair at every
multiple of its frequency, so a killed run continues from its latest complete pair.

The continuation runs in the point's own folder. What a new run of the folder writes again (the
configuration, its history, the logs and the job files) is moved to ``continued/<n>/`` before it
runs (:func:`evidence` names them); the restart files, the mesh, the zone files and the files of
a frozen-rotor stage stay, because the continuation reads them or does not write them.
"""

from __future__ import annotations

import re
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.plan import workspace_path
from cfdbender.solvers.su2._config import (
    config,
    extra_files,
    staged_mesh,
)
from cfdbender.solvers.su2._history import (
    _cfg,
    _keys,
    history_name,
    history_outputs,
    step_offset,
)
from cfdbender.solvers.su2._run_types import (
    SEGMENT_TWO,
    SEGMENTS_SCRIPT,
    SU2_RESTART_FILENAME,
    restart_order,
)

#: The files of a run a later run of the same folder writes again (besides configs and histories).
JOB_FILES = (
    "stdout.log",
    "stderr.log",
    "execution.json",
    "job.exit",
    "submit.stdout.txt",
    "submit.stderr.txt",
)
#: The row variables that set the length of a time-accurate run.
RUN_LENGTH_KEYS = ("REVOLUTIONS", "TIME_ITERATIONS")


def _yes(keys: dict[str, str], key: str, default: bool) -> bool:
    value = keys.get(key)
    if value is None:
        return default
    return value.strip().upper() in ("YES", "TRUE")


def _zones(keys: dict[str, str]) -> int:
    if not _yes(keys, "MULTIZONE", False):
        return 0
    return len([item for item in keys.get("CONFIG_LIST", "").strip("() ").split(",") if item])


def restart_iteration(folder: Path) -> int:
    """Return the SU2 time iteration the point continues at: one after its latest restart pair.

    The pair is the ``restart_order`` iterations SU2 reads, present and not empty for every
    zone, under the restart file name of the point's last configuration.

    Raises
    ------
    ValueError
        No complete pair: the message names the files of the run's last iterations that are
        missing.
    """
    keys = _cfg(folder)
    name = keys.get("RESTART_FILENAME", SU2_RESTART_FILENAME)
    stem = re.sub(r"[.](dat|csv)$", "", name)
    suffix = ".dat" if _yes(keys, "WRT_BINARY_RESTART", True) else ".csv"
    order = restart_order(keys)
    zones = _zones(keys)

    def file(zone: int, k: int) -> str:
        return f"{stem}_{zone}_{k:05d}{suffix}" if zones else f"{stem}_{k:05d}{suffix}"

    pattern = re.compile(
        rf"{re.escape(stem)}_(?:(\d+)_)?(\d{{5}}){re.escape(suffix)}"
        if zones
        else rf"{re.escape(stem)}_(\d{{5}}){re.escape(suffix)}"
    )
    found: set[tuple[int, int]] = set()
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size == 0:
            continue
        if zones:
            if match.group(1) is None:
                continue
            found.add((int(match.group(1)), int(match.group(2))))
        else:
            found.add((0, int(match.group(1))))
    for k in sorted({k for _, k in found}, reverse=True):
        levels = range(k - order + 1, k + 1)
        if k - order + 1 >= 0 and all(
            (zone, level) in found for zone in range(max(zones, 1)) for level in levels
        ):
            return k + 1
    end = int(keys.get("TIME_ITER", "0"))
    wanted = [file(z, k) for z in range(max(zones, 1)) for k in range(end - order, end) if k >= 0]
    absent = [name for name in wanted if not (folder / name).is_file()] or wanted
    raise ValueError(
        f"{folder}: the point keeps no restart files to continue from: {', '.join(absent)} "
        f"{'is' if len(absent) == 1 else 'are'} missing or empty (the last "
        f"{order} iteration(s) of every zone SU2 reads to continue)"
    )


def evidence(folder: Path) -> list[str]:
    """Name the files of the folder's latest run that a continuation writes again.

    The configuration (``su2.cfg``, ``su2_segment2.cfg`` and ``run_segments.sh``), the history
    of each configuration (``su2.csv`` for a multizone run) and the job files, those present.
    The restart files, the mesh, the zone files and a frozen-rotor stage's files stay.
    """
    names: list[str] = []
    for config_name in ("su2.cfg", SEGMENT_TWO):
        path = folder / config_name
        if path.is_file():
            names += [config_name, history_name(_keys(path))]
    names += [SEGMENTS_SCRIPT, *JOB_FILES]
    return [name for name in dict.fromkeys(names) if (folder / name).is_file()]


def continuation(point: AdapterPoint) -> dict[str, object]:
    """Resolve how a recorded time-accurate point continues with its grown row.

    ``point`` is the newly planned point (the row with its grown ``REVOLUTIONS`` or
    ``TIME_ITERATIONS``), whose folder holds the earlier run. Returns the SU2 time iteration
    it restarts at (``restart_iter``), the first and last steps of the continuation in the
    row's own steps, the declared outputs of the continued point and the files to move aside
    (``evidence``).

    Raises
    ------
    ValueError
        The folder holds no run, it keeps no restart pair, or nothing is left to march (the
        run length did not grow).
    """
    folder = workspace_path(point.root, point.point.directory)
    if not (folder / "su2.cfg").is_file():
        raise ValueError(
            f"POL {point.point.pol}: {folder} holds no su2.cfg, so there is no run to continue"
        )
    start = restart_iteration(folder)
    offset = step_offset(folder)
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
        restarted=point.point.restart_from is not None,
        precursor=point.point.precursor,
        continue_from=start,
    )
    total = int(str(values.get(f"{SEGMENT_TWO}/TIME_ITER", values["TIME_ITER"])))
    if start >= total:
        key = next((k for k in RUN_LENGTH_KEYS if k in point.point.variables), RUN_LENGTH_KEYS[0])
        raise ValueError(
            f"POL {point.point.pol}: {key} did not grow: the point already reached step "
            f"{start - offset}, the last of {key} {point.point.variables.get(key)!s}; raise "
            f"{key} in the matrix row to continue it"
        )
    names = (
        "su2.cfg",
        staged_mesh(values),
        "stdout.log",
        "stderr.log",
        "execution.json",
        *history_outputs(values),
        *extra_files(values),
    )
    return {
        "restart_iter": start,
        "first_step": start - offset + 1,
        "last_step": total - offset,
        "outputs": [f"{point.point.directory}/{name}" for name in dict.fromkeys(names)],
        "evidence": evidence(folder),
    }

"""The steady frozen-rotor stage that starts a time-accurate sliding-mesh rotor point.

A row whose RESTART cell holds ``MRF`` runs two solver stages in one point folder and one job.
Stage 1 (``precursor.cfg``) is the same two-zone mesh at the same condition solved steady, the
turning zone in a rotating frame; stage 2 (``su2.cfg``) is the time-accurate sliding run, which
restarts from stage 1's solution (``RESTART_SOL``, ``RESTART_ITER``, ``SOLUTION_FILENAME``).

Both stages come from one rendering of the setup, so they share the flow and turbulence model:
the stage-1 keys are the point's own keys minus those only a time-accurate run reads. SU2 8.6.0
reads ``<stem of SOLUTION_FILENAME>_<zone>_<k:05d>.dat`` for the ``restart_order`` iterations
before ``RESTART_ITER`` and does not refuse a restart written by another model, so the staging
below copies stage 1's restart file of each zone to each of those names, with plain ``cp``.

The job runs the stages through ONE launcher (the profile's application wrapper included):
``run_mrf.sh`` calls the command it is given once per stage with the stage's configuration file
as its last argument, which is how a command already receives ``su2.cfg``. Names of this
module that are not SU2's are PROVISIONAL: the ``[precursor]`` setup table (``iterations``,
``converge_to``), ``precursor.cfg``, ``run_mrf.sh`` and the exit statuses.
"""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: The master configuration of stage 1; its stem names stage 1's history (``precursor.csv``).
PRECURSOR_CFG = "precursor.cfg"
#: The stage-1 zone files and the job script that runs both stages.
ZONE_FILE = "pre_zone_{}.cfg"
PRECURSOR_SCRIPT = "run_mrf.sh"
PRECURSOR_LOG = "precursor.log"
PRECURSOR_HISTORY = "precursor.csv"
#: Stage 1's own file names (SU2 adds ``_<zone>`` to a restart name) and stage 2's input.
RESTART_NAME = "precursor_restart.dat"
SOLUTION_NAME = "solution.dat"
#: Exit statuses of a point's job, apart from 97 and 98 of an in-job RESTART source.
EXIT_FAILED = 91
EXIT_DIVERGED = 92
EXIT_NO_RESTART = 93
EXIT_STAGING = 94
#: Stage-1 iterations when the setup has no ``[precursor]`` table.
DEFAULT_ITERATIONS = 300
#: The keys of the ``[precursor]`` setup table.
TABLE_KEYS = ("iterations", "converge_to")
#: Where a NaN in a history is, as the job script and :func:`check` look for it.
NAN = r"(^|[^a-z])-?nan([^a-z]|$)"
#: The line SU2 8.6.0 prints when a steady run stops at its iteration cap.
CAP_LINE = "before convergence"

#: Keys only a time-accurate or restarted run reads, or that name stage-2 files.
_DROPPED = frozenset(
    {
        "TIME_MARCHING",
        "TIME_STEP",
        "TIME_ITER",
        "MAX_TIME",
        "WINDOW_START_ITER",
        "WINDOW_FUNCTION",
        "UNST_CFL_NUMBER",
        "RESTART_SOL",
        "RESTART_ITER",
        "SOLUTION_FILENAME",
        "READ_BINARY_RESTART",
        "OUTPUT_WRT_FREQ",
        "OUTPUT_FILES",
        "OUTER_ITER",
        "INNER_ITER",
        "ITER",
        "CONFIG_LIST",
        "CONV_FIELD",
        "CONV_RESIDUAL_MINVAL",
        "CONV_STARTITER",
        "CONV_CAUCHY_ELEMS",
        "CONV_CAUCHY_EPS",
        "RESTART_FILENAME",
        "VOLUME_FILENAME",
        "SURFACE_FILENAME",
        "WRT_RESTART_OVERWRITE",
        "WRT_SURFACE_OVERWRITE",
        "WRT_VOLUME_OVERWRITE",
    }
)


@dataclass(frozen=True)
class Settings:
    """The ``[precursor]`` table: stage 1's outer iterations and its optional stop level."""

    iterations: int = DEFAULT_ITERATIONS
    converge_to: float | None = None


def settings(setup: Mapping[str, object]) -> Settings:
    """Read the setup's ``[precursor]`` table; defaults apply where it states nothing.

    Raises
    ------
    ValueError
        The table is not a table, has an unknown key, ``iterations`` is not a whole number of
        at least 1, or ``converge_to`` is not a finite number.
    """
    table = setup.get("precursor")
    if table is None:
        return Settings()
    if not isinstance(table, Mapping):
        raise ValueError("precursor must be a table: iterations and converge_to")
    unknown = sorted(set(table) - set(TABLE_KEYS))
    if unknown:
        raise ValueError(
            f"precursor has unknown keys: {', '.join(unknown)} ({', '.join(TABLE_KEYS)})"
        )
    iterations = table.get("iterations", DEFAULT_ITERATIONS)
    if type(iterations) is not int or iterations < 1:
        raise ValueError(
            "precursor.iterations must be a whole number of at least 1 (a cold start is "
            "the RESTART cell -, not zero iterations)"
        )
    converge = table.get("converge_to")
    if converge is not None and (
        isinstance(converge, bool)
        or not isinstance(converge, (int, float))
        or not math.isfinite(converge)
    ):
        raise ValueError("precursor.converge_to must be a finite residual level such as -9")
    return Settings(iterations, None if converge is None else float(converge))


def stage_one(
    values: Mapping[str, object], setup: Mapping[str, object], turning: int
) -> dict[str, object]:
    """Return the keys of stage 1 as files: ``precursor.cfg`` and one ``pre_zone_<i>.cfg`` a zone.

    Each key is written ``<file>/<KEY>``, so :func:`~cfdbender.solvers.su2._config.extra_files`
    renders the file whole. The stage is the point's own keys without those only a
    time-accurate run reads, ``TIME_DOMAIN= NO``, ``OUTER_ITER`` the precursor iterations,
    ``INNER_ITER= 1``, the turning zone in ``ROTATING_FRAME`` with the rotation of stage 2 and
    the other zones as they are.

    Raises
    ------
    ValueError
        ``KIND_INTERPOLATION`` is ``WEIGHTED_AVERAGE`` (a steady run with it diverges at once on
        SU2 8.6.0), or the restart format is ASCII (the stage-2 files are read as binary).
    """
    from cfdbender.solvers.su2._config import staged_mesh

    chosen = settings(setup)
    interface = str(values.get("KIND_INTERPOLATION", "")).strip().upper()
    if interface == "WEIGHTED_AVERAGE":
        raise ValueError(
            "RESTART MRF: KIND_INTERPOLATION WEIGHTED_AVERAGE diverges in a steady run; "
            "state NEAREST_NEIGHBOR or ISOPARAMETRIC for the sliding interface"
        )
    for key in ("WRT_BINARY_RESTART", "READ_BINARY_RESTART"):
        if values.get(key) is False or str(values.get(key, "")).strip().upper() == "NO":
            raise ValueError(
                f"RESTART MRF stages binary restart files: remove {key}= NO from the setup"
            )
    zones = [str(name) for name in _list(values["CONFIG_LIST"])]
    stage: dict[str, object] = {
        key: value
        for key, value in values.items()
        if "/" not in key and key not in _DROPPED and not key.startswith("HISTORY_WRT_FREQ")
    }
    stage["MESH_FILENAME"] = staged_mesh(values)
    stage["TIME_DOMAIN"] = False
    stage["CONFIG_LIST"] = [ZONE_FILE.format(i) for i in range(len(zones))]
    stage["OUTER_ITER"] = chosen.iterations
    stage["INNER_ITER"] = 1
    stage["OUTPUT_FILES"] = ["RESTART"]
    stage["OUTPUT_WRT_FREQ"] = chosen.iterations
    stage["RESTART_FILENAME"] = RESTART_NAME
    stage["VOLUME_FILENAME"] = "precursor_flow"
    stage["SURFACE_FILENAME"] = "precursor_surface"
    if chosen.converge_to is not None:
        stage["CONV_FIELD"] = f"RMS_DENSITY[{turning}]"
        stage["CONV_RESIDUAL_MINVAL"] = chosen.converge_to
    if "HISTORY_OUTPUT" in stage:
        stage["HISTORY_OUTPUT"] = [
            item for item in _list(stage["HISTORY_OUTPUT"]) if str(item) != "TAVG_AERO_COEFF"
        ]
    found: dict[str, object] = {f"{PRECURSOR_CFG}/{key}": value for key, value in stage.items()}
    for index, name in enumerate(zones):
        target = ZONE_FILE.format(index)
        for key, value in values.items():
            if key.startswith(f"{name}/"):
                native = key.partition("/")[2]
                if native == "GRID_MOVEMENT" and index == turning:
                    value = "ROTATING_FRAME"
                found[f"{target}/{native}"] = value
    return found


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


@dataclass(frozen=True)
class Stages:
    """What a point's job does, read from its SU2 keys: zones, restart levels, segments."""

    zones: int
    levels: int
    converge: bool
    segments: tuple[str, ...]


def stages(values: Mapping[str, object]) -> Stages | None:
    """Return the stages of a point with a precursor, or None for a point without."""
    if not any(key.startswith(f"{PRECURSOR_CFG}/") for key in values):
        return None
    from cfdbender.solvers.su2._run_types import SEGMENT_TWO, restart_order

    second = any(key.startswith(f"{SEGMENT_TWO}/") for key in values)
    return Stages(
        len(_list(values["CONFIG_LIST"])),
        restart_order(values),
        f"{PRECURSOR_CFG}/CONV_RESIDUAL_MINVAL" in values,
        ("su2.cfg", SEGMENT_TWO) if second else ("su2.cfg",),
    )


def staged_names(zones: int, levels: int) -> list[tuple[str, str]]:
    """Return ``(stage 1 restart file, staged name)`` for each zone and each restart level."""
    stem = Path(RESTART_NAME).stem
    solution = Path(SOLUTION_NAME)
    return [
        (f"{stem}_{zone}.dat", f"{solution.stem}_{zone}_{level:05d}{solution.suffix}")
        for zone in range(zones)
        for level in range(levels)
    ]


def _sh(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def failure(what: str) -> str:
    """Return the one line a failed stage 1 leaves in ``stderr.log``."""
    return f"PRECURSOR_FAILED: {what}; the time-accurate stage was not run"


#: What each failure says (the script and the local run use the same words).
NAN_FOUND = f"{PRECURSOR_HISTORY} holds NaN"
CAP_WARNING = (
    "PRECURSOR_WARNING: the steady stage reached its iteration cap "
    "before the residual level precursor.converge_to"
)


def _say(text: str, *, error: bool = True) -> str:
    """Return the shell that prints ``text`` (double quotes, so ``$rc`` expands) on a stream."""
    return f"printf '%s\\n' \"{text}\"" + (" >&2" if error else "")


def script_text(spec: Stages) -> str:
    """Return ``run_mrf.sh``: stage 1, its three checks, the staging, then the stage-2 file(s).

    POSIX shell and ``cp``, ``grep`` only: no Python on the node. The command it is given (a
    launcher and the solver, or a profile's application wrapper) is called once per stage with
    the stage's configuration file as its LAST argument. A stage-1 failure ends the script with
    status 91 (the command failed, or did not run the file it was given), 92 (NaN in
    ``precursor.csv``), 93 (a restart file missing or empty) or 94 (a copy failed), with one
    line on standard error; stage 2 is then never started. Stage 2's own status is the
    script's.
    """
    ran = f"the steady stage ended with status $rc (see {PRECURSOR_LOG})"
    hint = "; the command must run the configuration file it is given as its last argument"
    lines = [
        "#!/bin/sh",
        "# Run a point that starts from a steady frozen-rotor solution of its own mesh.",
        "# Usage: sh run_mrf.sh [launcher and solver command, default SU2_CFD]",
        "# The command is called once per stage, the stage's configuration file as its LAST",
        "# argument: precursor.cfg, then su2.cfg"
        + (", then su2_segment2.cfg." if len(spec.segments) > 1 else "."),
        "# Status 91: stage 1 failed (or the command ignored the file it was given); 92: NaN in",
        "# precursor.csv; 93: a stage-1 restart file is missing; 94: staging failed.",
        'if [ "$#" -eq 0 ]; then set -- SU2_CFD; fi',
        f'"$@" {PRECURSOR_CFG} > {PRECURSOR_LOG} 2>&1',
        "rc=$?",
        'if [ "$rc" -ne 0 ]; then',
        f"  {_say(failure(ran + hint))}",
        f"  exit {EXIT_FAILED}",
        "fi",
        f"if grep -qiE {_sh(NAN)} {PRECURSOR_HISTORY} 2> /dev/null; then",
        f"  {_say(failure(NAN_FOUND))}",
        f"  exit {EXIT_DIVERGED}",
        "fi",
    ]
    for source, _ in staged_names(spec.zones, 1):
        lines += [
            f"if [ ! -s {source} ]; then",
            f"  {_say(failure(f'the steady stage wrote no {source}'))}",
            f"  exit {EXIT_NO_RESTART}",
            "fi",
        ]
    for source, name in staged_names(spec.zones, spec.levels):
        lines += [
            f"cp {source} {name} || {{",
            f"  {_say(failure(f'could not stage {name}'))}",
            f"  exit {EXIT_STAGING}",
            "}",
        ]
    if spec.converge:
        lines += [
            f"if grep -q {_sh(CAP_LINE)} {PRECURSOR_LOG}; then",
            f"  {_say(CAP_WARNING, error=False)}",
            "fi",
        ]
    for index, segment in enumerate(spec.segments):
        lines += [f'"$@" {segment}', "rc=$?"]
        if index + 1 < len(spec.segments):
            lines += ['[ "$rc" -eq 0 ] || exit "$rc"']
    lines += ['exit "$rc"']
    return chr(10).join(lines) + chr(10)


def after_precursor(folder: Path, spec: Stages) -> tuple[int, str] | None:
    """Check stage 1's output as ``run_mrf.sh`` does and stage it; None when stage 2 may start.

    For a local run, which has no shell script: the same three checks and the same copies,
    returning the exit status and the one line for ``stderr.log`` of the first that fails.
    """
    history = folder / PRECURSOR_HISTORY
    if history.is_file() and re.search(NAN, history.read_text(errors="replace"), re.I | re.M):
        return EXIT_DIVERGED, failure(NAN_FOUND)
    for source, _ in staged_names(spec.zones, 1):
        path = folder / source
        if not path.is_file() or path.stat().st_size == 0:
            return EXIT_NO_RESTART, failure(f"the steady stage wrote no {source}")
    for source, name in staged_names(spec.zones, spec.levels):
        try:
            shutil.copyfile(folder / source, folder / name)
        except OSError:
            return EXIT_STAGING, failure(f"could not stage {name}")
    return None


def capped(folder: Path) -> bool:
    """Whether stage 1's log says it stopped at its iteration cap."""
    log = folder / PRECURSOR_LOG
    return log.is_file() and CAP_LINE in log.read_text(errors="replace")

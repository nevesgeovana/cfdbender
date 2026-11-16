"""A loop of parameter sets over the geometry pipeline: one geometry and one report per set.

Each set runs :func:`cfdbender.geometry.pipeline.run_geometry` in a worker process of its own,
since the geometry kernel can fault natively, and writes into its own folder under ``out``: the
set's ``name`` when it states one (the matrix row it feeds, for example), else ``set-0001``,
``set-0002`` and so on. A failed or refused set never stops the loop; a set whose worker dies is
reported ``failed`` at stage ``worker`` with a report and a log left in its folder.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cfdbender._errors import InputValueError
from cfdbender.geometry._run_log import LOG_NAME, RunLog
from cfdbender.geometry.pipeline import _now, _package_version, run_geometry
from cfdbender.geometry.stage_table import stage_index
from cfdbender.geometry.stages import geometry_report, write_record
from cfdbender.mesh._gmsh_worker import run_isolated

__all__ = ["run_doe", "run_set", "set_names"]

#: A set name is one folder name: letters, digits, ``.``, ``_`` and ``-``.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def run_set(
    config: Any,
    out: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    configuration: str | None = None,
    variant: str = "clean",
    until: str = "S13",
) -> dict[str, Any]:
    """Run one parameter set and return its report.

    Parameters
    ----------
    config : Mapping or str or Path
        A configuration, or the path of a TOML file.
    out : str or Path
        The report file of the set; the records are written beside it.
    overrides : Mapping, optional
        The values of the set, by dotted path.
    configuration : str, optional
        The configuration to build.
    variant : str, optional
        The variant to build; ``"clean"`` by default.
    until : str, optional
        The last stage to run; ``"S13"`` by default.

    Returns
    -------
    dict
        The report, as :func:`cfdbender.geometry.stages.geometry_report`.
    """
    return run_geometry(
        config,
        out=Path(out).parent,
        configuration=configuration,
        variant=variant,
        overrides=overrides,
        until=until,
    )


def set_names(parameter_sets: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the folder name of each set: its ``name`` key, else ``set-<n>`` from 0001.

    Parameters
    ----------
    parameter_sets : sequence of Mapping
        The sets; a ``name`` key is the set's name, never an override.

    Returns
    -------
    list of str
        One folder name per set, in order.

    Raises
    ------
    InputValueError
        If a name is not one folder name or two sets share a name.
    """
    names = [str(row.get("name", f"set-{i + 1:04d}")) for i, row in enumerate(parameter_sets)]
    for name in names:
        if not _NAME.match(name):
            raise InputValueError(
                f"[[sets]] name {name!r}: expected one folder name of letters, digits, '.', '_' "
                "and '-'"
            )
    seen = sorted({n for n in names if names.count(n) > 1})
    if seen:
        raise InputValueError(f"[[sets]] name: {seen} given to more than one set")
    return names


def _one_set(job: tuple[str, Mapping[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Run one parameter set in a worker; ``job`` is its folder name, overrides and arguments."""
    name, params, common = job
    arguments = {
        **common,
        "out": str(Path(common["out"]) / name / "report.json"),
        "overrides": {k: v for k, v in params.items() if k != "name"},
    }
    # A report or a log left by an earlier run in this set folder would be read as this run's
    # when the worker dies before writing its own: neither survives the start of the set.
    for stale in ("report.json", LOG_NAME):
        (Path(arguments["out"]).parent / stale).unlink(missing_ok=True)
    try:
        return run_isolated(f"{__name__}:run_set", arguments, run_set)
    except Exception as error:  # a worker fault must end this set only
        report: dict[str, Any] = geometry_report(
            "failed",
            None,
            stage="worker",
            check="worker.completed",
            message=f"{type(error).__name__}: {error}",
        )
        return _record_worker_fault(Path(arguments["out"]).parent, arguments, report)


def _record_worker_fault(
    folder: Path, arguments: Mapping[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    """Leave a report and a log in the folder of a set whose worker died; return the report."""
    clock = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    log = RunLog(folder)
    if not log.path.exists():
        config = arguments["config"]
        log.start(
            config=config if isinstance(config, str) else "(mapping given in memory)",
            package_version=_package_version(),
            configuration=arguments["configuration"],
            variant=arguments["variant"],
            until=arguments["until"],
            started=_now(),
            overrides=arguments["overrides"],
        )
    if not (folder / "report.json").exists():
        write_record(folder / "report.json", report)
    log.finish(report, [], time.perf_counter() - clock)
    return report


def run_doe(
    config: Any,
    parameter_sets: Sequence[Mapping[str, Any]],
    *,
    out: str | Path,
    configuration: str | None = None,
    variant: str = "clean",
    until: str = "S13",
    workers: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield one report per parameter set, in order; a failed or refused set never stops the loop.

    Parameters
    ----------
    config : Mapping or str or Path
        A configuration, or the path of a TOML file.
    parameter_sets : sequence of Mapping
        One table of overrides per set, by dotted path; a ``name`` key names the set's folder.
    out : str or Path
        The folder that receives one sub-folder per set.
    configuration : str, optional
        The configuration to build, for a file that holds several.
    variant : str, optional
        The variant to build; ``"clean"`` by default.
    until : str, optional
        The last stage to run; ``"S13"`` by default.
    workers : int, optional
        The number of sets run at once; 1 runs them in order.

    Yields
    ------
    dict
        The report of each set, as :func:`cfdbender.geometry.stages.geometry_report`.

    Raises
    ------
    InputValueError
        If ``workers`` is below 1, ``until`` is not a stage id, or a set name is unusable.

    Examples
    --------
    >>> from cfdbender.geometry.doe import run_doe
    >>> next(run_doe({}, [], out="doe", workers=0))
    Traceback (most recent call last):
    ...
    cfdbender._errors.InputValueError: workers: at least 1 required
    """
    if workers < 1:
        raise InputValueError("workers: at least 1 required")
    stage_index(until)
    names = set_names(parameter_sets)
    common = {
        "config": config if isinstance(config, Mapping) else str(config),
        "out": str(out),
        "configuration": configuration,
        "variant": variant,
        "until": until,
    }
    jobs = [(name, params, common) for name, params in zip(names, parameter_sets, strict=True)]
    if workers == 1:
        for job in jobs:
            yield _one_set(job)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_one_set, jobs)

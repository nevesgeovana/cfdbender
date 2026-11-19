"""Read SU2's recorded restart output and select a point-local solution input."""

import re
from collections.abc import Mapping
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.restart import require_solution, source_folder


def _cfg_value(text: str, key: str) -> str | None:
    """Return the last value a configuration states for ``key``, or None."""
    found = None
    for line in text.splitlines():
        name, _, value = line.partition("=")
        if name.strip() == key:
            found = value
    return None if found is None else found.split("%")[0].strip().strip('"')


def predicted_restart_name(point: AdapterPoint) -> str:
    """Name the restart file a source of the same batch job will write, from its rendering.

    The name and format follow from the source's rendered configuration alone when it is a
    steady point, ``OUTPUT_FILES`` (when stated) holds ``RESTART``, and ``RESTART_FILENAME``
    (default ``restart_flow.dat``) ends in ``.dat`` or has no suffix. Anything that leaves
    the format to run time is refused naming the key to set.
    """
    text = point.restart_source_cfg
    source = point.point.restart_from
    if text is None:
        raise ValueError(f"RESTART {source}: the rendered source configuration is missing")
    if (_cfg_value(text, "TIME_DOMAIN") or "NO").upper() == "YES":
        raise ValueError(
            f"RESTART {source}: a source in the same job must be a steady point; run a "
            "time-accurate source first and name it once completed"
        )
    outputs = _cfg_value(text, "OUTPUT_FILES")
    if outputs is not None and "RESTART" not in [
        word.strip() for word in outputs.strip("()").split(",")
    ]:
        raise ValueError(
            f"RESTART {source}: its restart format is left to run time; "
            "add RESTART to the OUTPUT_FILES of its setup"
        )
    if (_cfg_value(text, "WRT_BINARY_RESTART") or "YES").upper() == "NO":
        raise ValueError(
            f"RESTART {source}: its restart format is left to run time; "
            "set WRT_BINARY_RESTART to YES in its setup"
        )
    name = _cfg_value(text, "RESTART_FILENAME") or "restart_flow.dat"
    if not Path(name).suffix:
        name += ".dat"
    if Path(name).suffix != ".dat" or Path(name).name != name:
        raise ValueError(
            f"RESTART {source}: its restart format is left to run time; set "
            "RESTART_FILENAME in its setup to a plain file name ending in .dat"
        )
    return name


def restart_files(point: AdapterPoint) -> dict[str, Path]:
    """Select the configured RESTART_FILENAME, defaulting to restart_flow.dat."""
    folder = source_folder(point)
    if point.point.restart_in_job:
        return {"restart-input.dat": folder / predicted_restart_name(point)}
    name = "restart_flow.dat"
    config = folder / "su2.cfg"
    if config.is_file():
        for line in config.read_text(encoding="ascii").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "RESTART_FILENAME":
                name = value.split("%")[0].strip().strip('"')
    path = folder / name
    if not path.is_file() and not Path(name).suffix:
        path = folder / (name + ".dat")
    if (
        not path.is_file()
        and name == "restart_flow.dat"
        and (folder / "restart_flow.csv").is_file()
    ):
        path = folder / "restart_flow.csv"
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError(
            f"RESTART {point.point.restart_from}: solution must be inside its point folder"
        )
    files = {"restart-input" + path.suffix: path}
    require_solution(point, files)
    return files


def staged_solution(files: dict[str, Path], values: Mapping[str, object]) -> dict[str, Path]:
    """Name the copied solution as SU2 reads it.

    A steady point reads ``SOLUTION_FILENAME`` itself. A time-accurate point started from that
    solution restarts at ``RESTART_ITER`` and SU2 8.6.0 reads ``<stem>_<k:05d><ext>`` for the
    previous ``restart_order`` iterations, so the one solution is staged under each of those
    names (for second-order dual time, ``restart-input_00000.dat`` and ``_00001.dat``) and
    not under its own name, which SU2 would not read.
    """
    if not files or values.get("TIME_DOMAIN") is not True or "RESTART_ITER" not in values:
        return files
    from cfdbender.solvers.su2._run_types import restart_order

    first = int(str(values["RESTART_ITER"]))
    staged: dict[str, Path] = {}
    for name, path in files.items():
        stem, suffix = Path(name).stem, Path(name).suffix
        for k in range(first - restart_order(values), first):
            staged[f"{stem}_{k:05d}{suffix}"] = path
    return staged


def restart_values(point: AdapterPoint, values: dict[str, object]) -> dict[str, object]:
    """Set the SU2 restart keys to the copied solution, overriding a cold-start setting."""
    if point.point.restart_from is not None:
        solution = next(iter(restart_files(point)))
        saved = (
            point.restart_source_cfg
            if point.restart_source_cfg is not None
            else (source_folder(point) / "su2.cfg").read_text(encoding="ascii")
        )
        binary = (
            not solution.endswith(".csv")
            and re.search(r"^\s*WRT_BINARY_RESTART\s*=\s*NO\b", saved, re.MULTILINE | re.IGNORECASE)
            is None
        )
        values = dict(
            values, RESTART_SOL=True, SOLUTION_FILENAME=solution, READ_BINARY_RESTART=binary
        )
    return values

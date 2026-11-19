"""Read SU2's recorded restart output and select a point-local solution input."""

import re
from collections.abc import Mapping
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.restart import require_solution, source_folder


def restart_files(point: AdapterPoint) -> dict[str, Path]:
    """Select the configured RESTART_FILENAME, defaulting to restart_flow.dat."""
    folder = source_folder(point)
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
        saved = (source_folder(point) / "su2.cfg").read_text(encoding="ascii")
        binary = (
            not solution.endswith(".csv")
            and re.search(r"^\s*WRT_BINARY_RESTART\s*=\s*NO\b", saved, re.MULTILINE | re.IGNORECASE)
            is None
        )
        values = dict(
            values, RESTART_SOL=True, SOLUTION_FILENAME=solution, READ_BINARY_RESTART=binary
        )
    return values

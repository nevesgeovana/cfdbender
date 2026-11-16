"""Select a reconstructed OpenFOAM solution at the latest numeric time."""

import math
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.restart import require_solution, source_folder


def restart_files(point: AdapterPoint) -> dict[str, Path]:
    """Select every field in the latest positive time directory, requiring U and p."""
    folder = source_folder(point)
    times = []
    if folder.is_dir():
        for path in folder.iterdir():
            try:
                time = float(path.name)
            except ValueError:
                continue
            if path.is_dir() and math.isfinite(time) and time > 0:
                times.append((time, path))
    latest = max(times, key=lambda item: item[0])[1] if times else folder / "missing-solution"
    files = {
        path.relative_to(folder).as_posix(): path for path in latest.rglob("*") if path.is_file()
    }
    require_solution(point, {"U": latest / "U", "p": latest / "p"})
    require_solution(point, files)
    return files

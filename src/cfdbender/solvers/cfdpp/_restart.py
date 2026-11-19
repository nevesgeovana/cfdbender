"""Name the saved solution a CFD++ restart starts from."""

from __future__ import annotations

from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.restart import require_solution, source_folder

#: The solution file a run leaves behind; a restart point receives a copy under this name
#: and the input states that the run starts from it. This is an unverified layout assumption.
SOLUTION = "cdepsout.bin"


def restart_files(point: AdapterPoint) -> dict[str, Path]:
    """Return the source point's saved solution, refusing a missing or empty one."""
    files = {SOLUTION: source_folder(point) / SOLUTION}
    require_solution(point, files)
    return files

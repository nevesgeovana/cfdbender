"""Select the most recently written PyFR solution file of a recorded point."""

import configparser
import importlib
import math
import re
from pathlib import Path

from cfdbender.core.adapter import AdapterPoint
from cfdbender.core.restart import require_solution, source_folder


def restart_files(point: AdapterPoint) -> dict[str, Path]:
    """Select the latest .pyfrs file, excluding the input copied for an earlier restart."""
    folder = source_folder(point)
    files = [path for path in folder.glob("*.pyfrs") if path.name != "restart.pyfrs"]
    source = (
        max(files, key=lambda path: (path.stat().st_mtime_ns, path.name))
        if files
        else folder / "missing.pyfrs"
    )
    result = {"restart.pyfrs": source, "mesh.pyfrm": folder / "mesh.pyfrm"}
    mesh = str(point.setup.get("mesh") or point.geometry or "")
    expected = point.point.inputs_sha256.get(mesh)
    original_mesh = folder / "mesh.msh"
    if (
        expected
        and original_mesh.is_file()
        and point.inputs().digest_via(mesh, original_mesh) != expected
    ):
        raise ValueError(
            f"RESTART {point.point.restart_from}: mesh differs from the saved solution"
        )
    require_solution(point, result)
    return result


def source_time(point: AdapterPoint) -> float:
    """Read the writer's saved time from its standard filename or native HDF5 stats."""
    path = restart_files(point)["restart.pyfrs"]
    match = re.fullmatch(r"solution-([0-9.eE+-]+)\.pyfrs", path.name)
    if match:
        time = float(match[1])
    else:
        try:
            h5py = importlib.import_module("h5py")
            with h5py.File(path, "r") as data:
                raw = data["stats"][()]
            stats = configparser.ConfigParser()
            stats.read_string(raw.decode() if isinstance(raw, bytes) else str(raw))
            time = stats.getfloat("solver-time-integrator", "tcurr")
        except (ImportError, OSError, KeyError, ValueError, configparser.Error) as error:
            raise ValueError(
                f"RESTART {point.point.restart_from}: cannot read PyFR solution time"
            ) from error
    if not math.isfinite(time) or time < 0:
        raise ValueError(f"RESTART {point.point.restart_from}: invalid PyFR solution time")
    return time

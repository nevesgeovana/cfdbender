"""Offline SU2_CFD 8.6.0 stand-in for time-accurate runs; never makes physical predictions.

It models only the file behaviour measured on a real SU2 8.6.0 run, so the adapter's staging and
reading are tested against what SU2 does:

* a transient run refuses ``WRT_RESTART_OVERWRITE``, ``WRT_SURFACE_OVERWRITE`` and
  ``WRT_VOLUME_OVERWRITE`` set to ``NO``;
* a restart at ``RESTART_ITER = R`` reads ``<SOLUTION_FILENAME stem>_<k:05d><ext>`` for
  ``k = R - 2, R - 1`` (second-order dual time) or ``k = R - 1`` (otherwise), and refuses a
  negative ``k`` or a missing file;
* the history of a restarted run is ``history_<R:05d>.csv``, of a cold start ``history.csv``;
* each file of ``OUTPUT_FILES`` is written at time iteration ``k`` when ``k % frequency == 0`` or
  ``k`` is the last, and under second-order dual time a restart file also one iteration
  before each multiple and before the last (SU2's CFlowOutput, measured); a restart refuses
  ``WINDOW_START_ITER`` below ``RESTART_ITER`` (measured); the files are restart
  ``<RESTART_FILENAME stem>_<k:05d>.dat``, volume ``flow_<k:05d>.vtu`` and surface
  ``surface_flow_<k:05d>.vtu``.
"""

import csv
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="ascii").splitlines()
cfg = {key.strip(): value.strip() for key, value in (line.split("=", 1) for line in lines)}
print('SU2 Release 8.6.0 "Harrier"', flush=True)
print("Three dimensional problem.", flush=True)


def listed(key, default):
    text = cfg.get(key)
    if text is None:
        return list(default)
    return [item.strip() for item in text.strip("() ").split(",") if item.strip()]


def stem(name, default):
    return re.sub(r"[.](dat|csv)$", "", cfg.get(name, default))


def stop(message):
    print("Error in fake SU2: " + message, flush=True)
    sys.exit(1)


if cfg.get("TIME_DOMAIN") != "YES":
    with Path("history.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Inner_Iter", "rms[Rho]", "CL", "CD", "CSF", "CFx", "CFy", "CFz"])
        for i in range(int(cfg["ITER"])):
            writer.writerow([i, -2 - i * 0.1, 0.3, 0.02, 0.01, 0.025, 0.0, 0.3])
    Path(stem("RESTART_FILENAME", "restart") + ".dat").write_bytes(b"steady solution")
    print("Maximum number of iterations reached.", flush=True)
    sys.exit(0)

for key in ("WRT_RESTART_OVERWRITE", "WRT_SURFACE_OVERWRITE", "WRT_VOLUME_OVERWRITE"):
    if cfg.get(key) == "NO":
        stop(f"Appending iterations to the filename ({key}=NO) is incompatible with transient")
total = int(cfg["TIME_ITER"])
inner = int(cfg.get("INNER_ITER", "1"))
second = cfg.get("TIME_MARCHING", "") == "DUAL_TIME_STEPPING-2ND_ORDER"
restart = cfg.get("RESTART_SOL") == "YES"
start = int(cfg.get("RESTART_ITER", "0")) if restart else 0
if restart and int(cfg.get("WINDOW_START_ITER", start)) < start:
    stop("WINDOW_START_ITER must be larger than or equal to: RESTART_ITER!")
if restart:
    solution = cfg.get("SOLUTION_FILENAME", "solution")
    extension = ".csv" if solution.endswith(".csv") else ".dat"
    base = re.sub(r"[.](dat|csv)$", "", solution)
    for k in range(start - (2 if second else 1), start):
        if k < 0:
            stop("Requesting a negative iteration number for the restart file!!")
        name = f"{base}_{k:05d}{extension}"
        if not Path(name).is_file():
            stop(f"Unable to open SU2 restart file {name}")
        print(f"Read flow solution from: {name}.", flush=True)
history = f"history_{start:05d}.csv" if restart else "history.csv"
markers = (
    listed("MARKER_MONITORING", []) if "AERO_COEFF_SURF" in cfg.get("HISTORY_OUTPUT", "") else []
)
loads = ["CD", "CL", "CSF", "CMx", "CMy", "CMz", "CFx", "CFy", "CFz"]
header = ["Time_Iter", "Outer_Iter", "Inner_Iter", "rms[Rho]", *loads]
header += [f"{load}({marker})" for marker in markers for load in loads]
header += [f"tavg[{load}]" for load in loads]
files = listed("OUTPUT_FILES", ["RESTART", "PARAVIEW", "SURFACE_PARAVIEW"])
frequencies = [int(f) for f in listed("OUTPUT_WRT_FREQ", ["1"] * len(files))]
names = {
    "RESTART": stem("RESTART_FILENAME", "restart") + "_{k:05d}.dat",
    "PARAVIEW": stem("VOLUME_FILENAME", "flow") + "_{k:05d}.vtu",
    "SURFACE_PARAVIEW": stem("SURFACE_FILENAME", "surface_flow") + "_{k:05d}.vtu",
}
with Path(history).open("w", newline="", encoding="ascii") as stream:
    writer = csv.writer(stream)
    writer.writerow(header)
    for k in range(start, total):
        for i in range(inner):
            value = 0.01 * (k % 36) + 0.001 * i
            row = [k, 0, i, -3.0, *([value] * len(loads))]
            row += [value] * (len(loads) * len(markers)) + [value] * len(loads)
            writer.writerow(row)
        for index, kind in enumerate(files):
            frequency = frequencies[min(index, len(frequencies) - 1)]
            # CFlowOutput::WriteVolumeOutput: under second-order dual time a restart file is
            # also written one iteration before each multiple and before the last.
            pair = second and kind == "RESTART" and ((k + 1) % frequency == 0 or k + 2 == total)
            if k % frequency == 0 or k == total - 1 or pair:
                Path(names[kind].format(k=k)).write_bytes(f"{kind} {k}".encode("ascii"))
print("Exit Success (SU2_CFD)", flush=True)

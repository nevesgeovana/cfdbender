"""Offline SU2_CFD stand-in for a 2D alpha sweep; never makes physical predictions.

It states the dimension its mesh's NDIME line gives and writes history.csv with rms[Rho],
CL, CD and CMz made from the AOA of su2.cfg: CL = 0.1 AOA, CD = 0.01 + 0.0005 AOA^2,
CMz = 0.01 AOA, all settling within a few iterations. An AOA of 15 or more runs to ITER
with rms[Rho] falling only two orders and the forces still drifting, so it meets neither
R nor C of the polar convergence rule; any other AOA reaches rms[Rho] -12.5 (R) in 60
iterations.
"""

import csv
import sys
from pathlib import Path

cfg = dict(line.split("=", 1) for line in Path(sys.argv[1]).read_text().splitlines() if "=" in line)
cfg = {key.strip(): value.strip() for key, value in cfg.items()}
alpha = float(cfg["AOA"])
limit = int(cfg["ITER"])
stalled = alpha >= 15
print('SU2 Release 8.6.0 "Harrier"', flush=True)
planar = "NDIME= 2" in Path(cfg["MESH_FILENAME"]).read_text()
print(("Two" if planar else "Three") + " dimensional problem.", flush=True)
count = limit if stalled else 60
with Path("history.csv").open("w", newline="", encoding="ascii") as stream:
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow([" Inner_Iter ", " rms[Rho] ", " CL ", " CD ", " CMz "])
    for n in range(count):
        if stalled:
            rms = -4 - 2 * n / max(count - 1, 1)
            settle = 1 + 1e-3 * n
        else:
            rms = -4 - 8.5 * n / (count - 1)
            settle = 1 - 0.5 * 0.7**n
        writer.writerow(
            [n, rms, 0.1 * alpha * settle, (0.01 + 0.0005 * alpha**2) * settle, 0.01 * alpha]
        )
message = (
    "Maximum number of iterations reached." if stalled else "All convergence criteria satisfied."
)
print(message, flush=True)

"""Tiny offline SU2_CFD stand-in; never makes physical predictions."""

import csv
import os
import sys
import time
from pathlib import Path

cfg = dict(line.split("=", 1) for line in Path(sys.argv[1]).read_text().splitlines())
cfg = {key.strip(): value.strip() for key, value in cfg.items()}
case = cfg.get("FAKE_CASE", "converged")
limit = int(cfg["ITER"])
print('SU2 Release 8.6.0 "Harrier"', flush=True)
print("OMP_NUM_THREADS=" + os.environ.get("OMP_NUM_THREADS", "missing"), flush=True)
print("fake diagnostic channel", file=sys.stderr, flush=True)
if case == "timeout":
    time.sleep(10)
if case != "missing":
    with Path("history.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerow([" Inner_Iter ", " rms[Rho] ", " CL ", " CD ", " CSF ", " CFx ", " CMy "])
        count = 2 if case == "converged" else limit
        for i in range(count):
            writer.writerow(
                [i, -2 - i * 0.1, "nan" if case == "nan" else 0.3, 0.02, 0.01, 0.025, -0.05]
            )
message = (
    "All convergence criteria satisfied."
    if case == "converged"
    else ("Maximum number of iterations reached." if case == "iteration_limit" else "Exit Success")
)
Path("solver.log").write_text(message, encoding="ascii")
print(message, flush=True)
sys.exit(1 if case == "failed" else 0)

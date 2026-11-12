"""Offline serial wrapper double; produces evidence, never physical predictions."""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path


class CSinglezoneDriver:
    """Check the measured serial constructor and normal driver lifecycle."""

    def __init__(self, conf, zones, comm):
        assert zones == 1 and comm == 0
        self.cfg = dict(line.split("=", 1) for line in Path(conf).read_text().splitlines())
        self.cfg = {key.strip(): value.strip() for key, value in self.cfg.items()}
        self.started = False
        print('SU2 Release 8.6.0 "Harrier"', flush=True)
        print(f"worker pid={os.getpid()}", flush=True)
        print("OMP_NUM_THREADS=" + os.environ.get("OMP_NUM_THREADS", "missing"), flush=True)
        print("fake diagnostic channel", file=sys.stderr, flush=True)

    def StartSolver(self):  # noqa: N802 - match the measured binding
        """Write independently specified numerical and process outcomes."""
        self.started = True
        case = self.cfg.get("FAKE_CASE", "converged")
        if case == "timeout":
            if marker := self.cfg.get("FAKE_MARKER"):
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import time,pathlib; time.sleep(2); "
                        + f"pathlib.Path({marker!r}).write_text('survived')",
                    ]
                )
            time.sleep(10)
        if case != "missing":
            with Path("history.csv").open("w", newline="", encoding="ascii") as stream:
                writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
                writer.writerow(["Inner_Iter", "rms[Rho]", "CL", "CD"])
                count = 2 if case == "converged" else int(self.cfg["ITER"])
                for i in range(count):
                    writer.writerow([i, -2 - i * 0.1, 0.3, 0.02])
        message = (
            "All convergence criteria satisfied."
            if case in ("converged", "failed")
            else "Maximum number of iterations reached."
        )
        Path("solver.log").write_text(message, encoding="ascii")
        print(message, flush=True)
        if case == "failed":
            print("SU2 hard exit", file=sys.stderr, flush=True)
            os._exit(1)

    def Finalize(self):  # noqa: N802 - match the measured binding
        """Make the completed lifecycle observable independently of exit zero."""
        assert self.started
        Path("finalized.txt").write_text("StartSolver then Finalize", encoding="ascii")
        print("Exit Success", flush=True)

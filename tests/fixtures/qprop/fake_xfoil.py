"""Offline XFOIL stand-in for the QPROP adapter: writes an analytic section polar.

The polar depends on the Reynolds number so every station gets different data:
CL = CL0 + 0.11 alpha with CL0 = 0.05 + 1e-7 Re, and CD = 0.01 + 0.02 (CL - CLCD0)^2
with CLCD0 = CL0 + 0.22, the lift at alpha = 2 degrees.

CFDBENDER_QPROP_FAKE_XFOIL lists failures as name=fail pairs, where name is the
folder the process runs in (sec_02, or sec_02_narrow for the retry).
"""

import os
import sys
from pathlib import Path


def main():
    data = sys.stdin.buffer.read()
    assert data == Path("xfoil.in").read_bytes(), "stdin must be the saved keystroke script"
    lines = data.decode("ascii").splitlines()
    assert lines[:3] == ["PLOP", "G", ""]
    assert lines[3] == "LOAD airfoil.dat" and Path("airfoil.dat").is_file()
    reynolds = float(next(x[5:] for x in lines if x.startswith("VISC ")))
    assert next(x for x in lines if x.startswith("MACH ")) and "PACC" in lines
    sweep = next(x for x in lines if x.startswith("ASEQ ")).split()
    start, end, step = float(sweep[1]), float(sweep[2]), float(sweep[3])
    print(" XFOIL Version 6.99", flush=True)
    failing = dict(
        item.split("=")
        for item in os.environ.get("CFDBENDER_QPROP_FAKE_XFOIL", "").split(",")
        if item
    )
    header = [
        "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr",
        "  ------ -------- --------- --------- -------- -------- --------",
    ]
    rows = []
    if failing.get(Path.cwd().name) == "fail":
        print("VISCAL:  Convergence failed", flush=True)
    else:
        cl0 = 0.05 + 1e-7 * reynolds
        alpha = start
        while alpha <= end + 1e-9:
            cl = cl0 + 0.11 * alpha
            cd = 0.01 + 0.02 * (cl - (cl0 + 0.22)) ** 2
            rows.append(f" {alpha:7.3f} {cl:8.4f} {cd:9.5f}   0.00046  -0.0000   0.6870   0.6870")
            alpha += step
    Path("polar.txt").write_text("\n".join(header + rows) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    sys.exit(main())

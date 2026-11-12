"""Offline XFOIL process stand-in; polar formatting comes from a measured 6.99 file."""

import os
import sys
import time
from pathlib import Path


def main():
    """Check stdin and write one measured-format row, or diagnostic failure evidence."""
    data = sys.stdin.buffer.read()
    assert data == Path("xfoil.in").read_bytes(), "stdin must be the saved keystroke script"
    lines = data.decode("ascii").splitlines()
    assert lines[:3] == ["PLOP", "G", ""]
    assert lines[3].startswith(("NACA ", "LOAD "))
    if lines[3].startswith("LOAD "):
        assert Path(lines[3][5:]).is_file(), "airfoil must be staged in the point cwd"
    assert "PANE" in lines and "OPER" in lines
    assert "MACH 0" in lines and "ITER 200" in lines
    index = lines.index("PACC")
    assert lines[index + 1 : index + 3] == ["polar.txt", ""]
    assert lines[-3:] == ["", "", "QUIT"]
    alpha = float(next(line[5:] for line in lines if line.startswith("ALFA ")))
    print(" XFOIL Version 6.99", flush=True)
    print("stand-in diagnostic", file=sys.stderr, flush=True)
    mode = os.environ.get("CFDBENDER_XFOIL_FAKE_MODE", "converged")
    if mode == "timeout":
        time.sleep(30)
    template = Path(__file__).with_name("polar.txt").read_text(encoding="ascii").splitlines()
    dashed = next(i for i, line in enumerate(template) if line.startswith("  ------"))
    header = template[: dashed + 1]
    rows = template[dashed + 1 :]
    if mode == "failed":
        print("VISCAL:  Convergence failed", flush=True)
        rows = []
    elif mode == "exit_only":
        return 0
    elif mode == "wrong_alpha":
        rows = [rows[1] if alpha == 0 else rows[0]]
    else:
        rows = [line for line in rows if float(line[:8]) == alpha]
        assert len(rows) == 1, "fixture supports only its measured alpha values"
    Path("polar.txt").write_bytes((chr(10).join(header + rows) + chr(10)).encode("ascii"))
    return 7 if mode == "nonzero" else 0


if __name__ == "__main__":
    sys.exit(main())

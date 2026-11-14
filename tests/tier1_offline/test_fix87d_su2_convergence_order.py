"""Q3: the SU2 convergence message counts only when it follows the last iteration line.

A banner printed before later iterations belongs to an earlier state of the log; the solver
went on, so the last iteration is not the one that converged.
"""

from __future__ import annotations

import json

import pytest
import test_p0000_item3d_config as su2_case

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main

STANDIN = """
import csv, sys
from pathlib import Path
print('SU2 Release 8.6.0 "Harrier"')
print("Two dimensional problem.")
with Path("history.csv").open("w", newline="", encoding="ascii") as stream:
    writer = csv.writer(stream)
    writer.writerow(["Inner_Iter", "rms[Rho]", "CL", "CD", "CSF", "CMx", "CMy", "CMz"])
    for i in range(3):
        writer.writerow([i, -3 - i, 0.3, 0.02, 0.0, 0.0, 0.0, -0.02])
BANNER_FIRST
for i in range(3):
    print("|%9d|  -3.0|  0.3|" % i)
BANNER_LAST
"""


def _classification(root, banner_first, banner_last):
    su2_case.workspace(root)
    message = "print('All convergence criteria satisfied.')"
    script = root / "standin_su2.py"
    script.write_text(
        STANDIN.replace("BANNER_FIRST", message if banner_first else "pass").replace(
            "BANNER_LAST", message if banner_last else "pass"
        ),
        encoding="ascii",
    )
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(
        setup.read_text(encoding="ascii")
        + f'\n[solver]\nroute = "executable"\nexecutable = {json.dumps(script.as_posix())}\n',
        encoding="ascii",
    )
    assert main(["plan", "study", "--workspace", str(root)]) == 0
    main(["run", "study", "--local", "--workspace", str(root)])
    return RunRegistry(root).read()[0].classification


@pytest.mark.parametrize(
    "first,last,expected",
    [(False, True, "converged"), (True, False, "failed"), (True, True, "converged")],
)
def test_the_banner_must_follow_the_last_iteration_line(tmp_path, first, last, expected):
    got = _classification(tmp_path, first, last)
    if expected == "converged":
        assert got == "converged"
    else:
        assert got != "converged", "a banner before later iterations is not the last state"

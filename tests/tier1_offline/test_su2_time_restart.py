"""A time-accurate SU2 point started from a steady point, with and without an export window.

The SU2 stand-in (``fixtures/su2/fake_su2_transient.py``) models the file behaviour a real SU2
8.6.0 run showed: the two previous-iteration files a second-order restart reads, the
``history_<RESTART_ITER>.csv`` of a restarted run, the per-file output frequencies and the
transient refusal of the overwrite keys. Each test states what the generated files must be so
that the point runs as generated.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import test_runtypes_b1_workflows as rt

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.solvers import load_solver
from cfdbender.solvers.su2._products import product_inputs
from cfdbender.solvers.su2._time_history import time_history

FAKE = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2_transient.py"
HEADER = rt.HEADER + " | RESTART"
CONDITION = "MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:0.4"
CLOCK = "DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_REVS_AVG: 0.25"
WINDOW = CLOCK + " / EXPORT_UNSTEADY_LAST_ITER: 9"


def call(argv):
    try:
        return main(argv)
    except SystemExit as error:
        return error.code


def matrix(root, name, pol, workflow, var, restart="-"):
    row = (
        f" {pol} | 1 | rotor | {CONDITION} | - | rotor.su2 | 001 | 001 | 001 | - | - "
        f"| {workflow} | {var} | {restart} "
    )
    path = root / "inputs/matrices" / f"{name}.fs"
    path.write_text("\n".join([HEADER, "-" * 60, row]) + "\n", encoding="ascii")


def workspace(root, var, *, restart=True, setup=""):
    """A steady point run first, then the time-accurate row named ``ta``."""
    rt.workspace(
        root,
        setup='RESTART_FILENAME = "restart_flow.dat"\n'
        + setup
        + f'[solver]\nroute = "executable"\nexecutable = {json.dumps(FAKE.as_posix())}\n',
    )
    source = "-"
    if restart:
        matrix(root, "src", 7001, "qsteady_rotor", "-")
        assert call(["run", "src", "--local", "--workspace", str(root)]) == 0
        source = RunRegistry(root).read()[0].point_id
    matrix(root, "ta", 7002, "unsteady_rotor", var, source)
    return root


def run_ta(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert call(["run", "ta", "--local", "--workspace", str(root)]) == 0
    record = RunRegistry(root).read()[-1]
    return record, root / record.point.directory


def keys(path):
    return dict(line.split("= ", 1) for line in path.read_text(encoding="ascii").splitlines())


def collected(root, record):
    point = load_point(root, record.point, solver="su2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_solver("su2").collect(point).to_dict()


def test_a_steady_start_stages_the_two_previous_states_and_counts_the_rows_steps(tmp_path):
    record, folder = run_ta(workspace(tmp_path, CLOCK))
    assert record.classification != "failed", record.error
    cfg = keys(folder / "su2.cfg")
    # 36 steps of the row are SU2 time iterations 2 to 37; the steady solution is 0 and 1.
    assert (cfg["RESTART_SOL"], cfg["RESTART_ITER"], cfg["TIME_ITER"]) == ("YES", "2", "38")
    assert cfg["SOLUTION_FILENAME"] == "restart-input.dat"
    # The averaging window is steps 28 to 36, SU2 time iterations 29 to 37.
    assert cfg["WINDOW_START_ITER"] == "29"
    for name in ("restart-input_00000.dat", "restart-input_00001.dat"):
        assert (folder / name).read_bytes() == b"steady solution"
    assert not (folder / "restart-input.dat").exists()
    assert list(time_history(folder).steps) == list(range(1, 37))
    result = collected(tmp_path, record)
    average = result["time_average"]
    assert (average["FIRST_STEP"], average["LAST_STEP"], average["STEPS"]) == (28, 36, 9)
    assert [row["FIRST_STEP"] for row in average["per_revolution"]] == [1]
    assert result["history"][-1]["iteration"] == 36
    # Step 36, the last, is SU2's time iteration 37: the product reads that step's files.
    inputs = product_inputs(load_point(tmp_path, record.point, solver="su2"))
    assert [item.path.rsplit("/", 1)[-1] for item in inputs.surface] == ["surface_flow_00037.vtu"]
    assert [item.step for item in inputs.surface] == [36]

"""P2D: a 2D alpha sweep of SU2 or OpenFOAM, judged per point and written as a polar.

The SU2 cases run the offline stand-in fixtures/su2/fake_su2_polar.py through
cfd-matrix plan, run and collect: no solver is launched. Its AOA 15 point meets neither
criterion of the convergence rule; every other angle reaches R.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from test_p0600_polar2d_rule import openfoam_point, recorded_2007

from cfdbender.core.refusals import warning_refusals
from cfdbender.core.settings import WorkspaceError
from cfdbender.core.workspace import init_workspace
from cfdbender.matrix.cli import main
from cfdbender.matrix.collection import collect_workspace
from cfdbender.matrix.polar2d import judge_rows, write_polars
from cfdbender.matrix.post import post_workspace
from cfdbender.solvers import available_solvers
from cfdbender.solvers.qprop._polar import fit_polar, read_polar_rows
from cfdbender.solvers.xfoil._polar import read_polar

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FAKE = FIXTURES / "su2/fake_su2_polar.py"
HEADER = "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES|GEOMETRY"


def su2_workspace(
    root: Path,
    sweep: str = "4,-2,15,0,2",
    *,
    condition: str = "MACH:0.15, REmi:6, ALPHA:sweep",
    ndime: int = 2,
) -> Path:
    init_workspace(root, solver="su2", available_solvers=available_solvers())
    files = {
        "inputs/setups/s001.toml": 'iterations = 300\nSOLVER = "EULER"\n'
        f"[solver]\nexecutable = {json.dumps(FAKE.as_posix())}\n",
        "inputs/references/r001.toml": "area = 1.0\nlength = 1.0\nmoment_origin = [0.25, 0, 0]\n",
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/mesh/naca.su2": f"NDIME= {ndime}\n",
        "inputs/matrices/polar.fs": f"{HEADER}\n{'-' * 60}\n"
        f"1001|1|001|001|001|{condition}|{sweep}|naca.su2\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    return root


def run(root: Path) -> None:
    assert main(["plan", "polar", "--workspace", str(root)]) == 0
    assert main(["run", "polar", "--workspace", str(root), "--local"]) == 0


def table(root: Path) -> list[dict[str, str]]:
    with (root / "post/polar/polars/FAM-ACOEF-P1001.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def test_p0600_csv_carries_the_rule_verdict_of_every_point(tmp_path):
    root = su2_workspace(tmp_path)
    run(root)
    main(["collect", "polar", "--workspace", str(root)])
    rows = {float(row["ALPHA"]): row for row in table(root)}
    assert sorted(rows) == [-2.0, 0.0, 2.0, 4.0, 15.0]
    for alpha, row in rows.items():
        if alpha == 15.0:
            assert row["rule_verdict"] == "none"
            assert row["rule_iteration"] == "NA"
            assert float(row["rule_drop_orders"]) == 2.0
        else:
            assert row["rule_verdict"] == "R"
            assert int(row["rule_iteration"]) > 0
    header = list(rows[0.0])
    position = header.index("solver_version")
    assert header[position + 1 : position + 4] == [
        "rule_verdict",
        "rule_iteration",
        "rule_drop_orders",
    ]
    folder = root / "post/polar"
    assert not (folder / "campaign_sweep.csv").exists()
    assert not (folder / "polars/P1001.csv").exists()
    products = json.loads((folder / "products.json").read_bytes())
    for path in (folder / "polars").glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            group_rows = list(csv.DictReader(stream))
        assert {float(row["ALPHA"]): row["rule_verdict"] for row in group_rows} == {
            alpha: row["rule_verdict"] for alpha, row in rows.items()
        }
        entry = next(p for p in products["products"] if p["path"].endswith(path.name))
        block = [c["name"] for c in products["run_block"]]
        columns = [c["name"] for c in products["kinds"][entry["kind"]]["columns"]]
        assert list(group_rows[0]) == [*block, *columns]


# P2D-3: the polar file.

COLUMNS = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
RULE = "  ------ -------- --------- --------- -------- -------- --------"
#: The stand-in's converged points: alpha to CMz (CL = 0.1 alpha).
CONVERGED = {alpha: 0.01 * alpha for alpha in (-2.0, 0.0, 2.0, 4.0)}


def polar_faults(text: str, converged: dict[float, float], scratch: Path) -> list[str]:
    """Return what is wrong with a written polar against the converged points."""
    faults = []
    lines = text.splitlines()
    named = [n for n, line in enumerate(lines) if {"alpha", "CL", "CD"} <= set(line.split())]
    if len(named) != 1 or lines[named[0]].split() != COLUMNS:
        faults.append(f"column lines {named}")
    rows = read_polar_rows(text)
    alphas = [row[0] for row in rows]
    if alphas != sorted(converged):
        faults.append(f"QPROP rows at {alphas}, wanted {sorted(converged)}")
    if len(converged) >= 3 and fit_polar(rows) is None:
        faults.append("QPROP cannot fit the polar")
    scratch.write_text(text, encoding="ascii")
    for alpha, cmz in converged.items():
        values = read_polar(scratch, alpha)
        if not values:
            faults.append(f"the XFOIL reader finds no row at {alpha}")
            continue
        if abs(values["CL"].value - 0.1 * alpha) > 1e-4:
            faults.append(f"CL at {alpha}")
        if abs(values["CM"].value + cmz) > 1e-4:
            faults.append(f"CM at {alpha} is not minus CMz")
        if values["CDp"].value != 0.0:
            faults.append(f"CDp at {alpha}")
    return faults


def collected(root: Path) -> str:
    report = collect_workspace(root, "polar")
    polar = root / "post/polar/polars/P1001.txt"
    assert polar in report.tables
    return polar.read_text(encoding="ascii")


def test_p0600_polar_is_read_by_the_xfoil_and_qprop_readers(tmp_path, recwarn):
    root = su2_workspace(tmp_path)
    run(root)
    text = collected(root)
    assert polar_faults(text, CONVERGED, tmp_path / "check.txt") == []
    rows = [line.split() for line in text.splitlines()[-4:]]
    assert [row[0] for row in rows] == ["-2.000", "0.000", "2.000", "4.000"]
    assert all(row[5:] == ["0.0000", "0.0000"] for row in rows)
    left = [w for w in recwarn.list if "ALPHA 15" in str(w.message)]
    assert len(left) == 1
    message = str(left[0].message)
    assert "left out of polars/P1001.txt" in message
    assert "not converged by the rule" in message and "R not reached" in message


def test_p0600_polar_header_names_the_run_and_breaks_no_reader(tmp_path):
    root = su2_workspace(tmp_path)
    run(root)
    text = collected(root)
    lines = text.splitlines()
    column = next(n for n, line in enumerate(lines) if line.split() == COLUMNS)
    assert lines[column + 1] == RULE
    head = chr(10).join(lines[:column])
    assert "XFOIL" not in text
    assert "SU2 8.6.0" in head and "cfdbender" in head
    assert "naca.su2" in head
    assert hashlib.sha256(b"NDIME= 2" + bytes([10])).hexdigest() in head
    assert "Mach =   0.150" in head and "Re =     6.000 e 6" in head
    assert "fully turbulent" in head
    assert "CDp not measured" in head and "0.00000" in head
    assert "15.000" in head, "the left-out angle is named in the header"
    assert text.isascii()


def test_p0600_post_rebuilds_the_same_polar(tmp_path):
    root = su2_workspace(tmp_path)
    run(root)
    first = collected(root)
    post_workspace(root, "polar")
    assert (root / "post/polar/polars/P1001.txt").read_text(encoding="ascii") == first
    products = json.loads((root / "post/polar/products.json").read_text())
    polars = [p for p in products["products"] if p["path"].endswith("P1001.txt")]
    assert polars == [
        {
            "kind": "polar",
            "polar": "1001",
            "name": "P1001",
            "path": "post/polar/polars/P1001.txt",
            "sha256": hashlib.sha256(
                (root / "post/polar/polars/P1001.txt").read_bytes()
            ).hexdigest(),
        }
    ]


def test_p0600_left_out_point_is_refused_under_warning_refusals(tmp_path):
    root = su2_workspace(tmp_path)
    run(root)
    with warning_refusals(True), pytest.raises(WorkspaceError, match="ALPHA 15"):
        collect_workspace(root, "polar")
    assert not (root / "post/polar/polars/P1001.txt").exists()


def test_p0600_qprop_fit_range_warning(tmp_path, recwarn):
    root = su2_workspace(tmp_path, "8,10,12")
    run(root)
    collected(root)
    assert any("fewer than three" in str(w.message) for w in recwarn.list)


def test_p0600_no_polar_for_a_3d_mesh_or_another_sweep(tmp_path, recwarn):
    three = su2_workspace(tmp_path / "three", ndime=3)
    run(three)
    other = su2_workspace(tmp_path / "mach", "0.1,0.2", condition="MACH:sweep, REmi:6, ALPHA:2")
    run(other)
    for root in (three, other):
        report = collect_workspace(root, "polar")
        assert not (root / "post/polar/polars/P1001.txt").exists()
        assert all(path.suffix == ".csv" for path in report.tables)
    assert not [w for w in recwarn.list if "polars/P1001.txt" in str(w.message)]


def test_p0600_mutants_are_caught(tmp_path):
    root = su2_workspace(tmp_path)
    run(root)
    lines = collected(root).splitlines()
    scratch = tmp_path / "check.txt"
    unconverged = [*lines, "  15.000   1.5000   0.12250   0.00000  -0.1500   0.0000   0.0000"]
    assert polar_faults(chr(10).join(unconverged), CONVERGED, scratch)
    broken = [" alpha CL CD of this run", *lines]
    assert polar_faults(chr(10).join(broken), CONVERGED, scratch)
    flipped = [
        line[:37] + f"{-float(line[37:46]):9.4f}" + line[46:]
        if line[:8].strip() == "2.000"
        else line
        for line in lines
    ]
    assert flipped != lines
    assert polar_faults(chr(10).join(flipped), CONVERGED, scratch)


def foam_row(alpha: float) -> dict[str, object]:
    return {
        "point_id": f"2001/DP-AL{alpha:+.0f}",
        "pol": "2001",
        "condition": {"MACH": 0.15, "REmi": 6.0, "ALPHA": alpha},
        "status": "COMPLETED",
        "classification": "iteration_limit",
        "inputs_sha256": {"inputs/mesh/naca.msh": "ab" * 32},
        "result": {
            "coefficients": {
                "Cl": {"value": 0.1 * alpha},
                "Cd": {"value": 0.01},
                "CmPitch": {"value": -0.001 * alpha},
            },
            "references": {"area": 1.0, "length": 1.0},
            "history": [{"iteration": 20000, "residual": 1e-5}],
            "solver_version": "v2412",
        },
    }


def test_p0600_openfoam_polar_from_recorded_outputs(tmp_path):
    root = tmp_path
    (root / "inputs/matrices").mkdir(parents=True)
    (root / "inputs/matrices/foam.fs").write_text(
        f"{HEADER}\n{'-' * 60}\n2001|1|001|001|001|MACH:0.15, REmi:6, ALPHA:sweep|0,2|naca.msh\n",
        encoding="ascii",
    )
    rows = [foam_row(0.0), foam_row(2.0)]
    recorded_2007(root / "sims/sim_2001/datapoints/DP-AL+0")
    openfoam_point(root / "sims/sim_2001/datapoints/DP-AL+2", 1e-7, 1e-4)
    verdicts = judge_rows(root, rows, "openfoam")
    with pytest.warns(UserWarning, match="ALPHA 0 ") as caught:
        written = write_polars(root, "inputs/matrices/foam.fs", rows, "openfoam", verdicts)
    assert len([w for w in caught.list if "left out" in str(w.message)]) == 1
    text = written[0].read_text(encoding="ascii")
    assert [row[0] for row in read_polar_rows(text)] == [2.0]
    assert "OpenFOAM v2412" in text and "incompressible" in text
    assert "naca.msh" in text and "ab" * 32 in text
    assert read_polar(written[0], 2.0)["CM"].value == pytest.approx(-0.002, abs=1e-4)


def test_review_empty_polar_retains_flight_conditions(tmp_path):
    root = su2_workspace(tmp_path, "15")
    run(root)
    with pytest.warns(UserWarning):
        text = collected(root)
    assert read_polar_rows(text) == []
    assert "Mach =   0.150" in text
    assert "Re =     6.000 e 6" in text

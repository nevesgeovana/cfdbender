"""P0000: run and collect QPROP points with stand-in XFOIL and QPROP executables.

Oracles are the analytic stand-in laws and hand arithmetic, never production code.
"""

from __future__ import annotations

import json
import math
import sys

import pytest
from qprop_support import beta_law
from qprop_workspace import (
    CONDITION,
    FAKE_QPROP,
    adapter,
    local_reynolds,
    prepared,
    prop_stations,
    setup_text,
    table_set,
    tree,
    workspace,
)

from cfdbender.core.registry import RunRegistry
from cfdbender.geometry.blade_slicing import slice_blade
from cfdbender.matrix.cli import main as matrix_cli
from cfdbender.workspace.planning import plan_workspace

RHO, MU, SPEED, RPM = 1.2, 1.8e-5, 20.0, 6000.0
SLOPE_PER_RAD = 0.11 * 180 / math.pi


def run_point(root, **kwargs):
    point = prepared(root, **kwargs)
    result = adapter().execute(point, check_ram=lambda: None)
    return point, result, root / point.point.directory


def test_p0000_qprop_run_makes_missing_polars_with_xfoil_and_writes_the_prop_file(tmp_path):
    root = workspace(tmp_path)
    checks = []
    point = prepared(root)
    result = adapter().execute(point, check_ram=lambda: checks.append(1))
    folder = root / point.point.directory
    assert len(checks) >= 5
    assert adapter().classify(result) == "converged" and result.solver_version == "1.22"
    sections = slice_blade(
        (root / "inputs/geometry/blade.obj").read_text(), n_sections=5, cut="plane"
    )
    rows = prop_stations(folder)
    assert len(rows) == 5
    for row, s in zip(rows, sections, strict=True):
        assert row[0] == pytest.approx(s.radius, abs=1e-5)
        assert row[1] == pytest.approx(s.chord, abs=1e-5)
        re = local_reynolds(RHO, MU, SPEED, RPM, s.radius, s.chord)
        assert row[11] == pytest.approx(re, abs=1.0)
        cl0 = 0.05 + 1e-7 * row[11]
        assert row[3] == pytest.approx(cl0, abs=2e-4)
        assert row[4] == pytest.approx(SLOPE_PER_RAD, rel=1e-3)
        assert row[5] == pytest.approx(cl0 - 8 * 0.11, abs=2e-3)
        assert row[6] == pytest.approx(cl0 + 16 * 0.11, abs=2e-3)
        assert row[7] == pytest.approx(0.01, abs=1e-4)
        assert row[8] == pytest.approx(0.02, rel=2e-2) and row[9] == pytest.approx(0.02, rel=2e-2)
        assert row[10] == pytest.approx(cl0 + 0.22, abs=2e-3)
    record = json.loads((folder / "sections.json").read_text())
    assert [s["source"] for s in record["sections"]] == ["xfoil"] * 5
    for index in range(1, 6):
        section = folder / f"xfoil/sec_{index:02d}"
        assert (section / "polar.txt").is_file() and (section / "airfoil.dat").is_file()
        assert json.loads((section / "execution.json").read_text())["exit_code"] == 0
    meta = json.loads((folder / "execution.json").read_text())
    assert meta["argv"] == [sys.executable, str(FAKE_QPROP), "propeller.prop", "none", "20", "6000"]
    assert "QPROP Version 1.22" in (folder / "stdout.log").read_text()


@pytest.mark.parametrize(
    "condition,badd",
    [
        ("V:20,RPM:6000,BETA07:30", 30 - beta_law(0.7 * 0.2)),
        ("V:20,RPM:6000,DBETA:2.5", 2.5),
        ("V:20,RPM:6000", 0.0),
    ],
)
def test_p0000_qprop_beta_offset_is_the_target_at_seven_tenths_radius_or_plain(
    tmp_path, condition, badd
):
    root = workspace(tmp_path, condition=condition)
    _, _, folder = run_point(root)
    line = next(x for x in (folder / "propeller.prop").read_text().splitlines() if "Badd" in x)
    assert float(line.split()[2]) == pytest.approx(badd, abs=0.02)
    assert json.loads((folder / "sections.json").read_text())["beta_add"] == pytest.approx(
        badd, abs=0.02
    )


def test_p0000_qprop_collect_returns_both_normalisations_and_the_standard_ones(tmp_path):
    root = workspace(tmp_path)
    point, _, _ = run_point(root)
    result = adapter().collect(point)
    n, d = RPM / 60, 0.4
    j = SPEED / (n * d)
    ct, cp = 0.12 - 0.06 * j, 0.04 + 0.01 * j
    value = {k: c.value for k, c in result.coefficients.items()}
    assert set(value) == {"J", "CT", "CP", "CT_QPROP", "CP_QPROP", "EFFICIENCY", "T", "Q", "P"}
    assert value["J"] == pytest.approx(j, rel=1e-5)
    assert value["CT"] == pytest.approx(ct, rel=1e-4) and value["CP"] == pytest.approx(cp, rel=1e-4)
    assert value["T"] == pytest.approx(ct * RHO * n**2 * d**4, rel=1e-4)
    assert value["P"] == pytest.approx(cp * RHO * n**3 * d**5, rel=1e-4)
    assert value["Q"] == pytest.approx(value["P"] / (2 * math.pi * n), rel=1e-4)
    assert value["EFFICIENCY"] == pytest.approx(j * ct / cp, rel=1e-4)
    tip_speed_area = 0.5 * RHO * (2 * math.pi * n * 0.2) ** 2 * math.pi * 0.2**2
    assert value["CT_QPROP"] == pytest.approx(value["T"] / tip_speed_area, rel=1e-4)
    assert value["CP_QPROP"] == pytest.approx(
        value["P"] / (tip_speed_area * 2 * math.pi * n * 0.2), rel=1e-4
    )
    assert result.classification == "converged" and result.solver_version == "1.22"
    assert result.history[-1].converged
    assert result.references.area == pytest.approx(math.pi * 0.2**2)
    assert result.references.length == pytest.approx(0.4)
    assert result.coefficients["T"].axis == "x"


def test_p0000_qprop_run_collect_cli_and_rerun_refusal(tmp_path):
    root = workspace(tmp_path, condition="V:20,J:sweep,RHO:1.2,MU:1.8e-5", sweep="0.8,1.0")
    assert matrix_cli(["run", "study", "--workspace", str(root), "--local"]) == 0
    records = RunRegistry(root).read()
    assert len(records) == 2
    assert all(r.status == "COMPLETED" and r.classification == "converged" for r in records)
    assert matrix_cli(["collect", "study", "--workspace", str(root)]) == 0
    summary = json.loads((root / "post/study/campaign_summary.json").read_bytes())
    assert summary["solver"] == "qprop" and len(summary["points"]) == 2
    for row, j in zip(summary["points"], (0.8, 1.0), strict=True):
        coefficients = row["result"]["coefficients"]
        assert coefficients["J"]["value"] == pytest.approx(j, rel=1e-5)
        assert coefficients["CT"]["value"] == pytest.approx(0.12 - 0.06 * j, rel=1e-4)
        assert row["classification"] == "converged" and row["error"] is None
    before = tree(root)
    assert matrix_cli(["collect", "study", "--workspace", str(root)]) == 0
    assert tree(root) == before
    assert matrix_cli(["run", "study", "--workspace", str(root), "--local"]) == 2
    assert tree(root) == before


def test_p0000_qprop_xfoil_failure_retries_with_a_narrower_alpha_range(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_XFOIL", "sec_02=fail")
    root = workspace(tmp_path)
    _, result, folder = run_point(root)
    assert adapter().classify(result) == "converged"
    record = json.loads((folder / "sections.json").read_text())
    assert [s["source"] for s in record["sections"]] == [
        "xfoil", "xfoil_narrow", "xfoil", "xfoil", "xfoil"
    ]  # fmt: skip
    deck = (folder / "xfoil/sec_02_narrow/xfoil.in").read_text().splitlines()
    assert "ASEQ -4 10 0.5" in deck
    assert (folder / "xfoil/sec_02/execution.json").is_file()


def test_p0000_qprop_xfoil_failure_twice_reuses_the_nearest_good_section(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_XFOIL", "sec_03=fail,sec_03_narrow=fail")
    root = workspace(tmp_path)
    _, _, folder = run_point(root)
    sections = slice_blade(
        (root / "inputs/geometry/blade.obj").read_text(), n_sections=5, cut="plane"
    )
    record = json.loads((folder / "sections.json").read_text())["sections"]
    assert [s["source"] for s in record] == ["xfoil", "xfoil", "reused", "xfoil", "xfoil"]
    radii = [s.radius for s in sections]
    nearest = min((0, 1, 3, 4), key=lambda i: abs(radii[i] - radii[2])) + 1
    assert record[2]["reused_from"] == nearest
    rows = prop_stations(folder)
    assert rows[2][3:11] == rows[nearest - 1][3:11]
    assert rows[2][0] == pytest.approx(radii[2], abs=1e-5), "geometry stays the section's own"


def test_p0000_qprop_every_xfoil_failure_fails_the_point(tmp_path, monkeypatch):
    fails = ",".join(f"sec_{i:02d}{s}=fail" for i in range(1, 6) for s in ("", "_narrow"))
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_XFOIL", fails)
    root = workspace(tmp_path)
    with pytest.raises(ValueError, match="no section polar"):
        adapter().execute(prepared(root), check_ram=lambda: None)
    assert matrix_cli(["run", "study", "--workspace", str(root), "--local"]) == 2


@pytest.mark.parametrize("mode", ["failed", "nonzero"])
def test_p0000_qprop_failed_solver_output_is_failed(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_MODE", mode)
    root = workspace(tmp_path)
    point, result, _ = run_point(root)
    assert adapter().classify(result) == "failed"
    assert adapter().collect(point).classification == "failed"


def test_p0000_qprop_timeout_and_existing_directory_and_low_ram(tmp_path, monkeypatch):
    root = workspace(tmp_path, setup=setup_text(n_sections=2), options="timeout_s = 0.6\n")
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_MODE", "timeout")
    point, result, folder = run_point(root)
    assert result.exit_code == 124 and adapter().classify(result) == "failed"
    assert json.loads((folder / "execution.json").read_text())["timed_out"] is True
    with pytest.raises(ValueError, match="existing"):
        adapter().execute(point, check_ram=lambda: None)
    other = workspace(tmp_path / "ram")
    from cfdbender.solvers._host import _ram

    monkeypatch.setattr(_ram, "free_ram_gib", lambda: 0.6)
    with pytest.raises(ValueError, match="RAM"):
        adapter().execute(prepared(other), check_ram=lambda: None)
    assert not (other / prepared(other).point.directory).exists()


def test_p0000_qprop_table_route_uses_given_polars_and_runs_xfoil_only_for_gaps(tmp_path):
    root = workspace(tmp_path, setup=setup_text(blade="inputs/polars-2D/set1", n_sections=None))
    table_set(root)
    point, result, folder = run_point(root)
    assert adapter().classify(result) == "converged"
    assert {"inputs/polars-2D/set1/blade.csv", "inputs/polars-2D/set1/p1.txt"} <= set(
        point.point.inputs_sha256
    )
    record = json.loads((folder / "sections.json").read_text())["sections"]
    assert [s["source"] for s in record] == ["table", "table", "xfoil"]
    assert sorted(p.name for p in (folder / "xfoil").iterdir()) == ["sec_03"]
    rows = prop_stations(folder)
    assert [r[0] for r in rows] == [0.06, 0.12, 0.19]
    assert [r[2] for r in rows] == [50.0, 38.0, 22.0]
    assert rows[0][3] == pytest.approx(0.2, abs=1e-3) and rows[1][3] == pytest.approx(0.3, abs=1e-3)
    assert rows[2][3] == pytest.approx(0.05 + 1e-7 * rows[2][11], abs=2e-4)
    assert rows[0][11] == pytest.approx(local_reynolds(RHO, MU, SPEED, RPM, 0.06, 0.030), abs=1.0)


def test_p0000_qprop_table_route_needs_no_xfoil_when_every_polar_is_given(tmp_path):
    root = workspace(
        tmp_path, setup=setup_text(blade="inputs/polars-2D/set1", n_sections=None), xfoil=False
    )
    table_set(
        root,
        rows=[
            "0.06,0.030,50,a1.dat,p1.txt",
            "0.12,0.026,38,a1.dat,p2.txt",
            "0.19,0.021,22,,p1.txt",
        ],
    )
    _, result, folder = run_point(root)
    assert adapter().classify(result) == "converged"
    assert not (folder / "xfoil").exists()


@pytest.mark.parametrize(
    "rows,setup,problem",
    [
        (["0.06,0.030,50,,"], {}, "airfoil or polar"),
        (["0.06,0.030,50,a1.dat,p1.txt"], {"n_sections": 4}, "carries 1"),
        (["0.12,0.03,40,a1.dat,p1.txt", "0.06,0.03,50,a1.dat,p1.txt"], {}, "ascending"),
        (["0.06,abc,50,a1.dat,p1.txt"], {}, "chord"),
    ],
)
def test_p0000_qprop_table_route_refuses_bad_tables(tmp_path, rows, setup, problem):
    overrides = {"blade": "inputs/polars-2D/set1", "n_sections": None, **setup}
    root = workspace(tmp_path, setup=setup_text(**overrides))
    table_set(root, rows=rows)
    with pytest.raises(ValueError, match=problem):
        plan_workspace(root, "study")


def test_p0000_qprop_registered_and_condition_keys_are_documented_by_the_module(tmp_path):
    assert adapter().SOLVER_NAME == "qprop"
    assert CONDITION.startswith("V:")
    assert "BETA07" in (adapter().__doc__ or "") and "N_SECTIONS" in (adapter().__doc__ or "")

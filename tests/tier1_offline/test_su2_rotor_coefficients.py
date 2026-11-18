"""The rotor quantities of a collected SU2 rotating-frame point and the rotor table."""

from __future__ import annotations

import math

import pytest
import test_su2_matrix_columns as columns
from test_su2_advance_ratio import (
    DIAMETER,
    FRAME,
    GAMMA,
    GAS,
    MACH,
    REFERENCE_BLOCK,
    TEMPERATURE,
    build,
    row,
)

from cfdbender.workspace.planning import plan_workspace

SETUP_COLLECT = FRAME + 'INIT_OPTION = "TD_CONDITIONS"\nfreestream_pressure = 101325.0\n'
MOMENT_ORIGIN = (0.1, 0.02, 0.01)
HISTORY = (
    "Inner_Iter,rms[Rho],CFx,CFy,CFz,CMx,CMy,CMz\n"
    "0,-3.0,-0.001,0.001,0.001,0.0001,0.0,0.0\n"
    "1,-5.0,-0.012,0.003,0.004,0.0004,0.0002,0.0001\n"
)
FORCE = (-0.012, 0.003, 0.004)
MOMENT = (0.0004, 0.0002, 0.0001)


def collected(root, monitored=None, advance=0.4):
    """Plan, write the deck and a fabricated history, and collect the first point."""
    from cfdbender.core.adapter import load_point
    from cfdbender.core.plan import workspace_path
    from cfdbender.solvers import load_solver

    block = REFERENCE_BLOCK + ("" if monitored is None else f"monitored_blades = {monitored}\n")
    build(
        root,
        f"MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:{advance}",
        setup=SETUP_COLLECT,
        block=block,
    )
    reference = root / "inputs/references/r001.toml"
    reference.write_text(
        reference.read_text(encoding="ascii").replace(
            "moment_origin = [0.0, 0.0, 0.0]", f"moment_origin = {list(MOMENT_ORIGIN)}"
        ),
        encoding="ascii",
    )
    plan = plan_workspace(root, "dpw")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    solver = load_solver("su2")
    (folder / "su2.cfg").write_text(solver.translate(point), encoding="ascii")
    (folder / "history.csv").write_text(HISTORY, encoding="ascii")
    return solver.collect(point), plan


def expected(advance=0.4, k=6.0):
    """The rotor quantities by independent arithmetic from the fabricated coefficients."""
    sonic = math.sqrt(GAMMA * GAS * TEMPERATURE)
    speed = MACH * sonic
    rho = 101325.0 / (GAS * TEMPERATURE)
    q = 0.5 * rho * speed**2
    n = speed / (advance * DIAMETER)
    force = [q * 1.0 * c for c in FORCE]
    moment = [q * 1.0 * 0.5 * c for c in MOMENT]
    # moment about the hub (origin 0, 0, 0): add (reference point - hub) x force
    _, ay, az = MOMENT_ORIGIN
    cross_x = ay * force[2] - az * force[1]
    thrust = k * (-force[0])  # the shaft points toward minus x
    torque = k * (moment[0] + cross_x)  # the rotation is about minus x: Q = -M . omega_hat
    ct = thrust / (rho * n**2 * DIAMETER**4)
    cq = torque / (rho * n**2 * DIAMETER**5)
    cp = 2.0 * math.pi * cq
    return {
        "n_rps": n,
        "advance_ratio": advance,
        "thrust_n": thrust,
        "torque_nm": torque,
        "ct": ct,
        "cq": cq,
        "cp": cp,
        "efficiency": advance * ct / cp,
        "velocity_m_s": speed,
        "density_kg_m3": rho,
        "dynamic_pressure_pa": q,
    }


@pytest.mark.parametrize(("monitored", "k"), [(1, 6.0), (6, 1.0), (None, 1.0)])
def test_the_rotor_quantities_of_the_whole_rotor_from_the_monitored_blades(tmp_path, monitored, k):
    result, _ = collected(tmp_path, monitored)
    rotor = result.to_dict()["rotor"]
    for key, value in expected(k=k).items():
        assert rotor[key] == pytest.approx(value, rel=1e-9), key
    assert rotor["rpm"] == pytest.approx(60.0 * rotor["n_rps"])
    assert rotor["monitored_blades"] == (6 if monitored is None else monitored)
    assert rotor["blades"] == 6


def test_the_rotor_quantities_follow_the_advance_ratio_of_the_point(tmp_path):
    result, _ = collected(tmp_path, 1, advance=0.7)
    rotor = result.to_dict()["rotor"]
    assert rotor["advance_ratio"] == pytest.approx(0.7, rel=1e-12)
    assert rotor["ct"] == pytest.approx(expected(0.7, 6.0)["ct"], rel=1e-9)


def test_a_point_without_a_rotor_block_has_no_rotor_quantities(tmp_path):
    from cfdbender.core.adapter import load_point
    from cfdbender.core.plan import workspace_path
    from cfdbender.solvers import load_solver

    columns.workspace(tmp_path, row=row("MACH:0.0529, ALPHA:0"), setup=SETUP_COLLECT)
    plan = plan_workspace(tmp_path, "dpw")
    folder = workspace_path(tmp_path, plan.points[0].directory)
    folder.mkdir(parents=True)
    point = load_point(tmp_path, plan.points[0], solver="su2")
    solver = load_solver("su2")
    (folder / "su2.cfg").write_text(solver.translate(point), encoding="ascii")
    (folder / "history.csv").write_text(HISTORY, encoding="ascii")
    assert "rotor" not in solver.collect(point).to_dict()


def test_the_rotor_table_is_a_product_of_its_own(tmp_path):
    from cfdbender.matrix.exports import write_tables

    result, plan = collected(tmp_path, 1)
    point = plan.points[0]
    rows = [
        {
            "point_id": point.point_id,
            "pol": point.pol,
            "condition": dict(point.condition),
            "status": "COMPLETED",
            "result": result.to_dict(),
        }
    ]
    paths = write_tables(tmp_path, "inputs/matrices/dpw.fs", rows, "su2")
    table = next(p for p in paths if p.name == "rotor.csv")
    header, line = table.read_text(encoding="ascii").splitlines()
    cells = dict(zip(header.split(","), line.split(","), strict=True))
    assert cells["POL"] == "5001" and cells["BLADES"] == "6" and cells["MONITORED_BLADES"] == "1"
    for column, key in (("CT", "ct"), ("CQ", "cq"), ("CP", "cp"), ("ETA", "efficiency")):
        assert float(cells[column]) == pytest.approx(expected()[key], rel=1e-9)
    assert float(cells["J"]) == pytest.approx(0.4, rel=1e-12)
    assert header.split(",")[:2] == ["POL", "POINT"]
    # the polar files keep their documented columns and gain none of these
    for polar in (p for p in paths if p.name.startswith("FAM-")):
        assert ",CQ," not in polar.read_text(encoding="ascii")

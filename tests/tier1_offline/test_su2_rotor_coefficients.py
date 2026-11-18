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
    TEMPERATURE,
    build,
    row,
)

from cfdbender.workspace.planning import plan_workspace

SETUP_COLLECT = (
    FRAME
    + 'INIT_OPTION = "TD_CONDITIONS"\nfreestream_pressure = 101325.0\n'
    + 'MARKER_PERIODIC = ["right", "left", 0.0, 0.0, 0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n'
)
MOMENT_ORIGIN = (0.1, 0.02, 0.01)
HISTORY = (
    "Inner_Iter,rms[Rho],CFx,CFy,CFz,CMx,CMy,CMz\n"
    "0,-3.0,-0.001,0.001,0.001,0.0001,0.0,0.0\n"
    "1,-5.0,-0.012,0.003,0.004,0.0004,0.0002,0.0001\n"
)
FORCE = (-0.012, 0.003, 0.004)
MOMENT = (0.0004, 0.0002, 0.0001)


def collected(root, copies=None, advance=0.4, setup=None, loads="-"):
    """Plan, write the deck and a fabricated history, and collect the first point."""
    from cfdbender.core.adapter import load_point
    from cfdbender.core.plan import workspace_path
    from cfdbender.solvers import load_solver

    var = "-" if copies is None else f"PERIODIC_COPIES: {copies}"
    symmetry = "-" if copies is None else "PERIODIC"
    build(
        root,
        f"MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:{advance}",
        setup=setup or SETUP_COLLECT,
        var=var,
        symmetry=symmetry,
        loads=loads,
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
    thrust = k * force[0]  # T = F . shaft, the shaft being +x (pyflightstream's sign)
    torque = k * (moment[0] + cross_x)  # Q = M_hub . shaft
    ct = thrust / (rho * n**2 * DIAMETER**4)
    cq = torque / (rho * n**2 * DIAMETER**5)
    cp = 2.0 * math.pi * cq * -1.0  # rpm_sign -1: n is negative
    return {
        "RPM": -60.0 * n,
        "J": advance,
        "T": thrust,
        "Q": torque,
        "CT": ct,
        "CQ": cq,
        "CP": cp,
        "ETA": advance * ct / cp,
    }, {"speed_m_s": speed, "density_kg_m3": rho, "dynamic_pressure_pa": q}


@pytest.mark.parametrize(("copies", "k"), [(1, 1.0), (6, 6.0), (None, 1.0)])
def test_the_rotor_row_of_the_whole_rotor_from_the_periodic_copies(tmp_path, copies, k):
    result, _ = collected(tmp_path, copies)
    rotor = result.to_dict()["rotor"]
    row, stream = expected(k=k)
    for key, value in row.items():
        assert rotor["row"][key] == pytest.approx(value, rel=1e-9), key
    assert rotor["loads"]["speed_m_s"] == pytest.approx(stream["speed_m_s"], rel=1e-12)
    assert rotor["loads"]["density_kg_m3"] == pytest.approx(stream["density_kg_m3"], rel=1e-12)
    q = stream["dynamic_pressure_pa"]
    assert rotor["loads"]["force_n"] == pytest.approx([q * c for c in FORCE], rel=1e-12)
    assert rotor["loads"]["moment_nm"] == pytest.approx([q * 0.5 * c for c in MOMENT], rel=1e-12)
    assert rotor["loads"]["about_m"] == list(MOMENT_ORIGIN)
    assert rotor["periodic_copies"] == (1 if copies is None else copies)
    assert rotor["blades"] == 6 and rotor["rotor"] == "ROTOR"


def test_the_rotor_row_follows_the_advance_ratio_of_the_point(tmp_path):
    result, _ = collected(tmp_path, 6, advance=0.7)
    rotor = result.to_dict()["rotor"]
    assert rotor["row"]["J"] == pytest.approx(0.7, rel=1e-12)
    assert rotor["row"]["CT"] == pytest.approx(expected(0.7, 6.0)[0]["CT"], rel=1e-9)


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


def test_symmetry_loads_off_reports_the_sector_as_computed(tmp_path):
    result, _ = collected(tmp_path, 6, loads="false")
    rotor = result.to_dict()["rotor"]
    assert rotor["symmetry_loads"] == "SECTOR" and rotor["periodic_copies"] == 6
    assert rotor["row"]["T"] == pytest.approx(expected(k=1.0)[0]["T"], rel=1e-9)
    # the in-plane parts of one sector do not cancel
    assert abs(rotor["row"]["CN"]) > 0.0


def test_the_whole_rotor_has_no_in_plane_load(tmp_path):
    rotor = collected(tmp_path, 6)[0].to_dict()["rotor"]
    assert rotor["symmetry_loads"] == "WHOLE"
    for key in ("CN", "CS", "CMN", "CMS"):
        assert rotor["row"][key] == pytest.approx(0.0, abs=1e-9)

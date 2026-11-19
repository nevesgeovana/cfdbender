"""A sliding-mesh point: the interpolator it states, and the history a multizone run writes."""

from __future__ import annotations

import pytest
from test_runtypes_b1_workflows import deck, workspace

from cfdbender.solvers.su2._history import history_name, history_outputs, read, residual_field

SLIDING = '[sliding]\nzones = 2\nrotating_zone = 0\ninterface = ["X_OUT", "X_IN"]\n'
ZONE_COLUMNS = ("rms[Rho]", "rms[RhoU]", "RefForce", "CD", "CL", "CFx", "CMx")


def _sliding(tmp_path, setup=""):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 3 / REVOLUTIONS: 5 / LAST_REVS_AVG: 2.5",
        setup=setup + SLIDING,
    )
    return deck(tmp_path)


def test_a_sliding_interface_defaults_to_the_nearest_neighbour_interpolation(tmp_path):
    point, lines, _ = _sliding(tmp_path)
    assert lines["KIND_INTERPOLATION"] == "NEAREST_NEIGHBOR"
    assert point.origins["KIND_INTERPOLATION"] == "run type unsteady_rotor"


def test_a_setup_that_states_the_interpolation_keeps_it(tmp_path):
    _, lines, _ = _sliding(tmp_path, 'KIND_INTERPOLATION = "ISOPARAMETRIC"\n')
    assert lines["KIND_INTERPOLATION"] == "ISOPARAMETRIC"


def test_a_multizone_history_is_named_after_the_master_configuration():
    assert history_name({"MULTIZONE": "YES", "TIME_DOMAIN": "YES"}) == "su2.csv"
    assert history_name({"TIME_DOMAIN": "YES"}) == "history.csv"
    assert history_outputs({"MULTIZONE": True, "TIME_DOMAIN": True}) == ["su2.csv"]
    assert history_outputs({"TIME_DOMAIN": True}) == ["history.csv"]


def test_a_sliding_point_expects_the_history_it_will_get(tmp_path):
    point, _, _ = _sliding(tmp_path)
    names = [output.rsplit("/", 1)[1] for output in point.outputs]
    assert "su2.csv" in names and "history.csv" not in names


def _folder(tmp_path, steps=3, outer=2):
    """A two-zone point as SU2 8.6.0 leaves it: the driver history under the master's stem."""
    folder = tmp_path / "point"
    folder.mkdir()
    (folder / "su2.cfg").write_text(
        "MULTIZONE= YES\nCONFIG_LIST= ( zone_0.cfg, zone_1.cfg )\nTIME_DOMAIN= YES\n"
        f"TIME_ITER= {steps}\nREF_AREA= 1\nREF_LENGTH= 0.5\n"
        "REF_ORIGIN_MOMENT_X= 0\nREF_ORIGIN_MOMENT_Y= 0\nREF_ORIGIN_MOMENT_Z= 0\n"
    )
    (folder / "zone_0.cfg").write_text("GRID_MOVEMENT= NONE\n")
    (folder / "zone_1.cfg").write_text("GRID_MOVEMENT= RIGID_MOTION\nROTATION_RATE= 1 0 0\n")
    header = ["Time_Iter", "Outer_Iter", "Inner_Iter"] + [
        f"{name}[{zone}]" for zone in (0, 1) for name in ZONE_COLUMNS
    ]
    rows = [",".join(f'"{h}"' for h in header)]
    for step in range(steps):
        for inner in range(outer):
            zone_values = [-9.0 - step - inner, -8.0, 700.0, 0.0, 0.0, 0.0, 0.0]
            turning = [-7.0 - step - inner, -6.0, 700.0, 0.2, 0.5, 0.2, 0.1]
            rows.append(",".join(str(v) for v in (step, inner, 0, *zone_values, *turning)))
    (folder / "su2.csv").write_text("\n".join(rows) + "\n")
    return folder


def test_the_history_of_a_multizone_point_is_read_from_the_turning_zone(tmp_path):
    folder = _folder(tmp_path)
    result, coefficients, _ = read(folder)
    assert result.evidence_error is None
    assert [item.iteration for item in result.history] == [1, 2, 3]
    assert coefficients["CL"].value == 0.5 and coefficients["CFx"].value == 0.2
    assert residual_field(folder) == "rms[Rho][0], rms[Rho][1]"
    assert result.history[-1].residual == pytest.approx(1e-10)


def test_a_missing_multizone_history_names_the_file_looked_for(tmp_path):
    folder = _folder(tmp_path)
    (folder / "su2.csv").unlink()
    result, _, _ = read(folder)
    assert "su2.csv" in str(result.evidence_error)

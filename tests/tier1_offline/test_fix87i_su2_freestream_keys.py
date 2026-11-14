"""FIX87I: the SU2 deck states no free-stream value that SU2 recomputes and ignores."""

from __future__ import annotations

import math

import test_su2_matrix_columns as columns


def _deck(root, solver="RANS", setup=""):
    columns.workspace(root, setup=setup)
    path = root / "inputs/setups/s001.toml"
    path.write_text(
        path.read_text(encoding="ascii").replace('SOLVER = "RANS"', f'SOLVER = "{solver}"'),
        encoding="ascii",
    )
    return columns.deck(root)


def test_viscous_run_with_reynolds_states_neither_pressure_nor_density(tmp_path):
    values = _deck(tmp_path)
    assert "REYNOLDS_NUMBER" in values and "FREESTREAM_TEMPERATURE" in values
    assert "FREESTREAM_PRESSURE" not in values
    assert "FREESTREAM_DENSITY" not in values


def test_euler_run_states_the_pressure_it_uses_but_no_density(tmp_path):
    values = _deck(tmp_path, solver="EULER")
    assert float(values["FREESTREAM_PRESSURE"]) > 0
    assert "FREESTREAM_DENSITY" not in values


def test_thermodynamic_initialisation_keeps_the_pressure(tmp_path):
    values = _deck(tmp_path, setup='INIT_OPTION = "TD_CONDITIONS"')
    assert float(values["FREESTREAM_PRESSURE"]) > 0
    assert "FREESTREAM_DENSITY" not in values


def test_density_free_stream_option_keeps_the_density(tmp_path):
    values = _deck(tmp_path, solver="EULER", setup='FREESTREAM_OPTION = "DENSITY_FS"')
    density, pressure = float(values["FREESTREAM_DENSITY"]), float(values["FREESTREAM_PRESSURE"])
    temperature = float(values["FREESTREAM_TEMPERATURE"])
    assert math.isclose(pressure, density * 287.05287 * temperature, rel_tol=1e-9)

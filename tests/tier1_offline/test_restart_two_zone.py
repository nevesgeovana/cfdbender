"""A RESTART that meets a two-zone (sliding-mesh) point on either side is refused at planning."""

from __future__ import annotations

from test_restart_in_job import CLOCK, batch, build, row, source_id_for, write_matrix

SLIDING = '[sliding]\nzones = 2\nrotating_zone = 0\ninterface = ["X_OUT", "X_IN"]\n'
MESSAGE = "a two-zone (sliding-mesh) restart is not supported; run the sliding point cold"


def test_a_sliding_source_with_a_steady_dependent_in_one_batch_is_refused(tmp_path, capsys):
    build(tmp_path, [row(7001, "unsteady_rotor", CLOCK)], setup=SLIDING)
    source = source_id_for(tmp_path)
    write_matrix(
        tmp_path,
        [row(7001, "unsteady_rotor", CLOCK), row(7002, "qsteady_rotor", "-", source)],
    )
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 2
    assert f"RESTART {source}: {MESSAGE}" in capsys.readouterr().err


def test_a_steady_source_with_a_sliding_dependent_in_one_batch_is_refused(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")], setup=SLIDING)
    source = source_id_for(tmp_path)
    write_matrix(
        tmp_path,
        [row(7001, "qsteady_rotor"), row(7002, "unsteady_rotor", CLOCK, source)],
    )
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 2
    assert "two-zone" in capsys.readouterr().err


def test_a_completed_two_zone_source_is_refused_from_its_saved_configuration(tmp_path, capsys):
    from test_restart_in_job import cli_m, point_of, run_script

    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    assert batch(tmp_path) == 0
    assert run_script(tmp_path).returncode == 0
    assert cli_m(tmp_path, "collect") != 2
    cfg = tmp_path / point_of(tmp_path, 7001).point.directory / "su2.cfg"
    cfg.write_text(cfg.read_text() + "MULTIZONE= YES\n")
    write_matrix(
        tmp_path,
        [row(7001, "qsteady_rotor"), row(7002, "qsteady_rotor", "-", source)],
    )
    assert cli_m(tmp_path, "plan") in (0, 2)
    capsys.readouterr()
    assert cli_m(tmp_path, "run") == 2
    assert "two-zone" in capsys.readouterr().err

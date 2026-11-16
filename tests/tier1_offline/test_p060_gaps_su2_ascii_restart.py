"""P060: a saved SU2 CSV solution selects the ASCII restart reader."""

from test_p060_gaps_restart import call, prepared

from cfdbender.core.registry import RunRegistry


def test_p060_su2_csv_restart_disables_the_binary_reader(tmp_path):
    source, argv = prepared(tmp_path, "su2")
    folder = tmp_path / source.point.directory
    (folder / "restart_flow.csv").write_bytes(b"PointID,Value\n0,1\n")
    assert call([*argv, "--restart"]) == 0
    record = RunRegistry(tmp_path).read()[0]
    text = (tmp_path / record.point.directory / "su2.cfg").read_text()
    assert "READ_BINARY_RESTART= NO" in text
    assert "SOLUTION_FILENAME= restart-input.csv" in text

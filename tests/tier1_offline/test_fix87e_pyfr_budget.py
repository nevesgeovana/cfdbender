"""PyFR budget exhaustion tolerates one output interval before tend."""

from test_conformance_workspace import build, prepared


def test_pyfr_last_sample_one_output_interval_short_of_tend_is_iteration_limit(tmp_path):
    root = build(tmp_path, "pyfr")
    adapter, point = prepared(root, "pyfr")
    adapter.stage(point)
    # The fixture setup has tend 3 and the samples are one unit apart; the last is at 2.
    rows = "0,0.1,0.3,0,0\n1,0.1,0.9,0,0\n2,0.1,0.2,0,0\n"
    (root / point.point.directory / "forces.csv").write_text("t,px,py,vx,vy\n" + rows)
    assert adapter.collect(point).classification == "iteration_limit"

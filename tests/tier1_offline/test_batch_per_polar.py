"""cfd-matrix run --batch-per-polar: one scheduler job per polar, its points in sweep order."""

import json

from test_p0000_item3f_plan import cli, prepared
from test_submit_batch import APP, FIELDS, STUB

from cfdbender.core.registry import RunRegistry

HEADER = "POL|RUN|FLIGHT_CONDITION|SWEEP_VALUES|SETUP|REF|PPROC|VAR_NAMES_VALUES"


def setup(tmp_path):
    prepared(tmp_path, fields=FIELDS)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/stub.sh").write_text(STUB, newline="\n")
    rows = [
        f"1001|1|MACH:0.2, ALPHA:sweep|0,2,4|001|001|001|cluster: one / {APP}",
        f"1002|1|MACH:0.3, ALPHA:sweep|0,2|001|001|001|cluster: one / {APP}",
    ]
    (tmp_path / "inputs/matrices/study.fs").write_text(HEADER + "\n" + "\n".join(rows) + "\n")


def test_each_polar_is_one_job_with_its_points_in_sweep_order(tmp_path, capsys):
    setup(tmp_path)
    assert cli(tmp_path, "plan", "--batch-per-polar") == 0
    capsys.readouterr()
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    jobs = receipt["grouping"]["jobs"]
    assert [j["sims"] for j in jobs] == [["1001"], ["1002"]]
    assert [len(j["points"]) for j in jobs] == [3, 2]
    assert [p.split("/")[0] for p in jobs[0]["points"]] == ["1001"] * 3
    assert cli(tmp_path, "run", "--batch-per-polar") == 0
    records = RunRegistry(tmp_path).read()
    labels = {r.point.pol: r.submission["job"]["label"] for r in records}
    assert labels == {"1001": "study_b1", "1002": "study_b2"}
    script = (tmp_path / "sims/batch/study_b1/BATCH-1001-1001.sh").read_text()
    order = [script.index(p.split("/", 1)[1]) for p in jobs[0]["points"]]
    assert order == sorted(order)


def test_batch_per_polar_refusals(tmp_path, capsys):
    setup(tmp_path)
    assert cli(tmp_path, "plan", "--batch-per-polar", "--local") == 2
    assert "cannot be combined with --local" in capsys.readouterr().err
    assert cli(tmp_path, "plan", "--batch-per-polar", "--batch", "2") == 2
    assert "--batch" in capsys.readouterr().err
    assert cli(tmp_path, "plan", "--batch-per-polar") == 0
    capsys.readouterr()
    for option in ("--resume", "--restart"):
        assert cli(tmp_path, "run", "--batch-per-polar", option) == 2
        assert option in capsys.readouterr().err
    assert cli(tmp_path, "run", "--batch-per-polar", "--force-rerun", "1001/DP-M200AL+000") == 2
    assert "--force-rerun" in capsys.readouterr().err


def test_a_selection_plans_the_polars_it_touches(tmp_path, capsys):
    setup(tmp_path)
    assert cli(tmp_path, "plan", "--batch-per-polar", "--sims", "1002") == 0
    capsys.readouterr()
    jobs = json.loads((tmp_path / "post/study/plan.json").read_text())["grouping"]["jobs"]
    assert [j["sims"] for j in jobs] == [["1002"]]

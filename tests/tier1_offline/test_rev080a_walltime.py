"""The walltime cap of a --batch-per-polar job advises what that mode can do (REV080A-004)."""

from test_batch_per_polar import HEADER, setup
from test_p0000_item3f_plan import cli
from test_submit_batch import APP


def capped(tmp_path):
    setup(tmp_path)
    profile = tmp_path / "inputs/hpc/h001.toml"
    profile.write_text('max_walltime = "01:00:00"\n' + profile.read_text(encoding="utf-8"))
    rows = [
        f"1001|1|MACH:0.2, ALPHA:sweep|0,2,4|001|001|001|cluster: one / {APP}|40m",
        f"1002|1|MACH:0.3, ALPHA:sweep|0,2|001|001|001|cluster: one / {APP}|20m",
    ]
    (tmp_path / "inputs/matrices/study.fs").write_text(
        HEADER + "|WALLTIME\n" + "\n".join(rows) + "\n"
    )


def test_rev080a_004_the_cap_warning_of_a_polar_job_does_not_advise_batch_n(tmp_path, capsys):
    capped(tmp_path)
    assert cli(tmp_path, "plan", "--batch-per-polar") == 0
    err = capsys.readouterr().err
    assert "above the cluster's max_walltime" in err
    assert "--batch n" not in err
    assert "split the polar" in err and "max_walltime" in err


def test_rev080a_004_the_cap_warning_of_batch_n_is_unchanged(tmp_path, capsys):
    capped(tmp_path)
    assert cli(tmp_path, "plan", "--batch", "1") == 0
    assert "a larger --batch n makes shorter jobs" in capsys.readouterr().err

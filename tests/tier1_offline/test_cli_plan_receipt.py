"""cfd-matrix plan writes the receipt, run refuses without it, and post rebuilds the products."""

from __future__ import annotations

from test_p0000_item3c_adapter import workspace

from cfdbender.matrix.cli import main


def test_run_refuses_until_the_matrix_is_planned(tmp_path, capsys):
    workspace(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 2
    assert "Run `cfd-matrix plan <matrix>` first" in capsys.readouterr().err
    assert not (tmp_path / "sims").exists() or not any((tmp_path / "sims").iterdir())
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "post/study/plan.json").is_file()
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0


def test_run_refuses_after_the_matrix_changes(tmp_path, capsys):
    workspace(tmp_path)
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    matrix = tmp_path / "inputs/matrices/study.fs"
    matrix.write_text(matrix.read_text().rstrip() + "\n1002|1|001|001|001\n", encoding="ascii")
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 2
    assert "the matrix changed since it was planned" in capsys.readouterr().err


def test_post_rebuilds_the_products_from_the_records(tmp_path, capsys):
    workspace(tmp_path)
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    summary = tmp_path / "post/study/campaign_summary.json"
    assert not summary.exists()
    capsys.readouterr()
    assert main(["post", "study", "--workspace", str(tmp_path)]) == 0
    assert summary.is_file()
    assert "post read 1" in capsys.readouterr().out

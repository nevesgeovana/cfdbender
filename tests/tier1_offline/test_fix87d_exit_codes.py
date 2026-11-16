"""collect and post exit codes with failed points follow pyflightstream's semantics.

pyflightstream: collect exits 1 when a point it tried to settle failed, 3 when points are still
outstanding, 0 when everything settled; post rebuilds the products from the records and exits 0
whenever they were written (3 only for a partial rebuild). A failed point is in the tables, not
a failure of the rebuild.
"""

from __future__ import annotations

from test_cli_verbs import grown

from cfdbender.matrix.cli import main


def test_collect_exits_1_and_post_exits_0_when_a_point_failed(tmp_path, capsys):
    grown(tmp_path, "failed")
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 2
    ws = str(tmp_path)
    assert main(["collect", "study", "--workspace", ws]) == 1
    capsys.readouterr()
    assert main(["post", "study", "--workspace", ws]) == 0
    assert "failed 2" in capsys.readouterr().out
    assert (tmp_path / "post/study/polars/P1001-TOTAL-ACOEF.csv").is_file()

"""P058-10: the campaign sweep table is retired: not written, refused by name, named once."""

from __future__ import annotations

import json
import re
from pathlib import Path

from test_matrix_selection import three

from cfdbender.matrix.cli import main
from cfdbender.workspace.cli import main as workspace_main

SRC = Path(__file__).resolve().parents[2] / "src"
OLD = b"POL,point\n1001,DP-AL+000\n"


def test_collect_writes_no_sweep_table(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert not (tmp_path / "post/study/campaign_sweep.csv").exists()
    data = json.loads((tmp_path / "post/study/products.json").read_bytes())
    assert not any("campaign_sweep" in p["path"] for p in data["products"])


def test_an_old_sweep_table_is_left_and_named_once(tmp_path, capsys):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    old = tmp_path / "post/study/campaign_sweep.csv"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(OLD)
    capsys.readouterr()
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    printed = capsys.readouterr()
    said = [line for line in (printed.out + printed.err).splitlines() if "is retired" in line]
    assert len(said) == 1 and "P<POL>-TOTAL-ACOEF.csv" in said[0]
    assert old.read_bytes() == OLD
    data = json.loads((tmp_path / "post/study/products.json").read_bytes())
    assert data["migrated"] == {"campaign_sweep.csv": "polars/P<POL>-TOTAL-ACOEF.csv"}


def test_tracing_the_sweep_table_is_refused_by_name(tmp_path, capsys):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    code = workspace_main(["trace", "post/study/campaign_sweep.csv", "--workspace", str(tmp_path)])
    printed = capsys.readouterr()
    assert code == 2
    assert "campaign_sweep.csv is retired" in printed.out + printed.err


def test_the_source_names_the_sweep_table_only_as_retired_or_removed():
    for path in SRC.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "campaign_sweep" in line:
                assert re.search(r"removed|retired", line, re.I), f"{path.name}:{number}"

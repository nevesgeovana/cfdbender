"""P058-8: an earlier-layout file is left in place, named once, and read as a fallback."""

from __future__ import annotations

import json
import warnings

import pytest
from test_matrix_selection import three

from cfdbender.matrix._tree_migration import polar_table, retired_products
from cfdbender.matrix.cli import main

OLD = b"POL,point,ALPHA\n1001,DP-AL+000,0.0\n"


def planted(root, *names):
    polars = root / "post/study/polars"
    polars.mkdir(parents=True, exist_ok=True)
    for name in names:
        (polars / name).write_bytes(OLD)
    return polars


def test_earlier_polar_tables_are_left_named_once_and_migrated(tmp_path, capsys):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    polars = planted(tmp_path, "P1001.csv", "P1002.csv")
    capsys.readouterr()
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    printed = capsys.readouterr()
    about = [line for line in (printed.out + printed.err).splitlines() if "left in place." in line]
    assert len(about) == 1, about
    assert "polars/P1001.csv, polars/P1002.csv" in about[0]
    assert "docs/results.md#the-earlier-layout" in about[0]
    assert (polars / "P1001.csv").read_bytes() == OLD
    data = json.loads((tmp_path / "post/study/products.json").read_bytes())
    assert data["migrated"] == {
        "polars/P1001.csv": "polars/FAM-ACOEF-P1001.csv",
        "polars/P1002.csv": "polars/FAM-ACOEF-P1002.csv",
    }
    log = (tmp_path / "post/study/post.log").read_text(encoding="utf-8")
    assert "left polars/P1001.csv in place; polars/FAM-ACOEF-P1001.csv replaces it" in log


def test_a_current_file_is_not_an_earlier_one(tmp_path):
    polars = planted(tmp_path, "FAM-ACOEF-P1001.csv", "P1001-TOTAL-DIM.csv", "P1001.txt")
    assert retired_products(polars.parent) == []


def test_warning_refusals_refuse_an_earlier_file(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    planted(tmp_path, "P1001.csv")
    assert main(["--warning-refusals", "collect", "study", "--workspace", str(tmp_path)]) == 2


def test_the_reader_falls_back_to_the_earlier_table_with_a_warning(tmp_path):
    polars = planted(tmp_path, "P1001.csv")
    folder = polars.parent
    with pytest.warns(UserWarning, match="the 0.5 polar table"):
        assert polar_table(folder, "1001") == polars / "P1001.csv"
    (polars / "FAM-ACOEF-P1001.csv").write_bytes(b"x\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert polar_table(folder, "1001") == polars / "FAM-ACOEF-P1001.csv"
    with pytest.raises(FileNotFoundError, match="P1003"):
        polar_table(folder, "1003")

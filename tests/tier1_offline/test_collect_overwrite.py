"""``collect --overwrite`` and ``post --overwrite``: replace edited or unrecorded products.

Without the flag a product or table file that is not the one the manifest recorded stops the
command, and the message names the file and the flag. With it every point is collected and the
files are replaced; the manifest then records the new bytes.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from test_matrix_columns_all_solvers import build

from cfdbender.matrix.cli import main


@pytest.fixture
def collected(tmp_path):
    root = build(tmp_path, "mock")
    matrix = root / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii").replace("| 2h | 1.0 |", "| 2h | - |"))
    for moved in (root / "inputs/geometries").iterdir():
        moved.rename(root / "inputs/mesh" / moved.name)
    (root / "inputs/geometries").rmdir()
    for argv in (["plan"], ["run", "--local"], ["collect"]):
        assert main([argv[0], "full", "--workspace", str(root), *argv[1:]]) == 0
    return root


def _edit(root):
    target = root / "post/full/polars/P7001-TOTAL-DIM.csv"
    target.write_bytes(b"edited by hand\n")
    return target


@pytest.mark.parametrize("verb", ["collect", "post"])
def test_a_stop_names_the_file_and_the_flag(collected, capsys, verb):
    target = _edit(collected)
    capsys.readouterr()
    code = main([verb, "full", "--workspace", str(collected)])
    err = capsys.readouterr().err
    assert code == 2
    assert "P7001-TOTAL-DIM.csv" in err and "--overwrite" in err, "".join(l for l in err.splitlines(True) if not l.startswith("#"))[-900:]
    assert target.read_bytes() == b"edited by hand\n"


@pytest.mark.parametrize("verb", ["collect", "post"])
def test_overwrite_replaces_the_file_and_the_manifest_follows(collected, capsys, verb):
    target = _edit(collected)
    capsys.readouterr()
    code = main([verb, "full", "--workspace", str(collected), "--overwrite"])
    assert code == 0, capsys.readouterr().err
    assert target.read_bytes() != b"edited by hand\n"
    manifest = json.loads((collected / "post/full/products.json").read_bytes())
    entry = next(p for p in manifest["products"] if p["path"].endswith("P7001-TOTAL-DIM.csv"))
    assert entry["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    # a second run without the flag is clean again: the records are coherent
    assert main([verb, "full", "--workspace", str(collected)]) == 0

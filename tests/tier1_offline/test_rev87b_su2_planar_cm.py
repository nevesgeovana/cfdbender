"""F8-001: the neutral CM of a two-dimensional SU2 case is CMz, its pitching moment.

SU2 solves a planar case in the x-y plane: it prints "Two dimensional problem.",
reports the pitching moment as CMz and leaves CMx and CMy at zero. The stand-in
below writes what a real 2D run writes.
"""

from __future__ import annotations

import csv
import json
import sys

import pytest
import test_p0000_item3d_config as su2_case

from cfdbender.core.neutral import neutral_values
from cfdbender.matrix.cli import main

STANDIN = """
import csv, sys
from pathlib import Path
cfg = dict(line.split("=", 1) for line in Path(sys.argv[1]).read_text().splitlines())
print('SU2 Release 8.6.0 "Harrier"')
print(DIMENSION_LINE)
with Path("history.csv").open("w", newline="", encoding="ascii") as stream:
    writer = csv.writer(stream)
    writer.writerow(["Inner_Iter", "rms[Rho]", "CL", "CD", "CSF", "CMx", "CMy", "CMz"])
    for i in range(2):
        writer.writerow([i, -3 - i, 0.3, 0.02, 0.0, 0.0, CMY, CMZ])
print("All convergence criteria satisfied.")
"""


def _run(root, line, cmy, cmz):
    su2_case.workspace(root)
    script = root / "standin_su2.py"
    script.write_text(
        STANDIN.replace("DIMENSION_LINE", repr(line))
        .replace("CMY", repr(cmy))
        .replace("CMZ", repr(cmz)),
        encoding="ascii",
    )
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(
        setup.read_text(encoding="ascii")
        + f'\n[solver]\nroute = "executable"\nexecutable = {json.dumps(script.as_posix())}\n',
        encoding="ascii",
    )
    assert main(["plan", "study", "--workspace", str(root)]) == 0
    assert main(["run", "study", "--local", "--workspace", str(root)]) == 0
    assert main(["collect", "study", "--workspace", str(root)]) == 0
    folder = root / "post/study/polars"
    with next(folder.glob("FAM-ACOEF-P*.csv")).open(newline="") as stream:
        return next(csv.DictReader(stream))


def test_a_two_dimensional_case_reports_cmz_as_the_neutral_cm(tmp_path):
    row = _run(tmp_path, "Two dimensional problem.", 0.0, -0.0258)
    assert float(row["CMy"]) == 0.0 and float(row["CMz"]) == -0.0258
    assert float(row["CmW"]) == pytest.approx(0.0258), (
        "Cm is the nose-up pitching moment: minus CMz in 2D"
    )
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert summary["points"][0]["result"]["dimensions"] == 2


def test_a_three_dimensional_case_keeps_cmy(tmp_path):
    row = _run(tmp_path, "Three dimensional problem.", -0.05, 0.001)
    assert float(row["CmW"]) == pytest.approx(-0.05)


def test_the_dimension_falls_back_to_the_native_mesh_and_may_be_unstated(tmp_path):
    from cfdbender.solvers.su2._history import dimensions

    assert dimensions(tmp_path) is None
    (tmp_path / "mesh.su2").write_text("% a comment\nNDIME= 2\nNELEM= 0\n", encoding="ascii")
    assert dimensions(tmp_path) == 2
    (tmp_path / "stdout.log").write_text("Three dimensional problem.\n", encoding="ascii")
    assert dimensions(tmp_path) == 3, "what SU2 printed outranks the mesh header"


@pytest.mark.parametrize("stated,expected", [(2, -7.0), (3, 5.0), (None, 5.0)])
def test_only_a_stated_planar_case_changes_the_name(stated, expected):
    native = {"CL": 1.0, "CD": 2.0, "CMy": 5.0, "CMz": 7.0}
    assert neutral_values("su2", native, dimensions=stated)["CM"] == expected
    assert neutral_values("xfoil", {"CM": 3.0}, dimensions=stated)["CM"] == 3.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

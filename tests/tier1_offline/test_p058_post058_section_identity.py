"""P058-13: equal point names in two polars keep distinct sectional distributions.

The point identity is POL plus POINT. Two outputs with different radii must remain
readable after a post, with their own run block and paths in the product list.
"""

import csv
import hashlib

from cfdbender.matrix.exports import write_polar_files


def test_p058_sections_keep_both_polars_with_the_same_point_name(tmp_path):
    rows = []
    for pol, radius in (("1001", 0.1), ("1002", 0.2)):
        output = f"sims/sim_{pol}/datapoints/DP-V200/stdout.log"
        path = tmp_path / output
        path.parent.mkdir(parents=True)
        path.write_text(f"# radius chord beta\n{radius} 0.03 40\n", encoding="ascii")
        rows.append(
            {
                "pol": pol,
                "point_id": f"{pol}/DP-V200",
                "condition": {"V": 20, "RPM": 6000},
                "result": {},
                "outputs_sha256": {output: hashlib.sha256(path.read_bytes()).hexdigest()},
            }
        )
    product = write_polar_files(tmp_path, "study", rows, "qprop")
    assert len(product.sections) == len(set(product.sections)) == 2
    found = {}
    for path in product.sections:
        with path.open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        found[row["POL"]] = float(row["radius"])
        assert path.relative_to(tmp_path).as_posix() == (
            f"post/study/sections/P{row['POL']}/DP-V200_sections.csv"
        )
    assert found == {"1001": 0.1, "1002": 0.2}

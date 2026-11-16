"""R2-01: the rotor tables state J once, in the run block."""

from cfdbender.matrix.exports import write_tables


def test_rotor_tables_have_unique_columns_and_one_advance_ratio(tmp_path):
    rows = [
        {
            "pol": "1001",
            "point_id": "1001/DP-test",
            "condition": {},
            "status": "COMPLETED",
            "classification": "converged",
            "result": {"coefficients": {"J": {"value": 0.75}}},
        }
    ]
    paths = write_tables(tmp_path, "inputs/matrices/study.fs", rows, "qprop")
    assert paths and not any(p.name == "campaign_sweep.csv" for p in paths)
    for path in paths:
        header = path.read_text().splitlines()[0].split(",")
        assert len(header) == len(set(header)), "a coefficient must have one column"
        assert header.count("J") == 1, path.name
    rcoef = next(p for p in paths if p.name == "FAM-RCOEF-P1001.csv")
    header, values = [line.split(",") for line in rcoef.read_text().splitlines()]
    assert values[header.index("CFZ")] == "NA"

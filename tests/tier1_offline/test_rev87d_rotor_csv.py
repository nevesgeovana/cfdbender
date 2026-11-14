"""R2-01: the rotor columns share J with the fixed CSV context."""

from cfdbender.matrix.exports import CONTEXT, write_tables


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
    for path in write_tables(tmp_path, "inputs/matrices/study.fs", rows, "qprop"):
        header, values = [line.split(",") for line in path.read_text().splitlines()]
        assert len(header) == len(set(header)), "a coefficient must have one column"
        assert header[2:18] == list(CONTEXT)
        assert values[header.index("J")] == "0.75"
        assert values[header.index("CN")] == "NA"

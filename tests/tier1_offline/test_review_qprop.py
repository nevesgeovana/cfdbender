"""Incomplete operating tables are failed evidence, not programming exceptions."""

import pytest

from cfdbender.solvers.qprop._output import parse_table


@pytest.mark.parametrize("header", ["V(m/s) rpm", "V(m/s) rpm CT CT"])
def test_incomplete_or_duplicate_qprop_columns_are_not_operating_evidence(header):
    assert parse_table("# " + header + "\n" + " ".join(["1"] * len(header.split()))) == {}

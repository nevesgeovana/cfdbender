"""Ambiguous or incomplete time histories must be refused as input errors."""

from __future__ import annotations

import pytest

from cfdbender.solvers.su2._time_history import time_history


@pytest.mark.parametrize("column,zone", [("CFx", None), ("CFx[1]", 1)])
def test_duplicate_requested_load_column_is_refused(tmp_path, column, zone):
    (tmp_path / "history.csv").write_text(
        f'Time_Iter," {column} ",{column}\n0,1,99\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicated"):
        time_history(tmp_path, zone=zone, columns=("CFx",))


@pytest.mark.parametrize(
    "body",
    [
        "",
        "Time_Iter,CFx\n",
        "Time_Iter,CFx\n0,1\n1\n",
        "Time_Iter,CFx\n0.5,1\n",
        "Time_Iter,CFx\n-1,1\n",
        "Time_Iter,CFx\n1,1\n0,2\n",
        "Time_Iter,CFx\ninf,1\n",
        "Time_Iter,CFx\n0,nan\n",
    ],
)
def test_malformed_rows_have_a_named_input_error(tmp_path, body):
    (tmp_path / "history.csv").write_text(body, encoding="utf-8")
    try:
        time_history(tmp_path, columns=("CFx",))
    except Exception as error:
        assert isinstance(error, ValueError), type(error).__name__
        assert "history.csv" in str(error)
    else:
        pytest.fail("malformed history was accepted")

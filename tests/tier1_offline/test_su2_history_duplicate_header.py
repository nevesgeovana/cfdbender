"""SU2 8.6.0 writes "Time(sec)" twice with WALL_TIME; a duplicate it does not read is kept apart."""

from __future__ import annotations

from cfdbender.solvers.su2._history import read

CFG = "ITER= 3\nREF_AREA= 1\nREF_LENGTH= 1\n" + "".join(
    f"REF_ORIGIN_MOMENT_{a}= 0\n" for a in "XYZ"
)
# The header line of a real SU2 8.6.0 steady rotating-frame history, shortened.
HEADER = (
    '"Time_Iter","Outer_Iter","Inner_Iter",    "Time(sec)"   ,    "Time(sec)"   ,'
    '    "rms[Rho]"    ,    "rms[RhoU]"   ,       "CD"       ,       "CL"       ,'
    '       "CMx"      ,       "CFx"      '
)
ROWS = [
    "0, 0, 0, 2.69, 2.69, -3.2, -1.1, 0.01, 0.2, 0.003, -0.06",
    "0, 0, 1, 5.10, 2.41, -3.4, -1.3, 0.02, 0.3, 0.004, -0.07",
]


def folder_with(tmp_path, header=HEADER):
    (tmp_path / "su2.cfg").write_text(CFG, encoding="ascii")
    (tmp_path / "history.csv").write_text("\n".join([header, *ROWS]) + "\n", encoding="ascii")
    return tmp_path


def test_a_duplicated_column_the_reader_does_not_use_is_kept_apart(tmp_path):
    result, coefficients, _ = read(folder_with(tmp_path))
    assert [i.iteration for i in result.history] == [1, 2]
    assert coefficients["CFx"].value == -0.07 and coefficients["CMx"].value == 0.004
    assert result.evidence_error is None


def test_a_duplicated_column_the_reader_uses_is_a_named_refusal(tmp_path):
    header = HEADER.replace('"CD"', '"CFx"')
    result, coefficients, _ = read(folder_with(tmp_path, header))
    assert result.history == () and coefficients == {}
    assert result.evidence_error is not None
    assert "CFx" in result.evidence_error and "history.csv" in result.evidence_error
    assert "duplicated" in result.evidence_error


def test_an_unreadable_history_says_why(tmp_path):
    (tmp_path / "su2.cfg").write_text(CFG, encoding="ascii")
    result, _, _ = read(tmp_path)
    assert result.evidence_error is not None and "history.csv" in result.evidence_error

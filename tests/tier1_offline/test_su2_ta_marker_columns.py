"""Per-marker load columns of an SU2 history carry the marker in round brackets.

Measured on a real SU2 8.6.0 time-accurate history with ``AERO_COEFF_SURF``: ``CFx(blade)``;
SU2's multizone output appends the zone index first, ``CFx[1](blade)``.
"""

from __future__ import annotations

from cfdbender.solvers.su2._time_history import time_history, zone_column

LOADS = ("CFx", "CFy", "CFz", "CMx", "CMy", "CMz")


def test_the_name_table_spells_columns_as_su2_writes_them():
    assert zone_column("CFx", None) == "CFx"
    assert zone_column("CFx", None, "blade") == "CFx(blade)"
    assert zone_column("CFx", 1) == "CFx[1]"
    assert zone_column("CFx", 1, "blade") == "CFx[1](blade)"


def test_one_markers_time_history_is_read_from_its_own_columns(tmp_path):
    header = ["Time_Iter", "Inner_Iter", "rms[Rho]", *LOADS, *(f"{n}(blade)" for n in LOADS)]
    rows = [",".join(header)]
    rows += [f"{k},0,-3," + ",".join(["1"] * 6) + ",0.1,0.2,0.3,0.4,0.5,0.6" for k in (0, 1)]
    (tmp_path / "history.csv").write_text("\n".join(rows) + "\n", encoding="ascii")
    history = time_history(tmp_path, marker="blade")
    assert list(history.columns["CFz"]) == [0.3, 0.3]
    assert list(time_history(tmp_path).columns["CFz"]) == [1.0, 1.0]

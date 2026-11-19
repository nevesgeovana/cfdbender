"""Interleaved A, B, A rows are contiguous within each batch group."""

from cfdbender.hpc._batch_split import RowUnit, split_rows


def test_interleaved_profiles_keep_group_order():
    rows = [
        RowUnit(str(i), i, (profile, "4", "build"), 1.0)
        for i, profile in enumerate(("A", "B", "A"), 1)
    ]
    jobs, warnings = split_rows(rows, 2)
    assert [[row.sim_id for row in job] for job in jobs] == [["1", "3"], ["2"]]
    assert warnings == []

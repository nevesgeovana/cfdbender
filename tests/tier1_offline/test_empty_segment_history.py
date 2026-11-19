"""A header-only segment cannot supply evidence for a joined time history."""

import pytest
from test_su2_ta_history_files import _history, _segments

from cfdbender.solvers.su2._history import read
from cfdbender.solvers.su2._time_history import time_history


@pytest.mark.parametrize("empty", ["history.csv", "history_00003.csv"])
def test_empty_segment_is_named_and_refused(tmp_path, empty):
    folder = tmp_path / "p"
    _segments(folder)
    for name, steps in (("history.csv", [0, 1, 2]), ("history_00003.csv", [3, 4, 5])):
        _history(folder, name, [] if name == empty else steps)
    result = read(folder)[0]
    assert result.evidence_error is not None, "a zero-row segment was silently accepted"
    assert empty in result.evidence_error
    with pytest.raises(ValueError, match="no data rows"):
        time_history(folder)

"""F8-006: post rebuilds the tables and leaves a submitted point to collect."""

from __future__ import annotations

from test_p0000_item3f_collect import outputs
from test_p0000_item3f_registry import submitted

from cfdbender.matrix.cli import main


def test_post_does_not_pick_up_a_submitted_point(tmp_path, capsys):
    registry, point, _ = submitted(tmp_path)
    outputs(tmp_path, point)
    before = registry.path.read_bytes()
    assert main(["post", "study", "--workspace", str(tmp_path)]) == 3
    assert "outstanding 1" in capsys.readouterr().out
    assert registry.path.read_bytes() == before, "post never changes a record"
    assert registry.read()[0].status == "SUBMITTED"
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert registry.read()[0].status == "COMPLETED"

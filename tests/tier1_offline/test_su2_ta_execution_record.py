"""execution.json states the thread count only when cfdbender set it for SU2 itself.

A real run through a launcher script (a ``.cmd`` that starts SU2 in WSL with four threads)
recorded ``"threads": 1``, the default cfdbender set for the script, not the count SU2 ran.
"""

from __future__ import annotations

import json

import pytest
from test_segment_execution import point_with_exports

from cfdbender.solvers.su2 import _execute


@pytest.mark.parametrize(
    ("executable", "threads"),
    [("solver", 1), ("C:/ws/inputs/su2_wsl.cmd", "unknown"), ("/ws/run_su2.sh", "unknown")],
)
def test_the_recorded_thread_count_is_unknown_behind_a_launcher_script(
    tmp_path, monkeypatch, executable, threads
):
    point = point_with_exports(tmp_path, segmented=False)
    values = _execute.config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )

    class Child:
        def __init__(self, argv, **kwargs):
            self.status = 0

        def wait(self, **kwargs):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(_execute.subprocess, "Popen", Child)
    monkeypatch.setattr(_execute._ram, "require_ram", lambda *a, **k: None)
    folder = tmp_path / point.point.directory
    _execute._run_owned(
        point,
        lambda: None,
        {"threads": 1, "ram_estimate_gib": 0.25},
        values,
        [executable, "su2.cfg"],
        5.0,
        folder,
        "executable",
        {},
    )
    assert json.loads((folder / "execution.json").read_bytes())["threads"] == threads

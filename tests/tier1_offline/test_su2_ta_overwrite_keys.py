"""SU2 8.6.0 refuses the overwrite keys set to NO for a transient run; the plan refuses them first.

Measured on a real run: ``Appending iterations to the filename (WRT_RESTART_OVERWRITE=NO) is
incompatible with transient problems.``; SU2's source refuses ``WRT_SURFACE_OVERWRITE`` and
``WRT_VOLUME_OVERWRITE`` the same way. The adapter refuses a native key whose value SU2 cannot
run at plan time, naming the key, as it does for ``CONV_FILENAME`` and ``GRID_MOVEMENT``.
"""

from __future__ import annotations

import pytest
import test_runtypes_b1_workflows as rt

from cfdbender.core.plan import PlanError
from cfdbender.workspace.planning import plan_workspace

CLOCK = "DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_REVS_AVG: 0.25"
KEYS = ["WRT_RESTART_OVERWRITE", "WRT_SURFACE_OVERWRITE", "WRT_VOLUME_OVERWRITE"]


@pytest.mark.parametrize("key", KEYS)
def test_an_overwrite_key_set_to_no_is_refused_for_a_time_accurate_row(tmp_path, key):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=CLOCK, setup=f"{key} = false\n")
    with pytest.raises(PlanError, match=f"{key}= NO.*time-accurate"):
        plan_workspace(tmp_path, "rt")


@pytest.mark.parametrize("key", KEYS)
def test_yes_passes_and_a_steady_row_keeps_no(tmp_path, key):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=CLOCK, setup=f"{key} = true\n")
    assert rt.deck(tmp_path)[1][key] == "YES"
    other = rt.workspace(
        tmp_path / "steady", condition="MACH:0.0529, ALPHA:0", rotor="", setup=f"{key} = false\n"
    )
    assert rt.deck(other)[1][key] == "NO"


def test_the_adapter_writes_no_overwrite_key_itself(tmp_path):
    window = CLOCK + " / EXPORT_UNSTEADY_LAST_ITER: 10"
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=window)
    _, lines, files = rt.deck(tmp_path)
    second = dict(line.split("= ", 1) for line in files["su2_segment2.cfg"].splitlines())
    for text in (lines, second):
        assert not [key for key in text if key in KEYS]

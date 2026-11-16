"""RC-4: each warn-and-skip site refuses under warning_refusals and warns without it.

Only two of the twelve routed sites had a test. A site that goes back to warnings.warn
would pass every other test, so each is named here with the text it must refuse with.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, run

from cfdbender.core.executables import check_builds
from cfdbender.core.refusals import warning_refusals
from cfdbender.core.settings import WorkspaceError
from cfdbender.hpc._profiles import read_hpc_profile
from cfdbender.solvers.qprop._execute import _unconverged
from cfdbender.solvers.su2._config import _fluid_model, _history_output


def _hpc(tmp_path):
    profile = tmp_path / "p.toml"
    profile.write_text('application_id = "x"\n[defaults]\nfoo = 1\n', encoding="utf-8")

    def read():
        # The profile is otherwise incomplete: only the warning before that matters here.
        try:
            read_hpc_profile(profile)
        except WorkspaceError:
            raise
        except ValueError:
            pass

    return read


def _builds(tmp_path):
    adapter = SimpleNamespace(EXECUTABLE_KEYS=("executable",))
    return lambda: check_builds(tmp_path, "su2", adapter, ["b1"])


SITES = {
    "hpc defaults block": (_hpc, "defaults block"),
    "carried build undeclared": (_builds, "is carried but no"),
    "qprop dropped angles": (
        lambda tmp: lambda: _unconverged({}, "section 1", [2.0]),
        "did not converge",
    ),
    "su2 history output added": (
        lambda tmp: lambda: _history_output({"HISTORY_OUTPUT": ["RMS_RES"]}),
        "HISTORY_OUTPUT lacks",
    ),
    "su2 fluid model gas": (
        lambda tmp: lambda: _fluid_model({"SOLVER": "EULER", "FLUID_MODEL": "VAN_DER_WAALS"}),
        "FLUID_MODEL",
    ),
}


@pytest.mark.parametrize("site", SITES)
def test_rc4_a_routed_site_warns_by_default_and_refuses_under_the_option(site, tmp_path):
    make, text = SITES[site]
    call = make(tmp_path)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        call()
    assert any(text in str(w.message) for w in seen)
    with warning_refusals(True), pytest.raises(WorkspaceError, match=text):
        call()


def test_rc4_su2_without_a_mach_number_refuses_through_the_command(tmp_path, capsys):
    root = build(tmp_path / "su2", "su2")
    edit(
        root / "inputs/matrices/full.fs",
        "MACH:0.5, ALTFT:10000, ALPHA:sweep, BETA:0",
        "ALPHA:sweep",
    )
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    code, _, err = run(capsys, "--warning-refusals", "plan", "full", "--workspace", root)
    assert code == 2 and "state no MACH_NUMBER" in err


def test_rc4_xfoil_without_a_reynolds_number_refuses_through_the_command(tmp_path, capsys):
    root = build(tmp_path / "xfoil", "xfoil")
    edit(root / "inputs/matrices/full.fs", "MACH:0.1, REmi:1, ALPHA:sweep", "ALPHA:sweep")
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    code, _, err = run(capsys, "--warning-refusals", "plan", "full", "--workspace", root)
    assert code == 2 and "inviscid" in err

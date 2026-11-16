"""P060: the opt-in guard composes with workspace defaults and warning refusals.

Read init.toml with a solver guard, then compose a setup over the defaults.
Both options predate this merge and must survive together.
"""

import pytest

from cfdbender.core.settings import read_init, setup_defaults


@pytest.mark.parametrize("guard", [False, True])
def test_p060_int060_guard_keeps_defaults_and_warning_refusals(tmp_path, guard):
    (tmp_path / "init.toml").write_text(
        "schema_version = 1\nwarning_refusals = true\n"
        '[solver]\nname = "mock"\n'
        f"one_run_at_a_time = {str(guard).lower()}\n"
        "[defaults]\niterations = 300\n[defaults.solver]\nthreads = 2\n",
        encoding="utf-8",
    )
    config = read_init(tmp_path, available_solvers=("mock",))
    assert getattr(config, "one_run_at_a_time", False) is guard
    assert config.solver_options == {}
    assert config.warning_refusals is True
    assert "one_run_at_a_time" not in config.solver_options
    assert setup_defaults(config.defaults, {"solver": {"threads": 4}}) == {
        "iterations": 300,
        "solver": {"threads": 4},
    }
    assert config.defaults == {"iterations": 300, "solver": {"threads": 2}}

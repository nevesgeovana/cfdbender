"""P060: init.toml [defaults] supplies setup values below setup and row overrides."""

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import RowRequirements, plan_matrix


def workspace(root, defaults):
    for kind in ("setups", "references", "pproc", "matrices"):
        (root / "inputs" / kind).mkdir(parents=True)
    (root / "init.toml").write_text('schema_version = 1\nsolver = "mock"\n' + defaults)
    (root / "inputs/setups/s001.toml").write_text("[mock]\niterations = 4\n")
    (root / "inputs/references/r001.toml").write_text("area = 1\n")
    (root / "inputs/pproc/p001.toml").write_text("history = true\n")
    (root / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|VAR_NAMES_VALUES\n"
        "1|1|001|001|001|ALPHA:0|mock.iterations:6\n"
    )


def test_p060_defaults_merge_nested_tables_and_load_the_same_snapshot(tmp_path):
    workspace(tmp_path, '[defaults.mock]\noutcome = "converged"\niterations = 2\n')
    seen = []

    def validate(row, inputs):
        seen.append(inputs.setup)
        return RowRequirements(("run",), ("sims/sim_{pol}/datapoints/{point}/out",))

    try:
        plan = plan_matrix(tmp_path, "study", validators={"mock": validate})
    except ValueError as error:
        pytest.fail(f"[defaults] must be accepted and composed: {error}")
    assert seen == [{"mock": {"outcome": "converged", "iterations": 6}}]
    context = load_point(tmp_path, plan.points[0], solver="mock")
    assert context.setup == seen[0]
    assert "init.toml" in plan.points[0].inputs_sha256


def test_p060_defaults_unknown_key_uses_setup_validation(tmp_path):
    from cfdbender.workspace.planning import plan_workspace

    workspace(tmp_path, '[defaults.mock]\noutcome = "converged"\niteratons = 2\n')
    with pytest.raises(ValueError) as caught:
        plan_workspace(tmp_path, "study")
    text = str(caught.value)
    assert "iteratons" in text and "iterations" in text and "data row 1" in text


def test_p060_defaults_setup_wins_without_mutating_defaults(tmp_path):
    workspace(tmp_path, '[defaults.mock]\noutcome = "converged"\niterations = 2\n')
    path = tmp_path / "inputs/matrices/study.fs"
    path.write_text(path.read_text().replace("mock.iterations:6", "-"))
    seen = []

    def validate(row, inputs):
        seen.append(inputs.setup["mock"]["iterations"])
        return RowRequirements(("run",), ("sims/sim_{pol}/datapoints/{point}/out",))

    try:
        plan_matrix(tmp_path, "study", validators={"mock": validate})
    except ValueError as error:
        pytest.fail(f"[defaults] must be accepted: {error}")
    assert seen == [4]

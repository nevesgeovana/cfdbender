"""Skipped verification must name a working recovery action and warn once per input."""

import pytest
from test_optcheck_identity import cli, small_threshold, workspace  # noqa: F401

from cfdbender.core import plan


def test_unverified_input_and_output_explain_how_to_verify(tmp_path):
    path = tmp_path / "mesh"
    path.write_bytes(b"x" * 2048)
    with plan.large_input_check("identity"), pytest.warns(UserWarning) as caught:
        files = plan.InputFiles(tmp_path)
        assert files.agrees("mesh", "a" * 64)
        assert files.output_agrees(path, ["mesh"], "a" * 64)
    assert len(caught) == 2
    assert all("--rehash-inputs" in str(item.message) for item in caught)


def test_dimension_warning_is_once_per_mesh_and_has_recovery(tmp_path, capsys):
    root = workspace(tmp_path / "ws", "su2")
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace('[groups]\nwing = ["wall"]\n', ""))
    assert cli(root, "run", "--local", "--no-read-large-inputs") == 0
    capsys.readouterr()
    assert cli(root, "collect", "--no-read-large-inputs") == 0
    output = capsys.readouterr().err
    assert output.count("problem dimension not read") == 1
    assert "--rehash-inputs" in output


def test_skipped_check_recovery_overrides_environment():
    assert "--rehash-inputs" in plan.skipped_content("mesh", "markers")

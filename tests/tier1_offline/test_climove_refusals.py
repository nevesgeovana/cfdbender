"""CLIM-2: --warning-refusals turns every warn-and-skip of cfdbender into a refusal.

Without it a skipped entry warns and the command exits 0; with the option, or with
warning_refusals = true in init.toml, the same skip is a refusal with exit 2.
"""

from __future__ import annotations

import test_p0000_xfoil_plan as xfoil_case

from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import main as workspace_main


def _planned(tmp_path):
    xfoil_case.workspace(tmp_path)
    (tmp_path / "inputs/references/r001.toml").write_text("length = 1.0\narea = 2.0\n")


def test_clim2_a_skipped_entry_warns_and_exits_zero_by_default(tmp_path, capsys):
    _planned(tmp_path)
    assert matrix_main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert "area" in capsys.readouterr().err


def test_clim2_the_option_turns_the_skip_into_a_refusal_in_cfd_matrix(tmp_path, capsys):
    _planned(tmp_path)
    code = matrix_main(["--warning-refusals", "plan", "study", "--workspace", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert "area" in err and "--warning-refusals" in err


def test_clim2_init_toml_carries_the_option_too(tmp_path, capsys):
    _planned(tmp_path)
    init = tmp_path / "init.toml"
    init.write_text(init.read_text(encoding="utf-8") + "warning_refusals = true\n", "utf-8")
    assert matrix_main(["plan", "study", "--workspace", str(tmp_path)]) == 2
    assert "area" in capsys.readouterr().err


def test_clim2_a_non_boolean_option_in_init_toml_is_refused(tmp_path, capsys):
    _planned(tmp_path)
    init = tmp_path / "init.toml"
    init.write_text(init.read_text(encoding="utf-8") + 'warning_refusals = "yes"\n', "utf-8")
    assert matrix_main(["plan", "study", "--workspace", str(tmp_path)]) == 2
    assert "warning_refusals" in capsys.readouterr().err


def test_clim2_the_option_reaches_the_workspace_verbs(tmp_path, capsys):
    xfoil_case.workspace(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "activity.log.jsonl").write_text("not json\n", encoding="utf-8")
    assert workspace_main(["log", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    code = workspace_main(["--warning-refusals", "log", "--workspace", str(tmp_path)])
    assert code == 2
    assert "could not be read" in capsys.readouterr().err

"""PyFR reads no REF or PPROC content but one length and history: the rest is warned and skipped."""

from __future__ import annotations

from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, run


def test_unread_reference_and_pproc_keys_are_warned_and_skipped(tmp_path, capsys):
    root = build(tmp_path / "pyfr", "pyfr")
    (root / "inputs/references/r001.toml").write_text("area = 1.0\n", encoding="ascii")
    (root / "inputs/pproc/p001.toml").write_text("history = true\nextra = 3\n", encoding="ascii")
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 0
    assert "'area' of inputs/references/r001.toml is skipped; PyFR does not read it" in err
    assert "'extra' of inputs/pproc/p001.toml is skipped" in err


def test_a_reference_length_is_read_not_warned(tmp_path, capsys):
    root = build(tmp_path / "pyfr", "pyfr")
    (root / "inputs/references/r001.toml").write_text("length = 1.0\n", encoding="ascii")
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 0 and "skipped" not in err


def test_history_must_still_be_true(tmp_path, capsys):
    root = build(tmp_path / "pyfr", "pyfr")
    edit(root / "inputs/pproc/p001.toml", "history = true", "history = false")
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 2 and "inputs/pproc/p001.toml history must be true" in err

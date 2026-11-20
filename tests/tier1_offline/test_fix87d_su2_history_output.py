"""A real SU2 run exited 0 yet failed: SU2's default history.csv lacks the columns read."""

from __future__ import annotations

import warnings

import pytest
import test_p0000_item3d_config as su2_case


def _text(root, history=None):
    su2_case.workspace(root)
    if history is not None:
        setup = root / "inputs/setups/s001.toml"
        setup.write_text(setup.read_text() + f"\nHISTORY_OUTPUT = {history}\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = su2_case.api().translate(su2_case.prepared(root))
    return text, [str(w.message) for w in caught]


def test_no_history_output_writes_the_default_groups(tmp_path):
    text, messages = _text(tmp_path)
    assert "HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF, AERO_COEFF_SURF )" in text
    assert not any("HISTORY_OUTPUT" in m for m in messages)


def test_a_history_output_missing_a_group_is_extended_with_a_warning(tmp_path):
    text, messages = _text(tmp_path, '["ITER", "RMS_RES"]')
    assert "HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF, AERO_COEFF_SURF )" in text
    assert any("HISTORY_OUTPUT lacks AERO_COEFF" in m for m in messages)


@pytest.mark.parametrize("history", ['["aero_coeff", "rms_res", "iter"]'])
def test_a_complete_history_output_is_kept_without_a_warning(tmp_path, history):
    text, messages = _text(tmp_path, history)
    assert "HISTORY_OUTPUT= ( aero_coeff, rms_res, iter, AERO_COEFF_SURF )" in text
    assert not any("HISTORY_OUTPUT" in m for m in messages)

"""The turbulence model is a row variable, fully turbulent, translated by the adapter."""

from __future__ import annotations

import warnings

import pytest
from test_runtypes_b1_workflows import deck, workspace

from cfdbender.core.plan import PlanError


def build(root, model, *, setup_model=None):
    workspace(root, condition="MACH:0.0529, ALPHA:0", rotor="", var=f"TURBULENCE_MODEL: {model}")
    path = root / "inputs/setups/s001.toml"
    text = path.read_text(encoding="ascii").replace('KIND_TURB_MODEL = "SA"\n', "")
    if setup_model:
        text += f'KIND_TURB_MODEL = "{setup_model}"\n'
    path.write_text(text, encoding="ascii")
    return root


@pytest.mark.parametrize(("model", "kind"), [("SST", "SST"), ("sst", "SST"), ("SA", "SA")])
def test_the_row_names_the_model_and_the_adapter_writes_its_keys(tmp_path, model, kind):
    lines = deck(build(tmp_path, model))[1]
    assert lines["KIND_TURB_MODEL"] == kind
    assert "TURBULENCE_MODEL" not in lines, "a row variable is never a native key"
    assert "KIND_TRANS_MODEL" not in lines, "every model is fully turbulent"
    if kind == "SA":
        assert lines["SA_OPTIONS"] == "( NEGATIVE, WITHFT2 )"
    else:
        assert "SA_OPTIONS" not in lines


def test_a_model_the_adapter_does_not_offer_is_refused_listing_those_it_does(tmp_path):
    build(tmp_path, "k_omega")
    with pytest.raises(PlanError, match="'k_omega' is not a model SU2 offers.*known: SA, SST"):
        deck(tmp_path)


def test_the_setup_model_wins_with_one_warning(tmp_path):
    build(tmp_path, "SST", setup_model="SA")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lines = deck(tmp_path)[1]
    assert lines["KIND_TURB_MODEL"] == "SA"
    assert any("the setup file wins" in str(w.message) for w in caught)

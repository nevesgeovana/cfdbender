"""P0000: reusable assertions must pass the mock and reject each broken rule."""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_p0000_item3c_adapter import api, prepared

from cfdbender.core.matrix import read_matrix
from cfdbender.core.plan import InputFiles, RowInputs

RULES = (
    "reads_init",
    "validation",
    "deterministic",
    "classifications",
    "exit_evidence",
    "results",
    "no_overwrite",
)


def helpers():
    try:
        return importlib.import_module("cfdbender.testing")
    except ModuleNotFoundError:
        return SimpleNamespace(**{f"assert_{rule}": lambda *a, **kw: None for rule in RULES})


def invoke(rule, adapter, root):
    case = prepared(root)
    check = getattr(helpers(), f"assert_{rule}")
    if rule == "validation":
        path = root / "inputs/setups/s001.toml"
        path.write_text("[mock]" + chr(10) + "iterations = 4")
        row = read_matrix(root / "inputs/matrices/study.fs")[0]
        check(adapter, row, RowInputs(root, row, InputFiles(root)), "outcome")
    elif rule == "classifications":
        cases = {
            outcome: prepared(root / outcome, outcome)
            for outcome in ("converged", "iteration_limit", "failed")
        }
        check(adapter, cases)
    else:
        check(adapter, case)


@pytest.mark.parametrize("rule", RULES)
def test_p0000_item3c_conformance_mock(tmp_path, rule):
    invoke(rule, api(), tmp_path)


@pytest.mark.parametrize("rule", RULES)
def test_p0000_item3c_conformance_catches_each_mutant(tmp_path, rule):
    real = api()
    mutant = SimpleNamespace(
        **{
            name: getattr(real, name)
            for name in ("validate_row", "translate", "execute", "classify", "collect")
        }
    )
    if rule == "reads_init":
        mutant.translate = lambda point: "ignores init"
    elif rule == "validation":
        mutant.validate_row = lambda row, workspace: None
    elif rule == "deterministic":
        counter = iter(range(100))
        mutant.translate = lambda point: str(next(counter))
    elif rule in ("classifications", "exit_evidence"):
        mutant.classify = lambda result: "converged"
    elif rule == "results":

        def bad_collect(point):
            result = real.collect(point)
            return replace(result, coefficients={"CX": SimpleNamespace(value=1, axis="", kind="")})

        mutant.collect = bad_collect
    else:
        saved = []

        def overwrite(point, **kwargs):
            if not saved:
                saved.append(real.execute(point, **kwargs))
            return saved[0]

        mutant.execute = overwrite
    with pytest.raises(AssertionError, match=rule):
        invoke(rule, mutant, tmp_path)

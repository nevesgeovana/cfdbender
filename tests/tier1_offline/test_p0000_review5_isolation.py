"""Execution conformance must detect mutation outside the point."""

import importlib
from types import SimpleNamespace

import pytest


def helper():
    try:
        module = importlib.import_module("cfdbender.testing")
    except ModuleNotFoundError:
        return lambda *args: None
    return getattr(module, "assert_execution_isolation", lambda *args: None)


@pytest.mark.parametrize("mutation", ["outside", "input"])
def test_p0000_review5_execution_conformance_rejects_mutations(tmp_path, mutation):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/setup.toml").write_text("original")
    point = SimpleNamespace(root=tmp_path, point=SimpleNamespace(directory="sims/point"))

    def execute(case, **kwargs):
        target = case.root / ("stray.txt" if mutation == "outside" else "inputs/setup.toml")
        target.write_text("changed")

    adapter = SimpleNamespace(execute=execute)
    with pytest.raises(AssertionError, match="execution_isolation"):
        helper()(adapter, point)


def test_p0000_review5_execution_conformance_accepts_mock(tmp_path):
    from test_p0000_item3c_adapter import prepared

    from cfdbender.solvers import mock

    point = prepared(tmp_path)
    helper()(mock, point)

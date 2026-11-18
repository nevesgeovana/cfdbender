"""Optional adapter behaviours have one vocabulary and checked declarations."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfdbender.core import adapter as contract
from cfdbender.solvers import available_solvers, load_solver

SOLVERS = ("mock", "openfoam", "pyfr", "qprop", "su2", "xfoil")


def reader():
    function = getattr(contract, "capabilities", None)
    assert callable(function), "the core must provide the capability reader"
    return function


def test_p051_f3c_closed_vocabulary_and_missing_declaration():
    assert getattr(contract, "CAPABILITIES", None) == frozenset(
        {"restart", "unsteady", "boundary_groups", "cluster_staging", "numerics_file"}
    )
    assert reader()(SimpleNamespace()) == frozenset()


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "capability", ("restart", "unsteady", "boundary_groups", "cluster_staging", "numerics_file")
)
def test_p051_f3c_each_declared_capability(solver, capability):
    adapter = load_solver(solver)
    declared = getattr(adapter, "CAPABILITIES", None)
    assert isinstance(declared, frozenset), "each adapter must state its supported behaviours"
    expected = (
        capability == "cluster_staging"
        or (solver == "pyfr" and capability == "unsteady")
        or (solver in {"su2", "openfoam", "pyfr"} and capability == "restart")
        or (solver in {"su2", "openfoam"} and capability == "boundary_groups")
        or (solver in {"su2", "openfoam", "pyfr"} and capability == "numerics_file")
    )
    assert (capability in reader()(adapter)) == expected


def test_p051_f3c_unknown_capability_refused_at_load(monkeypatch):
    adapter = import_module("cfdbender.solvers.mock")
    monkeypatch.setattr(adapter, "CAPABILITIES", frozenset({"restat"}), raising=False)
    with pytest.raises(ValueError) as caught:
        load_solver("mock")
    text = str(caught.value)
    assert "restat" in text and "mock" in text
    for name in ("restart", "unsteady", "boundary_groups", "cluster_staging", "numerics_file"):
        assert name in text


@pytest.mark.parametrize("declaration", ["restart", None, {"restart"}, frozenset({1})])
def test_p051_f3c_declaration_is_a_frozenset_of_names(declaration):
    read = reader()
    with pytest.raises(ValueError, match="CAPABILITIES"):
        read(SimpleNamespace(CAPABILITIES=declaration))


def test_p051_f3c_documented_capabilities_match_declarations():
    text = (Path(__file__).resolve().parents[2] / "docs/solvers.md").read_text(encoding="utf-8")
    assert "## What each solver supports" in text
    section = text.split("## What each solver supports", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    columns = rows[0].strip("| ").split(" | ")
    assert columns == ["Capability", *SOLVERS]
    assert set(SOLVERS) == set(available_solvers()) - {"flightstream"}
    data = rows[2:]
    assert len(data) == len(contract.CAPABILITIES)
    assert {line.split(" | ")[0].strip("| `") for line in data} == contract.CAPABILITIES
    for line in data:
        name, *cells = line.strip("| ").split(" | ")
        assert cells == [
            "Yes" if name.strip("`") in reader()(load_solver(solver)) else "No"
            for solver in SOLVERS
        ]

"""One solver per workspace, selected by init.toml.

Usage: read_init(root, available_solvers=available_solvers()).
"""

import importlib
import importlib.util
from pathlib import Path

import pytest


def _module(name):
    assert importlib.util.find_spec(name) is not None, f"required module: {name}"
    return importlib.import_module(name)


def test_p0000_item2a_mock_registration_and_loader():
    solvers = _module("cfdbender.solvers")
    assert solvers.available_solvers() == (
        "flightstream",
        "mock",
        "openfoam",
        "pyfr",
        "qprop",
        "su2",
        "xfoil",
    )
    adapter = solvers.load_solver("mock")
    assert adapter.__name__ == "cfdbender.solvers.mock"
    assert adapter.SOLVER_NAME == "mock"
    assert solvers.load_solver("mock") is adapter


@pytest.mark.parametrize("name", ["no-such-solver", "", "../mock", "MOCK"])
def test_p0000_item2a_unknown_solver_is_named_with_available_choices(name):
    solvers = _module("cfdbender.solvers")
    with pytest.raises(ValueError) as caught:
        solvers.load_solver(name)
    assert repr(name) in str(caught.value)
    assert "available solvers: flightstream, mock, openfoam, pyfr, qprop, su2, xfoil" in str(
        caught.value
    )


def test_p0000_item2a_reads_root_init_without_loading_an_adapter(tmp_path):
    core = _module("cfdbender.core")
    (tmp_path / "init.toml").write_text(
        "schema_version = 1" + chr(10) + 'solver = "mock"', encoding="utf-8"
    )
    config = core.read_init(tmp_path, available_solvers=("mock",))
    assert config.schema_version == 1
    assert config.solver == "mock"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("schema_version = 1" + chr(10) + 'solver = "missing"', "unknown solver 'missing'"),
        ("schema_version = 1" + chr(10) + "solver = [", "invalid TOML"),
        ('solver = "mock"', "schema_version"),
        ("schema_version = 2" + chr(10) + 'solver = "mock"', "schema_version"),
        ("schema_version = true" + chr(10) + 'solver = "mock"', "schema_version"),
        ("schema_version = 1.0" + chr(10) + 'solver = "mock"', "schema_version"),
        ('schema_version = "1"' + chr(10) + 'solver = "mock"', "schema_version"),
        ("schema_version = 1", "solver"),
        ("schema_version = 1" + chr(10) + "solver = 42", "solver"),
        ("schema_version = 1" + chr(10) + 'solver = ""', "solver"),
        ("schema_version = 1" + chr(10) + 'solver = "mock"' + chr(10) + "dna = 3", "dna"),
    ],
)
def test_p0000_item2a_invalid_init_names_file_problem_and_choices(tmp_path, text, problem):
    core = _module("cfdbender.core")
    source = tmp_path / "init.toml"
    source.write_text(text, encoding="utf-8")
    before = source.read_bytes()
    with pytest.raises(core.WorkspaceError) as caught:
        core.read_init(tmp_path, available_solvers=("mock",))
    message = str(caught.value)
    assert "init.toml" in message
    assert problem in message
    assert "available solvers: mock" in message
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid-encoding"])
def test_p0000_item2a_unreadable_init_is_a_named_refusal(tmp_path, kind):
    core = _module("cfdbender.core")
    source = tmp_path / "init.toml"
    if kind == "directory":
        source.mkdir()
    elif kind == "invalid-encoding":
        source.write_bytes(bytes([255]))
    with pytest.raises(core.WorkspaceError) as caught:
        core.read_init(tmp_path, available_solvers=("mock",))
    assert str(source) in str(caught.value)
    assert "available solvers: mock" in str(caught.value)


def test_p0000_item2a_core_accepts_solver_names_as_data(tmp_path):
    core = _module("cfdbender.core")
    (tmp_path / "init.toml").write_text(
        "schema_version = 1" + chr(10) + 'solver = "example"', encoding="utf-8"
    )
    config = core.read_init(Path(tmp_path), available_solvers=("example",))
    assert config.solver == "example"

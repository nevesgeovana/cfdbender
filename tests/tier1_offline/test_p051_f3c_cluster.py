"""Cluster requests consult declared support before staging or submission."""

import pytest
from test_p0000_item3c_adapter import workspace
from test_p0000_item3f_plan import prepared

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.solvers import load_solver


@pytest.mark.parametrize("command", ["plan", "run"])
@pytest.mark.parametrize("declaration", [None, frozenset()])
def test_p051_f3c_unsupported_cluster_warns_once_and_skips(
    tmp_path, monkeypatch, capsys, command, declaration
):
    prepared(tmp_path)
    adapter = load_solver("mock")
    if declaration is None:
        monkeypatch.delattr(adapter, "CAPABILITIES", raising=False)
    else:
        monkeypatch.setattr(adapter, "CAPABILITIES", declaration, raising=False)

    # A skipped request must not reach staging, local execution or a submit client.
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported cluster staging was attempted")

    monkeypatch.setattr(adapter, "stage", forbidden)
    monkeypatch.setattr(adapter, "execute", forbidden)
    options = ["--workspace", str(tmp_path)]
    assert main(["plan", "study", *options]) == 0
    capsys.readouterr()
    if command == "run":
        assert main([command, "study", *options, "--no-local"]) == 0
        output = capsys.readouterr()
    else:
        assert main([command, "study", *options]) == 0
        output = capsys.readouterr()
    assert "Warnings (1)" in output.err
    assert output.err.count("cluster_staging") == 1
    assert "mock" in output.err and "skipped" in output.err
    assert "staging and submission" in output.err
    assert "would submit:" not in output.out
    assert not (tmp_path / "sims").exists()
    assert RunRegistry(tmp_path).read() == ()


def test_p051_f3c_missing_cluster_support_does_not_change_local_default(tmp_path, monkeypatch):
    workspace(tmp_path)
    adapter = load_solver("mock")
    monkeypatch.setattr(adapter, "CAPABILITIES", frozenset(), raising=False)
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert RunRegistry(tmp_path).read()[0].classification == "converged"

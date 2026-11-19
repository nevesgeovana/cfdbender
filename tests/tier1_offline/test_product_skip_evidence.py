"""Unsupported declarations remain visible in the collect manifest."""

import json

import pytest
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from test_wiring_products import PPROC

from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("supported", [(), ("probes",)])
def test_unsupported_products_are_named_in_manifest(tmp_path, monkeypatch, supported):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(PPROC)
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    adapter = load_solver("mock")
    monkeypatch.setattr(adapter, "PRODUCTS", {k: adapter.PRODUCTS[k] for k in supported})
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    post = tmp_path / "post/study"
    manifest = json.loads((post / "products.json").read_text())
    reasons = "\n".join(item["reason"] for item in manifest["skipped"])
    for product in ("sections", "volume_section", "probes"):
        if product not in supported:
            assert product in reasons and "does not supply" in reasons
    assert not (post / "sections").exists()
    assert not (post / "fields").exists()
    assert list((post / "polars").glob("*.csv"))

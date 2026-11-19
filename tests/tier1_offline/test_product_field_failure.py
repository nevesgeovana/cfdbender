"""A scalar Velocity field must not abort collect or suppress unrelated products."""

import json
import xml.etree.ElementTree as ET

import pytest
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from test_wiring_products import PPROC

from cfdbender.core.plan import workspace_path
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("filename", ["surface.vtu", "volume.vtu"])
def test_bad_velocity_shape_is_isolated(tmp_path, filename):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(PPROC)
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    path = workspace_path(tmp_path, plan.points[0].directory) / filename
    tree = ET.parse(path)
    velocity = tree.find(".//PointData/DataArray[@Name='Velocity']")
    assert velocity is not None
    velocity.set("NumberOfComponents", "1")
    velocity.text = " ".join(velocity.text.split()[::3])
    tree.write(path)
    try:
        status = cli(["collect", "study", "--workspace", str(tmp_path)])
    except Exception as error:
        pytest.fail(f"a malformed product field escaped collect: {error!r}")
    assert status == 0
    post = tmp_path / "post/study"
    manifest = json.loads((post / "products.json").read_text())
    assert any("column" in item["reason"] for item in manifest["skipped"])
    other = "probes" if filename == "surface.vtu" else "sections_sloads"
    assert any(item.get("product") == other for item in manifest["products"])
    assert list((post / "polars").glob("*.csv"))

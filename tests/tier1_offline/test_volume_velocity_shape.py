"""A volume cut cannot publish scalar Velocity as a velocity vector."""

import json
import xml.etree.ElementTree as ET

from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from test_wiring_products import PPROC

from cfdbender.core.plan import workspace_path
from cfdbender.workspace.planning import plan_workspace


def test_scalar_velocity_cannot_be_written_as_a_volume_product(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(PPROC.split("[[probes]]")[0])
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    path = workspace_path(tmp_path, plan.points[0].directory) / "volume.vtu"
    tree = ET.parse(path)
    velocity = tree.find(".//PointData/DataArray[@Name='Velocity']")
    assert velocity is not None
    velocity.set("NumberOfComponents", "1")
    velocity.text = " ".join(velocity.text.split()[::3])
    tree.write(path)
    cli(["collect", "study", "--workspace", str(tmp_path)])
    assert not list((tmp_path / "post/study/fields").glob("**/*.vtk"))
    data = json.loads((tmp_path / "post/study/products.json").read_text())
    assert any("Velocity" in item["reason"] for item in data["skipped"])

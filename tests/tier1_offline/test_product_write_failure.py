"""A blocked product destination must not suppress unrelated products."""

import json

from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from test_wiring_products import PPROC, _su2_window

from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import _cell
from cfdbender.workspace.planning import plan_workspace


def test_blocked_sections_destination_keeps_probes(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(PPROC)
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    post = tmp_path / "post/study"
    post.mkdir(parents=True, exist_ok=True)
    (post / "sections").write_text("unrelated file")
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert (post / "sections").read_text() == "unrelated file"
    assert list((post / "probes").rglob("*.csv"))
    assert list((post / "polars").glob("*.csv"))
    manifest = json.loads((post / "products.json").read_text())
    assert any("sections" in item["product"] for item in manifest["skipped"])


def test_refused_sections_target_keeps_probes(tmp_path):
    root, point, adapter = _su2_window(tmp_path)

    def target(relative):
        if relative.startswith("sections/"):
            raise ValueError("destination is outside workspace")
        return root / relative

    found = point_products(
        root,
        "7001",
        "DP",
        point,
        adapter,
        lead={},
        moment_point=(0, 0, 0),
        cell=_cell,
        target=target,
    )
    assert any(item["product"] == "probes" for item in found.files.values())
    assert any("outside workspace" in reason for _, reason in found.skipped)

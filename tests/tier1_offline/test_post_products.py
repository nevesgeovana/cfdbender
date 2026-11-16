"""collect writes post/<matrix>/products.json and post.log, naming each product and each skip."""

from __future__ import annotations

import json
from dataclasses import replace

from test_matrix_selection import three

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.matrix.collection import collect_workspace
from cfdbender.matrix.post import post_workspace
from cfdbender.workspace.planning import plan_workspace


def products(root):
    return json.loads((root / "post/study/products.json").read_bytes())


def test_a_complete_collect_lists_every_product_and_skips_none(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    data = products(tmp_path)
    assert data["matrix"] == "inputs/matrices/study.fs" and data["solver"] == "mock"
    assert data["complete"] is True
    assert [s["product"] for s in data["skipped"]] == [
        f"polars/P{p}-ROTOR" for p in ("1001", "1002", "1003")
    ]
    paths = sorted(p["path"] for p in data["products"])
    assert paths == sorted(
        [
            "post/study/campaign_summary.json",
            *(
                f"post/study/polars/{name}"
                for pol in ("1001", "1002", "1003")
                for name in sorted(
                    [f"FAM-{k}-P{pol}.csv" for k in ("DIM", "ACOEF")]
                    + [
                        f"P{pol}-{g}-{k}.csv"
                        for g in ("TOTAL", "AIRFRAME")
                        for k in ("DIM", "ACOEF")
                    ]
                )
            ),
        ]
    )
    assert all((tmp_path / p["path"]).is_file() for p in data["products"])
    log = (tmp_path / "post/study/post.log").read_text(encoding="utf-8")
    assert "19 product(s) written, 3 skipped" in log


def test_a_point_never_run_is_a_named_skip_with_its_reason(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path), "--sims", "1001"]) == 0
    plan = plan_workspace(tmp_path, "study")
    RunRegistry(tmp_path).record(replace(plan, points=plan.points[2:3]))
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 3
    data = products(tmp_path)
    assert data["complete"] is False
    points = [s for s in data["skipped"] if s["product"] == "result"]
    assert [s["point"] for s in points] == ["1002/DP-AL+000"]
    data["skipped"] = points
    assert "PLANNED" in data["skipped"][0]["reason"]
    log = (tmp_path / "post/study/post.log").read_text(encoding="utf-8")
    assert "skipped 1002/DP-AL+000: " in log and "PLANNED" in log


def test_post_rebuilds_the_same_bytes_without_running_anything(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    first = collect_workspace(tmp_path, "study")
    before = {p: p.read_bytes() for p in (tmp_path / "post/study").rglob("*") if p.is_file()}
    registry = (tmp_path / "runs.json").read_bytes()
    second = post_workspace(tmp_path, "study")
    assert second.exit_code == first.exit_code == 0
    after = {p: p.read_bytes() for p in (tmp_path / "post/study").rglob("*") if p.is_file()}
    assert after == before and (tmp_path / "runs.json").read_bytes() == registry

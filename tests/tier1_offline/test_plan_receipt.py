"""A plan receipt records the matrix digest and each point's inputs; a run needs a fresh one."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_matrix_selection import three

from cfdbender.matrix.receipt import receipt_path, require_receipt, write_receipt
from cfdbender.matrix.selection import select_plan
from cfdbender.workspace.planning import plan_workspace


def test_the_receipt_records_the_matrix_digest_and_every_points_inputs(tmp_path):
    three(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    path = write_receipt(plan)
    assert path == tmp_path / "post/study/plan.json" == receipt_path(tmp_path, "study")
    data = json.loads(path.read_bytes())
    matrix = (tmp_path / "inputs/matrices/study.fs").read_bytes()
    assert data["matrix_sha256"] == hashlib.sha256(matrix).hexdigest()
    assert data["solver"] == "mock" and data["matrix"] == "inputs/matrices/study.fs"
    assert sorted(p["point_id"] for p in data["points"]) == sorted(p.point_id for p in plan.points)
    first = data["points"][0]
    assert first["inputs_sha256"] == dict(plan.points[0].inputs_sha256)
    before = path.read_bytes()
    write_receipt(plan)
    assert path.read_bytes() == before


def test_a_run_without_a_receipt_is_refused_naming_the_plan_command(tmp_path):
    three(tmp_path)
    with pytest.raises(ValueError, match=r"no plan for this matrix.*cfd-matrix plan"):
        require_receipt(plan_workspace(tmp_path, "study"))


def test_a_planned_matrix_passes(tmp_path):
    three(tmp_path)
    write_receipt(plan_workspace(tmp_path, "study"))
    require_receipt(plan_workspace(tmp_path, "study"))


def test_a_changed_matrix_makes_the_receipt_stale(tmp_path):
    three(tmp_path)
    write_receipt(plan_workspace(tmp_path, "study"))
    matrix = tmp_path / "inputs/matrices/study.fs"
    matrix.write_text(matrix.read_text(encoding="ascii") + "1004|1|001|001|001|ALPHA:0|\n")
    with pytest.raises(ValueError, match=r"matrix changed since it was planned.*Plan it again"):
        require_receipt(plan_workspace(tmp_path, "study"))


def test_a_changed_input_makes_the_receipt_stale(tmp_path):
    three(tmp_path)
    write_receipt(plan_workspace(tmp_path, "study"))
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text(encoding="ascii") + "# edited\n")
    with pytest.raises(ValueError, match=r"input changed since it was planned.*s001"):
        require_receipt(plan_workspace(tmp_path, "study"))


def test_planning_a_selection_keeps_the_other_points_of_the_receipt(tmp_path):
    three(tmp_path)
    full = plan_workspace(tmp_path, "study")
    write_receipt(select_plan(full, ["1001"], []))
    require_receipt(select_plan(full, ["1001"], []))
    with pytest.raises(ValueError, match="1002"):
        require_receipt(select_plan(full, ["1002"], []))
    write_receipt(select_plan(full, ["1002"], []))
    require_receipt(select_plan(full, ["1001", "1002"], []))


def test_an_unreadable_receipt_is_refused(tmp_path):
    three(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    path = receipt_path(tmp_path, "study")
    path.parent.mkdir(parents=True)
    path.write_text("[1]", encoding="ascii")
    with pytest.raises(ValueError, match="not a plan receipt"):
        require_receipt(plan)


def test_a_run_and_a_forced_rerun_leave_the_receipt_valid(tmp_path):
    from cfdbender.matrix.cli import main

    three(tmp_path)
    write_receipt(plan_workspace(tmp_path, "study"))
    assert main(["run", "study", "--workspace", str(tmp_path), "--sims", "1001"]) == 0
    rerun = ["run", "study", "--workspace", str(tmp_path), "--force-rerun", "1001/DP-AL+000"]
    assert main(rerun) == 0
    require_receipt(plan_workspace(tmp_path, "study"))

"""Gap 13: a matrix lives in the root or in inputs/matrices/; artifact ids may be named."""

from __future__ import annotations

import json
import shutil

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.plan import PlanError
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.workspace.planning import plan_workspace


def move_matrix_to_root(root):
    shutil.move(root / "inputs/matrices/study.fs", root / "study.fs")


def test_a_matrix_in_the_workspace_root_plans_runs_and_collects(tmp_path, capsys):
    workspace(tmp_path)
    move_matrix_to_root(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    assert plan.matrix == "study.fs"
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert {r.matrix for r in RunRegistry(tmp_path).read()} == {"study.fs"}
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "post/study/campaign_summary.json").is_file()
    capsys.readouterr()


def test_the_matrix_home_inputs_matrices_still_works(tmp_path):
    workspace(tmp_path)
    assert plan_workspace(tmp_path, "study.fs").matrix == "inputs/matrices/study.fs"


def test_a_matrix_in_both_homes_is_ambiguous_and_refused(tmp_path):
    workspace(tmp_path)
    shutil.copy(tmp_path / "inputs/matrices/study.fs", tmp_path / "study.fs")
    with pytest.raises(PlanError, match=r"study.*both"):
        plan_workspace(tmp_path, "study")
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 2


def test_a_missing_matrix_names_both_homes(tmp_path):
    workspace(tmp_path)
    with pytest.raises(
        PlanError, match=r"inputs/matrices/.*workspace root|workspace root.*inputs/matrices/"
    ):
        plan_workspace(tmp_path, "nothere")


def named(root):
    workspace(root)
    for kind, old, new in (
        ("setups", "s001", "sxbbase"),
        ("references", "r001", "rwing"),
        ("pproc", "p001", "pforces"),
    ):
        (root / f"inputs/{kind}/{old}.toml").rename(root / f"inputs/{kind}/{new}.toml")
    path = root / "inputs/matrices/study.fs"
    path.write_text("POL|RUN|SET|REF|PPROC\n1001|1|sxbbase|rwing|pforces\n", encoding="ascii")


def test_named_artifact_ids_plan_run_and_record(tmp_path):
    named(tmp_path)
    point = plan_workspace(tmp_path, "study").points[0]
    assert (point.setup_id, point.reference_id, point.post_id) == ("sxbbase", "rwing", "pforces")
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    record = RunRegistry(tmp_path).read()[0]
    assert record.point.setup_id == "sxbbase" and record.status == "COMPLETED"
    assert "inputs/setups/sxbbase.toml" in record.point.inputs_sha256


def test_numeric_and_suffixed_ids_keep_their_meaning(tmp_path):
    workspace(tmp_path)
    path = tmp_path / "inputs/matrices/study.fs"
    for cell in ("001", "s001", "s001.toml"):
        path.write_text(f"POL|RUN|SET|REF|PPROC\n1001|1|{cell}|001|001\n", encoding="ascii")
        assert plan_workspace(tmp_path, "study").points[0].setup_id == "s001"


@pytest.mark.parametrize("cell", ["wing", "r001", "s", "s/../x", "sx y"])
def test_an_id_of_the_wrong_kind_or_shape_is_refused(tmp_path, cell):
    workspace(tmp_path)
    path = tmp_path / "inputs/matrices/study.fs"
    path.write_text(f"POL|RUN|SET|REF|PPROC\n1001|1|{cell}|001|001\n", encoding="ascii")
    with pytest.raises(Exception, match="SET"):
        plan_workspace(tmp_path, "study")


def test_json_receipt_of_a_named_plan_is_written(tmp_path):
    named(tmp_path)
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    assert receipt["matrix"] == "inputs/matrices/study.fs"

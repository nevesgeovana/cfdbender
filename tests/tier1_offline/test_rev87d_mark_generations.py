"""R2-04: marks judge one execution, not every future run of the point."""

from cli_route import main
from test_cli_verbs import FIRST, ran


def test_a_failed_mark_does_not_block_a_verdict_on_a_rerun(tmp_path):
    ran(tmp_path, "iteration_limit")
    ws = ["--workspace", str(tmp_path)]
    assert main(["mark-failed", "--sims", "1001", "--apply", *ws]) == 0
    assert main(["run", "study", "--force-rerun", FIRST, *ws]) == 0
    assert (
        main(["mark-converged", "--sims", "1001", "--reason", "reviewed new run", "--apply", *ws])
        == 0
    )
    assert main(["collect", "study", *ws]) == 0


def test_a_previous_converged_mark_does_not_excuse_a_new_record_mismatch(tmp_path):
    ran(tmp_path, "iteration_limit")
    ws = ["--workspace", str(tmp_path)]
    assert (
        main(["mark-converged", "--sims", "1001", "--reason", "reviewed old run", "--apply", *ws])
        == 0
    )
    assert main(["run", "study", "--force-rerun", FIRST, *ws]) == 0
    manifest = tmp_path / "runs.json"
    # Change only the live record, keeping the archived run and its mark intact.
    import json

    data = json.loads(manifest.read_bytes())
    next(r for r in data["records"] if r["point_id"] == FIRST)["classification"] = "converged"
    manifest.write_text(json.dumps(data))
    assert main(["collect", "study", *ws]) == 1

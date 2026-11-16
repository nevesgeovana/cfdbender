"""Plan and submit rows using hpc: alpha and hpc: beta."""

import json

from test_p0000_item3f_collect import outputs
from test_p0000_item3f_plan import cli, prepared, profile
from test_p053_rowcl_cluster import matrix

from cfdbender.core.registry import RunRegistry


def test_p053_rowhpc_two_profiles_record_and_collect(tmp_path, capsys):
    prepared(tmp_path, fields={"selected": "alpha", "queue": "{cluster}"})
    profile(tmp_path, "beta", fields={"selected": "beta", "queue": "{cluster}"})
    matrix(tmp_path, ["HPC: h001 / cluster: one", "hpc: beta / cluster: two"])
    assert cli(tmp_path) == 0
    assert '"selected": "alpha"' in capsys.readouterr().out
    assert cli(tmp_path, "plan", "--hpc", "beta") == 0
    output = capsys.readouterr().out
    assert '"selected": "alpha"' in output and '"selected": "beta"' in output
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    assert [p["profile"] for p in receipt["points"]] == [
        "inputs/hpc/h001.toml",
        "inputs/hpc/beta.toml",
    ]
    assert cli(tmp_path, "run", "--hpc", "beta") == 0
    records = RunRegistry(tmp_path).read()
    assert [r.submission["profile"] for r in records] == [
        "inputs/hpc/h001.toml",
        "inputs/hpc/beta.toml",
    ]
    for record in records:
        assert "hpc" not in (tmp_path / record.submission["config_path"]).read_text().lower()
        outputs(tmp_path, record.point)
    assert cli(tmp_path, "collect") == 0


def test_p053_rowhpc_unknown_refused_and_local_ignores(tmp_path, capsys):
    prepared(tmp_path)
    profile(tmp_path, "beta")
    matrix(tmp_path, ["hpc: missing"])
    assert cli(tmp_path) == 2
    error = capsys.readouterr().err
    assert "missing" in error and "h001" in error and "beta" in error
    assert cli(tmp_path, "plan", "--local") == 0


def test_p053_rowhpc_default_and_changed_profile(tmp_path, capsys):
    prepared(tmp_path)
    profile(tmp_path, "beta")
    matrix(tmp_path, ["hpc: h001", "-"])
    assert cli(tmp_path) == 2
    assert cli(tmp_path, "plan", "--hpc", "beta") == 0
    assert cli(tmp_path, "run", "--hpc", "h001") == 2
    assert "profile changed" in capsys.readouterr().err


def test_rhp_1_legacy_submission_without_profile_collects(tmp_path):
    from test_p0000_item3f_registry import submitted

    registry, point, _ = submitted(tmp_path)
    document = json.loads(registry.path.read_text())
    del document["records"][0]["submission"]["profile"]
    registry.path.write_text(json.dumps(document))
    outputs(tmp_path, point)
    assert cli(tmp_path, "collect") == 0
    assert registry.read()[0].status == "COMPLETED"
    assert "profile" not in registry.read()[0].submission


def test_rhp_2_profile_bytes_changed_after_plan_refused(tmp_path, capsys):
    path = prepared(tmp_path, fields={"queue": "before"})
    matrix(tmp_path, ["hpc: h001"])
    assert cli(tmp_path) == 0
    path.write_text(path.read_text().replace("before", "after"))
    assert cli(tmp_path, "run") == 2
    assert "profile" in capsys.readouterr().err.lower()
    assert not (tmp_path / "runs.json").exists()


def test_rhp_3_old_hpc_receipt_refusal_has_migration_notice(tmp_path, capsys):
    from pathlib import Path

    prepared(tmp_path)
    assert cli(tmp_path) == 0
    receipt_path = tmp_path / "post/study/plan.json"
    receipt = json.loads(receipt_path.read_text())
    for point in receipt["points"]:
        point.pop("profile", None)
        point.pop("profile_sha256", None)
    receipt_path.write_text(json.dumps(receipt))
    assert cli(tmp_path, "run") == 2
    assert "plan it again" in capsys.readouterr().err.lower()
    changelog = (Path(__file__).resolve().parents[2] / "CHANGELOG.md").read_text()
    release = changelog.split("## [0.6.0]", 1)[1].split("## [0.5.2]", 1)[0]
    assert "### Migration" in release
    assert "HPC plan receipts" in release and "plan again" in release


def test_rhp_4_group_rejects_different_row_profiles(tmp_path):
    import pytest

    from cfdbender.hpc import plan_submission, resolve_hpc_profile

    prepared(tmp_path, fields={"queue": "default"})
    profile(tmp_path, "beta")
    chosen = resolve_hpc_profile(tmp_path / "inputs", "h001", solver_versions={"mock-1"})
    points = [
        {"row_id": "1001", "point": "1", "workdir": str(tmp_path), "hpc": "h001"},
        {"row_id": "1002", "point": "2", "workdir": str(tmp_path), "hpc": "beta"},
    ]
    with pytest.raises(ValueError, match="profile"):
        plan_submission(points, chosen)

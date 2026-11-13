"""Record, query and storage verbs of cfd-matrix and cfd-workspace archive, on the mock solver."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.workspace.cli import main as workspace_main

FIRST = "1001/DP-NOCOND"
SECOND = "1002/DP-NOCOND"


def grown(root: Path, outcome: str = "converged") -> Path:
    workspace(root, outcome)
    path = root / "inputs/matrices/study.fs"
    path.write_text(path.read_text().rstrip() + "\n1002|1|001|001|001\n", encoding="ascii")
    return root


def ran(root: Path, outcome: str = "converged") -> Path:
    grown(root, outcome)
    assert main(["run", "study", "--workspace", str(root)]) == 0
    return root


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def out(capsys) -> str:
    return capsys.readouterr().out


def test_status_lists_sims_and_points_and_filters(tmp_path, capsys):
    ran(tmp_path)
    assert main(["status", "--workspace", str(tmp_path)]) == 0
    text = out(capsys)
    assert "1001" in text and "1002" in text and "COMPLETED=1" in text
    assert (
        main(["status", "--points", "--sims", "1002", "--json", "--workspace", str(tmp_path)]) == 0
    )
    rows = json.loads(out(capsys))
    assert [r["point"] for r in rows] == [SECOND]
    assert main(["status", "--failed", "--workspace", str(tmp_path)]) == 0
    assert "1001" not in out(capsys)
    assert main(["status", "--status", "CONVER*", "--csv", "--workspace", str(tmp_path)]) == 0
    assert out(capsys).splitlines()[0] == "sim,matrix,points,status,failed"


def test_read_only_verbs_change_no_byte_and_no_mtime(tmp_path, capsys):
    ran(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path), "--force-rerun", FIRST]) == 0
    capsys.readouterr()
    before = snapshot(tmp_path)
    run_id = next(iter(RunRegistry(tmp_path).history()))["archive"]
    archived = f"{FIRST}@{Path(run_id).name}"
    for command in (
        ["status"],
        ["show", "1001"],
        ["history", FIRST],
        ["log"],
        ["log", "--problems"],
        ["trace", "sims/sim_1001/datapoints/DP-NOCOND/history.csv"],
        ["diff", FIRST, archived],
    ):
        code = main([*command, "--workspace", str(tmp_path)])
        assert code in (0, 1), command
    assert snapshot(tmp_path) == before


def test_show_and_history_name_the_archived_run_and_diff_compares_runs(tmp_path, capsys):
    ran(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path), "--force-rerun", FIRST]) == 0
    capsys.readouterr()
    assert main(["history", "1001", "--json", "--workspace", str(tmp_path)]) == 0
    rows = json.loads(out(capsys))
    assert [r["kind"] for r in rows] == ["current", "archived"]
    archived = rows[1]["run"]
    assert archived.startswith(f"{FIRST}@")
    assert main(["show", archived, "--json", "--workspace", str(tmp_path)]) == 0
    assert json.loads(out(capsys))[0]["point_id"] == FIRST
    assert main(["diff", FIRST, archived, "--json", "--workspace", str(tmp_path)]) == 1
    fields = {r["field"] for r in json.loads(out(capsys))}
    assert "started_at" in fields and "inputs_sha256" not in fields
    assert main(["diff", FIRST, FIRST, "--workspace", str(tmp_path)]) == 0
    assert main(["show", "9999", "--workspace", str(tmp_path)]) == 2


def test_trace_follows_a_product_to_its_run_and_a_post_product_to_its_matrix(tmp_path, capsys):
    ran(tmp_path)
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    product = "sims/sim_1001/datapoints/DP-NOCOND/history.csv"
    assert main(["trace", product, "--json", "--workspace", str(tmp_path)]) == 0
    row = json.loads(out(capsys))[0]
    assert row["run"] == FIRST and row["product_sha256"]
    assert (
        main(["trace", "post/study/campaign_summary.json", "--json", "--workspace", str(tmp_path)])
        == 0
    )
    assert {r["run"] for r in json.loads(out(capsys))} == {FIRST, SECOND}
    assert main(["trace", "post/other/x.csv", "--workspace", str(tmp_path)]) == 2


def test_mark_failed_previews_then_applies_with_an_archived_manifest(tmp_path, capsys):
    ran(tmp_path)
    before = (tmp_path / "runs.json").read_bytes()
    args = ["mark-failed", "--sims", "1001", "--reason", "wrong mesh", "--workspace", str(tmp_path)]
    assert main(args) == 0
    assert "would mark" in out(capsys) and (tmp_path / "runs.json").read_bytes() == before
    assert main([*args, "--apply"]) == 0
    capsys.readouterr()
    record = {r.point_id: r for r in RunRegistry(tmp_path).read()}[FIRST]
    assert record.classification == "failed" and record.error == "FAILED_MARKED: wrong mesh"
    saved = next((tmp_path / "archive").iterdir())
    assert (saved / "runs.json").read_bytes() == before
    marks = json.loads((tmp_path / "marks.json").read_text())["entries"]
    assert marks[0]["verdict"] == "failed" and marks[0]["from_classification"] == "converged"
    assert main(["log", "--problems", "--json", "--workspace", str(tmp_path)]) == 0
    assert any(e["stage"] == "mark" for e in json.loads(out(capsys)))
    assert main(["mark-failed", "--sims", "9", "--workspace", str(tmp_path)]) == 2


def test_mark_converged_needs_a_reason_a_completed_point_and_no_failed_mark(tmp_path, capsys):
    ran(tmp_path, "iteration_limit")
    base = ["--sims", "1001", "--workspace", str(tmp_path)]
    with pytest.raises(SystemExit):
        main(["mark-converged", *base])
    assert main(["mark-converged", *base, "--reason", "read the history", "--apply"]) == 0
    assert {r.point_id: r for r in RunRegistry(tmp_path).read()}[
        FIRST
    ].classification == "converged"
    assert main(["mark-failed", *base, "--apply"]) == 0
    capsys.readouterr()
    assert main(["mark-converged", *base, "--reason", "again", "--apply"]) == 2
    assert "marked failed" in capsys.readouterr().err


def test_mark_converged_refuses_a_point_that_has_not_completed(tmp_path, capsys):
    from cfdbender.workspace.planning import plan_workspace

    grown(tmp_path)
    RunRegistry(tmp_path).record(plan_workspace(tmp_path, "study"))
    assert (
        main(
            [
                "mark-converged",
                "--sims",
                "1001",
                "--reason",
                "x",
                "--workspace",
                str(tmp_path),
                "--apply",
            ]
        )
        == 2
    )
    assert "PLANNED, not COMPLETED" in capsys.readouterr().err


def test_restore_brings_back_the_archived_manifest(tmp_path, capsys):
    ran(tmp_path)
    original = (tmp_path / "runs.json").read_bytes()
    assert main(["mark-failed", "--sims", "1001", "--apply", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "runs.json").read_bytes() != original
    assert main(["restore", "runs", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "runs.json").read_bytes() != original, "a preview changes nothing"
    assert main(["restore", "runs", "--apply", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "runs.json").read_bytes() == original
    assert main(["restore", "runs", "--stamp", "nope", "--workspace", str(tmp_path)]) == 2


def test_rebuild_writes_records_from_the_sim_folders(tmp_path, capsys):
    ran(tmp_path)
    expected = {r.point_id: r.classification for r in RunRegistry(tmp_path).read()}
    (tmp_path / "runs.json").unlink()
    base = ["rebuild", "--workspace", str(tmp_path)]
    assert main(base) == 0
    assert not (tmp_path / "runs.json").exists(), "a preview writes nothing"
    assert main([*base, "--apply"]) == 0
    rebuilt = {r.point_id: r for r in RunRegistry(tmp_path).read()}
    assert {k: v.classification for k, v in rebuilt.items()} == expected
    assert all(r.status == "COMPLETED" for r in rebuilt.values())
    assert main([*base, "--out", "rebuilt.json", "--all-sims", "--apply"]) == 0
    assert (tmp_path / "rebuilt.json").is_file()
    # --inputs-from is FlightStream-only: it feeds a script comparison cfdbender has no
    # counterpart for, so the parser refuses it rather than accept and ignore it.
    with pytest.raises(SystemExit) as refused:
        main([*base, "--inputs-from", str(tmp_path)])
    assert refused.value.code == 2


def test_delete_sims_refuses_stale_products_then_deletes_folder_and_record(tmp_path, capsys):
    ran(tmp_path)
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    base = ["delete-sims", "--sims", "1001", "--workspace", str(tmp_path)]
    assert main([*base, "--apply"]) == 2
    assert "--matrix-products" in capsys.readouterr().err
    assert main([*base, "--matrix-products", "points-only"]) == 0
    assert (tmp_path / "sims/sim_1001").is_dir(), "a preview deletes nothing"
    assert main([*base, "--matrix-products", "points-only", "--apply"]) == 0
    assert not (tmp_path / "sims/sim_1001").exists()
    assert [r.point_id for r in RunRegistry(tmp_path).read()] == [SECOND]
    capsys.readouterr()
    assert main(["history", FIRST, "--json", "--workspace", str(tmp_path)]) == 0
    assert json.loads(out(capsys))[-1]["kind"] == "deleted"
    assert main(["delete-sims", "--sims", "7", "--workspace", str(tmp_path)]) == 2


def test_delete_sims_regenerate_rebuilds_the_products_without_the_sim(tmp_path, capsys):
    ran(tmp_path)
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert (
        main(
            [
                "delete-sims",
                "--sims",
                "1001",
                "--matrix-products",
                "regenerate",
                "--archive",
                "--apply",
                "--workspace",
                str(tmp_path),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_text())
    assert [p["point_id"] for p in summary["points"]] == [SECOND]
    assert list((tmp_path / "archive").glob("*/post/study/campaign_summary.json"))


def test_space_in_use_reports_three_groupings_and_records_the_call(tmp_path, capsys):
    ran(tmp_path)
    assert main(["space-in-use", "--top", "2", "--workspace", str(tmp_path)]) == 0
    text = out(capsys)
    assert "by folder" in text and "by sim" in text and "by extension" in text
    assert "sim_1001" in text
    journal = json.loads((tmp_path / "storage_management.json").read_text())["entries"]
    assert journal[0]["command"] == "space-in-use"
    with pytest.raises(SystemExit):
        main(["space-in-use", "--top", "0", "--workspace", str(tmp_path)])


def test_free_space_deletes_named_extensions_but_keeps_what_a_record_names(tmp_path, capsys):
    ran(tmp_path)
    folder = tmp_path / "sims/sim_1001/datapoints/DP-NOCOND"
    (folder / "big.vtu").write_bytes(b"x" * 100)
    recipes = tmp_path / "inputs/management"
    recipes.mkdir(parents=True)
    (recipes / "m1.toml").write_text('[[delete_extensions]]\nextensions = [".vtu", ".csv"]\n')
    base = ["free-space", "m1", "--workspace", str(tmp_path)]
    assert main([*base, "--list"]) == 0
    text = out(capsys)
    assert "delete sims/sim_1001/datapoints/DP-NOCOND/big.vtu" in text
    assert "keep sims/sim_1001/datapoints/DP-NOCOND/history.csv" in text
    assert (folder / "big.vtu").is_file()
    assert main([*base, "--apply"]) == 0
    assert not (folder / "big.vtu").exists() and (folder / "history.csv").is_file()
    assert main(["free-space", "m9", "--workspace", str(tmp_path)]) == 2


def test_sync_adds_missing_records_and_files_and_keeps_main_on_conflict(tmp_path, capsys):
    main_root, other = tmp_path / "main", tmp_path / "other"
    ran(other)
    workspace(main_root)
    (main_root / "inputs").mkdir(exist_ok=True)
    (main_root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n'
    )
    base = ["sync", "all", "--workspace", str(main_root)]
    assert main(base) == 0
    assert "would add" in out(capsys) and not (main_root / "runs.json").exists()
    assert main([*base, "--apply"]) == 0
    assert {r.point_id for r in RunRegistry(main_root).read()} == {FIRST, SECOND}
    assert (main_root / "sims/sim_1002/datapoints/DP-NOCOND/history.csv").is_file()
    assert main(["sync", "runs", "--from", "nope", "--workspace", str(main_root)]) == 2


def test_workspace_archive_moves_the_sim_and_its_records(tmp_path, capsys):
    ran(tmp_path)
    assert workspace_main(["archive", "1001", "--workspace", str(tmp_path)]) == 0
    assert not (tmp_path / "sims/sim_1001").exists()
    assert [r.point_id for r in RunRegistry(tmp_path).read()] == [SECOND]
    archived = next((tmp_path / "archive").iterdir())
    assert (archived / "sims/sim_1001/datapoints/DP-NOCOND/history.csv").is_file()
    assert (
        RunRegistry(tmp_path).history()[0]["archive"] == archived.relative_to(tmp_path).as_posix()
    )
    assert workspace_main(["archive", "5555", "--workspace", str(tmp_path)]) == 2


def test_runs_option_reads_another_manifest(tmp_path, capsys):
    ran(tmp_path)
    (tmp_path / "other.json").write_bytes((tmp_path / "runs.json").read_bytes())
    (tmp_path / "runs.json").unlink()
    capsys.readouterr()
    assert main(["status", "--workspace", str(tmp_path)]) == 0
    assert "no recorded sim yet" in out(capsys), "runs.json is absent"
    assert main(["status", "--runs", "other.json", "--workspace", str(tmp_path)]) == 0
    assert "1001" in out(capsys)
    assert main(["status", "--runs", "../x.json", "--workspace", str(tmp_path)]) == 2


def test_collect_archive_keeps_a_copy_of_each_replaced_product(tmp_path):
    ran(tmp_path)
    assert main(["collect", "study", "--archive", "--workspace", str(tmp_path)]) == 0
    assert not (tmp_path / "archive").exists() or not list(
        (tmp_path / "archive").glob("*/post/*")
    ), "nothing was replaced the first time"
    summary = tmp_path / "post/study/campaign_summary.json"
    old = summary.read_bytes()
    assert main(["mark-failed", "--sims", "1001", "--apply", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--archive", "--workspace", str(tmp_path)]) == 1
    assert summary.read_bytes() != old
    copies = list((tmp_path / "archive").glob("*/post/study/campaign_summary.json"))
    assert len(copies) == 1 and copies[0].read_bytes() == old


def test_collect_watch_stops_after_the_rounds_it_was_given(tmp_path, capsys):
    from cfdbender.workspace.planning import plan_workspace

    grown(tmp_path)
    RunRegistry(tmp_path).record(plan_workspace(tmp_path, "study"))
    code = main(
        [
            "collect",
            "study",
            "--watch",
            "--rounds",
            "2",
            "--watch-interval",
            "0",
            "--interval",
            "0",
            "--workspace",
            str(tmp_path),
        ]
    )
    assert code == 3
    assert out(capsys).count("outstanding 2") == 2
    with pytest.raises(SystemExit):
        main(["collect", "study", "--rounds", "0", "--workspace", str(tmp_path)])


def test_console_levels_and_no_local(tmp_path, capsys):
    grown(tmp_path)
    base = ["plan", "study", "--workspace", str(tmp_path)]
    assert main(base) == 0
    normal = out(capsys)
    assert main([*base, "--log-level", "quiet"]) == 0
    quiet = out(capsys)
    assert "would:" in normal and "would:" not in quiet and "2 point(s)" in quiet
    assert main([*base, "--verbose"]) == 0
    assert normal in out(capsys)
    with pytest.raises(SystemExit):
        main([*base, "--log-level", "loud"])
    assert main([*base, "--verbose", "--log-level", "quiet"]) == 2
    assert (
        main(["run", "study", "--workspace", str(tmp_path), "--no-local", "--log-level", "quiet"])
        == 0
    )
    assert "COMPLETED" not in out(capsys)


def test_warning_off_hides_the_console_warning_and_keeps_it_in_the_log(tmp_path, capsys):
    ran(tmp_path)
    legacy = tmp_path / "inputs/geometry"
    legacy.mkdir()
    (legacy / "wing.dat").write_text("x")
    matrix_file = tmp_path / "inputs/matrices/study.fs"
    matrix_file.write_text(
        "POL|RUN|SET|REF|PPROC|GEOMETRY\n1001|1|001|001|001|wing.dat\n", encoding="ascii"
    )
    from cfdbender.core import mesh

    mesh._warned.clear()
    assert main(["--warning-off", "plan", "study", "--workspace", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "Warnings" not in captured.err
    assert "legacy mesh folder" in (tmp_path / "logs/warnings.log").read_text()
    mesh._warned.clear()
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "Warnings (1)" in err


def test_init_and_archive_take_the_console_options(tmp_path, capsys):
    assert (
        workspace_main(
            [
                "--warning-off",
                "init",
                str(tmp_path / "c"),
                "--solver",
                "mock",
                "--log-level",
                "quiet",
            ]
        )
        == 0
    )
    assert out(capsys).strip() == f"campaign workspace ready at {(tmp_path / 'c').resolve()}"


def test_verbs_are_read_only_when_a_lock_is_held_elsewhere(tmp_path):
    ran(tmp_path)
    (tmp_path / "runs.json.lock").write_text("1")
    assert main(["status", "--workspace", str(tmp_path)]) == 0
    assert main(["mark-failed", "--sims", "1001", "--apply", "--workspace", str(tmp_path)]) == 2
    assert os.path.exists(tmp_path / "runs.json.lock"), "a lock held elsewhere is never removed"


def _link_folder(target: Path, link: Path) -> None:
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    link.symlink_to(target, target_is_directory=True)


def test_free_space_never_deletes_through_a_linked_folder(tmp_path, capsys):
    root = tmp_path / "ws"
    root.mkdir()
    ran(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.vtu").write_bytes(b"y" * 10)
    _link_folder(outside, root / "sims/sim_1002/link")
    (root / "inputs/management").mkdir(parents=True)
    recipe = '[[delete_extensions]]\nextensions = [".vtu"]\n'
    (root / "inputs/management/m1.toml").write_text(recipe)
    assert main(["free-space", "m1", "--apply", "--list", "--workspace", str(root)]) == 0
    assert (outside / "data.vtu").is_file()
    assert "link/data.vtu" not in out(capsys)


@pytest.mark.parametrize("sim", ["1001/../../../victim", "1001/../../inputs", "..", "1001."])
def test_delete_sims_never_deletes_outside_its_own_sim_folder(tmp_path, sim):
    root = tmp_path / "ws"
    root.mkdir()
    ran(root)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x")
    before = snapshot(root)
    code = main(["delete-sims", "--sims", sim, "--apply", "--workspace", str(root)])
    assert code == 2
    assert (victim / "keep.txt").is_file()
    assert snapshot(root) == before

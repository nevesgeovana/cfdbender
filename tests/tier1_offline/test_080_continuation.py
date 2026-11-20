"""A finished (or killed) time-accurate point continues with more revolutions: ``run --continue``.

The requirement: raise REVOLUTIONS in the matrix row of a finished ``unsteady_rotor``
point and run again; the point continues from its last time step instead of restarting. SU2
8.6.0 continues a dual-time run as a restart at ``RESTART_ITER = R``, reading the two files of
the iterations before it (``R - 2`` and ``R - 1`` under second order) of every zone, which a
finished run leaves as its last two (a real two-zone run, 9081, kept
``restart_flow_<zone>_00012.dat`` and ``_00013.dat`` with ``TIME_ITER= 14``). The SU2 stand-in
(``fixtures/su2/fake_su2_transient.py``) models that file behaviour, so the generated
configurations are checked by running them.
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
from test_su2_precursor import mrf_workspace
from test_su2_time_restart import CLOCK, call, collected, keys, matrix, run_ta, workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.solvers import load_solver
from cfdbender.solvers.su2._config import config, extra_files
from cfdbender.solvers.su2._products import product_inputs
from cfdbender.solvers.su2._time_history import time_history
from cfdbender.workspace.planning import plan_workspace

TWO = CLOCK.replace("REVOLUTIONS: 1", "REVOLUTIONS: 2")


def grow(root, var):
    """Edit the row of ``ta`` and plan it again, as a user does before ``run --continue``."""
    matrix(root, "ta", 7002, "unsteady_rotor", var)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert call(["plan", "ta", "--workspace", str(root)]) == 0


def cont(root, *options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(["run", "ta", "--local", "--continue", "--workspace", str(root), *options])


def finished(root, var=CLOCK):
    """A cold time-accurate point run to its end; its record and folder."""
    return run_ta(workspace(root, var, restart=False))


# ---- the continuation runs on from the last step ------------------------------------------


def test_a_finished_point_continues_from_its_last_step_with_more_revolutions(tmp_path):
    record, folder = finished(tmp_path)
    assert record.classification != "failed", record.error
    # A cold 36-step run ends at time iteration 35 and keeps its restart files 34 and 35.
    assert (folder / "restart_flow_00034.dat").is_file()
    grow(tmp_path, TWO)
    assert cont(tmp_path) == 0
    cfg = keys(folder / "su2.cfg")
    assert (cfg["RESTART_SOL"], cfg["RESTART_ITER"], cfg["TIME_ITER"]) == ("YES", "36", "72")
    assert cfg["SOLUTION_FILENAME"] == "restart_flow.dat"
    assert cfg["READ_BINARY_RESTART"] == "YES"
    # The first segment's configuration, history and logs are kept, not overwritten.
    kept = folder / "continued" / "1"
    for name in ("su2.cfg", "history.csv", "stdout.log", "stderr.log", "execution.json"):
        assert (kept / name).is_file(), name
    assert keys(kept / "su2.cfg")["TIME_ITER"] == "36"
    assert (folder / "history_00036.csv").is_file()
    # The collect joins both segments' histories in step order.
    assert list(time_history(folder).steps) == list(range(1, 73))
    current = RunRegistry(tmp_path).read()[-1]
    assert current.classification != "failed", current.error
    result = collected(tmp_path, current)
    average = result["time_average"]
    # LAST_REVS_AVG 0.25 counts back from the last step of the whole run: steps 64 to 72.
    assert (average["FIRST_STEP"], average["LAST_STEP"], average["STEPS"]) == (64, 72, 9)
    assert len(average["per_revolution"]) == 2
    assert result["history"][-1]["iteration"] == 72
    inputs = product_inputs(load_point(tmp_path, current.point, solver="su2"))
    assert [item.step for item in inputs.surface] == [72]


def test_the_record_shows_both_segments_and_keeps_the_first_in_its_history(tmp_path):
    finished(tmp_path)
    grow(tmp_path, TWO)
    assert cont(tmp_path) == 0
    registry = RunRegistry(tmp_path)
    current = registry.read()[-1]
    assert current.point.variables["REVOLUTIONS"] in (2, 2.0, "2")
    segments = current.point.continuation["segments"]
    assert [(s["first_step"], s["last_step"]) for s in segments] == [(1, 36), (37, 72)]
    assert segments[0]["folder"] == "continued/1"
    assert current.point.continuation["restart_iter"] == 36
    (entry,) = registry.history()
    assert entry["continued"] is True
    assert entry["record"]["variables"]["REVOLUTIONS"] in (1, 1.0, "1")
    assert entry["record"]["status"] == "COMPLETED"
    from cfdbender.matrix.queries import load_runs

    kinds = [run.kind for run in load_runs(registry)]
    assert kinds == ["current", "continued"]


def test_a_second_continuation_keeps_every_segment(tmp_path):
    record, folder = finished(tmp_path)
    grow(tmp_path, TWO)
    assert cont(tmp_path) == 0
    grow(tmp_path, CLOCK.replace("REVOLUTIONS: 1", "REVOLUTIONS: 3"))
    assert cont(tmp_path) == 0
    assert keys(folder / "su2.cfg")["RESTART_ITER"] == "72"
    assert (folder / "continued/2/history_00036.csv").is_file()
    assert list(time_history(folder).steps) == list(range(1, 109))
    segments = RunRegistry(tmp_path).read()[-1].point.continuation["segments"]
    assert [(s["first_step"], s["last_step"]) for s in segments] == [(1, 36), (37, 72), (73, 108)]


def test_a_point_started_from_a_steady_solution_keeps_its_offset(tmp_path):
    record, folder = run_ta(workspace(tmp_path, CLOCK))
    assert record.classification != "failed", record.error
    matrix(tmp_path, "ta", 7002, "unsteady_rotor", TWO, record.point.restart_from)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert call(["plan", "ta", "--workspace", str(tmp_path)]) == 0
    assert cont(tmp_path) == 0
    cfg = keys(folder / "su2.cfg")
    # Steps 1 to 36 were SU2's time iterations 2 to 37: the continuation starts at 38.
    assert (cfg["RESTART_ITER"], cfg["TIME_ITER"], cfg["WINDOW_START_ITER"]) == ("38", "74", "65")
    assert list(time_history(folder).steps) == list(range(1, 73))


def test_the_export_window_counts_in_the_whole_run(tmp_path):
    window = " / EXPORT_UNSTEADY_LAST_ITER: 9"
    record, folder = finished(tmp_path, CLOCK + window)
    assert record.classification != "failed", record.error
    grow(tmp_path, TWO + window)
    assert cont(tmp_path) == 0
    assert (folder / "continued/1/su2_segment2.cfg").is_file()
    first, second = keys(folder / "su2.cfg"), keys(folder / "su2_segment2.cfg")
    # Steps 64 to 72 are exported: segment one marches 37 to 63 writing restarts only.
    assert (first["RESTART_ITER"], first["TIME_ITER"]) == ("36", "63")
    assert (second["RESTART_ITER"], second["TIME_ITER"]) == ("63", "72")
    surfaces = sorted(p.name for p in folder.glob("surface_flow_*.vtu"))
    assert [f"surface_flow_{k:05d}.vtu" for k in range(63, 72)] == [
        s for s in surfaces if int(s[13:18]) >= 36
    ]
    assert list(time_history(folder).steps) == list(range(1, 73))


# ---- refusals ---------------------------------------------------------------------------------


def manifest(root):
    return (root / "runs.json").read_bytes()


def test_revolutions_that_did_not_grow_are_refused(tmp_path, capsys):
    finished(tmp_path)
    before = manifest(tmp_path)
    assert cont(tmp_path) == 2
    err = capsys.readouterr().err
    assert "REVOLUTIONS" in err and "did not grow" in err
    assert manifest(tmp_path) == before


def test_another_changed_key_of_the_row_is_refused_by_name(tmp_path, capsys):
    finished(tmp_path)
    grow(tmp_path, TWO.replace("LAST_REVS_AVG: 0.25", "LAST_REVS_AVG: 0.5"))
    before = manifest(tmp_path)
    assert cont(tmp_path) == 2
    err = capsys.readouterr().err
    assert "LAST_REVS_AVG" in err and "--force-rerun" in err
    assert manifest(tmp_path) == before


def test_missing_restart_files_are_refused_by_name(tmp_path, capsys):
    record, folder = finished(tmp_path)
    for path in folder.glob("restart_flow_*.dat"):
        path.unlink()
    grow(tmp_path, TWO)
    assert cont(tmp_path) == 2
    err = capsys.readouterr().err
    assert "restart_flow_00034.dat" in err and "restart_flow_00035.dat" in err
    assert not (folder / "continued").exists()


def test_a_point_with_no_record_is_refused(tmp_path, capsys):
    workspace(tmp_path, CLOCK, restart=False)
    assert cont(tmp_path) == 2
    assert "no record to continue" in capsys.readouterr().err


@pytest.mark.parametrize("other", ["--resume", "--restart", "--force-rerun-all"])
def test_continue_is_not_combined_with_another_rerun_option(tmp_path, capsys, other):
    finished(tmp_path)
    grow(tmp_path, TWO)
    assert cont(tmp_path, other) == 2
    assert "--continue cannot be combined" in capsys.readouterr().err


def test_a_steady_point_cannot_be_continued(tmp_path, capsys):
    workspace(tmp_path, CLOCK, restart=False)
    matrix(tmp_path, "ta", 7002, "qsteady_rotor", "-")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert call(["run", "ta", "--local", "--workspace", str(tmp_path)]) == 0
    assert cont(tmp_path) == 2
    assert "time-accurate" in capsys.readouterr().err


# ---- a killed point continues from its latest periodic restart ----------------------------


def kill_after(folder, last):
    """Leave the folder as a walltime kill after time iteration ``last`` would."""
    for path in folder.glob("restart_flow_*.dat"):
        if int(path.stem.rsplit("_", 1)[1]) > last:
            path.unlink()
    history = folder / "history.csv"
    with history.open(newline="") as stream:
        rows = list(csv.reader(stream))
    kept = [rows[0], *(r for r in rows[1:] if int(r[0]) <= last + 2)]
    with history.open("w", newline="") as stream:
        csv.writer(stream).writerows(kept)


def test_a_killed_point_continues_from_its_latest_periodic_restart(tmp_path):
    record, folder = finished(tmp_path, CLOCK + " / RESTART_EVERY_REV: 0.25")
    # Every 9 time iterations a restart pair: 8 and 9, 17 and 18, 26 and 27.
    kill_after(folder, 27)
    grow(tmp_path, CLOCK + " / RESTART_EVERY_REV: 0.25")
    assert cont(tmp_path) == 0
    cfg = keys(folder / "su2.cfg")
    assert (cfg["RESTART_ITER"], cfg["TIME_ITER"]) == ("28", "36")
    # The killed run's rows after the restart are superseded by the continuation's.
    assert list(time_history(folder).steps) == list(range(1, 37))


# ---- the two-zone point -----------------------------------------------------------------------


def two_zone(root, revolutions):
    mrf_workspace(root, extra="")
    path = root / "inputs/matrices/m.fs"
    path.write_text(path.read_text().replace("REVOLUTIONS: 1", f"REVOLUTIONS: {revolutions}"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (point,) = plan_workspace(root, "m").points
    return point


def rendered(root, point):
    context = load_point(root, point, solver="su2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = load_solver("su2").translate(context)
        values = config(
            context.setup,
            context.reference,
            context.post,
            point.condition,
            geometry=context.geometry,
            run_type=point.run_type,
            precursor=point.precursor,
            restarted=point.restart_from is not None,
            continue_from=point.continuation.get("restart_iter"),
        )
    return keys_of(text), extra_files(values)


def keys_of(text):
    return dict(line.split("= ", 1) for line in text.splitlines())


def test_the_two_zone_restart_is_found_for_every_zone(tmp_path):
    from cfdbender.solvers.su2._continuation import restart_iteration

    point = two_zone(tmp_path, 1)
    folder = tmp_path / point.directory
    folder.mkdir(parents=True)
    first, _ = rendered(tmp_path, point)
    (folder / "su2.cfg").write_text("".join(f"{k}= {v}\n" for k, v in first.items()))
    # 36 steps after the frozen-rotor stage: SU2 time iterations 2 to 37.
    for zone in (0, 1):
        for k in (36, 37):
            (folder / f"restart_flow_{zone}_{k:05d}.dat").write_bytes(b"state")
    assert restart_iteration(folder) == 38
    (folder / "restart_flow_1_00037.dat").unlink()
    with pytest.raises(ValueError, match="restart_flow_1_00037.dat"):
        restart_iteration(folder)


def test_the_two_zone_continuation_differs_from_the_first_segment_only_in_its_restart(tmp_path):
    first_point = two_zone(tmp_path / "a", 1)
    first, first_files = rendered(tmp_path / "a", first_point)
    grown = two_zone(tmp_path / "b", 2)
    second, files = rendered(tmp_path / "b", replace(grown, continuation={"restart_iter": 38}))
    changed = {key for key in first.keys() | second.keys() if first.get(key) != second.get(key)}
    assert changed == {
        "RESTART_ITER",
        "SOLUTION_FILENAME",
        "TIME_ITER",
        "WINDOW_START_ITER",
        "OUTPUT_WRT_FREQ",
        "READ_BINARY_RESTART",
    }
    assert (first["RESTART_ITER"], second["RESTART_ITER"]) == ("2", "38")
    assert (first["SOLUTION_FILENAME"], second["SOLUTION_FILENAME"]) == (
        "solution.dat",
        "restart_flow.dat",
    )
    assert (first["TIME_ITER"], second["TIME_ITER"]) == ("38", "74")
    # No frozen-rotor stage again: the zone files are the same, the stage-1 files are gone.
    assert {n for n in files if n.startswith("zone_")} == {
        n for n in first_files if n.startswith("zone_")
    }
    assert all(files[n] == first_files[n] for n in files if n.startswith("zone_"))
    assert not {"precursor.cfg", "run_mrf.sh", "pre_zone_0.cfg"} & set(files)
    command = load_solver("su2").job_command(
        load_point(tmp_path / "b", replace(grown, continuation={"restart_iter": 38}), solver="su2")
    )
    assert command[-1] == "su2.cfg" and "run_mrf.sh" not in command


def test_the_continued_history_of_a_two_zone_point_is_kept_before_it_is_overwritten(tmp_path):
    from cfdbender.solvers.su2._continuation import evidence

    point = two_zone(tmp_path, 1)
    folder = tmp_path / point.directory
    folder.mkdir(parents=True)
    first, _ = rendered(tmp_path, point)
    (folder / "su2.cfg").write_text("".join(f"{k}= {v}\n" for k, v in first.items()))
    for name in ("su2.csv", "stdout.log", "stderr.log", "precursor.csv", "precursor.log"):
        (folder / name).write_text("x")
    moved = evidence(folder)
    # The precursor's own files stay: the continuation does not run it again.
    assert {"su2.cfg", "su2.csv", "stdout.log", "stderr.log"} <= set(moved)
    assert not {"precursor.csv", "precursor.log"} & set(moved)


def test_the_continuation_mapping_is_recorded_in_runs_json(tmp_path):
    finished(tmp_path)
    grow(tmp_path, TWO)
    assert cont(tmp_path) == 0
    rows = json.loads((tmp_path / "runs.json").read_text())["records"]
    assert rows[-1]["continuation"]["restart_iter"] == 36
    assert Path(rows[-1]["directory"]).name.startswith("DP-")


def test_continue_is_refused_with_a_batch(tmp_path, capsys):
    finished(tmp_path)
    grow(tmp_path, TWO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = call(["run", "ta", "--continue", "--batch", "1", "--workspace", str(tmp_path)])
    assert code == 2
    assert "does not take --continue" in capsys.readouterr().err


def test_a_cluster_submission_continues_and_collect_joins_the_segments(tmp_path):
    import subprocess
    import sys

    from test_p0000_item3f_plan import profile
    from test_su2_time_restart import FAKE

    record, folder = finished(tmp_path)
    path = profile(
        tmp_path,
        fields={"name": "{job_name}", "working_directory": "{workdir}", "config": "{config_path}"},
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.index("[builds]")], encoding="utf-8")
    grow(tmp_path, TWO)
    assert cont_cluster(tmp_path) == 0
    current = RunRegistry(tmp_path).read()[-1]
    assert current.status == "SUBMITTED", current.error
    assert keys(folder / "su2.cfg")["RESTART_ITER"] == "36"
    assert (folder / "continued/1/history.csv").is_file()
    # The job runs SU2 in the point folder, as the scheduler would.
    with (folder / "stdout.log").open("wb") as log:
        subprocess.run([sys.executable, str(FAKE), "su2.cfg"], cwd=folder, stdout=log, check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert call(["collect", "ta", "--workspace", str(tmp_path)]) == 0
    done = RunRegistry(tmp_path).read()[-1]
    assert done.status == "COMPLETED" and done.classification != "failed", done.error
    assert list(time_history(folder).steps) == list(range(1, 73))


def cont_cluster(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert call(["plan", "ta", "--workspace", str(root)]) == 0
        return call(["run", "ta", "--continue", "--workspace", str(root)])

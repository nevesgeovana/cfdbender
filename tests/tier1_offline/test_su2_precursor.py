"""A time-accurate sliding rotor row whose RESTART cell is MRF starts from a frozen-rotor stage.

The point folder holds two solver stages. Stage 1 (``precursor.cfg``) is the same two-zone mesh at
the same condition solved steady, the turning zone in a rotating frame; stage 2 (``su2.cfg``) is
the time-accurate sliding run restarted from stage 1's solution. The job runs both through one
launcher with plain shell. The keys asserted here are those a real SU2 8.6.0 run accepted
(the two stages and the staged names), stated from the solver's file naming, not from the
output under test.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
from test_restart_in_job import (
    CLOCK,
    batch,
    build,
    cli_m,
    needs_sh,
    point_of,
    row,
    run_script,
    write_matrix,
)

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import _encode
from cfdbender.solvers import load_solver
from cfdbender.solvers.su2 import _precursor as pre
from cfdbender.solvers.su2._config import config, extra_files
from cfdbender.solvers.su2._history import step_offset
from cfdbender.solvers.su2._time_history import time_history
from cfdbender.workspace.planning import plan_workspace

SLIDING = '[sliding]\nzones = 2\nrotating_zone = 0\ninterface = ["X_OUT", "X_IN"]\n'
WRAPPER_ROW = ' / hpc.application_id: "tools/su2.sh"'
#: A toy SU2: it reads the configuration file named last, as the real command does.
TOY = """#!/bin/sh
for last; do :; done
echo "$last" >> calls.txt
flags=../../../../
case "$last" in
  precursor.cfg)
    [ -f ${flags}fail-pre ] && exit 3
    echo '"Time_Iter","rms[Rho][0]"' > precursor.csv
    value=-5.0
    [ -f ${flags}nan-pre ] && value=-nan
    echo "0,$value" >> precursor.csv
    if [ ! -f ${flags}no-restart ]; then
      echo state0 > precursor_restart_0.dat
      echo state1 > precursor_restart_1.dat
    fi
    cap="Maximum number of iterations reached (OUTER_ITER = 5)"
    [ -f ${flags}cap ] && echo "$cap" "before convergence."
    exit 0;;
  su2.cfg)
    for f in solution_0_00000.dat solution_0_00001.dat solution_1_00000.dat solution_1_00001.dat; do
      [ -s "$f" ] || { echo "Unable to open SU2 restart file $f" >&2; exit 1; }
    done
    echo done > su2.csv
    exit 0;;
esac
exit 9
"""


def mrf_workspace(root, *, setup="", restart="MRF", extra="", **kwargs):
    first = re.search(r"(?m)^\[", setup)
    plain, tables = (
        (setup, "") if first is None else (setup[: first.start()], setup[first.start() :])
    )
    build(
        root,
        [row(7001, "unsteady_rotor", CLOCK + extra, restart)],
        setup=plain + SLIDING + tables,
    )
    (root / "tools/su2.sh").write_text(TOY, newline="\n")
    return root


def deck(root, *, pol=7001):
    """The planned point, its keys with the precursor and the files beside su2.cfg."""
    plan = plan_workspace(root, "m")
    point = next(p for p in plan.points if p.pol == str(pol))
    context = load_point(root, point, solver="su2")
    values = config(
        context.setup,
        context.reference,
        context.post,
        point.condition,
        geometry=context.geometry,
        run_type=point.run_type,
        precursor=point.precursor,
        restarted=point.restart_from is not None,
    )
    cfg = dict(line.split("= ", 1) for line in load_solver("su2").translate(context).splitlines())
    return point, values, cfg, extra_files(values)


def keys(text):
    return dict(line.split("= ", 1) for line in text.splitlines())


# ---- the cell -----------------------------------------------------------------------------


def test_the_keyword_is_read_from_the_restart_cell(tmp_path):
    mrf_workspace(tmp_path)
    (point,) = plan_workspace(tmp_path, "m").points
    assert point.precursor is True and point.restart_from is None


@pytest.mark.parametrize("cell", ["MRF", "mrf", "Mrf"])
def test_the_keyword_is_case_insensitive(tmp_path, cell):
    mrf_workspace(tmp_path, restart=cell)
    assert plan_workspace(tmp_path, "m").points[0].precursor is True


def test_a_dash_is_a_cold_start(tmp_path):
    mrf_workspace(tmp_path, restart="-")
    (point,) = plan_workspace(tmp_path, "m").points
    assert point.precursor is False and "precursor" not in point.provenance


def test_a_cell_that_is_not_a_point_or_the_keyword_is_refused(tmp_path):
    mrf_workspace(tmp_path, restart="FROZEN")
    with pytest.raises(Exception, match="POL/DP-name, MRF or -"):
        plan_workspace(tmp_path, "m")


# ---- stage 1 ------------------------------------------------------------------------------


def test_stage_one_is_the_point_steady_with_a_rotating_frame(tmp_path):
    mrf_workspace(tmp_path)
    _, _, _, files = deck(tmp_path)
    one = keys(files["precursor.cfg"])
    assert one["TIME_DOMAIN"] == "NO"
    assert one["MULTIZONE"] == "YES"
    assert one["OUTER_ITER"] == "300" and one["INNER_ITER"] == "1"
    assert one["CONFIG_LIST"] == "( pre_zone_0.cfg, pre_zone_1.cfg )"
    assert one["RESTART_FILENAME"] == "precursor_restart.dat"
    assert one["VOLUME_FILENAME"] == "precursor_flow"
    assert one["SURFACE_FILENAME"] == "precursor_surface"
    assert one["OUTPUT_FILES"] == "( RESTART )" and one["OUTPUT_WRT_FREQ"] == "300"
    assert one["MESH_FILENAME"] == "mesh.su2"
    assert one["KIND_INTERPOLATION"] == "NEAREST_NEIGHBOR"
    assert one["MARKER_ZONE_INTERFACE"] == "( X_OUT, X_IN )"


def test_stage_one_zone_files_turn_the_rotating_zone_in_a_frame(tmp_path):
    mrf_workspace(tmp_path)
    _, _, _, files = deck(tmp_path)
    moving, still = keys(files["pre_zone_0.cfg"]), keys(files["pre_zone_1.cfg"])
    assert moving["GRID_MOVEMENT"] == "ROTATING_FRAME"
    assert still == {"GRID_MOVEMENT": "NONE"}
    rigid = keys(files["zone_0.cfg"])
    assert rigid["GRID_MOVEMENT"] == "RIGID_MOTION"
    for key in ("ROTATION_RATE", "MOTION_ORIGIN", "MACH_MOTION"):
        assert moving[key] == rigid[key]


def test_stage_one_carries_no_time_accurate_or_restart_key(tmp_path):
    mrf_workspace(tmp_path)
    _, _, _, files = deck(tmp_path)
    one = keys(files["precursor.cfg"])
    for key in (
        "TIME_MARCHING",
        "TIME_STEP",
        "TIME_ITER",
        "WINDOW_START_ITER",
        "RESTART_SOL",
        "RESTART_ITER",
        "SOLUTION_FILENAME",
        "CONV_FIELD",
        "CONV_RESIDUAL_MINVAL",
    ):
        assert key not in one
    assert "TAVG_AERO_COEFF" not in one["HISTORY_OUTPUT"]
    assert "RMS_RES[0]" in one["HISTORY_OUTPUT"]


def test_both_stages_state_one_flow_model(tmp_path):
    mrf_workspace(tmp_path)
    _, _, cfg, files = deck(tmp_path)
    one = keys(files["precursor.cfg"])
    shared = [
        key
        for key in cfg
        if key in one
        and key
        not in {
            "TIME_DOMAIN",
            "CONFIG_LIST",
            "OUTER_ITER",
            "INNER_ITER",
            "OUTPUT_FILES",
            "OUTPUT_WRT_FREQ",
            "RESTART_FILENAME",
            "HISTORY_OUTPUT",
        }
    ]
    assert {"SOLVER", "KIND_TURB_MODEL", "SA_OPTIONS", "REF_DIMENSIONALIZATION"} <= set(shared)
    assert all(one[key] == cfg[key] for key in shared)
    for key in one:
        assert key in cfg or key in ("VOLUME_FILENAME", "SURFACE_FILENAME")


def test_a_native_numerics_key_of_the_setup_reaches_both_stages(tmp_path):
    mrf_workspace(tmp_path, setup='CONV_NUM_METHOD_FLOW = "ROE"\nMUSCL_FLOW = false\n')
    _, _, cfg, files = deck(tmp_path)
    one = keys(files["precursor.cfg"])
    assert cfg["CONV_NUM_METHOD_FLOW"] == one["CONV_NUM_METHOD_FLOW"] == "ROE"
    assert cfg["MUSCL_FLOW"] == one["MUSCL_FLOW"] == "NO"


def test_a_stated_convergence_field_never_reaches_stage_one_without_converge_to(tmp_path):
    mrf_workspace(tmp_path, setup='CONV_FIELD = "RMS_DENSITY[0]"\nCONV_RESIDUAL_MINVAL = -6\n')
    _, _, _, files = deck(tmp_path)
    assert "CONV_FIELD" not in keys(files["precursor.cfg"])


# ---- the [precursor] table ----------------------------------------------------------------


def test_the_table_sets_the_iterations_and_the_stop_level(tmp_path):
    mrf_workspace(tmp_path, setup="[precursor]\niterations = 150\nconverge_to = -9\n")
    point, _, _, files = deck(tmp_path)
    one = keys(files["precursor.cfg"])
    assert one["OUTER_ITER"] == "150" and one["OUTPUT_WRT_FREQ"] == "150"
    assert one["CONV_FIELD"] == "RMS_DENSITY[0]" and one["CONV_RESIDUAL_MINVAL"] == "-9"
    assert point.provenance["precursor"] == {"iterations": 150, "converge_to": -9.0}


def test_the_convergence_field_is_the_rotating_zone(tmp_path):
    build(
        tmp_path,
        [row(7001, "unsteady_rotor", CLOCK, "MRF")],
        setup='[sliding]\nzones = 3\nrotating_zone = 2\ninterface = ["A", "B"]\n'
        "[precursor]\nconverge_to = -8\n",
    )
    _, _, _, files = deck(tmp_path)
    assert keys(files["precursor.cfg"])["CONV_FIELD"] == "RMS_DENSITY[2]"
    assert keys(files["pre_zone_2.cfg"])["GRID_MOVEMENT"] == "ROTATING_FRAME"
    assert keys(files["pre_zone_0.cfg"])["GRID_MOVEMENT"] == "NONE"


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ("iterations = 0", "whole number of at least 1"),
        ("iterations = 2.5", "whole number of at least 1"),
        ("iterations = true", "whole number of at least 1"),
        ('converge_to = "low"', "finite residual level"),
        ("steps = 3", "unknown keys: steps"),
    ],
)
def test_a_bad_table_is_refused(tmp_path, table, message):
    mrf_workspace(tmp_path, setup=f"[precursor]\n{table}\n")
    with pytest.raises(Exception, match=message):
        plan_workspace(tmp_path, "m")


def test_the_table_is_read_only_when_a_row_asks_for_the_precursor(tmp_path):
    mrf_workspace(tmp_path, setup="[precursor]\niterations = 0\n", restart="-")
    (point,) = plan_workspace(tmp_path, "m").points
    assert point.precursor is False


def test_the_defaults_apply_without_the_table():
    assert pre.settings({}) == pre.Settings(300, None)


# ---- stage 2 ------------------------------------------------------------------------------


def test_stage_two_restarts_from_the_staged_solution(tmp_path):
    mrf_workspace(tmp_path)
    _, _, cfg, _ = deck(tmp_path)
    assert cfg["RESTART_SOL"] == "YES" and cfg["RESTART_ITER"] == "2"
    assert cfg["SOLUTION_FILENAME"] == "solution.dat"
    assert cfg["TIME_ITER"] == "38"  # RESTART_ITER + the 36 steps asked
    assert cfg["TIME_DOMAIN"] == "YES" and cfg["MULTIZONE"] == "YES"


def test_first_order_time_marching_restarts_at_one_with_one_level(tmp_path):
    mrf_workspace(tmp_path, setup='TIME_MARCHING = "DUAL_TIME_STEPPING-1ST_ORDER"\n')
    _, _, cfg, files = deck(tmp_path)
    assert cfg["RESTART_ITER"] == "1" and cfg["TIME_ITER"] == "37"
    assert "solution_0_00000.dat" in files["run_mrf.sh"]
    assert "solution_0_00001.dat" not in files["run_mrf.sh"]


def test_a_row_without_the_keyword_renders_as_before(tmp_path):
    mrf_workspace(tmp_path, restart="-")
    point, values, cfg, files = deck(tmp_path)
    assert sorted(files) == ["zone_0.cfg", "zone_1.cfg"]
    assert not any(key.startswith(("precursor", "pre_zone")) for key in values)
    assert "RESTART_SOL" not in cfg and "SOLUTION_FILENAME" not in cfg
    assert cfg["TIME_ITER"] == "36"
    assert not any("precursor" in out or "run_mrf" in out for out in point.outputs)


def test_the_keyword_changes_only_the_restart_keys_of_the_time_accurate_file(tmp_path):
    mrf_workspace(tmp_path, restart="-")
    _, _, cold, _ = deck(tmp_path)
    mrf_workspace(tmp_path / "second", restart="MRF")
    _, _, warm, _ = deck(tmp_path / "second")
    changed = {key for key in set(cold) | set(warm) if cold.get(key) != warm.get(key)}
    assert changed == {
        "RESTART_SOL",
        "RESTART_ITER",
        "SOLUTION_FILENAME",
        "TIME_ITER",
        "OUTPUT_WRT_FREQ",
        "WINDOW_START_ITER",
    }
    assert int(warm["TIME_ITER"]) - int(cold["TIME_ITER"]) == 2
    assert int(warm["WINDOW_START_ITER"]) - int(cold["WINDOW_START_ITER"]) == 2


def test_the_clock_and_the_windows_are_those_of_the_cold_row(tmp_path):
    records = {}
    for name, cell in (("cold", "-"), ("warm", "MRF")):
        mrf_workspace(tmp_path / name, restart=cell)
        plan = plan_workspace(tmp_path / name, "m")
        context = load_point(tmp_path / name, plan.points[0], solver="su2")
        record: dict[str, object] = {}
        config(
            context.setup,
            context.reference,
            context.post,
            plan.points[0].condition,
            geometry=context.geometry,
            run_type="unsteady_rotor",
            record=record,
            precursor=plan.points[0].precursor,
        )
        records[name] = record
    cold, warm = records["cold"], records["warm"]
    assert warm["clock"] == cold["clock"]
    assert warm["averaging_window"] == cold["averaging_window"] == [28, 36]
    assert warm["export_window"] == cold["export_window"]
    assert warm["step_offset"] == 2 and "step_offset" not in cold


def test_the_history_counts_marched_steps_from_one(tmp_path):
    mrf_workspace(tmp_path)
    point, _, _, _ = deck(tmp_path)
    folder = tmp_path / "hist"
    folder.mkdir()
    context = load_point(tmp_path, point, solver="su2")
    (folder / "su2.cfg").write_text(load_solver("su2").translate(context))
    header = '"Time_Iter","Outer_Iter","Inner_Iter",' + ",".join(
        f'"{name}[1]"' for name in ("CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
    )
    rows = [header] + [f"{step},0,0,1,2,3,4,5,6" for step in range(2, 38)]
    (folder / "su2.csv").write_text("\n".join(rows) + "\n")
    assert step_offset(folder) == 2
    history = time_history(folder, zone=1)
    assert list(history.steps) == list(range(1, 37))


def test_the_precursor_files_are_not_a_history(tmp_path):
    from cfdbender.solvers.su2._history import history_names

    mrf_workspace(tmp_path)
    point, _, _, _ = deck(tmp_path)
    folder = tmp_path / "hist2"
    folder.mkdir()
    context = load_point(tmp_path, point, solver="su2")
    (folder / "su2.cfg").write_text(load_solver("su2").translate(context))
    (folder / "precursor.csv").write_text("a\n")
    assert history_names(folder) == ["su2.csv"]


def test_the_precursor_files_are_claimed_beside_the_point(tmp_path):
    mrf_workspace(tmp_path)
    point, _, _, _ = deck(tmp_path)
    names = {out.rsplit("/", 1)[1] for out in point.outputs}
    assert {"precursor.cfg", "pre_zone_0.cfg", "pre_zone_1.cfg", "run_mrf.sh", "su2.csv"} <= names
    assert "history.csv" not in names


# ---- rows it does not apply to ------------------------------------------------------------


def test_a_steady_row_warns_and_skips_the_keyword(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor", "-", "MRF")], setup=SLIDING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (point,) = plan_workspace(tmp_path, "m").points
    assert point.precursor is False
    messages = [str(w.message) for w in caught]
    assert any("POL 7001: RESTART MRF is skipped" in m and "qsteady_rotor" in m for m in messages)


def test_a_steady_row_with_the_keyword_runs_as_without_it(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor", "-", "MRF")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (with_key,) = plan_workspace(tmp_path, "m").points
    build(tmp_path / "other", [row(7001, "qsteady_rotor", "-", "-")])
    (without,) = plan_workspace(tmp_path / "other", "m").points
    assert replace(with_key, inputs_sha256={}) == replace(without, inputs_sha256={})


def test_a_time_accurate_row_without_a_sliding_table_is_refused(tmp_path):
    build(tmp_path, [row(7001, "unsteady_rotor", CLOCK, "MRF")])
    with pytest.raises(Exception, match="needs a time-accurate sliding-mesh rotor row"):
        plan_workspace(tmp_path, "m")


def test_a_non_rotor_time_accurate_row_is_refused(tmp_path):
    build(
        tmp_path,
        [
            row(
                7001,
                "unsteady",
                "DELTA_TIME: 0.001 / TIME_ITERATIONS: 10 / LAST_ITERS_AVG: 4",
                "MRF",
            )
        ],
    )
    with pytest.raises(Exception, match="needs a time-accurate sliding-mesh rotor row"):
        plan_workspace(tmp_path, "m")


def test_a_weighted_average_interface_is_refused(tmp_path):
    mrf_workspace(tmp_path, setup='KIND_INTERPOLATION = "WEIGHTED_AVERAGE"\n')
    with pytest.raises(Exception, match="WEIGHTED_AVERAGE diverges in a steady run"):
        plan_workspace(tmp_path, "m")


def test_an_ascii_restart_is_refused(tmp_path):
    mrf_workspace(tmp_path, setup="WRT_BINARY_RESTART = false\n")
    with pytest.raises(Exception, match="binary restart files"):
        plan_workspace(tmp_path, "m")


def test_a_saved_point_as_the_source_of_a_sliding_row_stays_refused(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")], setup=SLIDING)
    write_matrix(tmp_path, [row(7001, "qsteady_rotor")])
    source = plan_workspace(tmp_path, "m").points[0].point_id
    write_matrix(
        tmp_path,
        [row(7001, "qsteady_rotor"), row(7002, "unsteady_rotor", CLOCK, source)],
    )
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 2
    assert "two-zone (sliding-mesh) restart is not supported" in capsys.readouterr().err


def test_a_sliding_row_with_the_keyword_is_not_a_dependent(tmp_path):
    mrf_workspace(tmp_path)
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    record = point_of(tmp_path, 7001)
    assert record.point.restart_in_job is False and record.point.precursor is True
    assert not (tmp_path / record.point.directory / "restart-pending.json").exists()


# ---- the job script -----------------------------------------------------------------------


def run_batch(root):
    assert batch(root, command="plan") == 0
    assert batch(root) == 0
    return run_script(root)


def folder_of(root, pol=7001):
    return root / point_of(root, pol).point.directory


def batch_text(root):
    (path,) = (root / "sims/batch").glob("*/BATCH-*.sh")
    return path.read_text()


def test_the_batch_block_runs_the_stage_script_under_the_stated_launcher(tmp_path):
    mrf_workspace(tmp_path)
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    line = next(x for x in batch_text(tmp_path).splitlines() if "run_mrf.sh" in x)
    assert "&& sh run_mrf.sh " in line and line.rstrip().endswith("> stdout.log 2> stderr.log )")
    assert "su2.cfg" not in line and "precursor.cfg" not in line
    assert "tools/su2.sh" in line or "su2.sh" in line


def test_the_batch_block_puts_the_application_wrapper_under_the_stage_script(tmp_path):
    mrf_workspace(tmp_path, extra=WRAPPER_ROW)
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    line = next(x for x in batch_text(tmp_path).splitlines() if "run_mrf.sh" in x)
    assert "&& sh run_mrf.sh sh " in line and "tools/su2.sh >" in line
    assert "su2.cfg" not in line


def test_a_row_without_the_keyword_keeps_its_command(tmp_path):
    mrf_workspace(tmp_path, restart="-", extra=WRAPPER_ROW)
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    text = batch_text(tmp_path)
    assert "run_mrf" not in text and "tools/su2.sh su2.cfg >" in text


def test_the_stage_script_has_the_stages_in_order_and_no_python(tmp_path):
    mrf_workspace(tmp_path)
    _, _, _, files = deck(tmp_path)
    text = files["run_mrf.sh"]
    order = [
        text.index('"$@" precursor.cfg'),
        text.index("exit 91"),
        text.index("exit 92"),
        text.index("exit 93"),
        text.index("cp precursor_restart_0.dat solution_0_00000.dat"),
        text.index("cp precursor_restart_0.dat solution_0_00001.dat"),
        text.index("cp precursor_restart_1.dat solution_1_00000.dat"),
        text.index("cp precursor_restart_1.dat solution_1_00001.dat"),
        text.index('"$@" su2.cfg'),
    ]
    assert order == sorted(order)
    assert "exit 94" in text
    assert "python" not in text.lower() and "py " not in text
    assert text.startswith("#!/bin/sh")


def test_the_exit_statuses_do_not_collide_with_the_in_job_restart():
    from cfdbender.matrix.batch import SOURCE_FAILED, STAGING_FAILED

    codes = {pre.EXIT_FAILED, pre.EXIT_DIVERGED, pre.EXIT_NO_RESTART, pre.EXIT_STAGING}
    assert len(codes) == 4 and not codes & {SOURCE_FAILED, STAGING_FAILED, 0, 1, 124}


def test_a_stop_level_adds_the_cap_warning_line_to_the_script(tmp_path):
    mrf_workspace(tmp_path, setup="[precursor]\nconverge_to = -9\n")
    _, _, _, files = deck(tmp_path)
    assert "PRECURSOR_WARNING" in files["run_mrf.sh"]
    mrf_workspace(tmp_path / "plain")
    _, _, _, plain = deck(tmp_path / "plain")
    assert "PRECURSOR_WARNING" not in plain["run_mrf.sh"]


@needs_sh
@pytest.mark.parametrize("extra", ["", WRAPPER_ROW], ids=["launcher", "wrapper"])
def test_both_stages_run_in_order_and_the_solution_is_staged(tmp_path, extra):
    mrf_workspace(tmp_path, extra=extra)
    result = run_batch(tmp_path)
    assert result.returncode == 0
    folder = folder_of(tmp_path)
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg", "su2.cfg"]
    for zone in (0, 1):
        source = (folder / f"precursor_restart_{zone}.dat").read_bytes()
        for level in (0, 1):
            assert (folder / f"solution_{zone}_{level:05d}.dat").read_bytes() == source
    assert (folder / "job.exit").read_text().strip() == "0"
    assert (folder / "stderr.log").read_text() == ""


@needs_sh
def test_a_precursor_that_exits_non_zero_stops_the_point(tmp_path):
    mrf_workspace(tmp_path)
    (tmp_path / "fail-pre").write_text("")
    run_batch(tmp_path)
    folder = folder_of(tmp_path)
    assert (folder / "job.exit").read_text().strip() == "91"
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg"]
    lines = (folder / "stderr.log").read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("PRECURSOR_FAILED: the steady stage ended")
    assert "status 3" in lines[0]
    assert not list(folder.glob("solution_*"))


@needs_sh
def test_a_nan_in_the_precursor_history_stops_the_point(tmp_path):
    mrf_workspace(tmp_path)
    (tmp_path / "nan-pre").write_text("")
    run_batch(tmp_path)
    folder = folder_of(tmp_path)
    assert (folder / "job.exit").read_text().strip() == "92"
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg"]
    assert "precursor.csv holds NaN" in (folder / "stderr.log").read_text()
    assert not list(folder.glob("solution_*"))


@needs_sh
def test_a_missing_restart_file_stops_the_point(tmp_path):
    mrf_workspace(tmp_path)
    (tmp_path / "no-restart").write_text("")
    run_batch(tmp_path)
    folder = folder_of(tmp_path)
    assert (folder / "job.exit").read_text().strip() == "93"
    assert "wrote no precursor_restart_0.dat" in (folder / "stderr.log").read_text()
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg"]


@needs_sh
def test_a_wrapper_that_ignores_the_stage_file_is_caught(tmp_path):
    mrf_workspace(tmp_path, extra=WRAPPER_ROW)
    (tmp_path / "tools/su2.sh").write_text(
        "#!/bin/sh\necho su2.cfg >> calls.txt\nexit 1\n", newline="\n"
    )
    run_batch(tmp_path)
    folder = folder_of(tmp_path)
    assert (folder / "job.exit").read_text().strip() == "91"
    assert "last argument" in (folder / "stderr.log").read_text()
    assert not list(folder.glob("solution_*"))


@needs_sh
def test_stage_two_failing_returns_its_own_status(tmp_path):
    mrf_workspace(tmp_path)
    (tmp_path / "tools/su2.sh").write_text(
        TOY.replace("echo done > su2.csv", "exit 7"), newline="\n"
    )
    run_batch(tmp_path)
    assert (folder_of(tmp_path) / "job.exit").read_text().strip() == "7"


@needs_sh
def test_a_cap_hit_with_a_stop_level_is_a_warning_not_a_failure(tmp_path):
    mrf_workspace(tmp_path, setup="[precursor]\nconverge_to = -9\n")
    (tmp_path / "cap").write_text("")
    run_batch(tmp_path)
    folder = folder_of(tmp_path)
    assert (folder / "job.exit").read_text().strip() == "0"
    assert "PRECURSOR_WARNING" in (folder / "stdout.log").read_text()
    assert (folder / "stderr.log").read_text() == ""
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg", "su2.cfg"]


@needs_sh
def test_the_staging_needs_no_python_on_the_node(tmp_path):
    from test_restart_in_job import shell_only_path

    path = shell_only_path()
    if path is None:
        pytest.skip("a python is next to the shell")
    mrf_workspace(tmp_path)
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    assert run_script(tmp_path, search_path=path).returncode == 0
    assert (folder_of(tmp_path) / "job.exit").read_text().strip() == "0"


# ---- plan, force rerun, records -----------------------------------------------------------


def test_the_plan_shows_a_line_for_the_point(tmp_path, capsys):
    mrf_workspace(tmp_path, setup="[precursor]\niterations = 120\nconverge_to = -9\n")
    assert cli_m(tmp_path, "plan") == 0
    out = capsys.readouterr().out
    assert (
        "precursor: steady frozen-rotor stage first, 120 iterations at most, "
        "stopping at residual -9" in out
    )
    mrf_workspace(tmp_path / "plain")
    assert cli_m(tmp_path / "plain", "plan") == 0
    assert "precursor: steady frozen-rotor stage first, 300 iterations" in capsys.readouterr().out


def test_the_plan_of_a_cold_row_shows_no_such_line(tmp_path, capsys):
    mrf_workspace(tmp_path, restart="-")
    assert cli_m(tmp_path, "plan") == 0
    assert "precursor" not in capsys.readouterr().out


def test_the_record_keeps_the_flag_and_the_settings(tmp_path):
    mrf_workspace(tmp_path, setup="[precursor]\niterations = 120\n")
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    record = point_of(tmp_path, 7001)
    assert record.point.precursor is True
    assert record.point.provenance["precursor"] == {"iterations": 120, "converge_to": None}
    assert _encode(record)["precursor"] is True


def test_a_cold_record_omits_the_flag(tmp_path):
    mrf_workspace(tmp_path, restart="-")
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    record = point_of(tmp_path, 7001)
    assert record.point.precursor is False
    assert "precursor" not in _encode(record)
    assert "precursor" not in record.point.provenance


def test_a_forced_rerun_of_the_batch_writes_the_stage_files_again(tmp_path):
    mrf_workspace(tmp_path)
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 0
    assert batch(tmp_path, "--force-rerun-all") == 0
    record = point_of(tmp_path, 7001)
    folder = tmp_path / record.point.directory
    for name in ("precursor.cfg", "pre_zone_0.cfg", "pre_zone_1.cfg", "run_mrf.sh", "su2.cfg"):
        assert (folder / name).is_file()
    assert record.point.precursor is True and record.status == "SUBMITTED"


# ---- a local run --------------------------------------------------------------------------


def toy_python(path: Path) -> None:
    """A toy SU2 as a Python stand-in, for the local route (it reads its last argument)."""
    path.write_text(
        "import sys, pathlib\n"
        "cfg = sys.argv[-1]\n"
        "root = pathlib.Path.cwd().parents[3]\n"
        "pathlib.Path('calls.txt').open('a').write(cfg + chr(10))\n"
        "if cfg == 'precursor.cfg':\n"
        "    if (root / 'fail-pre').exists():\n"
        "        sys.exit(3)\n"
        "    nan = (root / 'nan-pre').exists()\n"
        "    value = '-nan' if nan else '1'\n"
        "    pathlib.Path('precursor.csv').write_text('Time_Iter' + chr(10) + value + chr(10))\n"
        "    if not (root / 'no-restart').exists():\n"
        "        for z in (0, 1):\n"
        "            pathlib.Path(f'precursor_restart_{z}.dat').write_text('s')\n"
        "    if (root / 'cap').exists():\n"
        "        print('Maximum number of iterations reached', 'before convergence.')\n"
        "else:\n"
        "    names = ['solution_%d_0000%d.dat' % (z, k) for z in (0, 1) for k in (0, 1)]\n"
        "    missing = [n for n in names if not pathlib.Path(n).exists()]\n"
        "    if missing:\n"
        "        sys.exit(1)\n"
        "    pathlib.Path('su2.csv').write_text('done')\n",
        encoding="ascii",
    )


def local(root, **kwargs):
    """Run the one point through the adapter's own executor; return its folder and the code."""
    from cfdbender.solvers.su2._execute import run

    plan = plan_workspace(root, "m")
    point = load_point(root, plan.points[0], solver="su2")
    try:
        run(point, lambda: None)
    except ValueError as error:
        # Only the host's own memory refusal skips; any other failure of the local path is a defect.
        if "would breach the 0.5 GiB floor" not in str(error):
            raise
        pytest.skip(f"the host is short of memory: {error}")
    folder = root / plan.points[0].directory
    return folder, json.loads((folder / "execution.json").read_text())["exit_code"]


def local_workspace(root, **kwargs):
    stand_in = root / "toy_su2.py"
    root.mkdir(parents=True, exist_ok=True)
    toy_python(stand_in)
    mrf_workspace(root, **kwargs)
    (root / "inputs/setups/s001.toml").write_text(
        (root / "inputs/setups/s001.toml")
        .read_text()
        .replace(
            "executable = ",
            f"executable = {json.dumps(stand_in.as_posix())}\n#",
        )
    )
    return root


def test_a_local_run_does_both_stages_and_stages_the_solution(tmp_path):
    folder, code = local(local_workspace(tmp_path))
    assert code == 0
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg", "su2.cfg"]
    assert (folder / "solution_1_00001.dat").is_file() and (folder / "precursor.log").is_file()


def test_a_local_run_stops_after_a_failed_precursor(tmp_path):
    root = local_workspace(tmp_path)
    (root / "fail-pre").write_text("")
    folder, code = local(root)
    assert code == pre.EXIT_FAILED
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg"]
    assert "PRECURSOR_FAILED" in (folder / "stderr.log").read_text()


def test_a_local_run_stops_on_nan_and_on_a_missing_restart(tmp_path):
    first = local_workspace(tmp_path / "a")
    (first / "nan-pre").write_text("")
    folder, code = local(first)
    assert code == pre.EXIT_DIVERGED and "holds NaN" in (folder / "stderr.log").read_text()
    second = local_workspace(tmp_path / "b")
    (second / "no-restart").write_text("")
    folder, code = local(second)
    assert code == pre.EXIT_NO_RESTART
    assert (folder / "calls.txt").read_text().split() == ["precursor.cfg"]


def test_a_local_cap_hit_with_a_stop_level_warns_in_the_point_log(tmp_path):
    root = local_workspace(tmp_path, setup="[precursor]\nconverge_to = -9\n")
    (root / "cap").write_text("")
    folder, code = local(root)
    assert code == 0
    assert "PRECURSOR_WARNING" in (folder / "stdout.log").read_text()

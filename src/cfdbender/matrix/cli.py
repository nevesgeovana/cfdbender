"""Plan, run and collect matrices through the workspace's declared adapter."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from cfdbender import __version__, _cli
from cfdbender._cli import cli_entrypoint, note_post_ran
from cfdbender.core.adapter import SolverAdapter, capabilities
from cfdbender.core.executables import check_builds, check_local_executables
from cfdbender.core.plan import large_input_check, new_post_file, overwrite_outputs, rehash_inputs
from cfdbender.core.progress import sink as progress_sink
from cfdbender.core.refusals import warn_or_refuse, warning_refusals
from cfdbender.core.registry import RunRecord, RunRegistry
from cfdbender.core.run import execute_plan, judge_environment
from cfdbender.core.settings import WorkspaceError, read_init, require_workspace
from cfdbender.hpc import HpcProfile
from cfdbender.matrix import queries
from cfdbender.matrix._console import (
    SIMS_HELP,
    WORKSPACE_HELP,
    Console,
    add_console_options,
    add_refusal_option,
    add_runs_option,
    add_warning_option,
    banner_lines,
    clock_text,
    command_help,
    held_warnings,
    level_of,
    open_live_log,
    opening_lines,
    record_activity,
    refusals_wanted,
    row_lines,
    status_table,
    status_word,
)
from cfdbender.matrix.batch import (
    BatchPlan,
    batch_count,
    batch_plans,
    grouped_options_refusal,
    grouping_receipt,
    submit_batches,
)
from cfdbender.matrix.collection import collect_workspace
from cfdbender.matrix.continuation import begin as begin_continuation
from cfdbender.matrix.continuation import prepare as prepare_continuation
from cfdbender.matrix.post import post_workspace
from cfdbender.matrix.receipt import require_receipt, write_receipt
from cfdbender.matrix.rerun import RerunDecision, decide
from cfdbender.matrix.restart import prepare as prepare_restart
from cfdbender.matrix.selection import select_plan, split_ids
from cfdbender.matrix.submission import (
    point_profiles,
    select_profile,
    submission_plans,
    submit_plan,
    warn_ignored_overrides,
)
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.solvers._host import _exe
from cfdbender.solvers.flightstream import dispatch
from cfdbender.workspace.planning import plan_workspace

#: The commands that leave a live log and activity events in logs/; plan only previews.
LOGGED = frozenset({"run", "collect", "post"})

#: The verbs that act on what the workspace holds, which cfd-workspace answers.
MOVED = (
    "space-in-use",
    "free-space",
    "delete-sims",
    "sync",
    "restore",
    "rebuild",
    "mark-failed",
    "mark-converged",
    "show",
    "log",
    "trace",
    "history",
    "diff",
    "setups",
)


_EPILOG = """\
first campaign (the folder is a workspace made by cfd-workspace init):
  cfd-matrix plan <matrix>            preview the work and write the plan receipt
  cfd-matrix run <matrix> --local     run every active point on this machine
  cfd-matrix status                   see how each simulation ended (--points: each point)
  cfd-matrix collect <matrix>         read the results into the tables of post/<matrix>/
  cfd-matrix post <matrix>            rebuild those tables again, with no solver and no pickup

cfd-matrix acts on a matrix: plan, run, status, collect, post.
cfd-workspace acts on what the workspace holds: show, history, log, trace, diff, mark-failed,
mark-converged, restore, rebuild, space-in-use, free-space, delete-sims, sync, archive, setups.
Every verb takes --workspace, --verbose and --log-level; the verbs that change files preview
until --apply. `cfd-matrix <verb> --help` says the rest.
"""


def _start_line(
    point_id: str, run_types: Mapping[str, str], number: int, total: int, doing: str
) -> str:
    """Return the line that names a point as it starts, with its run type."""
    kind = run_types.get(point_id, "steady")
    verb = "building and running" if doing == "running" else "building and submitting"
    return f"  -> {point_id}  [{kind}]  {verb}  ({number} of {total})"


def _end_line(record: RunRecord) -> str:
    """Return the line that says how a point ended, its error after the status word."""
    word = status_word(record.status, record.classification, record.error)
    return f"     {record.point_id}  {word}" + (f"  ({record.error})" if record.error else "")


def _typed(argv: list[str]) -> str:
    """Quote the arguments as the user's shell reads them: double quotes on Windows."""
    return subprocess.list2cmdline(argv) if sys.platform == "win32" else shlex.join(argv)


def moved_refusal(argv: list[str]) -> str | None:
    """Return the refusal for a verb that moved to cfd-workspace, or None for any other command."""
    for word in argv:
        if word.startswith("-"):
            continue
        if word not in MOVED:
            return None
        return (
            f"cfd-matrix {word} moved: it acts on what the workspace holds, not on a matrix. "
            f"Type: cfd-workspace {_typed(argv)}"
        )
    return None


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from error
    if value < 0:
        raise argparse.ArgumentTypeError("seconds cannot be negative")
    return value


def _rounds(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from error
    if value < 1:
        raise argparse.ArgumentTypeError("--rounds must be at least 1")
    return value


def build_parser() -> tuple[
    argparse.ArgumentParser, dict[str, Callable[[argparse.Namespace], int]]
]:
    """Return the cfd-matrix parser and the handlers of the verbs that read or change records."""
    parser = argparse.ArgumentParser(
        prog="cfd-matrix",
        description="Plan, run and collect a campaign matrix with the solver of its workspace.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_warning_option(parser)
    add_refusal_option(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser(
        "plan",
        help="print proposed work and write the plan receipt post/<matrix>/plan.json",
        allow_abbrev=False,
    )
    plan_parser.add_argument("matrix", help="matrix name in inputs/matrices/ (.fs is optional)")
    plan_parser.add_argument("--workspace", default=".", help=WORKSPACE_HELP)
    plan_parser.add_argument("--rehash-inputs", action="store_true", help=REHASH_HELP)
    plan_parser.add_argument("--no-read-large-inputs", action="store_true", help=NO_READ_HELP)
    run_parser = commands.add_parser(
        "run",
        help="run every active point; on this machine a row's SOLVER_BUILD picks its "
        "executable from inputs/executables.toml and executables.local.toml",
        allow_abbrev=False,
    )
    run_parser.add_argument("matrix", help="matrix name in inputs/matrices/ (.fs is optional)")
    run_parser.add_argument("--workspace", default=".", help=WORKSPACE_HELP)
    run_parser.add_argument("--rehash-inputs", action="store_true", help=REHASH_HELP)
    run_parser.add_argument("--no-read-large-inputs", action="store_true", help=NO_READ_HELP)
    run_parser.add_argument(
        "--restart",
        action="store_true",
        help="continue every selected point from its own last solution, "
        "archiving its previous run first",
    )
    run_parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="continue every selected time-accurate point in place from its latest restart "
        "files, after its REVOLUTIONS (or TIME_ITERATIONS) grew in the matrix row, or after its "
        "job ended early and collect recorded it failed (collect --ended for a job that wrote "
        "no history); the earlier run's configuration, history and logs move to "
        "continued/<n>/ in the point folder and its record joins the runs.json history",
    )
    run_parser.add_argument(
        "--one-run-at-a-time",
        action="store_true",
        help="reserve this machine for one guarded local run at a time",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="skip points already recorded COMPLETED or SUBMITTED and run the rest, so a "
        "grown matrix runs only its new points; a RUNNING point is skipped because its process "
        "may still be alive. Without it a recorded "
        "point refuses the run",
    )
    run_parser.add_argument(
        "--force-rerun",
        dest="force_rerun",
        action="append",
        default=[],
        metavar="POINT",
        help="redo this recorded point, named POL/DP-name or DP-name; repeat for several. "
        "runs.json is copied to archive/<stamp>/runs.json, the point folder moves to "
        "archive/<stamp>/ and its record joins the runs.json history; nothing is deleted",
    )
    run_parser.add_argument(
        "--force-rerun-all",
        dest="force_rerun_all",
        action="store_true",
        help="run every selected point again whatever its recorded state, a SUBMITTED or "
        "RUNNING one included; recorded points are archived as --force-rerun archives, "
        "and a job of an earlier submission is not canceled (a warning says so), "
        "cancel it in the scheduler. Still refused: invalid inputs or rows, a point folder "
        "or output with no record, and under --batch a grouping that differs from the plan "
        "in more than the batch folder numbers",
    )
    collect_parser = commands.add_parser(
        "collect",
        help="read saved results and rebuild post/<matrix>/campaign_summary.json, "
        "the polar files polars/FAM-<KIND>-P<POL>.csv and P<POL>-<GROUP>-<KIND>.csv",
        allow_abbrev=False,
    )
    collect_parser.add_argument("matrix", nargs="?", help="recorded matrix name (default: all)")
    collect_parser.add_argument("--workspace", default=".", help=WORKSPACE_HELP)
    collect_parser.add_argument("--rehash-inputs", action="store_true", help=REHASH_HELP)
    collect_parser.add_argument("--no-read-large-inputs", action="store_true", help=NO_READ_HELP)
    post_parser = commands.add_parser(
        "post",
        help="rebuild every product under post/<matrix>/ from runs.json and the saved "
        "outputs, without running anything",
        allow_abbrev=False,
    )
    post_parser.add_argument("matrix", nargs="?", help="recorded matrix name (default: all)")
    post_parser.add_argument("--workspace", default=".", help=WORKSPACE_HELP)
    post_parser.add_argument("--sims", nargs="+", metavar="POL", help=SIMS_HELP)
    post_parser.add_argument("--points", nargs="+", metavar="NAME", help="only these points")
    add_runs_option(post_parser)
    post_parser.add_argument(
        "--archive",
        action="store_true",
        help="keep a copy of each replaced product under archive/<stamp>/",
    )
    add_console_options(post_parser)
    for command_parser in (collect_parser, post_parser):
        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="replace a product or table file that the manifest does not record (edited by "
            "hand or left by another tool) and go on through every point; without it such a "
            "file stops the command, which names the file",
        )
        command_parser.add_argument(
            "--new-pproc",
            action="store_true",
            help="accept a PPROC file that differs from the planned one, so a new grouping of "
            "the products (an edited [plots] table) needs no new plan and no new run",
        )
    for command_parser in (plan_parser, run_parser, collect_parser):
        if command_parser is not collect_parser:
            command_parser.add_argument(
                "--warning-refusals",
                action="store_true",
                default=argparse.SUPPRESS,
                help="refuse warnings, including unused cluster values",
            )
        command_parser.add_argument(
            "--sims",
            nargs="+",
            metavar="POL",
            help=SIMS_HELP,
        )
        command_parser.add_argument(
            "--points",
            nargs="+",
            metavar="NAME",
            help="only these points, named DP-name or POL/DP-name, within --sims when "
            "given; a name no point carries is refused",
        )
        command_parser.add_argument(
            "--hpc",
            metavar="NAME",
            help="the HPC profile inputs/hpc/<NAME>.toml to use when the workspace holds several",
        )
        command_parser.add_argument(
            "--local",
            action="store_true",
            help=argparse.SUPPRESS
            if command_parser is collect_parser
            else "keep work on THIS machine instead of submitting it",
        )
    for command_parser in (plan_parser, run_parser):
        grouping_choice = command_parser.add_mutually_exclusive_group()
        grouping_choice.add_argument(
            "--batch",
            type=batch_count,
            metavar="N",
            help="submit the rows as N scheduler jobs, rows split across them in matrix order; "
            "cfdbender writes the shell script each job runs, one run after another with no "
            "warm start. Needs an HPC profile; not with --local",
        )
        grouping_choice.add_argument(
            "--batch-per-polar",
            action="store_true",
            help="submit one scheduler job per polar (a matrix row, POL): its points run one "
            "after another inside the job in sweep order, with the row's NCPUS and the sum of "
            "its points' WALLTIME (capped by the profile's max_walltime, with its warning). "
            "Needs an HPC profile; not with --local, --resume, --restart or --force-rerun; a "
            "selection (--sims, --points) plans only the polars it touches",
        )
    run_parser.add_argument(
        "--no-local",
        dest="local",
        action="store_false",
        help="the opposite of --local: submit when a profile exists (the default)",
    )
    run_parser.set_defaults(local=False)
    for command_parser in (plan_parser, run_parser, collect_parser):
        add_console_options(command_parser)
    add_runs_option(collect_parser)
    collect_parser.add_argument(
        "--archive",
        action="store_true",
        help="keep a copy of each replaced summary, table and polar under archive/<stamp>/; "
        "without it they are overwritten in place",
    )
    collect_parser.add_argument(
        "--watch",
        action="store_true",
        help="keep sweeping until no submitted point is outstanding, instead of once",
    )
    collect_parser.add_argument(
        "--interval",
        type=_seconds,
        default=None,
        help="seconds between the two observations that decide an output is settled "
        "(size and modification time stable); without it or --watch no settle check is made",
    )
    collect_parser.add_argument(
        "--watch-interval",
        dest="watch_interval",
        type=_seconds,
        default=None,
        help="seconds between sweeps under --watch (default: 60)",
    )
    collect_parser.add_argument(
        "--ended",
        nargs="+",
        metavar="POINT",
        default=None,
        help="declare these SUBMITTED points (POL/DP-name) as points whose cluster job ended "
        "without job.exit (killed by its walltime, cancelled): each is sealed COMPLETED with a "
        "failed classification, its files kept, so run --continue (from its latest restart "
        "pair) or run --force-rerun can follow. Refused for a point that is not SUBMITTED or "
        "that holds a job.exit (provisional name)",
    )
    collect_parser.add_argument(
        "--rounds",
        type=_rounds,
        default=None,
        help="stop a watch after this many sweeps, whatever is still outstanding",
    )
    verbs: dict[str, Callable[[argparse.Namespace], int]] = {}
    verbs.update(queries.register(commands))
    for name in verbs:
        add_console_options(commands.choices[name])
    return parser, verbs


NO_READ_HELP = (
    "do not open any input larger than 8 MiB (a mesh): record its size and modification time "
    "instead of its SHA-256, check it by those two, and skip with a warning the checks that "
    'need its contents; the same as large_input_check = "identity" in init.toml or '
    "CFDBENDER_NO_READ_LARGE_INPUTS=1; a mesh changed in place or replaced by a different file "
    "with the same size and modification time goes unnoticed; ignored with a warning on "
    "the FlightStream route"
)
REHASH_HELP = (
    "read every input again instead of trusting the digest cache of large files "
    "(.input-digests.cache); without it a file is read again when its file identity, size or "
    "timestamps changed, so a file rewritten with the same size and timestamps goes "
    "unnoticed. On a FlightStream workspace the option is ignored with a warning"
)


@cli_entrypoint
def main(argv: list[str] | None = None) -> int:
    """Plan, run or collect a matrix, printing refusals to stderr with exit code 2.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments, excluding the program name.

    Returns
    -------
    int
        Plan and successful runs return 0; run failures and refusals return 2.
        Collect returns 0 for complete, 1 for failures, 3 for outstanding work,
        or 2 for a refusal or summary write error. Help exits with zero.
    """
    argv = sys.argv[1:] if argv is None else argv
    if (forwarded := dispatch(argv, workspace=False)) is not None:
        return forwarded
    if (moved := moved_refusal(argv)) is not None:
        print(moved, file=sys.stderr)
        return 2
    parser, verbs = build_parser()
    args = parser.parse_args(argv)
    args.typed_argv = list(argv)
    try:
        level = level_of(args)
        root = Path(args.workspace).resolve()
        refuse = refusals_wanted(args, root)
        no_read = getattr(args, "no_read_large_inputs", False)
        if no_read and getattr(args, "rehash_inputs", False):
            raise WorkspaceError(
                "--no-read-large-inputs and --rehash-inputs contradict each other: the first "
                "never reads a large input, the second reads every input again; give one"
            )
    except WorkspaceError as error:
        print(str(error), file=sys.stderr)
        return 2
    # The commands that do work leave a live log and activity events; the read-only
    # verbs leave nothing behind.
    live = open_live_log(root, args.command) if args.command in LOGGED else None
    console = Console(level, live)
    console.block(
        f"cfd-matrix {args.command}",
        opening_lines(
            command_help(parser, args.command),
            workspace=root,
            live_log=live.path.relative_to(root).as_posix() if live is not None else None,
        ),
    )
    code = 2
    if args.command in LOGGED:
        record_activity(root, args.command, "started", argv=list(argv))
    try:
        with (
            held_warnings(root, show=args.console_warnings, log=live),
            warning_refusals(refuse),
            rehash_inputs(getattr(args, "rehash_inputs", False)),
            new_post_file(getattr(args, "new_pproc", False)),
            overwrite_outputs(getattr(args, "overwrite", False)),
            large_input_check("identity" if no_read else None),
        ):
            code = _execute(args, verbs, console)
        return code
    finally:
        if args.command in LOGGED:
            record_activity(
                root,
                args.command,
                "finished" if code == 0 else "failed",
                f"exit code {code}",
                exit_code=code,
            )
        if live is not None:
            live.write(f"# {args.command} ended with exit code {code}")
            live.close()


def _shown(workspace: str, path: Path) -> str:
    """Return a path relative to the workspace, with / separators."""
    try:
        return path.resolve().relative_to(Path(workspace).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _strays_line(strays: Mapping[str, tuple[str, ...]]) -> str:
    """Return the line naming recorded points the matrix no longer has."""
    parts = []
    for matrix, points in strays.items():
        shown = ", ".join(points[:3]) + (", ..." if len(points) > 3 else "")
        parts.append(
            f"{len(points)} recorded points are not in {matrix}.fs any more ({shown}); "
            "retire them with cfd-workspace delete-sims or cfd-workspace archive"
        )
    return "; ".join(parts)


def _collect(args: argparse.Namespace, console: Console) -> int:
    """Collect once, or sweep repeatedly under --watch; the last sweep's exit code wins."""
    watching = bool(args.watch)
    settle = args.interval if args.interval is not None else (2.0 if watching else None)
    pause = 60.0 if args.watch_interval is None else args.watch_interval
    sweeps = 0
    while True:
        sweeps += 1
        report = collect_workspace(
            args.workspace,
            args.matrix,
            sims=split_ids(args.sims),
            points=split_ids(args.points),
            runs=args.runs,
            archive=args.archive,
            settle=settle,
            ended=() if sweeps > 1 else split_ids(args.ended),
        )
        note_post_ran()
        for path in (*report.summaries, *report.tables):
            console.say(f"wrote {_shown(args.workspace, path)}")
        if report.strays:
            console.say(_strays_line(report.strays))
        console.final(
            f"collected {report.collected}, failed {report.failed}, "
            f"outstanding {report.outstanding}"
        )
        if not watching or not report.outstanding or (args.rounds and sweeps >= args.rounds):
            return report.exit_code
        console.detail(f"sweep {sweeps}: {report.outstanding} outstanding; next in {pause:g} s")
        time.sleep(pause)


def _post(args: argparse.Namespace, console: Console) -> int:
    report = post_workspace(
        args.workspace,
        args.matrix,
        sims=split_ids(args.sims),
        points=split_ids(args.points),
        runs=args.runs,
        archive=args.archive,
    )
    note_post_ran()
    for path in (*report.summaries, *report.tables):
        console.say(f"wrote {_shown(args.workspace, path)}")
    console.final(
        f"post read {report.collected}, failed {report.failed}, outstanding {report.outstanding}"
    )
    # pyflightstream's post exits 0 once the products are written, and 3 for a partial
    # rebuild; a failed point is a row of the tables, not a failure of the rebuild.
    return 3 if report.outstanding else 0


def _execute(
    args: argparse.Namespace,
    verbs: dict[str, Callable[[argparse.Namespace], int]],
    console: Console,
) -> int:
    """Run the command; a run is bracketed by its ``[run]`` stage lines."""
    if args.command != "run":
        return _execute_command(args, verbs, console)
    console.chatter(f"[run] started: {Path(args.workspace).resolve()}")
    args.run_outcomes = {}
    code = 2
    try:
        code = _execute_command(args, verbs, console)
        return code
    finally:
        outcomes = dict(sorted(args.run_outcomes.items()))
        detail = f"; outcomes={outcomes}" if outcomes else ""
        console.chatter(f"[run] {'finished' if code == 0 else 'failed'}{detail}")


def _execute_command(
    args: argparse.Namespace,
    verbs: dict[str, Callable[[argparse.Namespace], int]],
    console: Console,
) -> int:
    try:
        if args.command == "run":
            config = read_init(args.workspace, available_solvers=available_solvers())
            selected = select_plan(
                plan_workspace(args.workspace, args.matrix),
                split_ids(args.sims),
                split_ids(args.points),
            )
            profiles = () if args.local else point_profiles(selected, args.hpc)
            args.guard_active = (args.local or not any(profiles)) and (
                args.one_run_at_a_time or config.one_run_at_a_time
            )
            if args.guard_active:
                try:
                    lease = _exe.process_lease("solver")
                    lease.__enter__()
                except ValueError as error:
                    raise WorkspaceError(
                        "another cfdbender solver run holds this machine (one run at a time); "
                        "wait for it to end, then run again"
                    ) from error
                try:
                    console.final("one run at a time guard is on")
                    return _execute_inner(args, verbs, console)
                finally:
                    lease.__exit__(None, None, None)
        return _execute_inner(args, verbs, console)
    except (WorkspaceError, OSError, ValueError) as error:
        console.error("matrix not run: " + str(error))
        return 2


def _execute_inner(
    args: argparse.Namespace,
    verbs: dict[str, Callable[[argparse.Namespace], int]],
    console: Console,
) -> int:
    receipt_rel: str | None = None
    try:
        require_workspace(args.workspace)
        if args.command in verbs:
            return verbs[args.command](args)
        if args.command == "post":
            return _post(args, console)
        if args.command == "collect":
            if not (Path(args.workspace) / "runs.json").is_file():
                select_profile(Path(args.workspace).resolve(), args.hpc)
            return _collect(args, console)
        plan = select_plan(
            plan_workspace(args.workspace, args.matrix),
            split_ids(args.sims),
            split_ids(args.points),
        )
        if not plan.points:
            console.final(f"no active row: every row of {Path(plan.matrix).name} has RUN = 0")
            return 0
        if args.local:
            warn_ignored_overrides(plan)
        selected_profiles = () if args.local else point_profiles(plan, args.hpc)
        cluster = bool(selected_profiles) and all(p is not None for p in selected_profiles)
        profiles = cast(tuple[HpcProfile, ...], selected_profiles)
        downgraded = False
        if cluster and "cluster_staging" not in capabilities(load_solver(plan.solver)):
            local_argv = [word for word in args.typed_argv if word != "--no-local"] + ["--local"]
            warn_or_refuse(
                f"solver {plan.solver!r} does not support cluster_staging; "
                "cluster staging and submission skipped. "
                f"Type: cfd-matrix {_typed(local_argv)}",
                UserWarning,
                stacklevel=2,
            )
            cluster = False
            downgraded = True
            if args.command == "run":
                return 0
        if args.command == "plan":
            if not cluster:
                check_builds(
                    plan.root,
                    plan.solver,
                    load_solver(plan.solver),
                    (point.solver_build for point in plan.points),
                )
            if args.local or downgraded:
                # A local plan looks for the executable as a run does; a cluster plan does not.
                check_local_executables(
                    plan.root, plan.solver, load_solver(plan.solver), plan.points
                )
        batch = getattr(args, "batch", None)
        per_polar = bool(getattr(args, "batch_per_polar", False))
        flag = "--batch-per-polar" if per_polar else "--batch"
        batches: tuple[BatchPlan, ...] = ()
        grouping = None
        if batch is not None or per_polar:
            refusal = grouped_options_refusal(
                local=args.local,
                restart=getattr(args, "restart", False),
                resume=getattr(args, "resume", False),
                force_rerun=getattr(args, "force_rerun", None),
                continue_run=getattr(args, "continue_run", False),
                flag=flag,
            )
            if refusal is None and not cluster:
                refusal = f"{flag} submits rows as scheduler jobs, and no HPC profile is selected"
            if refusal is not None:
                raise WorkspaceError(refusal)
            # A source already COMPLETED, and not run again, feeds its dependents as before;
            # any other source that shares the job feeds them from the job script.
            done = (
                frozenset()
                if getattr(args, "force_rerun_all", False)
                else frozenset(
                    r.point_id for r in RunRegistry(plan.root).read() if r.status == "COMPLETED"
                )
            )
            batches, split_warnings = batch_plans(
                plan, profiles, batch or 0, completed=done, per_polar=per_polar
            )
            fed = {p.point_id: p for group in batches for p in group.points if p.restart_in_job}
            plan = replace(plan, points=tuple(fed.get(p.point_id, p) for p in plan.points))
            for text in split_warnings:
                warn_or_refuse(text, UserWarning, stacklevel=1)
            grouping = grouping_receipt(batches, batch or 0, per_polar=per_polar)
        jobs = submission_plans(plan, profiles) if cluster and not grouping else ()
        if args.command == "plan":
            profile_paths = (
                tuple(p.path.relative_to(plan.root).as_posix() for p in profiles) if cluster else ()
            )
            receipt = write_receipt(plan, profile_paths, grouping)
            console.detail(f"plan receipt: {receipt}")
            receipt_rel = _shown(args.workspace, receipt)
        if args.command == "run":
            profile_paths = (
                tuple(p.path.relative_to(plan.root).as_posix() for p in profiles) if cluster else ()
            )
            require_receipt(plan, profile_paths, grouping, renumbered_batches=args.force_rerun_all)
            submitting = cluster
            adapter = cast(SolverAdapter, load_solver(plan.solver))
            if args.restart and "restart" not in capabilities(adapter):
                warn_or_refuse(
                    f"solver {plan.solver!r} does not support restart; --restart skipped"
                )
                args.restart = False
            if args.restart and (args.resume or args.force_rerun or args.force_rerun_all):
                raise WorkspaceError(
                    "--restart cannot be combined with --resume or --force-rerun options"
                )
            continuing = getattr(args, "continue_run", False)
            if continuing and (
                args.restart or args.resume or args.force_rerun or args.force_rerun_all
            ):
                raise WorkspaceError(
                    "--continue cannot be combined with --restart, --resume or the "
                    "--force-rerun options: it runs on the recorded points it selects"
                )
            if continuing:
                plan, earlier = prepare_continuation(plan, adapter)
                if not submitting:
                    judge_environment(plan, adapter)
                planned = plan
                decision = RerunDecision(plan, (), begin_continuation(plan, earlier))
            else:
                plan = prepare_restart(plan, adapter, own=args.restart, local=not submitting)
                planned = plan
                decision = decide(
                    plan,
                    resume=args.resume,
                    force=args.force_rerun,
                    force_all=args.force_rerun_all or args.restart,
                    submitting=submitting,
                    # A local run this machine would refuse must refuse before a point is
                    # archived, not after: the refusal leaves the records as they were.
                    before_archive=None
                    if submitting
                    else lambda chosen: judge_environment(chosen, adapter),
                )
            if decision.archive is not None:
                console.chatter(f"archived previous evidence: {decision.archive}")
            for point_id, reason in decision.skipped:
                console.chatter(f"  {point_id}  skipped: {reason}")
            for point_id, was in decision.live:
                console.error(
                    f"warning: {point_id} was {was}; a job of the earlier submission may "
                    "still be running; cancelling it is your action"
                )
            plan = decision.plan
            if decision.archive is not None:
                plan = replace(
                    plan,
                    points=tuple(
                        replace(
                            p,
                            restart_directory=(decision.archive / p.restart_directory)
                            .relative_to(plan.root)
                            .as_posix(),
                        )
                        if p.restart_directory
                        and not p.restart_in_job
                        and (decision.archive / p.restart_directory).is_dir()
                        else p
                        for p in plan.points
                    ),
                )
            to_run = len(plan.points)
            began = time.perf_counter()
            if to_run:
                for line in banner_lines(__version__, Path(plan.matrix).stem, to_run, _cli._RNG):
                    console.chatter(line)
            run_types = {p.point_id: p.run_type for p in plan.points}
            profiles = (
                cast(tuple[HpcProfile, ...], point_profiles(plan, args.hpc)) if submitting else ()
            )
            jobs = submission_plans(plan, profiles) if submitting and not batches else ()
            live = not submitting

            def progress(
                kind: str, index: int, total: int, record: RunRecord | None, point_id: str
            ) -> None:
                if kind == "start":
                    console.chatter(_start_line(point_id, run_types, index, total, "running"))
                elif record is not None:
                    console.chatter(_end_line(record))

            if live:
                with progress_sink(console.chatter):
                    records = execute_plan(
                        plan, adapter, on_event=progress, one_run_at_a_time=args.guard_active
                    )
            elif submitting and batches:
                records = submit_batches(plan, batches, adapter)
            elif submitting:
                records = submit_plan(plan, jobs, profiles, adapter)
            else:
                raise WorkspaceError("no HPC profile selected")
            for number, record in enumerate(records, 1):
                if not live:
                    console.chatter(
                        _start_line(record.point_id, run_types, number, len(records), "submitting")
                    )
                    console.chatter(_end_line(record))
                if record.error:
                    # The point line already says it, except on a console that is quiet.
                    if console.level == "quiet":
                        console.error(f"  {record.point_id}: {record.error}")
                elif record.submission and record.submission["timed_out"]:
                    console.error(str(record.submission["stderr"]))
            words = [status_word(r.status, r.classification, r.error) for r in records]
            for word in words:
                args.run_outcomes[word] = args.run_outcomes.get(word, 0) + 1
            if records:
                spent = time.perf_counter() - began
                for line in status_table(words, spent):
                    console.chatter(line)
                ended = ", ".join(f"{w} {n}" for w, n in sorted(args.run_outcomes.items()))
                console.quiet_note(f"{len(words)} point(s) in {clock_text(spent)}: {ended}")
            if to_run:
                recorded = {r.point_id for r in RunRegistry(plan.root).read()}
                rows: dict[str, list[str]] = {}
                for point in planned.points:
                    rows.setdefault(str(point.pol), []).append(point.point_id)
                for line in row_lines(rows, recorded):
                    console.chatter(line)
            if any(
                record.classification == "failed"
                or (record.submission and record.submission["return_code"] != 0)
                for record in records
            ):
                console.error("matrix run with failures")
                return 2
            if any(record.status == "SUBMITTED" for record in records):
                console.final("submitted; extraction pending collect")
            return 0
    except (WorkspaceError, OSError, ValueError) as error:
        prefix = "matrix not run: " if args.command == "run" else ""
        console.error(prefix + str(error))
        return 2
    console.final(f"matrix: {plan.matrix}")
    console.final(f"solver: {plan.solver}")
    console.final(f"{len(plan.points)} point(s)")
    for point in plan.points:
        condition = ", ".join(f"{key}={value:g}" for key, value in sorted(point.condition.items()))
        console.say(
            f"  {point.point_id}: SETUP={point.setup_id} "
            f"{'NUMERICS=' + point.numerics_id + ' ' if point.numerics_id else ''}"
            f"REF={point.reference_id} "
            f"PPROC={point.post_id} WALLTIME={point.walltime or 'NA'}"
            f"  {condition or 'no condition stated'}"
        )
        stage = point.provenance.get("precursor")
        if point.precursor and isinstance(stage, Mapping):
            stop = stage.get("converge_to")
            console.say(
                f"    precursor: steady frozen-rotor stage first, {stage['iterations']} "
                "iterations" + ("" if stop is None else f" at most, stopping at residual {stop:g}")
            )
        labels = [
            f"{name}={value}"
            for name, value in (
                ("AIRCRAFT", point.aircraft),
                ("CONFIGURATION", point.configuration),
                ("DESCRIPTION", point.description),
            )
            if value
        ]
        if labels:
            console.detail(f"    {' '.join(labels)}")
        columns = [
            f"{name}={value}"
            for name, value in (
                ("MESH", point.geometry),
                ("NCPUS", point.ncpus),
                ("SOLVER_BUILD", point.solver_build),
                ("HIDDEN", 1 if point.hidden else None),
            )
            if value is not None
        ]
        if columns:
            console.detail(f"    {' '.join(columns)}")
        for key, value in point.variables.items():
            console.detail(f"    variable {key} = {value!r}")
        carried = f"run type {point.run_type}"
        if point.numerics_id or carried in point.origins.values():
            by_origin: dict[str, list[str]] = {}
            for key, origin in point.origins.items():
                by_origin.setdefault(origin, []).append(key)
            for origin in ("defaults", carried, "numerics", "setup", "row"):
                if origin in by_origin:
                    console.detail(f"    setup keys from {origin}: {', '.join(by_origin[origin])}")
        console.detail(f"    would: {'; '.join(point.actions)}")
        for output in point.outputs:
            console.detail(f"    output: {output}")
    for job, profile in zip(jobs, profiles if jobs else (), strict=True):
        console.say(f"profile: {profile.path.relative_to(plan.root).as_posix()}")
        for warning in job.warnings:
            console.error(f"warning: {warning}")
        console.say(f"descriptor: {job.descriptor_path.as_posix()}")
        if console.level != "quiet":
            print(job.descriptor_text, end="")
        console.say(f"would submit: {shlex.join(job.command)}")
    for group in batches:
        console.say(
            f"batch {group.batch_id}: {group.label} holds row(s) {', '.join(group.sims)}, "
            f"{len(group.points)} point(s); script {group.script}"
        )
        for point in group.points:
            if point.restart_in_job:
                console.say(
                    f"  {point.point_id} RESTART {point.restart_from}: fed inside the job, "
                    "staged by the script when the source ends with status 0"
                )
        for warning in group.submission.warnings:
            console.error(f"warning: {warning}")
        console.say(f"descriptor: {group.submission.descriptor_path.as_posix()}")
        if console.level != "quiet":
            print(group.submission.descriptor_text, end="")
        console.say(f"would submit: {shlex.join(group.submission.command)}")
    if receipt_rel is not None:
        console.final(
            f"{len(plan.points)} point(s) planned; receipt {receipt_rel}; "
            f"next: cfd-matrix run {Path(plan.matrix).stem}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

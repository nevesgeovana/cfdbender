"""Plan, run and collect matrices through the workspace's declared adapter."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import SolverAdapter
from cfdbender.core.run import execute_plan
from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix import queries, records, storage
from cfdbender.matrix._console import (
    Console,
    add_console_options,
    add_runs_option,
    add_warning_option,
    held_warnings,
    level_of,
    open_live_log,
    record_activity,
)
from cfdbender.matrix.collection import collect_workspace
from cfdbender.matrix.post import post_workspace
from cfdbender.matrix.receipt import require_receipt, write_receipt
from cfdbender.matrix.rerun import decide
from cfdbender.matrix.selection import select_plan, split_ids
from cfdbender.matrix.submission import select_profile, submission_plans, submit_plan
from cfdbender.solvers import load_solver
from cfdbender.solvers.flightstream import dispatch
from cfdbender.workspace.planning import plan_workspace

#: The commands that leave a live log and activity events in logs/.
LOGGED = frozenset({"plan", "run", "collect", "post"})


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
    parser = argparse.ArgumentParser(prog="cfd-matrix", allow_abbrev=False)
    add_warning_option(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser(
        "plan",
        help="print proposed work and write the plan receipt post/<matrix>/plan.json",
        allow_abbrev=False,
    )
    plan_parser.add_argument("matrix", help="matrix name in inputs/matrices/ (.fs is optional)")
    plan_parser.add_argument("--workspace", default=".", help="campaign workspace (default: .)")
    run_parser = commands.add_parser(
        "run",
        help="run every active point; on this machine a row's SOLVER_BUILD picks its "
        "executable from inputs/executables.toml and executables.local.toml",
        allow_abbrev=False,
    )
    run_parser.add_argument("matrix", help="matrix name in inputs/matrices/ (.fs is optional)")
    run_parser.add_argument("--workspace", default=".", help="campaign workspace (default: .)")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="skip points already recorded COMPLETED or SUBMITTED and run the rest, so a "
        "grown matrix runs only its new points; a point left RUNNING by a crashed local run "
        "runs again when no solver process holds this host's slot. Without it a recorded "
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
        help="redo every recorded point of the matrix, archived as --force-rerun archives; "
        "a SUBMITTED job is not cancelled, cancel it in the scheduler first",
    )
    collect_parser = commands.add_parser(
        "collect",
        help="read saved results and rebuild post/<matrix>/campaign_summary.json, "
        "campaign_sweep.csv and polars/P<POL>.csv",
        allow_abbrev=False,
    )
    collect_parser.add_argument("matrix", nargs="?", help="recorded matrix name (default: all)")
    collect_parser.add_argument("--workspace", default=".", help="campaign workspace (default: .)")
    post_parser = commands.add_parser(
        "post",
        help="rebuild every product under post/<matrix>/ from runs.json and the saved "
        "outputs, without running anything",
        allow_abbrev=False,
    )
    post_parser.add_argument("matrix", nargs="?", help="recorded matrix name (default: all)")
    post_parser.add_argument("--workspace", default=".", help="campaign workspace (default: .)")
    post_parser.add_argument("--sims", nargs="+", metavar="POL", help="only these simulations")
    post_parser.add_argument("--points", nargs="+", metavar="NAME", help="only these points")
    add_runs_option(post_parser)
    post_parser.add_argument(
        "--archive",
        action="store_true",
        help="keep a copy of each replaced product under archive/<stamp>/",
    )
    add_console_options(post_parser)
    for command_parser in (plan_parser, run_parser, collect_parser):
        command_parser.add_argument(
            "--sims",
            nargs="+",
            metavar="POL",
            help="only these simulations, by POL, space or comma separated "
            "(--sims 1001 1002 or --sims 1001,1002); an id the matrix lacks is refused",
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
            help="keep work on THIS machine instead of submitting it",
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
        "--rounds",
        type=_rounds,
        default=None,
        help="stop a watch after this many sweeps, whatever is still outstanding",
    )
    verbs: dict[str, Callable[[argparse.Namespace], int]] = {}
    for module in (queries, records, storage):
        verbs.update(module.register(commands))
    for name in verbs:
        add_console_options(commands.choices[name])
    return parser, verbs


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
    parser, verbs = build_parser()
    args = parser.parse_args(argv)
    try:
        level = level_of(args)
    except WorkspaceError as error:
        print(str(error), file=sys.stderr)
        return 2
    root = Path(args.workspace).resolve()
    # The commands that do work leave a live log and activity events; the read-only
    # verbs leave nothing behind.
    live = open_live_log(root, args.command) if args.command in LOGGED else None
    console = Console(level, live)
    code = 2
    if args.command in LOGGED:
        record_activity(root, args.command, "started", argv=list(argv))
    try:
        with held_warnings(root, show=args.console_warnings, log=live):
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
        )
        for path in report.summaries:
            console.say(f"collected summary: {path}")
        for path in report.tables:
            console.say(f"collected table: {path}")
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
    for path in (*report.summaries, *report.tables):
        console.say(f"post product: {path}")
    console.final(
        f"post read {report.collected}, failed {report.failed}, outstanding {report.outstanding}"
    )
    return report.exit_code


def _execute(
    args: argparse.Namespace,
    verbs: dict[str, Callable[[argparse.Namespace], int]],
    console: Console,
) -> int:
    try:
        if args.command in verbs:
            return verbs[args.command](args)
        if args.command == "post":
            return _post(args, console)
        profile = select_profile(Path(args.workspace).resolve(), args.hpc)
        if args.command == "collect":
            return _collect(args, console)
        plan = select_plan(
            plan_workspace(args.workspace, args.matrix),
            split_ids(args.sims),
            split_ids(args.points),
        )
        if args.command == "plan":
            receipt = write_receipt(plan)
            console.detail(f"plan receipt: {receipt}")
        jobs = submission_plans(plan, profile) if profile is not None and not args.local else ()
        if args.command == "run":
            require_receipt(plan)
            submitting = profile is not None and not args.local
            decision = decide(
                plan,
                resume=args.resume,
                force=args.force_rerun,
                force_all=args.force_rerun_all,
                submitting=submitting,
            )
            if decision.archive is not None:
                console.say(f"archived previous evidence: {decision.archive}")
            for point_id, reason in decision.skipped:
                console.say(f"  {point_id}  skipped: {reason}")
            plan = decision.plan
            jobs = submission_plans(plan, profile) if submitting and profile is not None else ()
            adapter = cast(SolverAdapter, load_solver(plan.solver))
            records = (
                submit_plan(plan, jobs, profile, adapter)
                if profile is not None and not args.local
                else execute_plan(plan, adapter)
            )
            for record in records:
                console.say(f"  {record.point_id}  {record.status} {record.classification}")
                if record.error:
                    console.error(f"  {record.point_id}: {record.error}")
                elif record.submission and record.submission["timed_out"]:
                    console.error(str(record.submission["stderr"]))
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
            f"  {point.point_id}: SET={point.setup_id} REF={point.reference_id} "
            f"PPROC={point.post_id} WALLTIME={point.walltime or 'NA'}"
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
            console.say(f"    {' '.join(labels)}")
        columns = [
            f"{name}={value}"
            for name, value in (
                ("GEOMETRY", point.geometry),
                ("NCPUS", point.ncpus),
                ("SOLVER_BUILD", point.solver_build),
                ("HIDDEN", 1 if point.hidden else None),
            )
            if value is not None
        ]
        if columns:
            console.say(f"    {' '.join(columns)}")
        for key, value in point.variables.items():
            console.say(f"    variable {key} = {value!r}")
        console.say(f"    {condition or 'no condition stated'}")
        console.say(f"    would: {'; '.join(point.actions)}")
        for output in point.outputs:
            console.say(f"    output: {output}")
    for job in jobs:
        for warning in job.warnings:
            console.error(f"warning: {warning}")
        console.say(f"descriptor: {job.descriptor_path.as_posix()}")
        if console.level != "quiet":
            print(job.descriptor_text, end="")
        console.say(f"would submit: {shlex.join(job.command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

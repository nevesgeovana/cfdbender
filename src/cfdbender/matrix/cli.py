"""Plan, run and collect matrices through the workspace's declared adapter."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import cast

from cfdbender.core.adapter import SolverAdapter
from cfdbender.core.run import execute_plan
from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix.collection import collect_workspace
from cfdbender.matrix.rerun import decide
from cfdbender.matrix.selection import select_plan, split_ids
from cfdbender.matrix.submission import select_profile, submission_plans, submit_plan
from cfdbender.solvers import load_solver
from cfdbender.solvers.flightstream import dispatch
from cfdbender.workspace.planning import plan_workspace


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
    parser = argparse.ArgumentParser(prog="cfd-matrix", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser(
        "plan", help="print proposed work without writing files", allow_abbrev=False
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
    args = parser.parse_args(argv)
    try:
        profile = select_profile(Path(args.workspace).resolve(), args.hpc)
        if args.command == "collect":
            report = collect_workspace(
                args.workspace,
                args.matrix,
                sims=split_ids(args.sims),
                points=split_ids(args.points),
            )
            for path in report.summaries:
                print(f"collected summary: {path}")
            for path in report.tables:
                print(f"collected table: {path}")
            print(
                f"collected {report.collected}, failed {report.failed}, "
                f"outstanding {report.outstanding}"
            )
            return report.exit_code
        plan = select_plan(
            plan_workspace(args.workspace, args.matrix),
            split_ids(args.sims),
            split_ids(args.points),
        )
        jobs = submission_plans(plan, profile) if profile is not None and not args.local else ()
        if args.command == "run":
            submitting = profile is not None and not args.local
            decision = decide(
                plan,
                resume=args.resume,
                force=args.force_rerun,
                force_all=args.force_rerun_all,
                submitting=submitting,
            )
            if decision.archive is not None:
                print(f"archived previous evidence: {decision.archive}")
            for point_id, reason in decision.skipped:
                print(f"  {point_id}  skipped: {reason}")
            plan = decision.plan
            jobs = submission_plans(plan, profile) if submitting and profile is not None else ()
            adapter = cast(SolverAdapter, load_solver(plan.solver))
            records = (
                submit_plan(plan, jobs, profile, adapter)
                if profile is not None and not args.local
                else execute_plan(plan, adapter)
            )
            for record in records:
                print(f"  {record.point_id}  {record.status} {record.classification}")
                if record.error:
                    print(f"  {record.point_id}: {record.error}", file=sys.stderr)
                elif record.submission and record.submission["timed_out"]:
                    print(record.submission["stderr"], file=sys.stderr)
            if any(
                record.classification == "failed"
                or (record.submission and record.submission["return_code"] != 0)
                for record in records
            ):
                print("matrix run with failures", file=sys.stderr)
                return 2
            if any(record.status == "SUBMITTED" for record in records):
                print("submitted; extraction pending collect")
            return 0
    except (WorkspaceError, OSError, ValueError) as error:
        prefix = "matrix not run: " if args.command == "run" else ""
        print(prefix + str(error), file=sys.stderr)
        return 2
    print(f"matrix: {plan.matrix}")
    print(f"solver: {plan.solver}")
    print(f"{len(plan.points)} point(s)")
    for point in plan.points:
        condition = ", ".join(f"{key}={value:g}" for key, value in sorted(point.condition.items()))
        print(
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
            print(f"    {' '.join(labels)}")
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
            print(f"    {' '.join(columns)}")
        for key, value in point.variables.items():
            print(f"    variable {key} = {value!r}")
        print(f"    {condition or 'no condition stated'}")
        print(f"    would: {'; '.join(point.actions)}")
        for output in point.outputs:
            print(f"    output: {output}")
    for job in jobs:
        for warning in job.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"descriptor: {job.descriptor_path.as_posix()}")
        print(job.descriptor_text, end="")
        print(f"would submit: {shlex.join(job.command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

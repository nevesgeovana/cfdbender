"""Count how often one large mesh is opened by `cfd-matrix` commands and time them.

Run with PYTHONPATH=src. A synthetic SU2 text mesh of ``--mib`` MiB is written under ``--dir``
(deleted afterwards) and ``--steps`` (default ``plan,plan,run``) are timed in turn on it; a step
may end in ``+rehash`` for ``--rehash-inputs``. The audit hook counts every open of the mesh by
any code path, so streamed hashing, scanning and whole-file reads are all counted.
"""

import argparse
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests" / "tier1_offline"))

from cfdbender.core.workspace import init_workspace  # noqa: E402
from cfdbender.matrix.cli import main  # noqa: E402
from cfdbender.solvers import available_solvers  # noqa: E402

FAKE = HERE.parent / "tests" / "fixtures" / "su2" / "fake_su2.py"
HEADER = (
    "POL | HIDDEN | RUN | AIRCRAFT | CONFIGURATION | DESCRIPTION | FLIGHT_CONDITION "
    "| SWEEP_VALUES | GEOMETRY | REF | SET | PPROC | SYMMETRY | SYMMETRY_LOADS | NCPUS "
    "| WALLTIME | FS_BUILD | WORKFLOW | VAR_NAMES_VALUES"
)


def build(root: Path, mib: int, rows: int, points: int) -> Path:
    """Write the workspace and the padded mesh; return the mesh path."""
    init_workspace(root, solver="su2", available_solvers=available_solvers())
    sweep = ",".join(str(p) for p in range(points))
    lines = "".join(
        f" {7001 + i} | 1 | 1 | Model | clean | row | MACH:0.5, ALTFT:10000, ALPHA:sweep, "
        f"BETA:0 | {sweep} | large.su2 | 001 | 001 | 001 | - | - | 4 | 2h | - | - | -\n"
        for i in range(rows)
    )
    files = {
        "inputs/setups/s001.toml": 'iterations = 40\nSOLVER = "EULER"\n[groups]\nwing = ["wall"]\n'
        f'[solver]\nexecutable = "{FAKE.as_posix()}"\n',
        "inputs/references/r001.toml": "area = 1.0\nlength = 1.0\nmoment_origin = [0, 0, 0]\n",
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": f"{HEADER}\n{'-' * 60}\n{lines}",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    mesh = root / "inputs/geometries/large.su2"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    with mesh.open("wb") as stream:
        stream.write(b"NDIME= 3\nNMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 0\n")
        block = (b"%" + b"x" * 1022 + b"\n") * 1024
        for _ in range(mib):
            stream.write(block)
    return mesh


def measure(args: argparse.Namespace) -> None:
    """Build one workspace and time each step in turn, printing opens of the mesh per step.

    A step is a command name, optionally followed by ``+rehash`` for ``--rehash-inputs``.
    """
    root = Path(tempfile.mkdtemp(dir=args.dir)).resolve()
    try:
        mesh = build(root, args.mib, args.rows, args.points).resolve()
        time.sleep(2.5)  # let the file settle, as a mesh that was not just written is
        opens = [0]
        names = {mesh.name, "mesh.su2"}  # the mesh and its link in each point folder

        def hook(event: str, info: tuple) -> None:
            if event == "open" and Path(str(info[0])).name in names:
                opens[0] += 1

        sys.addaudithook(hook)
        for step in args.steps.split(","):
            command, _, flag = step.partition("+")
            argv = [command, "study", "--workspace", str(root)]
            if command == "run":
                argv.append("--local")
            if flag == "rehash":
                argv.append("--rehash-inputs")
            opens[0] = 0
            if not args.no_memory:
                tracemalloc.start()
            start = time.perf_counter()
            code = main(argv)
            elapsed = time.perf_counter() - start
            peak = 0 if args.no_memory else tracemalloc.get_traced_memory()[1]
            if not args.no_memory:
                tracemalloc.stop()
            print(
                f"{step:14s} {args.mib} MiB, {args.rows} rows x {args.points} points: "
                f"exit {code}, {elapsed:.2f} s, peak {peak / 1024**2:.1f} MiB, "
                f"mesh opens {opens[0]}"
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def cli() -> None:
    """Parse arguments and measure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=128)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--steps", default="plan,plan,run")
    parser.add_argument("--dir", default="C:/t")
    parser.add_argument("--no-memory", action="store_true")
    measure(parser.parse_args())


if __name__ == "__main__":
    cli()

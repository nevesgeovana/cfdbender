"""Count how often one large mesh is read by the OpenFOAM and PyFR adapters, and time the steps.

Run with PYTHONPATH=src. A synthetic Gmsh mesh of ``--mib`` MiB is written under ``--dir`` (deleted
afterwards) in a workspace per adapter, using the offline stand-in executables, and ``--steps``
(default ``plan,plan,run,collect``) are timed in turn. A step may end in ``+rehash`` for
``--rehash-inputs``. Per step it prints the wall time, the peak Python memory, the opens of the
mesh by any code path (audit hook) and the bytes read from it through the tracked reader and
``Path.read_bytes``.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOAM_FAKE = HERE.parent / "tests" / "fixtures" / "openfoam" / "fake_openfoam.py"

from cfdbender.core import plan as plan_module  # noqa: E402
from cfdbender.matrix.cli import main  # noqa: E402
from cfdbender.workspace.cli import main as workspace  # noqa: E402

PYFR_STANDIN = (
    "import pathlib,sys\n"
    "if 'import' in sys.argv: pathlib.Path('mesh.pyfrm').write_text('mesh')\n"
    "else: pathlib.Path('forces.csv').write_text('t,px,py,vx,vy\\n'"
    " + '0,0.1,0.3,0,0\\n1,0.1,0.3,0,0\\n2,0.1,0.3,0,0\\n3,0.1,0.3,0,0\\n')\n"
)
FOAM_STANDIN = (
    "import pathlib, runpy, shutil\n"
    "try:\n"
    "    runpy.run_path({fake!r}, run_name='__main__')\n"
    "except SystemExit:\n"
    "    pass\n"
    "source = pathlib.Path('postProcessing/forceCoeffs/0/coefficient.dat')\n"
    "target = pathlib.Path('postProcessing/forceCoeffs_main/0/coefficient.dat')\n"
    "target.parent.mkdir(parents=True)\n"
    "shutil.copy(source, target)\n"
)
NAMES = '$PhysicalNames\n3\n2 1 "wing"\n2 2 "tail"\n2 3 "farfield"\n$EndPhysicalNames\n'


def mesh_bytes(stream, mib: int, version: str) -> None:
    """Write a Gmsh header, physical names and padding of ``mib`` MiB."""
    stream.write(f"$MeshFormat\n{version} 0 8\n$EndMeshFormat\n{NAMES}$Nodes\n".encode("ascii"))
    block = (b"1 0.5 0.25 0.125" + b" " * 1006 + b"\n") * 1024
    for _ in range(mib):
        stream.write(block)
    stream.write(b"$EndNodes\n")


def build(root: Path, adapter: str, mib: int, points: int) -> str:
    """Write one workspace and its mesh; return the mesh file name."""
    assert workspace(["init", "--solver", adapter, "--workspace", str(root)]) == 0
    sweep = ",".join(str(p) for p in range(points))
    if adapter == "openfoam":
        standin = root / "foam_standin.py"
        standin.write_text(FOAM_STANDIN.format(fake=FOAM_FAKE.as_posix()), encoding="ascii")
        setup = (
            'mesh = "inputs/meshes/large.msh"\napplication = "simpleFoam"\nspeed = 10.0\n'
            'nu = 1.5e-5\nturbulence = "SpalartAllmaras"\niterations = 4\n'
            '[groups]\nmain = ["wing"]\n[solver]\nexecutable = '
            f"{json.dumps(standin.as_posix())}\n"
        )
        reference = "area = 0.1\nlength = 1.0\nmoment_origin = [0.25, 0.0, 0.0]\n"
        version = "2.2"
    else:
        standin = root / "standin.py"
        standin.write_text(PYFR_STANDIN, encoding="ascii")
        setup = (
            'mesh = "inputs/meshes/large.msh"\nmach = 0.2\nreynolds = 10000\norder = 3\n'
            f"tend = 3\nwindow_start = 0\n[solver]\nexecutable = {json.dumps(standin.as_posix())}\n"
        )
        reference = ""
        version = "4.1"
    matrix = (
        "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        f"1001|1|001|001|001|ALPHA:sweep|{sweep}\n"
    )
    files = {
        "inputs/setups/s001.toml": setup,
        "inputs/references/r001.toml": reference,
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": matrix,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    mesh = root / "inputs/meshes/large.msh"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    with mesh.open("wb") as stream:
        mesh_bytes(stream, mib, version)
    return mesh.name


def measure_one(args: argparse.Namespace, adapter: str) -> None:
    """Build a workspace for ``adapter`` and time each step, printing mesh reads per step."""
    root = Path(tempfile.mkdtemp(dir=args.dir)).resolve()
    try:
        name = build(root, adapter, args.mib, args.points)
        time.sleep(2.5)  # let the file settle, as a mesh that was not just written is
        names = {name, "mesh.msh"}  # the mesh and its link in each point folder
        opens = [0]
        read = [0]

        def hook(event: str, info: tuple) -> None:
            if event == "open" and Path(str(info[0])).name in names:
                opens[0] += 1

        real_open = plan_module._open_input
        real_read_bytes = Path.read_bytes

        class Counting:
            def __init__(self, stream):
                self.stream = stream

            def read(self, *size):
                data = self.stream.read(*size)
                read[0] += len(data)
                return data

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *info):
                return self.stream.__exit__(*info)

        def counted_open(path):
            stream = real_open(path)
            return Counting(stream) if Path(path).name in names else stream

        def counted_read_bytes(self):
            data = real_read_bytes(self)
            if self.name in names:
                read[0] += len(data)
            return data

        plan_module._open_input = counted_open
        Path.read_bytes = counted_read_bytes  # type: ignore[method-assign]
        sys.addaudithook(hook)
        for step in args.steps.split(","):
            command, _, flag = step.partition("+")
            argv = [command, "study", "--workspace", str(root)]
            if command == "run":
                argv.append("--local")
            if flag == "rehash":
                argv.append("--rehash-inputs")
            opens[0] = read[0] = 0
            tracemalloc.start()
            start = time.perf_counter()
            code = main(argv)
            elapsed = time.perf_counter() - start
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            print(
                f"{adapter:9s} {step:14s} {args.mib} MiB: exit {code}, {elapsed:6.2f} s, "
                f"peak {peak / 1024**2:7.1f} MiB, mesh opens {opens[0]}, "
                f"mesh bytes read {read[0] / 1024**2:7.1f} MiB"
            )
        plan_module._open_input = real_open
        Path.read_bytes = real_read_bytes  # type: ignore[method-assign]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def cli() -> None:
    """Parse arguments and measure each adapter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=128)
    parser.add_argument("--points", type=int, default=2)
    parser.add_argument("--steps", default="plan,plan,run,collect")
    parser.add_argument("--adapters", default="openfoam,pyfr")
    parser.add_argument("--dir", default="C:/t")
    args = parser.parse_args()
    for adapter in args.adapters.split(","):
        measure_one(args, adapter)


if __name__ == "__main__":
    cli()

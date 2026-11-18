"""Time and measure `plan` on a synthetic SU2 workspace with one large mesh.

Run with PYTHONPATH=src. The mesh is a valid minimal SU2 text mesh padded with comment lines;
``--dir`` puts the workspace (and so the large file) where the disk allows, and it is deleted
afterwards. Prints the elapsed seconds, the tracemalloc peak and, when the planner exposes its
read seam, the number of times the mesh was opened and the bytes hashed.
"""

import argparse
import shutil
import tempfile
import time
import tracemalloc
from pathlib import Path

from cfdbender.core import plan as plan_module
from cfdbender.core.plan import plan_matrix
from cfdbender.solvers.su2 import validate_row

SETUP = "\n".join(
    [
        'mesh = "inputs/mesh/large.su2"',
        "iterations = 40",
        'convergence_field = "RMS_DENSITY"',
        "freestream_temperature = 288.15",
        "reynolds = 1000000",
        'SOLVER = "EULER"',
        "CFL_NUMBER = 2.0",
        '[groups]\nwing = ["wall"]',
    ]
)
REFERENCE = "area = 1.0\nlength = 1.0\nmoment_origin = [0.0, 0.0, 0.0]\n"


def build(root: Path, mib: int, rows: int, points: int) -> Path:
    """Write the workspace and return the mesh path."""
    sweep = ",".join(str(p) for p in range(points))
    files = {
        "init.toml": 'schema_version = 1\nsolver = "su2"\n',
        "inputs/setups/s001.toml": SETUP + "\n",
        "inputs/references/r001.toml": REFERENCE,
        "inputs/pproc/p001.toml": "history = true\noutput_frequency = 5\n",
        "inputs/matrices/study.fs": "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        + "".join(f"{1001 + i}|1|001|001|001|MACH:0.3,ALPHA:sweep|{sweep}\n" for i in range(rows)),
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii")
    mesh = root / "inputs/mesh/large.su2"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    with mesh.open("wb") as stream:
        stream.write(b"NDIME= 2\nNMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 0\n")
        block = (b"%" + b"x" * 1022 + b"\n") * 1024
        for _ in range(mib):
            stream.write(block)
    return mesh


def main() -> None:
    """Build the workspace, plan it once and print the measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=128)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--dir", default=None, help="parent folder of the temporary workspace")
    parser.add_argument("--no-memory", action="store_true", help="skip tracemalloc (faster)")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(dir=args.dir))
    try:
        mesh = build(root, args.mib, args.rows, args.points).resolve()
        opens = [0]
        fed = [0]
        seam_open = getattr(plan_module, "_open_input", None)
        seam_hasher = getattr(plan_module, "_hasher", None)
        if seam_open is not None and seam_hasher is not None:

            def open_input(path):
                opens[0] += path == mesh
                return seam_open(path)

            class Hasher:
                def __init__(self):
                    self.inner = seam_hasher()

                def update(self, data):
                    fed[0] += len(data)
                    self.inner.update(data)

                def hexdigest(self):
                    return self.inner.hexdigest()

            plan_module._open_input = open_input
            plan_module._hasher = Hasher
        if not args.no_memory:
            tracemalloc.start()
        start = time.perf_counter()
        result = plan_matrix(root, "study", validators={"su2": validate_row})
        elapsed = time.perf_counter() - start
        peak = 0 if args.no_memory else tracemalloc.get_traced_memory()[1]
        print(
            f"{args.mib} MiB, {args.rows} rows, {len(result.points)} points: {elapsed:.3f} s, "
            f"peak {peak / 1024**2:.1f} MiB, mesh opens {opens[0]}, "
            f"bytes hashed {fed[0] / (args.mib * 1024**2):.2f}x the mesh"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

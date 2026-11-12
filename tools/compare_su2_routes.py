"""Compare both SU2 routes on the committed tiny case in the same interpreter.

Run with --output NEW_DIRECTORY. Set CFDBENDER_PYSU2_PATH to the wrapper
installation and CFDBENDER_SU2_EXE to SU2_CFD, or place SU2_CFD beside the
wrapper. Both routes run sequentially with one thread. Existing output refuses.
This tool uses the installed cfdbender wheel and requires no gmsh or psutil.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from cfdbender.core.adapter import load_point
from cfdbender.core.run import execute_plan
from cfdbender.solvers import su2
from cfdbender.solvers.su2._execute import _argv
from cfdbender.solvers.su2._ram import require_ram
from cfdbender.solvers.su2._reference import prepare
from cfdbender.solvers.su2.routes import select_route


def executable() -> str:
    """Resolve the explicit executable or the one beside the wrapper install."""
    if os.environ.get("CFDBENDER_SU2_EXE"):
        # Same resolution rule as the executable route itself (explicit path first).
        try:
            argv = _argv({}, Path.cwd())
        except ValueError:
            raise ValueError("CFDBENDER_SU2_EXE does not name an executable") from None
        return str(Path(argv[-2]).resolve())
    if wrapper := os.environ.get("CFDBENDER_PYSU2_PATH"):
        for name in ("SU2_CFD", "SU2_CFD.exe"):
            candidate = Path(wrapper).expanduser().resolve() / name
            if candidate.is_file():
                return str(candidate)
    raise ValueError("set CFDBENDER_SU2_EXE or locate SU2_CFD via CFDBENDER_PYSU2_PATH")


def compare(output: Path, mesh: Path, exe: str) -> tuple[list[dict[str, object]], float | None]:
    """Execute both routes and calculate the final CL/CD maximum difference.

    Parameters
    ----------
    output : Path
        New comparison directory, containing one workspace per route.
    mesh : Path
        Same committed reference mesh for both routes.
    exe : str
        Explicit executable path.

    Returns
    -------
    tuple
        Table rows and maximum absolute final CL/CD difference, or None when
        either route has no coefficients. Peak RAM is explicitly not measured:
        cumulative child high-water marks cannot distinguish these two runs.
    """
    require_ram(0.25)
    wrapper = os.environ.get("CFDBENDER_PYSU2_PATH")
    select_route("pysu2", pysu2_path=str(Path(wrapper).expanduser().resolve()) if wrapper else None)
    # Resolve preflight inputs before creating either workspace.
    mesh.read_bytes()
    if output.exists():
        raise ValueError("comparison output already exists; choose a new directory")
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    coefficients: list[dict[str, float]] = []
    for route in ("executable", "pysu2"):
        require_ram(0.25)
        proposal = prepare(output / route, mesh, route, exe if route == "executable" else None)
        start = time.perf_counter()
        records = execute_plan(proposal, su2)
        elapsed = time.perf_counter() - start
        point = load_point(proposal.root, proposal.points[0], solver="su2")
        result = su2.collect(point)
        final = {
            key: result.coefficients[key].value
            for key in ("CL", "CD")
            if key in result.coefficients
        }
        coefficients.append(final)
        rows.append(
            {
                "route": result.to_dict()["route"],
                "wall_s": elapsed,
                "peak_ram": "not measured",
                "classification": records[0].classification,
                "CL": final.get("CL"),
                "CD": final.get("CD"),
            }
        )
    difference = (
        max(abs(coefficients[0][key] - coefficients[1][key]) for key in ("CL", "CD"))
        if all({"CL", "CD"} <= values.keys() for values in coefficients)
        else None
    )
    return rows, difference


def main(argv: list[str] | None = None) -> int:
    """Print the comparison; return 0 for matching numerical classifications.

    Parameters
    ----------
    argv : list[str], optional
        Command arguments, otherwise process arguments.

    Returns
    -------
    int
        0 when both classifications agree and are not failed; 1 for disagreement
        or failed runs; 2 when inputs or availability prevent a comparison.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/su2/naca0012_small.su2",
    )
    args = parser.parse_args(argv)
    try:
        require_ram(0.25)
        rows, difference = compare(args.output.resolve(), args.mesh.resolve(), executable())
    except (OSError, ValueError) as error:
        print(f"comparison refused: {error}", file=sys.stderr)
        return 2
    print("route | wall time (s) | peak RAM | classification | final CL | final CD")
    for row in rows:
        print(
            f"{row['route']} | {row['wall_s']:.6f} | {row['peak_ram']} | "
            f"{row['classification']} | {row['CL']} | {row['CD']}"
        )
    print(
        "max abs final CL/CD difference:", difference if difference is not None else "unavailable"
    )
    return 0 if rows[0]["classification"] == rows[1]["classification"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

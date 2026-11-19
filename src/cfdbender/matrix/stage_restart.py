"""Stage the RESTART solution of a point from its source, as a batch job script calls it.

``python -m cfdbender.matrix.stage_restart POINT_FOLDER`` runs after the source of a RESTART
in the same job has ended with status 0. It reads the point folder's ``restart-pending.json``,
checks that each source file exists and is not empty, copies it under the names the solver
reads and records the SHA-256 of each copy in ``restart-staged.json``. It never edits the
solver's input file. On a failure it appends one line to the point's ``stderr.log`` and exits 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cfdbender.core.restart import stage_pending


def main(argv: list[str] | None = None) -> int:
    """Stage one point folder's pending restart solution; return the process exit status."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m cfdbender.matrix.stage_restart POINT_FOLDER", file=sys.stderr)
        return 2
    folder = Path(args[0])
    try:
        stage_pending(folder)
    except (OSError, ValueError) as failure:
        with (folder / "stderr.log").open("a", encoding="utf-8", newline="") as stream:
            stream.write(f"{failure}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

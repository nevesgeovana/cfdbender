"""Offline argument-forwarding launcher; starts one stand-in, never MPI."""

import json
import subprocess
import sys
from pathlib import Path

assert sys.argv[1:3] == ["-np", "192"]
Path("mpi-argv.json").write_text(json.dumps(sys.argv[1:]), encoding="ascii")
raise SystemExit(subprocess.call(sys.argv[3:]))

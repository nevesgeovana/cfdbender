"""Exercise Allrun command order without OpenFOAM or MPI."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

tool, *args = sys.argv[1:]
with Path("commands.txt").open("a", encoding="ascii") as stream:
    stream.write(tool + " " + " ".join(args) + "\n")
if os.environ.get("FAKE_FAIL_TOOL") == tool:
    raise SystemExit(7)
if tool == "mpirun":
    assert args[:3] == ["-np", "12", "simpleFoam"]
    command = args[2:]
    if os.name == "nt":
        command = ["C:/Program Files/Git/bin/bash.exe", "-c", 'exec "$@"', "--", *command]
    raise SystemExit(subprocess.call(command))
if tool == "simpleFoam":
    assert args in ([], ["-parallel"])
    runpy.run_path(str(Path(__file__).with_name("fake_openfoam.py")), run_name="__main__")
if tool == "reconstructPar":
    assert args == ["-latestTime"]
    assert Path("postProcessing/forceCoeffs/0/coefficient.dat").is_file()

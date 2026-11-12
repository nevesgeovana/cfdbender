"""Record converter placeholders and create synthetic offline mesh outputs."""

import json
import os
import sys
from pathlib import Path

mesh, case = sys.argv[1:]
Path("converter-args.json").write_text(json.dumps([mesh, case]), encoding="ascii")
assert Path(mesh).read_bytes() == b"synthetic binary mesh\x00"
assert Path(case).resolve() == Path.cwd()
if os.environ.get("FAKE_CONVERTER_FAIL") == "1":
    raise SystemExit(9)
boundary = Path(case) / "constant/polyMesh/boundary"
boundary.parent.mkdir(parents=True, exist_ok=True)
boundary.write_text("synthetic boundary", encoding="ascii")
# Existing offline simpleFoam stand-in expects this sentinel; it is not a mesh reader.
Path("mesh.msh").write_bytes(b"synthetic imported sentinel")

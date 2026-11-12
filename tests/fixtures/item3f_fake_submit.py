"""Neutral offline submit client: record argv, print a job line, optionally fail or wait."""

import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("descriptor", type=Path)
parser.add_argument("--exit-code", type=int, default=0)
parser.add_argument("--sleep", type=float, default=0)
args = parser.parse_args()
(args.descriptor.parent / "fake-submit-argv.json").write_text(
    json.dumps(sys.argv[1:]), encoding="utf-8"
)
print("job offline-001 accepted", flush=True)
print("fake submit diagnostic", file=sys.stderr, flush=True)
time.sleep(args.sleep)
raise SystemExit(args.exit_code)

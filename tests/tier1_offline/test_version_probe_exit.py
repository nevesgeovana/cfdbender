"""The version probe trusts only an executable that answered the flag successfully."""

from __future__ import annotations

import sys
from pathlib import Path

from cfdbender.solvers._host._version import UNKNOWN, probe_version


def _script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "standin.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_a_failing_answer_with_a_dotted_number_in_its_error_is_unknown(tmp_path: Path) -> None:
    # A stand-in that does not know --version fails with a traceback naming the
    # interpreter path, which on a hosted runner holds its own dotted version.
    command = _script(
        tmp_path,
        "import sys\n"
        "sys.stderr.write('File \"/opt/hostedtoolcache/Python/3.12.16/x64/lib/x.py\"\\n')\n"
        "sys.exit(2)\n",
    )
    assert probe_version(command) == UNKNOWN


def test_a_successful_answer_gives_its_version(tmp_path: Path) -> None:
    command = _script(tmp_path, "print('Solver v8.6.0')\n")
    assert probe_version(command) == "8.6.0"

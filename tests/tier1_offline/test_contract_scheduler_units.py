"""Adapter contract C-48 (no scheduler in an adapter) and C-55 (SI units, units in key names)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

SOLVERS = Path(__file__).resolve().parents[2] / "src/cfdbender/solvers"
SCHEDULER_WORDS = re.compile(
    r"\b(sbatch|squeue|scancel|qsub|qstat|qdel|bsub|bjobs|#SBATCH|#PBS|#BSUB)\b", re.IGNORECASE
)
SCHEDULER_IMPORTS = re.compile(r"^\s*(from|import)\s+cfdbender\.(hpc|matrix\.submission)", re.M)


def _adapter_files():
    # FlightStream is forwarded to pyflightstream and is no library adapter.
    return [
        path
        for path in sorted(SOLVERS.rglob("*.py"))
        if "flightstream" not in path.relative_to(SOLVERS).parts
    ]


def test_c48_no_adapter_names_a_scheduler_command_or_imports_the_cluster_layer():
    files = _adapter_files()
    assert len(files) > 20, "the scan found no adapter sources"
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if SCHEDULER_WORDS.search(text) or SCHEDULER_IMPORTS.search(text):
            offenders.append(path.name)
    assert not offenders, offenders


def test_c48_the_scan_finds_a_scheduler_word_when_one_is_present():
    assert SCHEDULER_WORDS.search("run(['sbatch', 'job.sh'])")
    assert SCHEDULER_IMPORTS.search("from cfdbender.hpc import render")


def test_c55_qprop_converts_feet_to_metres_and_reports_velocity_in_metres_per_second():
    from cfdbender.solvers.qprop._config import matrix_condition

    held = matrix_condition({"MACH": 0.1, "ALTFT": 10000.0})
    temperature = 288.15 - 0.0065 * 10000.0 * 0.3048
    speed_of_sound = math.sqrt(1.4 * 287.05287 * temperature)
    assert held["V"] == pytest.approx(0.1 * speed_of_sound, rel=1e-4)
    assert held["A"] == pytest.approx(speed_of_sound, rel=1e-4)


def test_c55_qprop_takes_the_row_velocity_in_metres_per_second_unchanged():
    from cfdbender.solvers.qprop._config import matrix_condition

    assert matrix_condition({"V": 12.5})["V"] == 12.5

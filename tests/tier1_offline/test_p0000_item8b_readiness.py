"""Reproduce the tracked reference mesh refusal via inspect_files."""

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = "tests/fixtures/su2/naca0012_small.su2"


def guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "item8b_guard", ROOT / "tools/check_public_readiness.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan(files: dict[str, bytes], exemptions: list[str]) -> list[str]:
    function = guard().inspect_files
    if len(inspect.signature(function).parameters) == 2:
        return cast(list[str], function(files, []))
    return cast(list[str], function(files, [], exemptions))


def test_p0000_item8b_exact_exception() -> None:
    files = {
        "LICENSE": b"Copyright (c) 2026 Jane Doe\n",
        "CITATION.cff": b"authors:\n  - given-names: Jane\n    family-names: Doe\n",
        ".github/workflows/ci.yml": b"name: CI\n",
        FIXTURE: b"mesh",
        "tests/fixtures/su2/other.su2": b"mesh",
        "elsewhere/" + FIXTURE: b"mesh",
        "tests/fixtures/su2/NACA0012_small.su2": b"mesh",
    }
    exemptions: list[str] = []
    findings = scan(files, exemptions)
    assert f"solver or mesh output: {FIXTURE}" not in findings
    for name in list(files)[4:]:
        assert f"solver or mesh output: {name}" in findings
    assert len(exemptions) == 1 and exemptions[0].startswith(f"exempt: {FIXTURE} (")


def test_p0000_item8b_size_limit() -> None:
    for size in (300 * 1024, 300 * 1024 + 1):
        files = {FIXTURE: b"x" * size}
        exemptions: list[str] = []
        assert f"solver or mesh output: {FIXTURE}" in scan(files, exemptions)
        assert not exemptions

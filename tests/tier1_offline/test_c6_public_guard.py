"""The text guard rejects private planning identifiers and long dash characters."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("sample", ["CFB-" + "PLN-060", chr(0x2013), chr(0x2014)])
def test_text_patterns_cover_release_exclusions(sample):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "public_guard", root / "tools/check_public_text.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert any(pattern.search(sample) for pattern in module.patterns())

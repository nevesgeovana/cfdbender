"""Adapter documentation describes behavior without private design or work locations."""

import re
from pathlib import Path

import cfdbender.solvers.cfdpp as adapter


def test_adapter_text_has_no_private_design_or_work_location():
    root = Path(adapter.__file__).parent
    pattern = re.compile(r"question R\d|of the design|the licensed machine", re.I)
    found = [
        f"{path.name}:{number}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert found == []

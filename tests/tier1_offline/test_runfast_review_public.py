"""Public performance evidence obeys the repository text guard."""

import importlib.util
from pathlib import Path


def test_performance_report_has_no_prohibited_public_text():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "public_guard", root / "tools/check_public_text.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = next((root / "reports").glob("runfast-*.md")).read_text()
    assert not [m.group() for rule in module.patterns() for m in rule.finditer(report)]

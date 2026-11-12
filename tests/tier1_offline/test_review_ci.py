"""The CI environment must include the dependencies of the full suite."""

from pathlib import Path


def test_ci_installs_docs_dependencies():
    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    assert ".[dev,docs]" in workflow or (".[dev]" in workflow and ".[docs]" in workflow)

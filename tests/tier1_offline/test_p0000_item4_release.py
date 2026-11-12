"""P0000 item 4: release coordinates and metadata describe the delivered wheel."""

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_p0000_item4_release_coordinates():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    module = ast.parse((ROOT / "src/cfdbender/__init__.py").read_text(encoding="utf-8"))
    version = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    )
    citation = re.search(
        r"(?m)^version: (.+)$", (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    assert citation is not None
    assert project["version"] == version == citation[1]
    assert project["dependencies"] == [], "the wheel must install without runtime dependencies"


def test_p0000_item4_release_notes_and_status():
    path = ROOT / "CHANGELOG.md"
    notes = path.read_text(encoding="utf-8") if path.exists() else ""
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert f"## [{version}]" in notes, "release fragments must be collated for the current version"
    for text in ("pysu2", "--batch", "FlightStream", "OpenFOAM", "Fluent", "--hpc"):
        assert text in notes
    assert "not implemented" in notes
    assert f"Version {version}" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_does_not_describe_su2_as_absent():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "SU2 is the first planned solver" not in readme
    assert "These integrations are planned and are not present" not in readme


def test_p0200_item5_release_target():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.4.1"

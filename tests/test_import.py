import tomllib
from pathlib import Path

import cfdbender as cb


def test_version_is_declared():
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert cb.__version__ == project["project"]["version"]

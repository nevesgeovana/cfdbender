"""P0000: the SPDX license must build under every allowed build backend."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_p0000_item3c_spdx_requires_setuptools_77():
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    if isinstance(project["project"]["license"], str):
        backend = next(
            Requirement(item)
            for item in project["build-system"]["requires"]
            if Requirement(item).name == "setuptools"
        )
        assert "76.9" not in backend.specifier, "SPDX license needs setuptools >=77"
        assert "77.0" in backend.specifier

"""P0600: the guides describe the commands and products available in this release."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "guide/latex-sources"


def _text(folder):
    return "\n".join(
        line
        for path in sorted((SOURCES / folder).glob("text/*.tex"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("%")
    ).replace(r"\_", "_")


@pytest.mark.parametrize("folder", ["overview", "workspaces", "solvers", "cheatsheet"])
def test_p0600_each_guide_explains_the_current_workflow(folder):
    text = _text(folder)
    for term in (
        "cfd-geom",
        "cfd-mesh airfoil-2d",
        "P<POL>.txt",
        "FAM-ACOEF",
        "RESTART",
        "[groups]",
        "cluster:",
        "hpc:",
        "--one-run-at-a-time",
        "off by default",
    ):
        assert term in text, (folder, term)


def test_p0600_retired_products_are_not_promised_by_the_guides():
    for folder in ("overview", "workspaces", "solvers", "cheatsheet"):
        text = _text(folder)
        assert "campaign_sweep.csv" not in text, folder
        assert "polars/P<POL>.csv" not in text, folder


def test_p0600_cheatsheet_lists_geometry_and_mesh_controls():
    text = _text("cheatsheet")
    for term in (
        "cfd-geom validate",
        "cfd-geom run",
        "cfd-mesh mesh",
        "cfd-mesh refine",
        "--naca",
        "--reynolds",
        "--level",
        "--trailing-edge",
        "--warning-refusals",
        "--restart",
    ):
        assert term in text, term

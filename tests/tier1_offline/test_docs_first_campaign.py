"""The First campaign page is complete: its files and its commands make a campaign."""

from __future__ import annotations

import csv
import re
import shlex
from pathlib import Path

import pytest

from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import main as workspace_main

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "getting-started.md"
NAV_PAGES = (
    "index.md",
    "install.md",
    "getting-started.md",
    "workflow.md",
    "workspace-layout.md",
    "matrix-columns.md",
    "executables.md",
    "records.md",
    "results.md",
    "cluster-jobs.md",
    "storage.md",
    "troubleshooting.md",
    "solvers.md",
    "su2-workspaces.md",
    "openfoam-workspaces.md",
    "pyfr-workspaces.md",
    "xfoil-workspaces.md",
    "qprop-workspaces.md",
    "flightstream-workspaces.md",
    "cli.md",
    "architecture.md",
    "adapter-contract.md",
)
SOLVER_HEADINGS = ["Install", "Mesh", "Setup file", "What converged means", "Worked example"]


def _files(page: str) -> dict[str, str]:
    """Return each file the page states as a path line followed by a code block."""
    found = re.findall(r"^`(inputs/[^`]+)`:\n\n```\w+\n(.*?)```", page, flags=re.M | re.S)
    return dict(found)


def _commands(page: str) -> list[str]:
    lines: list[str] = []
    for block in re.findall(r"```console\n(.*?)```", page, flags=re.S):
        lines += [line for line in block.splitlines() if line.strip()]
    return lines


def test_first_campaign_page_runs_as_written(tmp_path, monkeypatch):
    page = PAGE.read_text(encoding="utf-8")
    files = _files(page)
    assert set(files) == {
        "inputs/matrices/polar.fs",
        "inputs/setups/s001.toml",
        "inputs/references/r001.toml",
        "inputs/pproc/p001.toml",
    }
    commands = _commands(page)
    assert commands[:2] == ["cfd-workspace init campaign --solver mock", "cd campaign"]
    monkeypatch.chdir(tmp_path)
    ran = []
    for line in commands:
        words = shlex.split(line)
        if words[0] == "cd":
            monkeypatch.chdir(tmp_path / words[1])
            for name, text in files.items():
                (Path.cwd() / name).write_bytes(text.encode("ascii"))
            continue
        main = {"cfd-workspace": workspace_main, "cfd-matrix": matrix_main}[words[0]]
        assert main(words[1:]) == 0, line
        ran.append(words[1])
    assert ran == ["init", "plan", "run", "status", "status", "collect"]
    table = tmp_path / "campaign" / "post" / "polar" / "campaign_sweep.csv"
    with table.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["point"] for row in rows] == ["DP-M200AL+000", "DP-M200AL+020", "DP-M200AL+040"]
    assert {row["classification"] for row in rows} == {"converged"}
    shown = re.search(r"^POL,point,ALPHA,BETA,MACH,\.\.\.,(.*)$", page, flags=re.M)
    assert shown is not None
    assert ",".join(rows[0]).endswith(shown.group(1))


def test_site_navigation_names_the_approved_pages():
    nav = (ROOT / "properdocs.yml").read_text(encoding="utf-8").split("nav:")[1]
    listed = re.findall(r":\s+([\w/-]+\.md)\s*$", nav.split("exclude_docs")[0], flags=re.M)
    assert listed == [*NAV_PAGES, "requirements/index.md"]
    on_disk = {path.name for path in (ROOT / "docs").glob("*.md")}
    assert on_disk == set(NAV_PAGES)
    for gone in ("workspace-files.md", "workspace-settings.md", "commands.md"):
        assert gone not in on_disk


@pytest.mark.parametrize("solver", ["su2", "openfoam", "pyfr", "xfoil", "qprop"])
def test_solver_pages_share_one_template(solver):
    page = (ROOT / "docs" / f"{solver}-workspaces.md").read_text(encoding="utf-8")
    assert re.findall(r"^## (.+)$", page, flags=re.M) == SOLVER_HEADINGS


def test_no_page_links_a_removed_page_or_carries_a_dash():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for gone in ("workspace-files.md", "workspace-settings.md", "commands.md"):
            assert gone not in text, f"{path.name} links {gone}"
        assert chr(0x2014) not in text and chr(0x2013) not in text, f"{path.name}: dash"

"""The built site exposes the authored requirements."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_su2_workspace_page_is_in_site_navigation(tmp_path):
    output = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "properdocs",
            "build",
            "--strict",
            "-f",
            str(ROOT / "properdocs.yml"),
            "-d",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    home = (output / "index.html").read_text(encoding="utf-8")
    assert 'href="su2-workspaces/"' in home


def test_p000_item5_requirements_are_on_strict_site(tmp_path):
    register = ROOT / "docs" / "unpublished" / "requirements" / "register.md"
    assert register.is_file(), "the SRS register must exist"
    sys.path.insert(0, str(ROOT / "scripts"))
    from _requirements_paths import read_requirements

    rows = read_requirements(ROOT)
    assert rows
    output = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "properdocs",
            "build",
            "--strict",
            "-f",
            str(ROOT / "properdocs.yml"),
            "-d",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    page = (output / "requirements" / "index.html").read_text(encoding="utf-8")
    assert all(re.search(rf"(?<![A-Z0-9-]){re.escape(row['uid'])}(?![0-9])", page) for row in rows)

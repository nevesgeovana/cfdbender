"""Generate the public requirements page from the authored register."""

import sys
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _requirements_paths import REGISTER, read_requirements  # noqa: E402

read_requirements(ROOT)
with mkdocs_gen_files.open("requirements/index.md", "w") as page:
    page.write((ROOT / REGISTER).read_text(encoding="utf-8"))

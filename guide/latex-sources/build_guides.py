# Copyright (c) 2026 Geovana Neves. Licensed under CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/). See LICENSE-AND-AUTHORSHIP.md.
"""Build the four cfdbender guides and copy each PDF into guide/pdf/.

    python guide/latex-sources/build_guides.py             all four guides
    python guide/latex-sources/build_guides.py cheatsheet  only that guide

Needs pdflatex (MiKTeX or TeX Live) with beamer, tcolorbox, listings, tikz,
adjustbox, microtype, enumitem and underscore. The compiler runs with shared/
as the working directory, so every ../shared/ in the preamble resolves inside
this folder, and each guide's own folder is handed to it as an include
directory. Nothing outside this folder is read. Auxiliary files stay in
build/ here, which is never versioned.

The build refuses a source with an em or en dash, and fails when a guide
reports an overfull box. Exit code 0 means every requested guide was built.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "pdf"
GUIDES = {
    "overview": "cfdb-guide-01-overview",
    "workspaces": "cfdb-guide-02-workspaces",
    "solvers": "cfdb-guide-03-solvers",
    "cheatsheet": "cfdb-guide-04-cheatsheet",
}
DASHES = (chr(0x2014), chr(0x2013))


def dashed_sources() -> list[str]:
    """Return the sources that hold an em or en dash."""
    found = []
    for path in sorted(HERE.rglob("*.tex")):
        if "build" in path.relative_to(HERE).parts:
            continue
        text = path.read_text(encoding="utf-8")
        if any(dash in text for dash in DASHES):
            found.append(path.relative_to(HERE).as_posix())
    return found


def compile_guide(folder: str, job: str, miktex: bool) -> int | None:
    """Run three passes; return the overfull box count, or None on failure."""
    deck = HERE / folder
    shared = HERE / "shared"
    out = HERE / "build" / folder
    out.mkdir(parents=True, exist_ok=True)
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    env = dict(os.environ)
    if miktex:
        command += [f"-include-directory={deck}", f"-include-directory={shared}"]
    else:
        env["TEXINPUTS"] = f"{deck}//{os.pathsep}{shared}//{os.pathsep}"
    command += [f"-output-directory={out}", f"-jobname={job}", (deck / "main.tex").as_posix()]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for _ in range(3):
        done = subprocess.run(
            command, cwd=shared, env=env, capture_output=True, creationflags=flags, check=False
        )
        if done.returncode != 0:
            return None
    log = (out / f"{job}.log").read_text(encoding="utf-8", errors="replace")
    PDF.mkdir(exist_ok=True)
    shutil.copyfile(out / f"{job}.pdf", PDF / f"{job}.pdf")
    return sum(1 for line in log.splitlines() if line.startswith("Overfull"))


def main(argv: list[str]) -> int:
    """Build the guides named in argv, or all four."""
    wanted = argv or list(GUIDES)
    unknown = [name for name in wanted if name not in GUIDES]
    if unknown:
        print(f"no guide named {', '.join(unknown)}; the guides are {', '.join(GUIDES)}")
        return 2
    if shutil.which("pdflatex") is None:
        print("pdflatex was not found; install MiKTeX or TeX Live")
        return 2
    dashed = dashed_sources()
    if dashed:
        print("em or en dash found in: " + ", ".join(dashed))
        return 1
    version = subprocess.run(
        ["pdflatex", "--version"],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    ).stdout
    status = 0
    for folder in wanted:
        job = GUIDES[folder]
        over = compile_guide(folder, job, "MiKTeX" in version)
        if over is None:
            print(f"FAILED {job}; see build/{folder}/{job}.log")
            status = 1
            continue
        print(f"built guide/pdf/{job}.pdf (overfull boxes: {over})")
        if over:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

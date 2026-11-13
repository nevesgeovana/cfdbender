<!-- Copyright (c) 2026 Geovana Neves. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). See LICENSE-AND-AUTHORSHIP.md. -->

# The guides

Four guides for the people who run studies with cfdbender in a workspace:
three slide decks and one cheatsheet, each written for one version of the
package (its title page says which).

| Guide | PDF | What it covers |
|---|---|---|
| 1 | `pdf/cfdb-guide-01-overview.pdf` | read first: what a solver-agnostic workspace is, the solvers as a spectrum of fidelities, the workflow in one picture, a first campaign on the mock solver, and a map of guides 2 to 4 |
| 2 | `pdf/cfdb-guide-02-workspaces.pdf` | the workspace tree, the inputs, plan, run, follow, collect and post, a workstation and an HPC cluster, keeping, correcting and cleaning |
| 3 | `pdf/cfdb-guide-03-solvers.pdf` | what each of the six library solvers (SU2, OpenFOAM, PyFR, XFOIL, QPROP and the mock) reads and writes, what differs between them, and how to choose |
| 4 | `pdf/cfdb-guide-04-cheatsheet.pdf` | the cheatsheet, eight pages: every command and option of `cfd-matrix` and `cfd-workspace`, and one page for each of the seven stages of the workflow |

A FlightStream workspace is run by pyflightstream, and its commands and inputs
are covered by the pyflightstream guides
(https://nevesgeovana.github.io/pyflightstream/). The two sets share one
design, so they read as one collection.

## Building the guides

`python guide/latex-sources/build_guides.py` from the repository root compiles
the four PDFs into `guide/pdf/`, overwriting them; a guide's folder name
(`overview`, `workspaces`, `solvers`, `cheatsheet`) as an argument builds that
guide alone. It needs `pdflatex` (MiKTeX or TeX Live) with beamer, tcolorbox,
listings, tikz, adjustbox, microtype, enumitem and underscore. The build
refuses a source with an em or en dash, and fails when a guide reports an
overfull box. The same sources give the same PDF, byte for byte, on one
installation, so a rebuild without a change leaves the committed files as
they are.

## Sources

`latex-sources/shared/` holds what every guide reads: the preamble, the
frame design, the boxes, the collection's mark and `info.tex`, the one place
that states the version, the author and the licence notice. Each guide has its
own folder with a `main.tex` and its parts under `text/`.

## The cheatsheet, guide 4

`latex-sources/cheatsheet/` is one document of eight pages, A4 landscape:

- page 1 is every command and every option of `cfd-matrix` and
  `cfd-workspace`, by stage;
- pages 2 to 8 are the seven stages of the workflow, one page each (create
  the workspace and write the inputs, plan, run, follow, collect and post,
  correct the records, manage storage): what the stage is for, its commands
  with one line per option, the files it reads and writes, examples, and its
  common mistakes with their fixes.

The tier-1 test `tests/tier1_offline/test_guide_cheatsheet_parsers.py` walks
both parsers and refuses the sheet when a command or an option is missing from
its entry on page 1, when the sheet names one no parser has, when a stage page
or a deck shows a command line the parsers do not accept, and when the
committed PDF is not eight pages for the version `info.tex` names.

The authorship and licence of this folder are in `LICENSE-AND-AUTHORSHIP.md`.

## Changed

- The documentation site has five tabs: Start here, Workflow, Solvers, Commands and Reference. The common workflow is told once (workflow, workspace layout, executables, records, results, cluster jobs, storage, troubleshooting) and each solver page states what differs, under the same five headings: install, mesh, setup file, what converged means, worked example.
- The Commands page lists every command and option of `cfd-workspace` and `cfd-matrix`, written from the argument parsers by `python tools/build_cli_page.py`; a test fails when a command or an option is missing from the page.
- First campaign is a complete campaign on the mock solver, with every file and command shown.

## Removed

- The pages Workspace files, Workspace settings and the earlier Commands page; their content is on the new pages.

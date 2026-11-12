# First prompt

Paste everything between the lines into your own assistant opened in this folder.

---

Set up and run this cfdbender 0.4.2 kit. Read README-FIRST.md and SMOKE-CASE.md.
Treat this unzipped folder as KIT. Do these steps in order. Stop on ANY failure,
print `STOP: step <number>: <cause and next action>`, and do not continue dependent
steps. Ask me only for a missing executable path. Do not scan my other folders.
Never write outside KIT: put temporary files in KIT/.tmp; set TEMP, TMP and TMPDIR
to that directory for child processes; set PYTHONDONTWRITEBYTECODE=1 and
PIP_NO_CACHE_DIR=1. Remove PYTHONPATH and PYTHONHOME from child process
environments so another Python installation cannot override this kit. Do not change PATH, create aliases or add launchers.

1. Find a Python interpreter with `py -3` or `python` on Windows, or `python3`
   on Linux. Run it to check `sys.version_info >= (3, 12)`. If none qualifies,
   stop and say Python 3.12 or newer is required. Check before creating the venv.
2. Create KIT/.tmp and KIT/.venv with that Python's `-m venv .venv`. Invoke the
   venv's Python and CLI by their direct paths; activation is unnecessary.
   Windows uses `.venv/Scripts/python.exe` and `.venv/Scripts/cfd-matrix.exe`;
   Linux uses `.venv/bin/python` and `.venv/bin/cfd-matrix`.
   Run venv Python `-I -m pip install --no-index ./cfdbender-0.4.2-py3-none-any.whl`
   from KIT. This installation puts the Python package in KIT/.venv. Verify
   installed version 0.4.2 and that its import comes from KIT/.venv. Verify wheel
   METADATA has no unconditional Requires-Dist entries: cfdbender has no runtime
   dependencies; its test/dev/docs extras are not needed. Stop on a failed pip
   command; do not download anything. Run both installed CLIs with `--help`.
3. Find the site's prebuilt SU2_CFD with `where.exe SU2_CFD` on Windows or
   `which SU2_CFD` on Linux. An existing CFDBENDER_SU2_EXE may supply the path
   when discovery fails. If neither works, ask me for the full executable path
   and wait. Verify the file exists, then export CFDBENDER_SU2_EXE for this tool
   session: `$env:CFDBENDER_SU2_EXE = <the path>` in PowerShell, or
   `export CFDBENDER_SU2_EXE=<the quoted path>` in a POSIX shell. Do not copy or
   install SU2. Use one SU2 process and one OpenMP thread only. The adapter checks
   a 0.5 GiB free RAM floor before and during execution; stop if it refuses.
4. In KIT/smoke run the installed `cfd-matrix plan smoke --local`, then
   `cfd-matrix run smoke --local`, then `cfd-matrix collect smoke --local`.
   Use direct venv CLI paths throughout. Always include --local because the
   bundled HPC profile is a TEMPLATE, with fake values that cannot submit work.
   Stop on any nonzero command status. Do not retry a recorded point or delete
   its results; use a freshly unzipped kit for a new smoke run.
5. Read and print `smoke/post/smoke/campaign_summary.json` with venv Python.
   Print the neutral coefficients, axes, references, history, solver version and
   classification. Check the single point is COMPLETED, with classification
   `iteration_limit` and 40 history entries. Never call process success convergence.
   If the measured classification differs, stop and show it with the logs.
6. Finish with exactly these five lines explaining how to use my own case:
   Copy your .cfg and mesh into a new case folder inside KIT, retaining the smoke.
   Put its SU2 keys and relative mesh path in inputs/setups/s002.toml; omit generated/output keys.
   Put reference area, length and moment_origin in inputs/references/r002.toml.
   Put history=true and output settings in inputs/pproc/p002.toml; cfdbender writes a new .cfg from s/r/p.
   Add a new named .fs matrix with SET/REF/PPROC=002 and a new POL; plan, run --local, then collect.

---

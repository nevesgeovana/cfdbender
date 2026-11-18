# Run-fast lane: a large mesh is read once, not once per command and point

Branch `lane/run-big-mesh`, cut from `69e5db0`. Commits: `b1a9788` (cache, session, streamed
output digests, flag, tests), `88ec5a6` (docs), and the commit that adds this report and
`tools/measure_run_big_mesh.py`.

## Measured cause (base 69e5db0)

The slowness was hashing, and it was worse than the hypothesis. For `cfd-matrix run --local` on a
synthetic SU2 text mesh (3 rows, 4 sweep values each = 12 points), every point read the mesh
whole several times, two of them into memory:

| Read of the mesh, per point | Where | Held in memory |
|---|---|---|
| `registry.record` (`hashlib.sha256(path.read_bytes())`), once per point | `core/registry.py` | whole mesh |
| `RunRegistry.transition(..., "RUNNING")`, once per point | `core/registry.py` | whole mesh |
| `load_point` -> `AdapterPoint.verify` (second `InputFiles`) | `core/adapter.py` | streamed |
| `translate`/`_execute.run` -> `verify` again | `solvers/su2/_execute.py` | streamed |
| `link_mesh` re-hashes the mesh through the point-folder link | `solvers/_host/_mesh.py` | streamed |
| `RunRegistry.transition(..., "COMPLETED")` hashes the outputs, and the mesh link is one | `core/registry.py` | whole mesh |

`collect` repeated it: `_verify_outputs` twice per point plus the SUBMITTED recount, all through
`read_bytes` of the mesh link. Stack evidence came from an audit hook on `open` with
`traceback.extract_stack` (script not committed; the committed tool counts the same opens).

Whole-file reads of large inputs on the paths asked about, and what remains:

| Site | Large file? | After this lane |
|---|---|---|
| `registry.record`, `transition` RUNNING and COMPLETED | mesh, mesh link | streamed, once per input per registry, served by cache |
| `matrix/collection._verify_outputs`, SUBMITTED recount, `matrix/records._digests` | mesh link | streamed or served as above |
| `solvers/_host/_mesh.link_mesh` | mesh through link | trusted when size, mtime, ctime match the command's read or the cache; otherwise streamed |
| `core/restart.py:45,56`, `matrix/restart.py:38` | restart solution files (can be large) | unchanged: `read_bytes`. They are not planned inputs and only run with `--restart`. Doubt below |
| `solvers/pyfr/_restart.py:30` | mesh, `--restart` only | unchanged |
| `core/registry.py` archive/restore, `matrix/storage.py`, `matrix/exports.py` copies and comparisons | workspace files | unchanged: not on run, submit or collect |
| `solvers/su2/_history.dimensions` | mesh through link | reads lines until `NDIME`, a bounded head read |
| `solvers/_host/_groups` marker scan, `solvers/qprop`, `xfoil` | mesh scan streamed in the same pass that hashes; qprop and xfoil inputs are small | unchanged |

## Design

* `InputSession` (`core/plan.py`): the reads of one command. `load_point(root, point, solver=...,
  session=None)` takes it as an optional keyword; without it behaviour is as before. The session
  is created by the composition layer in `core/run.py` (preflight and execute), `core/executables.py`,
  `matrix/submission.py`, `matrix/restart.py`, `matrix/records.py`, `matrix/collection.py`. The
  `AdapterPoint` carries it (`compare=False`) so `verify()`, `point_markers` and `link_mesh`
  reuse it. `RunRegistry` owns one for its own digest checks.
* Inside a session a small file (up to `LARGE_INPUT_BYTES`) is always read again, as today. A
  large file is trusted for the command only while size, `mtime_ns` and `ctime_ns` are unchanged
  when touched again (one `stat` per touch, not per byte); otherwise it is read again and
  compared with the planned digest, giving the unchanged refusal `input changed after planning:
  <name>; plan again`. Missing input gives the unchanged `<name>: cannot read: ...`.
* `DigestCache` (`core/plan.py`): `.input-digests.cache` in the workspace root, beside `runs.json`
  (the library has no separate state directory). Format field `1`. Entries only above
  `LARGE_INPUT_BYTES`: name, size, mtime_ns, ctime_ns, sha256, and the name-list scans (SU2
  marker names) of the same pass. Served only when all three identity values match. Not stored
  when the newest of mtime and ctime is under 2 s old at hashing time (`CACHE_SETTLE_NS`, clock
  injectable as `_clock_ns`). Atomic write (`mkstemp` in the same folder plus `os.replace`); a lost
  update is accepted; a missing, unreadable, corrupt, wrong-version or malformed-entry file is an
  empty cache; a failed write (read-only workspace) is ignored silently, because the cache is an
  optimisation and a warning on every command of a read-only workspace would be noise.
  It never enters `inputs_sha256`, records, exports or storage moves (they list planned inputs
  and run files; `storage` moves `sims/` and `post/` folders).
* `--rehash-inputs` on `plan`, `run` and `collect` (`submit` is `run` without `--local`) sets a
  context variable for the command; the cache is not read (it is still refreshed).
  Help text and `docs/cli.md` (generated from the parsers) carry the one-sentence limit.
* The stricter reading of rule (a): the 2 s rule uses the newer of mtime and ctime, because on
  Windows ctime is the creation time and a just-copied file with an old mtime must not be cached.

Provenance: `plan.json`, `runs.json` (all non-time fields), `inputs_sha256` key order, the polar
tables and `post/` outputs of one workspace (9 MiB mesh, 2 rows, 3 points) were compared base
versus this branch: identical except run timestamps and the digests of files that embed a
timestamp (`campaign_summary.json`, `products.json`), which differ between any two runs. The
cache file itself is the only new file and only appears for files above the threshold that are
older than 2 s.

## Measurements

Command: `PYTHONPATH=<src> python tools/measure_run_big_mesh.py --mib N --steps
plan,plan,run,collect` (base: `PYTHONPATH` at an extract of `69e5db0`), 3 rows x 4 points = 12
points, local fake SU2 stand-in, mesh in `C:/t`, `tracemalloc` peak, mesh opens counted by an audit
hook for the mesh and its point-folder link. The mesh is settled (2.5 s) before the first step. The
fake solver start is about 0.25 s per point and is part of every run time. Wall times are from the
final run of the script; the first runs of the lane differed by about 20 percent.

| | 128 MiB plan | plan again | run | collect | 512 MiB plan | plan again | run | collect |
|---|---|---|---|---|---|---|---|---|
| base time | 0.33 s | 0.30 s | 10.7 s | 4.4 s | 0.92 s | 0.89 s | 33.4 s | 16.0 s |
| base opens | 1 | 1 | 74 | 49 | 1 | 1 | 74 | 49 |
| base peak | 3.3 MiB | 3.1 MiB | 128.8 MiB | 128.2 MiB | 3.3 MiB | 3.1 MiB | 512.8 MiB | 512.2 MiB |
| new time | 0.33 s | 0.10 s | 3.2 s | 0.44 s | 4.09 s | 1.47 s | 6.8 s | 0.45 s |
| new opens | 1 | 0 | 0 | 12 (head) | 1 | 0 | 0 | 12 (head) |
| new peak | 3.3 MiB | 0.1 MiB | 1.8 MiB | 1.2 MiB | 3.3 MiB | 0.1 MiB | 1.8 MiB | 1.2 MiB |
| new, `--rehash-inputs` | 0.33 s, 1 open | | 3.8 s, 4 opens, 3.1 MiB | 0.75 s, 14 opens | 3.6 s, 1 open | | 6.3 s, 4 opens, 3.1 MiB | 1.7 s, 14 opens |

Reading it: the first command that meets a mesh (plan) reads it once, as before (512 MiB: the cold
read itself, 0.9 to 4 s depending on disk cache). Every later command reads it zero times; the 12
collect opens are `su2._history.dimensions` stopping at `NDIME` in the first lines. Even forced
(`--rehash-inputs`) a run now reads the mesh 4 times for 12 points (registry, executables,
preflight and execute sessions) instead of 74, and never holds it in memory. These are local-disk
numbers; the gain on a network file system scales with the bytes not read (74 x 512 MiB = 37 GiB
per run before, 0 after).

Submit was not run end to end here (no scheduler stand-in in the measurement tool); the same
session is used by `submission_plans` and `submit_plan`, and the counting tests cover `load_point`
through a session and the registry.

## Tests (`tests/tier1_offline/test_runfast_digest_cache.py`, 32 cases)

Written first; on the base source 24 of 27 then-existing cases failed (the remaining 3 pass
because they state behaviour that already held). Covered: one read per session for N points,
default API unchanged (two reads per point), constant opens whatever the number of points,
changed mesh between plan and run refused with the identical text, mesh changed during the
command (size, and same size with new mtime) refused, missing input unchanged, changed small
input refused inside a session, second command opens nothing, changed size / mtime / ctime each
force a re-hash and the refusal (each case moves one value only), corrupt cache (6 shapes),
wrong format version with valid entries, entry of another name, not stored for a just-written
file, flag forces the re-read and the documented limit (same size and timestamps not detected
without the flag, detected with it), flag on plan/run/collect parsers, markers from the cache
equal the scan, small files never cached, provenance identical cold, warm and rehashed, a
read-only cache location ignored, registry streams and reads once, warm run and collect read
nothing and record exact output digests, cold run does not stream through the link, output that
is the mesh takes its digest and other outputs are streamed, changed mesh behind the link is
hashed and refused.

Sabotage (each mutation applied to the source, the named tests run, source restored): 21
mutations, all red: cache ignores size, ignores mtime, ignores ctime; no 2 s rule; version not
checked; flag ignored; scans not stored; small files cached; session ignored; refresh trusts a
changed large file; corrupt cache raises; small file trusted in a session; registry reads whole;
cache hit skips the planned digest compare; markers not read from cache; flag not wired; cache
write not tolerant; any output takes the input digest; link digest trusts the cache without a
stat; `link_mesh` streams through the link; registry reads outputs whole. Three first-draft
tests were not caught by their mutation (size, version, small file) and were tightened so each
case moves one value only; the table above is after that.

Gates: 2538 passed, 28 skipped over 239 test files (every file naming load_point, execute_plan,
RunRegistry, InputFiles, plan_matrix, submit_plan, collect_workspace, link_mesh, outputs_sha256
or a `_mesh` name, plus `test_docs_cli.py`) before the last test-only edits; the new file alone:
32 passed after them. `ruff check`, `ruff format --check`, `mypy`, `tools/check_architecture.py`,
`tools/check_public_text.py` (0 findings) and `tools/build_cli_page.py --check` are clean.

## Proposed changelog sentences (not written to CHANGELOG.md)

* A large input (above 8 MiB, such as a mesh) is now read and hashed once when first seen and
  remembered in `.input-digests.cache` in the workspace root, so `plan`, `run`, `submit` and
  `collect` no longer read it again for every command and point. The entry is used only while the
  file's size, modification time and change time are unchanged.
* `plan`, `run` and `collect` take `--rehash-inputs`, which reads every input again. A file
  rewritten with the same size and timestamps is not noticed without it.
* The run record no longer reads a mesh into memory to digest it; digests are streamed, and the
  mesh link of a point folder takes the digest of the mesh it points at.

## Doubts for the orchestrator

* Restart solution files (`core/restart.py`, `matrix/restart.py`) are still read whole for
  `restart_sha256`; they are not planned inputs and were outside the ask, but they can be large.
* The settle rule uses the newer of mtime and ctime (stricter than "mtime within 2 s").
* On POSIX a hard link (the fallback when symbolic links are refused) changes the source's ctime,
  so with that fallback the cache entry is refreshed per point instead of served; correctness is
  unaffected.
* The tool's `NDIME` read in `su2._history.dimensions` stops at the first lines only because the
  synthetic mesh puts `NDIME` first, as SU2 meshes do.

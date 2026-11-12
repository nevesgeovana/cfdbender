"""P0000-FTS-RUN: run, collect and post use the upstream campaign engine.

The POC fixture helper is reused unchanged. The export-writing stand-in adapts
test_goal031_local_run_log.STUB: accept native -script arguments, read bundled
loads, emit the log directly, and use valid minimal VTK and Tecplot exports.
No solver installation or scheduler is used.
"""

import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
from test_p0000_fts_plan_parity import _call, _declare, _fixture

from cfdbender.matrix.cli import main as matrix_main

FIXTURES = Path(__file__).parents[1] / "fixtures/flightstream"

# One list, shared by both snapshots. No file, coefficient or status is omitted.
PARITY_EXCLUSIONS = (
    "absolute workspace and stand-in paths",
    "ISO timestamps and archive timestamp tokens",
    "run ids: retained because this fixture gives deterministic ids",
    "durations: wall_time_s, duration_s, elapsed_s and log elapsed seconds",
)

STUB = r"""
import pathlib, re, sys
from pyflightstream.cases import EXPORT_KINDS
script = pathlib.Path(sys.argv[sys.argv.index('-script') + 1])
loads = pathlib.Path(__file__).with_name('loads.txt').read_text(encoding='utf-8')
verbs = {kind[2] for kind in EXPORT_KINDS}
vtk = ('# vtk DataFile Version 3.0\nFlightStream vtk output\nASCII\nDATASET POLYDATA\n'
       'POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\nPOLYGONS 1 4\n3 0 1 2\n'
       'CELL_DATA 1\nSCALARS Cp_reference FLOAT\nLOOKUP_TABLE default\n-0.5\n')
native = ('TITLE="Native fixture"\nVARIABLES="X","Y","Z","Singularity_strength"\n'
          'ZONE T=Solver, NODES=3, ELEMENTS=1, FACES=3, DATAPACKING=BLOCK, '
          'ZONETYPE=FEPolygon, NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0\n'
          '0 1 0\n0 0 1\n0 0 0\n0.125 1.125 2.125\n1 2 2 3 3 1\n1 1 1\n0 0 0\n')
log = pathlib.Path(__file__).with_name('log.txt').read_text(encoding='utf-8')
alpha = 0.0
lines = script.read_text(encoding='utf-8').splitlines()
for index, line in enumerate(lines):
    verb = line.split(' ')[0]
    if verb == 'SOLVER_SET_AOA':
        alpha = float(line.split()[1])
    if verb == 'EXPORT_LOG' or verb in verbs:
        body = (log if verb == 'EXPORT_LOG' else
                vtk if verb == 'EXPORT_SOLVER_ANALYSIS_VTK' else
                native if verb == 'EXPORT_SOLVER_ANALYSIS_TECPLOT' else 'DATA')
        if verb == 'EXPORT_SOLVER_ANALYSIS_SPREADSHEET':
            body = re.sub(r'(Angle of attack \(Deg\)\s+)\S+',
                          lambda m: m.group(1) + str(alpha), loads)
        target = pathlib.Path(lines[index + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding='utf-8')
"""


@pytest.fixture
def engine():
    pytest.importorskip(
        "pyflightstream",
        reason="install cfdbender[flightstream]: pip install cfdbender[flightstream]",
    )
    return (
        importlib.import_module("pyflightstream.workspace.cli").main,
        importlib.import_module("pyflightstream.run.cli").main,
    )


def _stand_in(folder):
    folder.mkdir()
    (folder / "solver.py").write_text(STUB, encoding="utf-8")
    shutil.copyfile(FIXTURES / "loads.txt", folder / "loads.txt")
    shutil.copyfile(FIXTURES / "log.txt", folder / "log.txt")
    if os.name == "nt":
        launcher = folder / "solver.cmd"
        launcher.write_text(
            f'@echo off\n"{sys.executable}" "{folder / "solver.py"}" %*\n', encoding="utf-8"
        )
    else:
        launcher = folder / "solver"
        launcher.write_text(
            f"#!{sys.executable}\nexec(open({str(folder / 'solver.py')!r}).read())\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    return launcher


def _source(folder, engine, monkeypatch):
    # The same source as the POC, bundled so a wheel-only extra can run this test.
    monkeypatch.setenv("PYFLIGHTSTREAM_TEST_FIXTURES", str(FIXTURES))
    root = _fixture(folder / "source", engine[0])
    # Grouped local execution refuses unbounded jobs. This is the only setup
    # addition to the POC fixture, shared by both copies and every mode.
    with (root / "inputs/setups/s002.toml").open("a", encoding="utf-8") as stream:
        stream.write("timeout_s = 60\n")
    launcher = _stand_in(folder / "stand-in")
    (root / "inputs/executables.toml").write_text(
        '"26.120" = ' + json.dumps(launcher.as_posix()) + "\n", encoding="utf-8"
    )
    return root


def _normalise(data, root, stand_in):
    # Operate on bytes, preserving all formatting and every unexcluded byte.
    for path, token in ((root, b"<workspace>"), (stand_in, b"<stand-in>")):
        for spelling in (str(path), path.as_posix(), str(path).replace("\\", "\\\\")):
            data = data.replace(spelling.encode(), token)
    data = re.sub(rb"\d{4}-\d\d-\d\d[T ][\d:.]+(?:Z|[+-]\d\d:\d\d)?", b"<timestamp>", data)
    data = re.sub(
        rb'("(?:pyfs:)?(?:wall_time_s|duration_s|elapsed_s)"\s*:\s*)[0-9.eE+-]+',
        rb"\g<1>0",
        data,
    )
    data = re.sub(rb"(elapsed[=: ]+)[0-9.]+(\s*s)", rb"\g<1>0\2", data)
    return data


def _csv_durations(data):
    lines = data.splitlines(keepends=True)
    if not lines:
        return data
    columns = lines[0].rstrip(b"\r\n").split(b",")
    indices = [i for i, column in enumerate(columns) if column == b"wall_time_s"]
    if not indices:
        return data
    cell = re.compile(rb'(?:^|,)("(?:[^\"]|\"\")*"|[^,\r\n]*)')
    result = [lines[0]]
    for line in lines[1:]:
        cells = list(cell.finditer(line))
        for index in reversed(indices):
            start, end = cells[index].span(1)
            line = line[:start] + b"0" + line[end:]
        result.append(line)
    return b"".join(result)


def _snapshot(root, stand_in, *, source_root=None):
    files = {}
    for name in ("post", "sims"):
        for path in sorted((root / name).rglob("*")):
            if path.is_file():
                data = _normalise(path.read_bytes(), source_root or root, stand_in)
                files[path.relative_to(root).as_posix()] = (
                    _csv_durations(data) if path.suffix == ".csv" else data
                )
    ledger = importlib.import_module("pyflightstream.workspace.ledger").read_ledger(root)
    assert not ledger.unreadable, ledger.unreadable
    statuses = {
        (point["sim"], point["point"]): point["status"]
        for point in ledger.points()
        if point["run_id"] is not None
    }
    return statuses, files


def _compare(left, right, stand_in, *, right_source=None):
    a = _snapshot(left, stand_in)
    b = _snapshot(right, stand_in, source_root=right_source)
    differences = [key for key in a[1].keys() | b[1].keys() if a[1].get(key) != b[1].get(key)]
    assert a == b, (
        f"Parity exclusions: {PARITY_EXCLUSIONS!r}; "
        f"statuses: {a[0]!r} versus {b[0]!r}; "
        f"different files: {differences}"
    )


@pytest.mark.parametrize("grouping", [[], ["--batch", "2"]], ids=["P0000-LOCAL", "P0000-BATCH2"])
def test_p0000_fts_run_collect_post_parity(tmp_path, engine, monkeypatch, grouping):
    source = _source(tmp_path, engine, monkeypatch)
    direct, delegated = tmp_path / "direct/campaign", tmp_path / "delegated/campaign"
    for root, main in ((direct, engine[1]), (delegated, matrix_main)):
        shutil.copytree(source, root)
        if root == delegated:
            _declare(root)
        assert _call(main, ["plan", "m", "--workspace", str(root), *grouping]) == 0
        assert _call(main, ["run", "m", "--workspace", str(root), "--local", *grouping]) == 0
        assert _call(main, ["collect", "--workspace", str(root)]) == 0
        assert (
            _call(main, ["post", "m", "--workspace", str(root), "--force-overwrite", "--yes"]) == 0
        )
    _compare(direct, delegated, tmp_path / "stand-in")
    statuses, files = _snapshot(direct, tmp_path / "stand-in")
    # Build 26.120 cannot group its unsteady action commands; upstream leaves
    # those two rows out, retaining the two steady points as one grouped job.
    assert len(statuses) == (2 if grouping else 4), statuses
    assert files and any("polars/" in name and name.endswith(".csv") for name in files)
    print(f"Parity exclusions: {PARITY_EXCLUSIONS!r}; {len(statuses)} points; {len(files)} files")

    # Each mutant is a fresh scratch copy. Both must fail the actual comparator.
    for mutation in ("product", "status"):
        scratch = tmp_path / mutation
        shutil.copytree(direct, scratch)
        _compare(direct, scratch, tmp_path / "stand-in", right_source=direct)
        if mutation == "product":
            product = next((scratch / "post").rglob("*.csv"))
            data = product.read_bytes()
            product.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        else:
            manifest = scratch / "runs.json"
            storage = importlib.import_module("pyflightstream.workspace.record_document")
            rows = storage.read_rows(manifest)
            if rows[0].get("points_ran"):
                rows[0]["points_ran"][0]["status"] = "FAILED_EXECUTION"
            else:
                rows[0]["status"] = "FAILED_EXECUTION"
            manifest.write_text(json.dumps(storage.encode_rows(rows)), encoding="utf-8")
        with pytest.raises(AssertionError, match="Parity exclusions"):
            _compare(direct, scratch, tmp_path / "stand-in", right_source=direct)


def test_p0000_fts_hpc_planning_parity(tmp_path, engine, monkeypatch):
    """Run has no offline mode; plan --hpc exercises its delegated grouped plan."""
    import subprocess

    source = _source(tmp_path, engine, monkeypatch)
    for name in ("h001", "h002"):
        (source / f"inputs/hpc/{name}.toml").write_text(
            f'application_id = "offline-{name}"\n'
            f'max_walltime = "0{1 if name == "h001" else 2}:00:00"\n'
            '[descriptor]\nformat = "yaml"\nname = "submit.yaml"\n'
            '[descriptor.fields]\napplication = "{application_id}"\nscript = "{script_path}"\n'
            'workdir = "{work_dir}"\n[submit]\ncommand = ["never-submit", "{descriptor_path}"]\n',
            encoding="utf-8",
        )

    def no_launch(*args, **kwargs):
        pytest.fail("HPC planning must launch no process")

    monkeypatch.setattr(subprocess, "Popen", no_launch)
    direct, delegated = tmp_path / "direct/campaign", tmp_path / "delegated/campaign"
    for root, main in ((direct, engine[1]), (delegated, matrix_main)):
        shutil.copytree(source, root)
        if root == delegated:
            _declare(root)
        assert (
            _call(main, ["plan", "m", "--workspace", str(root), "--batch", "2", "--hpc", "h002"])
            == 0
        )
        assert not (root / "runs.json").exists()
    _compare(direct, delegated, tmp_path / "stand-in")
    plan = json.loads((direct / "post/m/plan.json").read_text(encoding="utf-8"))
    assert plan["grouping"]["max_walltime_s"] == 7200

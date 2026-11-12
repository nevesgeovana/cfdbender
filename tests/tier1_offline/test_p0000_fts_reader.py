"""P0000-FTS-READER: a read-only neutral view of public ledger and polar data.

Usage: read_workspace(root, group="1")[("7002", point)].result
The result uses coefficients exactly as the polar wrote them, with the run's
references. A submitted point has classification None because the neutral core
has no pending value. Exit code zero is never convergence evidence.
"""

import importlib
import json

import pytest

from cfdbender.core.results import Coefficient, NeutralResult, References

EXPECTED_STATUSES = {
    "CONVERGED": "converged",
    "COMPLETED_MAX_ITER": "iteration_limit",
    "MAX_ITER_REACHED": "iteration_limit",
    "FAILED_EXECUTION": "failed",
    "FAILED_SCRIPT": "failed",
    "FAILED_INCOMPLETE_OUTPUT": "failed",
    "FAILED_DIVERGED": "failed",
    "WALLTIME_REACHED": "failed",
    "SUBMITTED": None,
    "FAILED_MARKED": "failed",
    "RAN_MISSING_LOG": "failed",
}


@pytest.fixture
def upstream_reader():
    pytest.importorskip(
        "pyflightstream",
        reason="install cfdbender[flightstream]: pip install cfdbender[flightstream]",
    )
    return importlib.import_module("pyflightstream.workspace.ledger")


def _read(root, **kwargs):
    module = importlib.import_module("cfdbender.solvers.flightstream")
    # On the approved POC base, no reader means no neutral records. This shim
    # keeps the initial measurement an assertion about records, not ImportError.
    reader = getattr(module, "read_workspace", lambda *args, **kw: {})
    return reader(root, **kwargs)


def _record(status="CONVERGED", *, sim="7002", point="AL+000"):
    return {
        "run_id": f"campaign/sim_{sim}/{point}",
        "sim_id": sim,
        "point_name": point,
        "point": {"alpha": 0.0, "beta": 0.0},
        "matrix_stem": "m",
        "fs_version_requested": "26.120",
        "fs_build": "26.120",
        "package_version": "0.56.0",
        "script_sha256": "0" * 64,
        "raw_flag": False,
        "status": status,
        "iterations": 312,
        "residual": 9.6e-8,
        "reference": {"SREF": 10.0, "CREF": 1.2, "BREF": 8.0, "XMOM": 1, "YMOM": 2, "ZMOM": 3},
    }


def _workspace(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    encoder = importlib.import_module("pyflightstream.workspace.record_document").encode_rows
    (root / "runs.json").write_text(json.dumps(encoder(rows)), encoding="utf-8")
    polars = root / "post/m/polars"
    polars.mkdir(parents=True)
    # Synthetic product in the documented steady-polar column contract. Values
    # deliberately disagree with the raw loads and input-reference artifacts.
    (polars / "P7002-ALsweep_g1.csv").write_text(
        "POL,DESCRIPTION,GROUP,SREF,CREF,BREF,XMOM,YMOM,ZMOM,ALPHA,BETA,MACH,"
        "CDB,CYB,CLB,CRB25,CMB25,CNB25\n"
        "7002,test,1,999,999,999,0,0,0,0,0,0.1,0.123,0.234,0.345,0.456,0.567,0.678\n"
        "7002,test,1,999,999,999,0,0,0,2,0,0.1,0.223,0.334,0.445,0.556,0.667,0.778\n",
        encoding="utf-8",
    )
    return root


def test_p0000_fts_reader_reads_public_ledger_and_written_coefficients(
    tmp_path, upstream_reader, monkeypatch
):
    root = _workspace(tmp_path, [_record()])
    original = upstream_reader.read_ledger
    calls = []

    def observe(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(upstream_reader, "read_ledger", observe)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    records = _read(root, group="1")
    assert set(records) == {("7002", "AL+000")}
    point = records[("7002", "AL+000")]
    assert calls, "must use the public ledger reader"
    assert "read_ledger" in upstream_reader.__all__
    assert point.classification == "converged"
    assert isinstance(point.result, NeutralResult)
    assert point.result.references == References(
        10.0, 1.2, (1.0, 2.0, 3.0), "body", ("forward", "right", "down")
    )
    assert point.result.coefficients == {
        "CDB": Coefficient(0.123, "x", "force"),
        "CYB": Coefficient(0.234, "y", "force"),
        "CLB": Coefficient(0.345, "z", "force"),
        "CRB25": Coefficient(0.456, "x", "moment"),
        "CMB25": Coefficient(0.567, "y", "moment"),
        "CNB25": Coefficient(0.678, "z", "moment"),
    }
    assert point.result.history[-1].iteration == 312
    assert point.result.history[-1].residual == 9.6e-8
    assert point.result.solver_version == "26.120"
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "status,expected", EXPECTED_STATUSES.items(), ids=lambda value: f"P0000-{value}"
)
def test_p0000_fts_reader_every_status(tmp_path, upstream_reader, status, expected):
    root = _workspace(tmp_path, [_record(status)])
    records = _read(root, group="1")
    assert set(records) == {("7002", "AL+000")}
    point = records[("7002", "AL+000")]
    assert point.classification == expected, "exit 0 must not override the ledger status"
    if status == "SUBMITTED":
        assert point.result is None
    else:
        assert isinstance(point.result, NeutralResult)
        assert point.result.classification == expected


def test_p0000_fts_reader_status_inventory_is_exhaustive(upstream_reader):
    workspace = importlib.import_module("pyflightstream.workspace")
    assert {status.value for status in workspace.RunStatus} == set(EXPECTED_STATUSES), (
        "New upstream status: choose its neutral mapping explicitly"
    )


def test_p0000_fts_reader_grouped_points_use_effective_ledger_records(tmp_path, upstream_reader):
    row = _record()
    row["point_name"] = "JOB"
    row["run_id"] = "campaign/sim_7002/JOB"
    row["points_ran"] = [
        {
            "tag": "AL+000",
            "point": {"alpha": 0.0, "beta": 0.0},
            "status": "CONVERGED",
            "iterations": 312,
            "residual": 1e-8,
        },
        {
            "tag": "AL+020",
            "point": {"alpha": 2.0, "beta": 0.0},
            "status": "MAX_ITER_REACHED",
            "iterations": 800,
            "residual": 0.1,
        },
    ]
    root = _workspace(tmp_path, [row])
    records = _read(root, group="1")
    assert set(records) == {("7002", "AL+000"), ("7002", "AL+020")}
    assert records[("7002", "AL+000")].classification == "converged"
    second = records[("7002", "AL+020")]
    assert second.classification == "iteration_limit"
    assert second.result.coefficients["CLB"].value == 0.445
    assert second.result.history[-1].iteration == 800


def test_p0000_fts_reader_missing_evidence_is_not_invented(tmp_path, upstream_reader):
    row = _record()
    row["reference"] = None
    root = _workspace(tmp_path, [row])
    with pytest.raises(ValueError, match="reference"):
        _read(root, group="1")


def test_p0000_fts_reader_unknown_status_is_refused(tmp_path, upstream_reader):
    root = _workspace(tmp_path, [_record()])
    manifest = root / "runs.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["runs"][0]["status"] = "FUTURE_STATUS"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable|status"):
        _read(root, group="1")


def test_p0000_fts_reader_unsteady_uses_written_axes(tmp_path, upstream_reader):
    root = _workspace(tmp_path, [_record()])
    polar = root / "post/m/polars/P7002-ALsweep_g1.csv"
    body = polar.read_text(encoding="utf-8").replace("GROUP,", "").replace("test,1,", "test,")
    for name in ("CDB", "CYB", "CLB", "CRB25", "CMB25", "CNB25"):
        body = body.replace(name, f"{name}_MRP_TOTAL")
    polar.unlink()
    (polar.parent / "P7002_ALsweep_uns_avg.csv").write_text(body, encoding="utf-8")
    records = _read(root, group="MRP_TOTAL")
    assert records[("7002", "AL+000")].result.coefficients["CLB_MRP_TOTAL"].value == 0.345


def test_p0000_fts_reader_ignores_derived_polar_tables(tmp_path, upstream_reader):
    root = _workspace(tmp_path, [_record()])
    polar = root / "post/m/polars/P7002-ALsweep_g1.csv"
    (polar.parent / f"SUPER-{polar.name[1:]}").write_bytes(polar.read_bytes())
    (polar.parent / "P7002-ALsweep_g1_trim.csv").write_bytes(polar.read_bytes())
    records = _read(root, group="1")
    assert records[("7002", "AL+000")].result.coefficients["CLB"].value == 0.345


def test_p0000_fts_reader_refuses_ambiguous_polar_rows(tmp_path, upstream_reader):
    root = _workspace(tmp_path, [_record()])
    polar = root / "post/m/polars/P7002-ALsweep_g1.csv"
    body = polar.read_text(encoding="utf-8")
    polar.write_text(body + body.splitlines(keepends=True)[1], encoding="utf-8")
    with pytest.raises(ValueError, match="one polar row"):
        _read(root, group="1")


def test_p0000_fts_reader_reads_products_written_by_engine(tmp_path, upstream_reader, monkeypatch):
    from test_p0000_fts_plan_parity import _call
    from test_p0000_fts_run_parity import _source

    engine = (
        importlib.import_module("pyflightstream.workspace.cli").main,
        importlib.import_module("pyflightstream.run.cli").main,
    )
    root = _source(tmp_path, engine, monkeypatch)
    matrix = root / "inputs/matrices/m.fs"
    lines = matrix.read_text(encoding="utf-8").splitlines()
    matrix.write_text("\n".join([*lines[:2], lines[3]]) + "\n", encoding="utf-8")
    assert _call(engine[1], ["plan", "m", "--workspace", str(root)]) == 0
    assert _call(engine[1], ["run", "m", "--workspace", str(root), "--local"]) == 0
    records = _read(root, group="1")
    assert len(records) == 2
    for point in records.values():
        assert point.classification == EXPECTED_STATUSES[point.status]
        assert point.result.coefficients["CLB"].value == 0.431
        assert point.result.coefficients["CDB"].value == 0.0102
        assert point.result.references.area == 10.0
        assert point.result.references.length == 1.2
        assert point.reference_span == 8.0


def test_rfrun_001_documented_default_group(tmp_path, upstream_reader):
    root = _workspace(tmp_path, [_record()])
    assert _read(root) == _read(root, group="1")

"""The matrix column MESH, and the retired header GEOMETRY that is still read as MESH."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from setnum_support import PARTS, build, decks, digests

from cfdbender import migrate_setup_numerics as migrate
from cfdbender.core.matrix import MatrixError, parse_matrix
from cfdbender.core.registry import decode_record, encode_record
from cfdbender.workspace.planning import plan_workspace

REPO = Path(__file__).resolve().parents[2]
SOLVERS = ("su2", "openfoam", "pyfr")
HEAD = "POL|RUN|{col}|SETUP|REF|PPROC|FLIGHT_CONDITION"
ROW = "1001|1|{mesh}|001|001|001|{condition}"


def mesh_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if "column GEOMETRY is now MESH" in str(w.message)]


def with_column(root: Path, solver: str, column: str) -> Path:
    mesh = PARTS[solver][2][0].removeprefix("inputs/meshes/")
    return build(
        root,
        solver,
        columns=HEAD.format(col=column),
        cells=ROW.format(mesh=mesh, condition="{condition}"),
    )


def test_mesh_header_is_accepted_without_a_warning():
    text = b"POL|RUN|MESH|SETUP|REF|PPROC\n1|1|tiny.su2|001|001|001\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = parse_matrix(text, Path("m.fs"))
    assert rows[0].geometry == "tiny.su2"
    assert not mesh_warnings(caught)


def test_geometry_header_is_read_as_mesh_with_one_warning_per_matrix(tmp_path):
    with_column(tmp_path, "su2", "GEOMETRY")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = plan_workspace(tmp_path, "study")
    assert len(plan.points) == 1
    assert len(mesh_warnings(caught)) == 1
    message = mesh_warnings(caught)[0]
    assert "inputs/matrices/study.fs" in message
    assert "column GEOMETRY is now MESH; run the migration script" in message


def test_geometry_and_mesh_in_one_header_are_refused():
    text = b"POL|RUN|GEOMETRY|MESH|SETUP|REF|PPROC\n1|1|a|b|001|001|001\n"
    with pytest.raises(MatrixError, match="GEOMETRY and MESH name one column; keep MESH"):
        parse_matrix(text, Path("m.fs"))


def test_a_bad_mesh_cell_names_mesh_not_geometry():
    text = b"POL|RUN|MESH|SETUP|REF|PPROC\n1|1|a\\b.su2|001|001|001\n"
    with pytest.raises(MatrixError, match="MESH 'a"):
        parse_matrix(text, Path("m.fs"))


@pytest.mark.parametrize("solver", SOLVERS)
def test_plans_are_byte_identical_whichever_spelling_the_header_uses(tmp_path, solver):
    new = with_column(tmp_path / "new", solver, "MESH")
    old = with_column(tmp_path / "old", solver, "GEOMETRY")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan_new = plan_workspace(new, "study")
        plan_old = plan_workspace(old, "study")
        assert [p.point_id for p in plan_new.points] == [p.point_id for p in plan_old.points]
        assert [p.directory for p in plan_new.points] == [p.directory for p in plan_old.points]
        assert decks(new, solver) == decks(old, solver)
        assert digests(new) == digests(old)
    assert [p.outputs for p in plan_new.points] == [p.outputs for p in plan_old.points]
    assert plan_new.points[0].geometry is not None
    assert plan_new.points[0].geometry == plan_old.points[0].geometry


def test_records_are_identical_and_older_records_still_load(tmp_path):
    from cfdbender.core.registry import RunRecord

    raws = []
    for column in ("MESH", "GEOMETRY"):
        root = with_column(tmp_path / column, "su2", column)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point = plan_workspace(root, "study").points[0]
        record = RunRecord(
            point, "inputs/matrices/study.fs", "su2", "PLANNED", "2026-01-01T00:00:00+00:00"
        )
        raw = json.loads(json.dumps(encode_record(record)))
        assert decode_record(raw, root).point.geometry == point.geometry
        raws.append(raw)
    assert "GEOMETRY" not in json.dumps(raws[0])
    assert raws[0]["geometry"] == raws[1]["geometry"]
    assert raws[0]["point_id"] == raws[1]["point_id"]


def test_the_migration_renames_the_header_in_the_same_pass(tmp_path):
    root = with_column(tmp_path, "su2", "GEOMETRY")
    path = root / "inputs/matrices/study.fs"
    before = path.read_bytes()
    assert migrate.main([str(root)]) == 0
    assert path.read_bytes() == before  # a dry run changes nothing
    assert migrate.main([str(root), "--apply"]) == 0
    text = path.read_text()
    header = [c.strip() for c in text.splitlines()[0].split("|")]
    assert "MESH" in header and "GEOMETRY" not in header
    archives = list((root / "archive").glob("migration-setup-numerics-*/inputs/matrices/study.fs"))
    assert [a.read_bytes() for a in archives] == [before]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan_workspace(root, "study")
    assert not mesh_warnings(caught)


def test_the_migration_is_idempotent_for_the_mesh_rename(tmp_path, capsys):
    root = with_column(tmp_path, "su2", "GEOMETRY")
    migrate.main([str(root), "--apply"])
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert migrate.main([str(root), "--apply"]) == 0
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == after
    assert "nothing to change" in capsys.readouterr().out


def test_the_migration_summary_names_the_mesh_rename(tmp_path, capsys):
    root = with_column(tmp_path, "su2", "GEOMETRY")
    migrate.main([str(root)])
    assert "renamed GEOMETRY to MESH" in capsys.readouterr().out


def test_the_migration_leaves_a_header_with_both_names(tmp_path):
    text = "POL|RUN|GEOMETRY|MESH|SETUP|NUMERICS|REF|PPROC\n1|1|a|b|001||001|001\n"
    new, note = migrate.migrate_matrix(text, {})
    assert new == text and "both GEOMETRY and MESH" in note


def test_the_packaged_script_and_the_tools_copy_stay_identical():
    packaged = REPO / "src/cfdbender/migrate_setup_numerics.py"
    assert packaged.read_bytes() == (REPO / "tools/migrate_setup_numerics.py").read_bytes()

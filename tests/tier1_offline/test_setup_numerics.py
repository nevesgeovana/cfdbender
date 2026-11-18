"""The SETUP and NUMERICS matrix columns: header, merge, provenance and capability."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from setnum_support import build, decks, digests

from cfdbender.core.adapter import capabilities, load_point
from cfdbender.core.matrix import MatrixError, parse_matrix
from cfdbender.core.plan import PlanError
from cfdbender.core.registry import decode_record, encode_record
from cfdbender.core.settings import WorkspaceError
from cfdbender.core.workspace import init_workspace
from cfdbender.matrix.cli import main as matrix
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.planning import plan_workspace

SOLVERS = ("su2", "openfoam", "pyfr")
OLD = "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION"
NEW = "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION"
WITH = "POL|RUN|SETUP|NUMERICS|REF|PPROC|FLIGHT_CONDITION"


def retired_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if "column SET is now SETUP" in str(w.message)]


# ---- the header ------------------------------------------------------------------------


def test_setup_header_is_accepted_without_a_warning():
    text = b"POL|RUN|SETUP|REF|PPROC\n1|1|001|001|001\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = parse_matrix(text, Path("m.fs"))
    assert rows[0].setup_id == "s001" and rows[0].numerics_id == ""
    assert not retired_warnings(caught)


def test_set_header_is_read_as_setup_with_one_warning_per_matrix(tmp_path):
    build(tmp_path, "su2", columns=OLD, rows=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = plan_workspace(tmp_path, "study")
    assert [p.setup_id for p in plan.points] == ["s001"] * 3
    assert len(retired_warnings(caught)) == 1
    message = retired_warnings(caught)[0]
    assert "inputs/matrices/study.fs" in message and "migration script" in message


def test_set_and_setup_in_one_header_are_refused():
    text = b"POL|RUN|SET|SETUP|REF|PPROC\n1|1|001|001|001|001\n"
    with pytest.raises(MatrixError, match="SET and SETUP name one column"):
        parse_matrix(text, Path("m.fs"))


def test_numerics_cell_names_n_ids_and_refuses_other_text():
    text = b"POL|RUN|SETUP|NUMERICS|REF|PPROC\n1|1|001|002|001|001\n"
    assert parse_matrix(text, Path("m.fs"))[0].numerics_id == "n002"
    bad = b"POL|RUN|SETUP|NUMERICS|REF|PPROC\n1|1|001|x!|001|001\n"
    with pytest.raises(MatrixError, match="NUMERICS"):
        parse_matrix(bad, Path("m.fs"))


# ---- no numerics file: byte-identical to the base ---------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("cell", ["", "-"])
def test_an_unstated_numerics_cell_plans_byte_identically(tmp_path, solver, cell):
    base = build(tmp_path / "base", solver, columns=OLD)
    column = build(
        tmp_path / "column",
        solver,
        columns=WITH,
        cells="1001|1|001|" + cell + "|001|001|{condition}",
    )
    absent = build(tmp_path / "absent", solver, columns=NEW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = (decks(base, solver), digests(base))
        assert (decks(column, solver), digests(column)) == reference
        assert (decks(absent, solver), digests(absent)) == reference
    plan = plan_workspace(column, "study")
    assert plan.points[0].numerics_id == ""
    assert not any("numerics" in name for name in plan.points[0].inputs_sha256)


# ---- the merge --------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_numerical_keys_moved_to_a_numerics_file_give_the_same_input(tmp_path, solver):
    base = build(tmp_path / "base", solver, columns=NEW)
    split = build(
        tmp_path / "split",
        solver,
        columns=WITH,
        cells="1001|1|001|001|001|001|{condition}",
        split=True,
    )
    assert decks(split, solver) == decks(base, solver)
    point = plan_workspace(split, "study").points[0]
    assert point.numerics_id == "n001"
    assert "inputs/numerics/n001.toml" in point.inputs_sha256


def su2_precedence(
    root: Path, *, setup_extra=("CFL_NUMBER = 3.0",), variables="output_frequency: 7"
):
    build(
        root,
        "su2",
        columns=WITH + "|VAR_NAMES_VALUES",
        cells="1001|1|001|001|001|001|{condition}|" + variables,
        split=True,
        extra_setup=list(setup_extra),
        init_extra="[defaults]\niterations = 10\noutput_frequency = 3\nconvergence_residual = -8\n",
        numerics={
            "inputs/numerics/n001.toml": (
                "iterations = 20\noutput_frequency = 4\nCFL_NUMBER = 2.0\n"
            )
        },
    )
    return root


def deck(root: Path) -> dict[str, str]:
    text = decks(root, "su2")[0]
    return dict(line.split("= ", 1) for line in text.splitlines() if "= " in line)


def test_precedence_is_defaults_then_numerics_then_setup_then_row(tmp_path):
    su2_precedence(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = deck(tmp_path)
    assert values["ITER"] == "20", "the numerics file wins over the defaults"
    assert values["OUTPUT_WRT_FREQ"] == "7", "the row wins over the numerics file"
    assert values["CFL_NUMBER"] == "3", "the setup wins over the numerics file"
    assert values["CONV_RESIDUAL_MINVAL"] == "-8", "a default no layer replaces stays"


def test_a_key_in_both_files_warns_once_naming_the_key_and_both_files(tmp_path):
    su2_precedence(tmp_path)
    (tmp_path / "inputs/matrices/study.fs").write_text(
        (tmp_path / "inputs/matrices/study.fs").read_text().rstrip("\n")
        + "\n"
        + "1002|1|001|001|001|001|MACH:0.4,ALPHA:2.0|output_frequency: 7\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan_workspace(tmp_path, "study")
    both = [str(w.message) for w in caught if "stated in both" in str(w.message)]
    assert len(both) == 1, "one warning for the key, however many rows share the files"
    assert "CFL_NUMBER" in both[0]
    assert "inputs/numerics/n001.toml" in both[0] and "inputs/setups/s001.toml" in both[0]


def test_the_setup_wins_a_key_both_files_state(tmp_path):
    su2_precedence(tmp_path, variables="-")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert deck(tmp_path)["CFL_NUMBER"] == "3"
        assert deck(tmp_path)["OUTPUT_WRT_FREQ"] == "4"


@pytest.mark.parametrize(
    "text,word",
    [
        ("iterations = 40\n[solver]\nthreads = 2\n", "solver. table"),
        ('iterations = 40\nmesh = "inputs/meshes/tiny.su2"\n', "mesh key"),
        ('iterations = 40\nMESH_FILENAME = "tiny.su2"\n', "mesh key"),
    ],
)
def test_a_solver_table_or_a_mesh_in_a_numerics_file_is_refused(tmp_path, text, word):
    build(tmp_path, "su2", columns=WITH, cells="1001|1|001|001|001|001|{condition}", split=True)
    (tmp_path / "inputs/numerics/n001.toml").write_text(text)
    with pytest.raises(PlanError, match=word) as error:
        plan_workspace(tmp_path, "study")
    assert "inputs/numerics/n001.toml" in str(error.value)


def test_a_missing_numerics_file_is_refused_by_cell(tmp_path):
    build(tmp_path, "su2", columns=WITH, cells="1001|1|001|009|001|001|{condition}")
    with pytest.raises(PlanError, match="NUMERICS n009 names inputs/numerics/n009.toml"):
        plan_workspace(tmp_path, "study")


# ---- planned input, record, capability --------------------------------------------------


def split_su2(tmp_path: Path) -> Path:
    return build(
        tmp_path,
        "su2",
        columns=WITH,
        cells="1001|1|001|001|001|001|{condition}",
        split=True,
    )


def test_a_changed_numerics_file_is_refused_with_the_standard_text(tmp_path):
    root = split_su2(tmp_path)
    plan = plan_workspace(root, "study")
    path = root / "inputs/numerics/n001.toml"
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(
        ValueError, match="input changed after planning: inputs/numerics/n001.toml; plan again"
    ):
        load_point(root, plan.points[0], solver="su2")


def test_the_numerics_id_is_recorded_and_older_records_load(tmp_path):
    from dataclasses import replace

    from cfdbender.core.registry import RunRecord

    root = split_su2(tmp_path)
    point = plan_workspace(root, "study").points[0]
    record = RunRecord(
        point, "inputs/matrices/study.fs", "su2", "PLANNED", "2026-01-01T00:00:00+00:00"
    )
    raw = json.loads(json.dumps(encode_record(record)))
    assert raw["numerics_id"] == "n001"
    assert decode_record(raw, root).point.numerics_id == "n001"
    old = {k: v for k, v in raw.items() if k != "numerics_id"}
    old["inputs_sha256"] = {k: v for k, v in raw["inputs_sha256"].items() if "numerics" not in k}
    assert decode_record(old, root).point.numerics_id == ""
    bare = replace(point, numerics_id="", inputs_sha256=old["inputs_sha256"])
    assert "numerics_id" not in encode_record(replace(record, point=bare))


def test_a_record_naming_numerics_needs_its_digest(tmp_path):
    from cfdbender.core.registry import RegistryError, RunRecord

    root = split_su2(tmp_path)
    point = plan_workspace(root, "study").points[0]
    raw = json.loads(
        json.dumps(
            encode_record(
                RunRecord(
                    point, "inputs/matrices/study.fs", "su2", "PLANNED", "2026-01-01T00:00:00+00:00"
                )
            )
        )
    )
    del raw["inputs_sha256"]["inputs/numerics/n001.toml"]
    with pytest.raises(RegistryError, match="missing"):
        decode_record(raw, root)


def test_the_cfd_adapters_declare_the_numerics_capability():
    for solver in SOLVERS:
        assert "numerics_file" in capabilities(load_solver(solver))
    for solver in ("xfoil", "qprop", "mock"):
        if solver in available_solvers():
            assert "numerics_file" not in capabilities(load_solver(solver))


def test_a_solver_without_the_capability_warns_once_and_ignores_the_cell(tmp_path):
    init_workspace(tmp_path, solver="xfoil", available_solvers=available_solvers())
    files = {
        "inputs/setups/s001.toml": 'airfoil = "NACA 0012"\n',
        "inputs/references/r001.toml": "",
        "inputs/pproc/p001.toml": "",
        "inputs/matrices/study.fs": (
            "POL|RUN|SETUP|NUMERICS|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
            "1001|1|001|001|001|001|MACH:0,REmi:1,ALPHA:sweep|0,2\n"
            "1002|1|001|001|001|001|MACH:0,REmi:1,ALPHA:sweep|0,4\n"
        ),
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = plan_workspace(tmp_path, "study")
    ignored = [
        str(w.message) for w in caught if "NUMERICS is ignored for solver xfoil" in str(w.message)
    ]
    assert len(ignored) == 1
    assert all(p.numerics_id == "" for p in plan.points)
    assert not any("numerics" in name for p in plan.points for name in p.inputs_sha256)


# ---- init and the resolved view ---------------------------------------------------------


@pytest.mark.parametrize("solver", [*SOLVERS, "xfoil", "qprop"])
def test_init_creates_the_numerics_folder_for_the_cfd_solvers_and_documents_it(tmp_path, solver):
    init_workspace(tmp_path, solver=solver, available_solvers=available_solvers())
    cfd = solver in SOLVERS
    assert (tmp_path / "inputs/numerics").is_dir() is cfd
    guide = (tmp_path / "inputs/INPUTS.md").read_text()
    assert ("inputs/numerics/" in guide) is cfd
    assert ("NUMERICS" in guide) is cfd
    assert "SETUP 001" in guide and "SET 001" not in guide


def test_plan_verbose_says_where_each_setup_key_came_from(tmp_path, capsys):
    su2_precedence(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = matrix(["plan", "study", "--workspace", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert code == 0
    assert "SETUP=s001 NUMERICS=n001" in out
    assert "setup keys from defaults: convergence_residual" in out
    assert "setup keys from numerics: iterations" in out
    assert "setup keys from setup:" in out and "CFL_NUMBER" in out.split("from setup:")[1]
    assert "setup keys from row: output_frequency" in out


def test_a_workspace_error_is_still_a_workspace_error():
    assert issubclass(MatrixError, WorkspaceError)

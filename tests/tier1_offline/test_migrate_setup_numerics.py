"""The migration script: SET to SETUP, a NUMERICS column and split setups, byte-safe."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest
from setnum_support import build, decks

from cfdbender import migrate_setup_numerics as migrate

REPO = Path(__file__).resolve().parents[2]
SETUP_A = (
    "# the mesh of the first case\n"
    'mesh = "inputs/meshes/tiny.su2"\n'
    "# iteration budget\n"
    "iterations = 40\n"
    'convergence_field = "RMS_DENSITY"\n'
    "CFL_ADAPT_PARAM = [\n    0.5,\n    1.5,\n    1.0e-4,\n]\n"
    "freestream_temperature = 288.15\n"
    "reynolds = 1000000\n"
    'SOLVER = "EULER"\n'
    "\n[solver]\nthreads = 1\ntimeout_s = 5\n"
)
SETUP_B = SETUP_A.replace("reynolds = 1000000", "reynolds = 2000000")
SETUP_C = SETUP_A.replace("iterations = 40", "iterations = 80")
SETUP_D = 'mesh = "inputs/meshes/tiny.su2"\nfreestream_temperature = 288.15\nSOLVER = "EULER"\n'
HEADER = "POL | RUN | FLIGHT_CONDITION | REF | SET | PPROC | DESCRIPTION"
ROWS = [
    "# first the clean cases",
    "1001 | 1 | MACH:0.3, ALPHA:2.0 | 001 | 001 | 001 | case a",
    "1002 | 1 | MACH:0.3, ALPHA:2.0 | 001 | 002 | 001 | case b  ",
    "-----+---+-------------------+-----+-----+-----+--------",
    "1003 | 0 | MACH:0.3, ALPHA:2.0 | 001 | 003 | 001 | case c",
    "1004 | 1 | MACH:0.3, ALPHA:2.0 | 001 | 004 | 001 | case d",
]


def workspace(root: Path, *, eol: str = "\n") -> Path:
    build(root, "su2")
    for name, text in {"s001": SETUP_A, "s002": SETUP_B, "s003": SETUP_C, "s004": SETUP_D}.items():
        (root / f"inputs/setups/{name}.toml").write_bytes(text.replace("\n", eol).encode())
    matrix = eol.join([HEADER, *ROWS]) + eol
    (root / "inputs/matrices/study.fs").write_bytes(matrix.encode())
    return root


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def run(root: Path, *flags: str) -> int:
    return migrate.main([str(root), *flags])


def test_a_dry_run_prints_the_plan_and_changes_nothing(tmp_path, capsys):
    root = workspace(tmp_path)
    before = tree(root)
    assert run(root) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "would change: inputs/matrices/study.fs" in out
    assert "would create: inputs/numerics/n001.toml" in out
    assert tree(root) == before


def test_apply_archives_then_renames_adds_the_column_and_splits_the_setups(tmp_path, capsys):
    root = workspace(tmp_path)
    before = tree(root)
    assert run(root, "--apply") == 0
    after = tree(root)
    archive = "archive/migration-setup-numerics-1/"
    for name in ("inputs/matrices/study.fs", *(f"inputs/setups/s00{i}.toml" for i in (1, 2, 3))):
        assert after[archive + name] == before[name], "the original is archived byte for byte"
    assert archive + "inputs/setups/s004.toml" not in after, "an unchanged file is not archived"
    assert after["inputs/setups/s004.toml"] == before["inputs/setups/s004.toml"]
    numerics = after["inputs/numerics/n001.toml"].decode()
    assert "# iteration budget\niterations = 40\n" in numerics
    assert "CFL_ADAPT_PARAM = [\n    0.5,\n    1.5,\n    1.0e-4,\n]\n" in numerics
    assert "mesh" not in numerics and "[solver]" not in numerics
    kept = after["inputs/setups/s001.toml"].decode()
    assert "iterations = 40" not in kept.split("[solver]")[0]
    assert kept.startswith("# the mesh of the first case\nmesh = ")
    assert "[solver]\nthreads = 1\ntimeout_s = 5\n" in kept
    assert "n001.toml" in capsys.readouterr().out


def test_identical_numerics_share_one_file_and_each_row_names_its_numerics(tmp_path):
    root = workspace(tmp_path)
    run(root, "--apply")
    files = sorted(p.name for p in (root / "inputs/numerics").glob("*.toml"))
    assert files == ["n001.toml", "n002.toml"], "s001 and s002 differ only outside the numerics"
    text = (root / "inputs/matrices/study.fs").read_text().splitlines()
    cells = [[c.strip() for c in line.split("|")] for line in text if line[:1].isdigit()]
    assert [(c[0], c[5], c[6]) for c in cells] == [
        ("1001", "001", "001"),
        ("1002", "001", "001"),
        ("1003", "002", "001"),
        ("1004", "", "001"),
    ]


def test_the_matrix_changes_only_where_it_must(tmp_path):
    root = workspace(tmp_path)
    run(root, "--apply")
    new = (root / "inputs/matrices/study.fs").read_text().splitlines()
    old = [HEADER, *ROWS]
    assert new[0] == "POL | RUN | FLIGHT_CONDITION | REF | SETUP | NUMERICS | PPROC | DESCRIPTION"
    assert new[1] == old[1], "a comment line stays byte-identical"
    for before, after in zip(old[2:], new[2:], strict=True):
        got = after.split("|")
        assert [*got[:5], *got[6:]] == before.split("|"), "every other cell stays byte-identical"
    assert new[4] == old[4], "a rule line without pipes stays byte-identical"


def test_crlf_line_endings_are_kept(tmp_path):
    root = workspace(tmp_path, eol="\r\n")
    run(root, "--apply")
    data = (root / "inputs/matrices/study.fs").read_bytes()
    assert data.count(b"\r\n") == len(ROWS) + 1 and b"\n" not in data.replace(b"\r\n", b"")
    assert (root / "inputs/setups/s001.toml").read_bytes().count(b"\r\n") > 0


def test_a_second_run_changes_nothing_and_says_so(tmp_path, capsys):
    root = workspace(tmp_path)
    run(root, "--apply")
    settled = tree(root)
    capsys.readouterr()
    assert run(root, "--apply") == 0
    assert "nothing to change" in capsys.readouterr().out
    assert tree(root) == settled, "no new archive folder either"


def test_the_plan_before_and_after_is_byte_identical(tmp_path):
    root = workspace(tmp_path)
    (root / "inputs/matrices/study.fs").write_text(
        (root / "inputs/matrices/study.fs").read_text().replace("| 0 |", "| 1 |")
    )
    init = root / "init.toml"  # the setup without numerics takes its iterations from here
    init.write_text(init.read_text() + "[defaults]\niterations = 12\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = decks(root, "su2")
    run(root, "--apply")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        after = decks(root, "su2")
    assert len(before) == 4 and after == before
    assert not [w for w in caught if "SET is now SETUP" in str(w.message)]


def test_other_solvers_only_get_the_columns(tmp_path, capsys):
    root = build(tmp_path, "openfoam", columns="POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION")
    setup = (root / "inputs/setups/s001.toml").read_bytes()
    assert run(root, "--apply") == 0
    assert "only the columns are migrated" in capsys.readouterr().out
    assert (root / "inputs/setups/s001.toml").read_bytes() == setup
    assert not (root / "inputs/numerics").exists() or not list((root / "inputs/numerics").glob("*"))
    first = (root / "inputs/matrices/study.fs").read_text().splitlines()[:2]
    assert first[0] == "POL|RUN|SETUP|NUMERICS|REF|PPROC|FLIGHT_CONDITION"
    assert first[1].startswith("1001|1|001||001|001|")


def test_a_folder_that_is_not_a_workspace_is_refused(tmp_path, capsys):
    assert run(tmp_path) == 2
    assert "not a workspace" in capsys.readouterr().err


def test_the_script_is_standalone_and_the_tools_copy_is_the_same_file():
    tree_ = ast.parse(Path(migrate.__file__).read_text(encoding="utf-8"))
    imported = {
        name.split(".")[0]
        for node in ast.walk(tree_)
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert "cfdbender" not in imported
    assert (REPO / "tools/migrate_setup_numerics.py").read_bytes() == Path(
        migrate.__file__
    ).read_bytes()


@pytest.mark.parametrize("key", ["iterations", "CFL_NUMBER", "NUM_METHOD_GRAD", "cfl"])
def test_the_listed_numerical_keys_include_the_neutral_and_native_names(key):
    assert key in migrate.SU2_NUMERICAL_KEYS

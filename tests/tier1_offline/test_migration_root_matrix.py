"""Migration preserves the numerics of a supported root-level matrix."""

from __future__ import annotations

import warnings

from setnum_support import build, decks

from cfdbender.migrate_setup_numerics import apply, migrate


def test_root_matrix_keeps_its_setup_numerics(tmp_path):
    build(tmp_path, "su2", columns="POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION")
    (tmp_path / "inputs/matrices/study.fs").rename(tmp_path / "study.fs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = decks(tmp_path, "su2")
    result = migrate(tmp_path)
    assert "study.fs" in {edit.name for edit in result.edits}
    archive = apply(tmp_path, result)
    assert archive is not None and (archive / "study.fs").is_file()
    assert decks(tmp_path, "su2") == before
    assert not migrate(tmp_path).edits

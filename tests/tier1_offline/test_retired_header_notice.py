"""Retired headers retain values and remain notices under strict warning refusals."""

import warnings

from test_runtypes_b1_workflows import deck, workspace

from cfdbender.core.refusals import warning_refusals


def test_legacy_headers_plan_under_warning_refusals(tmp_path):
    workspace(tmp_path, condition="MACH:0.0529", rotor="")
    expected, values, files = deck(tmp_path)
    matrix = tmp_path / "inputs/matrices/rt.fs"
    matrix.write_text(matrix.read_text().replace("SETUP", "SET").replace("MESH", "GEOMETRY"))
    with warnings.catch_warnings(record=True) as caught, warning_refusals(True):
        warnings.simplefilter("always")
        actual, legacy_values, legacy_files = deck(tmp_path)
    assert actual.point_id == expected.point_id
    assert actual.directory == expected.directory
    assert (legacy_values, legacy_files) == (values, files)
    assert any("column SET is now SETUP" in str(w.message) for w in caught)
    assert any("column GEOMETRY is now MESH" in str(w.message) for w in caught)

"""F8-007: restore products and plan take a matrix name, never a folder step."""

from __future__ import annotations

import pytest
from cli_route import main
from test_cli_verbs import ran, snapshot


@pytest.mark.parametrize("kind", ["products", "plan"])
@pytest.mark.parametrize("name", [".", "..", "inputs/..", "./"])
def test_restore_refuses_a_matrix_that_is_not_a_name(tmp_path, capsys, kind, name):
    ran(tmp_path)
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["mark-failed", "--sims", "1001", "--apply", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--archive", "--workspace", str(tmp_path)]) == 1
    capsys.readouterr()
    before = snapshot(tmp_path)
    for apply in ([], ["--apply"]):
        code = main(["restore", kind, "--matrix", name, *apply, "--workspace", str(tmp_path)])
        assert code == 2
        assert "not a matrix name" in capsys.readouterr().err
    assert snapshot(tmp_path) == before

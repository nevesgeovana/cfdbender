"""RC-5: the refusal of an old cfd-matrix form shows the exact new command, quoted for the shell."""

from __future__ import annotations

import sys

import pytest

from cfdbender.matrix.cli import main as matrix_main


def test_rc5_the_new_command_repeats_the_arguments_exactly(capsys):
    assert matrix_main(["show", "1001", "--json", "--workspace", "camp"]) == 2
    assert (
        capsys.readouterr()
        .err.strip()
        .endswith("Type: cfd-workspace show 1001 --json --workspace camp")
    )


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows shell quotes with double quotes")
def test_rc5_a_path_with_a_space_is_double_quoted_on_windows(capsys):
    assert matrix_main(["sync", "--workspace", "my camp", "--apply"]) == 2
    assert (
        capsys.readouterr()
        .err.strip()
        .endswith('Type: cfd-workspace sync --workspace "my camp" --apply')
    )

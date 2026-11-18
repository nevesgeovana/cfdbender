"""A mode transition at the receipt boundary must disclose the weaker check."""

import pytest
from test_optcheck_identity import cli, small_threshold, workspace  # noqa: F401

from cfdbender.core.plan import DigestCache
from cfdbender.matrix import cli as cli_module
from cfdbender.matrix import receipt


@pytest.mark.parametrize("identity_plan", [True, False])
def test_mixed_mode_run_warns_once_at_receipt_boundary(
    tmp_path, monkeypatch, capsys, identity_plan
):
    monkeypatch.setattr(cli_module, "require_receipt", receipt.require_receipt)
    root = workspace(tmp_path / "ws", "su2")
    off = ("--no-read-large-inputs",)
    assert cli(root, "plan", *(off if identity_plan else ())) == 0
    (root / DigestCache.NAME).unlink(missing_ok=True)
    capsys.readouterr()
    assert cli(root, "run", "--local", *(() if identity_plan else off)) == 0
    output = capsys.readouterr()
    text = output.out + output.err
    expected = (
        "checked by size and modification time only"
        if identity_plan
        else "not verified against its planned SHA-256"
    )
    assert text.count(expected) == 1

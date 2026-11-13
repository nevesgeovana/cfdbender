"""C-09, second sentence, over the six adapters: an unread r or p key is warned and skipped."""

from __future__ import annotations

import warnings

import pytest
from test_conformance_workspace import case  # noqa: F401

from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("kind", ["references/r001", "pproc/p001"])
def test_c09_an_unread_reference_or_post_key_is_warned_and_skipped(case, kind):  # noqa: F811
    """C-09: the key is not refused; one warning names the row, the file and the key."""
    solver, root = case
    path = root / f"inputs/{kind}.toml"
    path.write_bytes(path.read_bytes() + b"\nzz_unread_key = 1\n")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        plan = plan_workspace(root, "full")
    assert plan.points, solver
    if solver == "mock":
        return  # the mock writes every r and p key into its input: none is unread
    named =[str(w.message) for w in seen if "zz_unread_key" in str(w.message)]
    assert named, f"{solver}: no warning names the unread key"
    assert all(kind.split("/")[1] in text and "POL 7001" in text for text in named)

"""P060: a skipped product uses the shared warning-refusal policy."""

import pytest

from cfdbender.core.refusals import warning_refusals
from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix.products import write_products


def skipped(root):
    summary = root / "post/study/campaign_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    return lambda: write_products(
        root,
        "study.fs",
        "mock",
        [{"point_id": "1001/DP-A", "status": "PLANNED", "result": None}],
        summary,
        [],
    )


def test_p060_product_skip_warns_with_point_and_reason(tmp_path):
    with pytest.warns(UserWarning, match="1001/DP-A.*PLANNED"):
        paths = skipped(tmp_path)()
    assert all(path.exists() for path in paths)


def test_p060_product_skip_refuses_under_the_shared_policy(tmp_path):
    write = skipped(tmp_path)
    with warning_refusals(True), pytest.raises(WorkspaceError, match="1001/DP-A"):
        write()
    assert not (tmp_path / "post/study/products.json").exists()

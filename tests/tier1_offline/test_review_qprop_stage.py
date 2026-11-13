"""QPROP must refuse staging if its job inputs require a local XFOIL run."""

import pytest
from qprop_workspace import adapter, prepared, workspace


def test_qprop_refuses_incomplete_cluster_staging_before_writing(tmp_path):
    workspace(tmp_path)
    point = prepared(tmp_path)
    with pytest.raises(ValueError, match="(?i)polar"):
        adapter().stage(point)
    assert not (tmp_path / point.point.directory).exists()

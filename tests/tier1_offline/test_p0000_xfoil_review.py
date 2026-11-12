"""Misspelled XFOIL setup keys must not silently change physics."""

import pytest
from test_p0000_xfoil_plan import prepared, tree, workspace


@pytest.mark.parametrize("key", ["reynold", "iteration", "executable"])
def test_p0000_xfoil_unknown_setup_keys(tmp_path, key):
    workspace(tmp_path, setup=f'airfoil="NACA 0012"\n{key}=1000000\n')
    before = tree(tmp_path)
    with pytest.raises(ValueError, match=key):
        prepared(tmp_path)
    assert tree(tmp_path) == before

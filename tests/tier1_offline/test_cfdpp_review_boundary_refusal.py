"""Unsupported conditions explain the supported boundary choices."""

import pytest
from test_cfdpp_real_case import NAMES, _case_of, _template

from cfdbender.solvers.cfdpp import _case


@pytest.mark.parametrize("wall", [False, True])
def test_missing_boundary_code_names_supported_choices(wall):
    setup = _template()
    names = dict(NAMES)
    if wall:
        setup["boundaries"]["wing"].update(
            treatment="wall-function", thermal="isothermal", temperature=310.0
        )
    else:
        names[3] = "mid"
        setup["boundaries"]["mid"] = {"type": "symmetry"}
    with pytest.raises(ValueError) as caught:
        _case.render(_case_of(setup, names=names))
    message = str(caught.value)
    assert "[boundaries." in message
    assert "supported" in message and "solve-to-wall" in message and "farfield" in message

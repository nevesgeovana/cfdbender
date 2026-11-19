"""Raw keyword arity follows the real grammar."""

import pytest
from test_cfdpp_real_case import _case_of, _template

from cfdbender.solvers.cfdpp import _case


@pytest.mark.parametrize("flags", [(0,), (0, 1, 0, 1)])
def test_variadic_raw_keyword_keeps_every_value(flags):
    setup = _template()
    setup["keywords"] = {"celltypes": 1, "celltype": [3, "fluxmasks", *flags]}
    text = _case.render(_case_of(setup))
    assert "celltype 3 fluxmasks " + " ".join(map(str, flags)) + "\r\n" in text

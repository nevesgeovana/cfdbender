"""The real grammar rejects malformed structure before building a model."""

import pytest
from test_cfdpp_real_grammar import TEXT

from cfdbender.solvers.cfdpp._schema import schema_for
from cfdbender.solvers.cfdpp._schema._model import parse


@pytest.mark.parametrize(
    "text",
    [
        TEXT.replace("\r\n", "\n", 1),
        "mbcons -1\r\nseq# type modi info\r\n",
        "infsets -2\r\n",
        "physics begin\r\nphysics end\r\noptions begin\r\noptions end\r\n",
    ],
)
def test_malformed_structure_is_refused(text):
    with pytest.raises(ValueError):
        parse(text, schema_for("22.5.U3"))

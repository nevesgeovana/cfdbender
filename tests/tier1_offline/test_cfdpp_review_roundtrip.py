"""Accepted text survives a read/write without lexical normalization."""

from dataclasses import replace

import pytest
from test_cfdpp_real_grammar import TEXT

from cfdbender.solvers.cfdpp._schema import schema_for
from cfdbender.solvers.cfdpp._schema._model import Trivia, parse, render


@pytest.mark.parametrize(
    "text",
    [
        TEXT.rstrip("\r\n"),
        TEXT.replace("\r\n", "\n"),
        TEXT.replace("infsets 2", "infsets   +2"),
        TEXT.replace("system begin", " system begin "),
        TEXT.replace("seq# type modi info", "seq#  type  modi info"),
        TEXT.replace("istart 0\r\nntstep 40", "ntstep 40\r\n# keep\r\nistart 0"),
    ],
)
def test_every_accepted_spelling_roundtrips(text):
    schema = schema_for("22.5.U3")
    assert render(parse(text, schema), schema) == text


def test_changing_trivia_does_not_replay_old_text():
    schema = schema_for("22.5.U3")
    model = parse(TEXT, schema)
    blocks = list(model.blocks)
    block = blocks[3]
    blocks[3] = replace(block, items=(*block.items, Trivia("# added")))
    assert "# added\r\n" in render(replace(model, blocks=tuple(blocks)), schema)

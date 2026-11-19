"""The grammar of mcfd.inp for CFD++ 22.5.U3: its layout in the interface database.

Written from the design's requirements (the layout lives in the interface database and is
version-gated; parse and render are inverse byte for byte on every input in schema order;
every promoted status cites a committed report). The input here is invented: no file the
vendor's tools wrote is part of the public tree.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cfdbender.solvers.cfdpp._schema import Status, schema_for
from cfdbender.solvers.cfdpp._schema._model import Pair, diff, parse, render

VERSION = "22.5.U3"
REPO = Path(__file__).resolve().parents[2]
REFERENCE = {"area": 1.0, "length": 1.0, "moment_origin": [0.25, 0.0, 0.0]}
CONDITION = {"MACH": 0.2, "REmi": 4.0, "TK": 290.0, "ALPHA": 3.0}
NAMES = {1: "wing", 2: "outer"}

# An invented input in the layout of 22.5.U3: comments before and inside sections, both
# system blocks, the bare cell-type and selection keywords, a repeated keyword with any
# number of values, multi-line information sets with their trailing spaces, boundary rows
# with family names, and reals spelt in several ways.
TEXT = "\r\n".join(
    [
        "#== invented ==",
        "system begin",
        "mc_filecopy cdepsout.bin cdepsin.bin",
        "system end",
        "#--",
        "iofiles begin",
        "iofiles end",
        "celltypes 1",
        "celltype 3 fluxmasks 0 0 0 0",
        "#",
        "celltype 3 vtxpmasks 0 0 0 0 0 0",
        "tsteps begin",
        "istart 0",
        "ntstep 40",
        "cfllbg 2",
        "cfllen 3.0e1",
        "tsteps end",
        "options begin",
        "xyzpol xy",
        "#----",
        "rtcord 4",
        "options end",
        "infsets 2",
        "#--",
        "seq.# 1 #vals 6 title primitive_variables_2",
        "values 1.5e+05 280 6.1e+01 0.000000e+00 0.0 ",
        "values 0.002 ",
        "#--",
        "seq.# 2 #vals 1 title temperature",
        "values 310 ",
        "#--",
        "eqslct 101",
        "icslct 1",
        "#",
        "mbcons 2",
        "seq# type modi info",
        "   1   13    0    2 wing",
        "   2   16    0    1 outer",
        "#seq mtyp info",
        "octree begin",
        "octree end",
        "physics begin",
        "gasnam Gas",
        "gasgam 1.3",
        "physics end",
        "probe begin",
        "probe end",
        "debug begin",
        "debug end",
        "guiopts begin",
        "guiopts end",
        "system begin",
        "system end",
        "#---",
        "",
    ]
)


@pytest.fixture
def schema():
    return schema_for(VERSION)


# --- the layout -------------------------------------------------------------------------


def test_cfdpp_g2_the_layout_of_the_version_is_established(schema):
    number_format = schema.require_grammar()
    assert number_format.line_ending == "\r\n"
    assert [s.name for s in schema.sections] == [
        "system", "iofiles", "celltypes", "tsteps", "options", "infsets", "eqslct",
        "mbcons", "octree", "physics", "probe", "debug", "guiopts", "system_last",
    ]  # fmt: skip
    assert schema.layout is not None and schema.layout.comment == "#"
    assert schema.raw_section is None


def test_cfdpp_g2_every_promoted_entry_cites_a_committed_report(schema):
    from test_cfdpp_database import _statuses

    promoted = [(label, report) for label, status, report in _statuses(schema)]
    promoted = [(label, report) for label, report in promoted if report is not None]
    assert len(promoted) > 300
    for label, report in promoted:
        assert (REPO / str(report)).is_file(), label
    assert all(status is not Status.VERIFIED for _, status, _ in _statuses(schema))


def test_cfdpp_g2_parse_and_render_keep_an_input_byte_for_byte(schema):
    model = parse(TEXT, schema)
    assert render(model, schema) == TEXT
    assert model.pair("cfllen") == Pair("cfllen", (30.0,))
    assert [b.section for b in model.blocks][:3] == ["system", "iofiles", "celltypes"]


def test_cfdpp_g2_an_edited_value_is_written_from_its_value(schema):
    model = parse(TEXT, schema)
    blocks = []
    for block in model.blocks:
        items = tuple(
            Pair("ntstep", (55,)) if isinstance(i, Pair) and i.name == "ntstep" else i
            for i in block.items
        )
        blocks.append(replace(block, items=items))
    text = render(replace(model, blocks=tuple(blocks)), schema)
    assert "ntstep 55\r\n" in text
    assert "cfllen 3.0e1\r\n" in text  # untouched entries keep their spelling
    assert [c.key for c in diff(model, parse(text, schema))] == ["ntstep"]


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("seq.# 2 #vals 1 title temperature", "seq.# 3 #vals 1 title temperature", "position 2"),
        ("#vals 1 title temperature", "#vals 2 title temperature", "takes 1 values"),
        ("infsets 2", "infsets 3", "states 3"),
        ("   2   16    0    1 outer", "   2   16    0    1", "family name"),
        ("gasnam Gas", "gasnam_x Gas", "not a keyword of section physics"),
        ("mbcons 2\r\nseq# type modi info", "mbcons 2\r\nseq# type info", "continues with"),
    ],
)
def test_cfdpp_g2_a_malformed_input_is_refused(schema, old, new, fragment):
    assert old in TEXT
    with pytest.raises(ValueError, match=fragment):
        parse(TEXT.replace(old, new, 1), schema)

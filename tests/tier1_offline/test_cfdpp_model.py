"""The grammar engine of the solver input, proved on the invented toy database."""

from __future__ import annotations

import random
import shutil
import string
from pathlib import Path
from typing import Any

import pytest
from cfdpp_support import toml_dumps, toml_load

from cfdbender.solvers.cfdpp._schema import Schema, schema_for
from cfdbender.solvers.cfdpp._schema._model import (
    Block,
    Boundary,
    Change,
    InfoSet,
    InputModel,
    Pair,
    diff,
    parse,
    render,
)
from cfdbender.solvers.cfdpp._schema._rules import Violation, check

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "cfdpp" / "toy_schema"

TEXT = (
    "begin control\n"
    "tsteps 500\n"
    "tmode 1\n"
    "tcfl 1.0 50.0\n"
    "tname alpha\n"
    "tset 1\n"
    "ninfo 2\n"
    "end control\n"
    "begin sets\n"
    "2\n"
    "prim 101325.0 300.0 50.0 0.0\n"
    "primt 101325.0 300.0 50.0 0.0 0.001\n"
    "end sets\n"
    "begin bcs\n"
    "2\n"
    "1 11 300.0\n"
    "2 20 1\n"
    "end bcs\n"
    "begin extras\n"
    "xgain 0.5\n"
    "zfree 3 word 2.5\n"
    "end extras\n"
    "shell begin\n"
    "cp a b\n"
    "shell end\n"
)


@pytest.fixture(scope="module")
def schema() -> Schema:
    return schema_for("1.0", TOY)


def _variant(tmp_path: Path, grammar: dict[str, Any]) -> Schema:
    root = tmp_path / "toy"
    shutil.copytree(TOY, root)
    path = root / "inp_sections.toml"
    data = toml_load(path)
    for key, value in grammar.items():
        # TOML has no null: an unestablished grammar entry is an omitted key.
        if value is None:
            data["grammar"].pop(key, None)
        else:
            data["grammar"][key] = value
    path.write_text(toml_dumps(data), encoding="utf-8")
    return schema_for("1.0", root)


def _control(*pairs: Pair) -> InputModel:
    return InputModel((Block("control", pairs),))


# Parse and render


def test_cfdpp_model_parse_gives_typed_values(schema):
    model = parse(TEXT, schema)
    assert [block.section for block in model.blocks] == [
        "control",
        "sets",
        "bcs",
        "extras",
        "shell",
    ]
    control = model.block("control")
    assert control is not None
    assert control.items[0] == Pair("tsteps", (500,))
    assert control.items[2] == Pair("tcfl", (1.0, 50.0))
    assert isinstance(control.items[2].values[0], float)
    assert control.items[3] == Pair("tname", ("alpha",))
    assert model.block("sets").items[1] == InfoSet("primt", (101325.0, 300.0, 50.0, 0.0, 0.001))
    assert model.block("bcs").items == (Boundary(1, 11, (300.0,)), Boundary(2, 20, (1,)))
    assert model.block("extras").items[1] == Pair("zfree", (3, "word", 2.5))
    assert model.block("shell").items == ("cp a b",)
    assert model.pair("tset") == Pair("tset", (1,))
    assert model.pair("absent") is None


def test_cfdpp_model_render_parse_round_trip_on_text(schema):
    assert render(parse(TEXT, schema), schema) == TEXT


def _word(rng: random.Random) -> str:
    return "w" + "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(0, 6)))


def _real(rng: random.Random) -> float:
    return rng.choice(
        [
            rng.uniform(0.0, 1.0e6),
            rng.uniform(0.0, 1.0) * 10.0 ** rng.randint(-30, 30),
            float(rng.randint(0, 1000)),
            0.0,
        ]
    )


def _value(kind: str, rng: random.Random, choices: Any) -> Any:
    if kind == "int":
        return rng.randint(1, 10**6)
    if kind == "real":
        return _real(rng)
    if kind == "word":
        return _word(rng)
    return rng.choice(choices)


def _random_model(schema: Schema, rng: random.Random) -> InputModel:
    blocks = []
    for section in schema.sections:
        if rng.random() < 0.2:
            continue
        items: list[Any] = []
        if section.kind == "pairs":
            known = sorted(
                (
                    keyword
                    for keyword in schema.keywords.values()
                    if keyword.section == section.name and keyword.types is not None
                ),
                key=lambda keyword: keyword.order or 0,
            )
            for keyword in known:
                if rng.random() < 0.6:
                    assert keyword.types is not None
                    values = tuple(_value(kind, rng, keyword.choices) for kind in keyword.types)
                    items.append(Pair(keyword.name, values))
            if section.name == schema.raw_section:
                for index in range(rng.randint(0, 3)):
                    values = tuple(
                        rng.choice([rng.randint(-50, 50), _real(rng), _word(rng)])
                        for _ in range(rng.randint(0, 3))
                    )
                    items.append(Pair(f"zz{index}{_word(rng)}", values))
        elif section.kind == "infosets":
            for _ in range(rng.randint(0, 3)):
                tag = rng.choice(sorted(schema.infosets))
                variables = schema.infosets[tag].variables or ()
                items.append(InfoSet(tag, tuple(_real(rng) for _ in variables)))
        elif section.kind == "boundaries":
            for selector in range(1, rng.randint(1, 4)):
                code = rng.choice(sorted(schema.boundary_codes))
                params = schema.boundary_codes[code].params or ()
                items.append(
                    Boundary(selector, code, tuple(_value(p.type, rng, None) for p in params))
                )
        else:
            items = [f"{_word(rng)} {_word(rng)}" for _ in range(rng.randint(0, 3))]
        blocks.append(Block(section.name, tuple(items)))
    return InputModel(tuple(blocks))


@pytest.mark.parametrize("seed", range(200))
def test_cfdpp_model_parse_and_render_are_inverse_on_random_models(schema, seed):
    model = _random_model(schema, random.Random(seed))
    text = render(model, schema)
    assert parse(text, schema) == model
    assert render(parse(text, schema), schema) == text


def test_cfdpp_model_render_writes_sections_in_schema_order(schema):
    model = parse(TEXT, schema)
    shuffled = InputModel(tuple(reversed(model.blocks)))
    assert render(shuffled, schema) == TEXT


def test_cfdpp_model_render_writes_keywords_in_schema_order(schema):
    model = InputModel(
        (
            Block("control", (Pair("ninfo", (2,)), Pair("tsteps", (5,)), Pair("tname", ("a",)))),
            Block("extras", (Pair("zlate", (1,)), Pair("xgain", (0.5,)), Pair("aearly", (2,)))),
        )
    )
    assert render(model, schema) == (
        "begin control\ntsteps 5\ntname a\nninfo 2\nend control\n"
        "begin extras\nxgain 0.5\nzlate 1\naearly 2\nend extras\n"
    )


def test_cfdpp_model_number_format_shortest(schema):
    model = _control(Pair("tcfl", (0.1, 2.0)), Pair("told", (1.0e-20,)))
    assert render(model, schema) == "begin control\ntcfl 0.1 2.0\ntold 1e-20\nend control\n"


def test_cfdpp_model_number_format_from_the_database(tmp_path):
    schema = _variant(tmp_path, {"number_format": {"real": ".3e"}})
    model = _control(Pair("tsteps", (12,)), Pair("tcfl", (0.1, 2.0)))
    assert (
        render(model, schema) == "begin control\ntsteps 12\ntcfl 1.000e-01 2.000e+00\nend control\n"
    )


def test_cfdpp_model_integer_given_for_a_real_is_written_as_a_real(schema):
    assert (
        render(_control(Pair("tcfl", (1, 2))), schema)
        == "begin control\ntcfl 1.0 2.0\nend control\n"
    )


def test_cfdpp_model_same_bytes_on_every_platform(schema):
    text = render(parse(TEXT, schema), schema)
    assert "\r" not in text
    assert text.encode("ascii") == TEXT.encode("ascii")


def test_cfdpp_model_crlf_line_ending_from_the_database(tmp_path):
    schema = _variant(tmp_path, {"line_ending": "crlf"})
    text = render(parse(TEXT, schema), schema)
    assert text == TEXT.replace("\n", "\r\n")
    assert parse(text, schema) == parse(TEXT, schema)


def test_cfdpp_model_blank_lines_between_entries_are_ignored(schema):
    spaced = TEXT.replace("tsteps 500\n", "\ntsteps 500\n\n")
    assert parse(spaced, schema) == parse(TEXT, schema)


def test_cfdpp_model_refuses_a_schema_whose_layout_is_not_established(tmp_path):
    schema = _variant(tmp_path, {"number_format": None})
    with pytest.raises(ValueError, match="not established"):
        parse(TEXT, schema)
    with pytest.raises(ValueError, match="not established"):
        render(InputModel(()), schema)


# Tokens outside the schema


@pytest.mark.parametrize(
    ("old", "new", "fragments"),
    [
        ("tname alpha\n", "bogus 1\n", ["bogus", "line 5", "control"]),
        ("tname alpha\n", "xgain 1.0\n", ["xgain", "extras", "control"]),
        ("tname alpha\n", "tvague 1\n", ["tvague", "line 5"]),
        ("prim 101325.0", "pram 101325.0", ["pram", "line 11"]),
        ("1 11 300.0\n", "1 99 300.0\n", ["99", "line 16"]),
        ("begin control\n", "stray\nbegin control\n", ["stray", "line 1"]),
        ("tsteps 500\n", "tsteps many\n", ["tsteps", "int", "many"]),
        ("tsteps 500\n", "tsteps 0\n", ["tsteps", "0", "range"]),
        ("tmode 1\n", "tmode 7\n", ["tmode", "7", "0, 1"]),
        ("tcfl 1.0 50.0\n", "tcfl 1.0\n", ["tcfl", "2", "1"]),
        ("prim 101325.0 300.0 50.0 0.0\n", "prim 101325.0 300.0\n", ["prim", "4"]),
        ("1 11 300.0\n", "1 11\n", ["11", "temperature"]),
        ("ninfo 2\n", "ninfo 2\ntsteps 3\n", ["tsteps", "twice"]),
        ("begin sets\n2\n", "begin sets\n3\n", ["sets", "3", "2"]),
        ("begin sets\n2\n", "begin sets\nmany\n", ["sets", "count"]),
        ("end extras\n", "", ["extras", "end extras"]),
        ("shell begin\ncp a b\nshell end\n", "begin control\nend control\n", ["control", "twice"]),
        ("tcfl 1.0 50.0\n", "tcfl 1.0 nan\n", ["tcfl", "nan"]),
    ],
)
def test_cfdpp_model_token_outside_the_schema_is_an_error_naming_it(schema, old, new, fragments):
    assert old in TEXT
    with pytest.raises(ValueError) as caught:
        parse(TEXT.replace(old, new, 1), schema)
    for fragment in fragments:
        assert fragment in str(caught.value)


@pytest.mark.parametrize(
    ("model", "fragments"),
    [
        (_control(Pair("tsteps", (1.5,))), ["tsteps", "int"]),
        (_control(Pair("tsteps", (True,))), ["tsteps", "int"]),
        (_control(Pair("tname", ("two words",))), ["tname", "two words"]),
        (_control(Pair("tcfl", (1.0,))), ["tcfl", "2"]),
        (_control(Pair("tcfl", (1.0, float("inf")))), ["tcfl", "inf"]),
        (_control(Pair("bogus", (1,))), ["bogus", "control"]),
        (InputModel((Block("nowhere", ()),)), ["nowhere"]),
        (InputModel((Block("sets", (InfoSet("prim", (1.0,)),)),)), ["prim", "4"]),
        (InputModel((Block("bcs", (Boundary(1, 99, ()),)),)), ["99"]),
        (InputModel((Block("bcs", (Pair("tsteps", (1,)),)),)), ["bcs", "Pair"]),
        (InputModel((Block("shell", ("shell end",)),)), ["shell end"]),
        (InputModel((Block("control", ()), Block("control", ()))), ["control", "twice"]),
    ],
)
def test_cfdpp_model_render_refuses_a_model_outside_the_schema(schema, model, fragments):
    with pytest.raises(ValueError) as caught:
        render(model, schema)
    for fragment in fragments:
        assert fragment in str(caught.value)


# Diff


def test_cfdpp_model_diff_of_equal_models_is_empty(schema):
    assert diff(parse(TEXT, schema), parse(TEXT, schema)) == []


def test_cfdpp_model_diff_reports_additions_removals_and_value_changes(schema):
    before = parse(TEXT, schema)
    after = parse(
        TEXT.replace("tsteps 500\n", "tsteps 800\n")
        .replace("tname alpha\n", "")
        .replace("xgain 0.5\n", "xgain 0.5\nznew 1\n")
        .replace("2 20 1\n", "2 20 1\n3 10\n")
        .replace("begin bcs\n2\n", "begin bcs\n3\n")
        .replace("shell begin\ncp a b\nshell end\n", ""),
        schema,
    )
    assert diff(before, after) == [
        Change("changed", "control", "tsteps", "500", "800"),
        Change("removed", "control", "tname", "alpha", None),
        Change("added", "bcs", "selector 3", None, "10"),
        Change("added", "extras", "znew", None, "1"),
        Change("removed", "shell", "section", None, None),
        Change("removed", "shell", "line 1", "cp a b", None),
    ]


def test_cfdpp_model_diff_of_information_sets_by_position(schema):
    before = parse(TEXT, schema)
    after = parse(
        TEXT.replace("prim 101325.0 300.0 50.0 0.0", "prim 101325.0 290.0 50.0 0.0"), schema
    )
    assert diff(before, after) == [
        Change(
            "changed",
            "sets",
            "set 1",
            "prim 101325.0 300.0 50.0 0.0",
            "prim 101325.0 290.0 50.0 0.0",
        )
    ]


# Interdependency rules


def test_cfdpp_rules_a_consistent_model_breaks_no_rule(schema):
    assert check(parse(TEXT, schema), schema) == []


@pytest.mark.parametrize(
    ("old", "new", "rule", "level"),
    [
        ("tcfl 1.0 50.0\n", "tcfl 60.0 50.0\n", "ramp-order", "hard"),
        ("ninfo 2\n", "", "set-needs-count", "hard"),
        ("ninfo 2\n", "ninfo 3\n", "set-count", "hard"),
        ("ninfo 2\n", "ninfo 2\ntlegacy 1\n", "label-or-legacy", "advisory"),
    ],
)
def test_cfdpp_rules_each_kind_and_level(schema, old, new, rule, level):
    violations = check(parse(TEXT.replace(old, new), schema), schema)
    assert [(violation.rule, violation.level) for violation in violations] == [(rule, level)]
    stated = next(item for item in schema.rules if item.id == rule)
    assert isinstance(violations[0], Violation)
    assert stated.message in violations[0].message


def test_cfdpp_rules_count_rule_counts_a_missing_section_as_empty(schema):
    model = parse(TEXT, schema)
    without_sets = InputModel(tuple(block for block in model.blocks if block.section != "sets"))
    assert [violation.rule for violation in check(without_sets, schema)] == ["set-count"]


def test_cfdpp_rules_a_rule_measured_broken_is_not_applied(tmp_path):
    root = tmp_path / "toy"
    shutil.copytree(TOY, root)
    path = root / "rules.toml"
    data = toml_load(path)
    data["rules"][0]["status"]["1.0"] = {"status": "broken", "report": "reports/toy/x.yaml"}
    path.write_text(toml_dumps(data), encoding="utf-8")
    schema = schema_for("1.0", root)
    model = parse(TEXT.replace("tcfl 1.0 50.0\n", "tcfl 60.0 50.0\n"), schema)
    assert check(model, schema) == []


@pytest.mark.parametrize("section", ["control", "extras"])
@pytest.mark.parametrize("name", ["", "tmode ", "x\ntmode"])
def test_model_rejects_invalid_free_pair_names_in_every_pair_section(schema, section, name):
    model = InputModel((Block(section, (Pair(name, (1,)),)),))
    with pytest.raises(ValueError, match="keyword name"):
        render(model, schema, free={name})

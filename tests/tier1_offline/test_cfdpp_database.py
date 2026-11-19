"""The CFD++ interface database of 22.5.U3: documentation and the first kit's harvest.

An entry the solver's help names is ``documented``, with a locator. An entry seen in
the inputs the GUI saved during the first kit is ``harvested``, citing that kit's
committed report. A status is promoted only by a committed report of a measured run or
harvest, never by editing the data files; nothing is ``verified`` before a generated
input has run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cfdpp_support import toml_load

from cfdbender.solvers.cfdpp._schema import (
    DATA,
    META,
    Status,
    schema_for,
    supported_versions,
)
from cfdbender.solvers.cfdpp._schema._model import InputModel, parse, render

VERSION = "22.5.U3"
KIT_A_REPORT = "reports/cfdpp/CMP-22.5.U3_2026-11-18_harvest-grammar.yaml"
QUOTE_MARKS = set("\"'`“”‘’«»")
LONGEST_MEANING = 120


def _files() -> list[Path]:
    return sorted(DATA.glob("*.toml"))


def _keys(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


def _meanings(schema: Any) -> Iterator[tuple[str, str]]:
    for section in schema.sections:
        yield f"section {section.name}", section.meaning
    for keyword in schema.keywords.values():
        yield f"keyword {keyword.name}", keyword.meaning
    for tag in schema.infosets.values():
        yield f"infoset {tag.tag}", tag.meaning
    for code in schema.boundary_codes.values():
        yield f"boundary code {code.code}", code.meaning
    for model in schema.turbulence.values():
        yield f"turbulence {model.name}", model.meaning
    for rule in schema.rules:
        yield f"rule {rule.id}", rule.message
    for table, entries in schema.entries.items():
        for entry in entries:
            yield f"{table} {entry.name}", entry.meaning


def _statuses(schema: Any) -> Iterator[tuple[str, Status, str | None]]:
    groups: list[Any] = [
        schema.sections,
        schema.keywords.values(),
        schema.infosets.values(),
        schema.boundary_codes.values(),
        schema.turbulence.values(),
        schema.rules,
    ]
    groups.extend(schema.entries.values())
    for group in groups:
        for item in group:
            label = getattr(item, "name", None) or getattr(item, "tag", None) or str(item)
            yield str(label), item.status, item.report


def test_cfdpp_database_loads_through_the_loader():
    schema = schema_for(VERSION)
    assert schema.solver == "CFD++"
    assert schema.input_file == "mcfd.inp"
    assert schema.version == VERSION
    assert len(schema.keywords) > 50
    assert schema.infosets and schema.boundary_codes and schema.turbulence
    for table in ("tools", "files"):
        assert schema.entries[table], table
    # Output layouts wait for a file returned from a run: none is committed.
    assert schema.entries["outputs"] == ()
    assert all(
        entry.status != "documented"
        for table in ("tools", "files", "outputs")
        for entry in schema.entries[table]
    )


def test_cfdpp_database_meta_holds_the_target_version_without_inheritance():
    assert supported_versions() == (VERSION,)
    meta = toml_load(DATA / META)
    assert [entry["canonical"] for entry in meta["versions"]] == [VERSION]
    assert meta["versions"][0]["inherits_base"] is False
    assert "prints" not in meta["versions"][0]


def test_cfdpp_database_promotions_are_harvested_by_the_first_kit_report_only():
    promoted = {
        (status.value, report)
        for _, status, report in _statuses(schema_for(VERSION))
        if status is not Status.DOCUMENTED or report is not None
    }
    assert promoted == {("harvested", KIT_A_REPORT)}


def test_cfdpp_database_no_combination_and_no_rule_without_evidence():
    schema = schema_for(VERSION)
    assert schema.entries["combinations"] == ()
    assert schema.rules == ()


@pytest.mark.parametrize("path", _files(), ids=lambda path: path.name)
def test_cfdpp_database_holds_no_value_field(path):
    data = toml_load(path)
    assert {"default", "defaults", "value"} & set(_keys(data)) == set()


def test_cfdpp_database_meanings_are_short_and_quote_free():
    long_or_quoted = [
        label
        for label, meaning in _meanings(schema_for(VERSION))
        if len(meaning) > LONGEST_MEANING or QUOTE_MARKS & set(meaning)
    ]
    assert long_or_quoted == []


def test_cfdpp_database_fixed_constants_of_the_first_release():
    fixed = {
        keyword.name: keyword.fixed
        for keyword in schema_for(VERSION).keywords.values()
        if keyword.fixed is not None
    }
    dimensional = dict.fromkeys(("refmwt", "reftem", "refden", "refpre", "refmuu", "refkap"), 1.0)
    assert fixed == {
        "simstr": 1,
        "inityp": 0,
        "mcgrps": 0,
        "ntout1": 1,
        "eqslct": 101,
        "grduni": "m",
        "univgc": 8314.0,
        **dimensional,
    }


def test_cfdpp_public_data_files_carry_no_locator_and_only_harvested_statuses():
    import re

    from cfdbender.solvers.cfdpp import _schema

    for path in sorted(Path(_schema.DATA).glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?m)^locator\s*=", text), path.name
        assert not re.search(r"status\s*=\s*\"(?!harvested\")", text), path.name
    schema = schema_for(VERSION)
    assert {status for _, status, _ in _statuses(schema)} <= {Status.HARVESTED}
    assert all(
        not locator
        for table, name, locator in _schema._labelled(schema)  # noqa: SLF001
    )


def test_cfdpp_database_generates_only_harvested_keywords_once_the_layout_is_established():
    schema = schema_for(VERSION)
    assert render(InputModel(()), schema) == ""
    assert parse("", schema) == InputModel(())
    emitted = {name for name in schema.keywords if schema.emittable(name)}
    harvested = {k.name for k in schema.keywords.values() if k.status is Status.HARVESTED}
    assert emitted == harvested

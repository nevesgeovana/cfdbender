"""The interface database loader and its version gate, proved on the invented toy database."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from cfdpp_support import toml_dumps, toml_load

from cfdbender.solvers.cfdpp._schema import (
    Status,
    canonical,
    locator_index,
    schema_for,
    supported_versions,
)

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "cfdpp" / "toy_schema"
REPORT_2 = "reports/toy/CMP-2.0_2026-01-01_toy.yaml"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "toy"
    shutil.copytree(TOY, target)
    return target


def _edit(root: Path, name: str, change: Callable[[dict[str, Any]], None]) -> None:
    path = root / name
    data = toml_load(path)
    change(data)
    path.write_text(toml_dumps(data), encoding="utf-8")


def _keyword(data: dict[str, Any], name: str) -> dict[str, Any]:
    return next(entry for entry in data["keywords"] if entry["name"] == name)


def test_cfdpp_schema_loads_the_toy_database():
    assert supported_versions(TOY) == ("1.0", "2.0")
    schema = schema_for("1.0", TOY)
    assert schema.solver == "ToySolver"
    assert schema.input_file == "toy.inp"
    assert schema.version == "1.0"
    assert [section.name for section in schema.sections] == [
        "control",
        "sets",
        "bcs",
        "extras",
        "shell",
    ]
    assert schema.raw_section == "extras"
    assert schema.keywords["tcfl"].types == ("real", "real")
    assert schema.keywords["tcfl"].arity == 2
    assert schema.keywords["tmode"].choices == (0, 1)
    assert schema.keywords["tmode"].fixed == 1
    assert schema.keywords["tsteps"].range == (1, None)
    assert schema.keywords["tvague"].section is None
    assert schema.infosets["primt"].variables == ("p", "t", "u", "v", "k")
    assert schema.boundary_codes[11].params[0].name == "temperature"
    assert schema.boundary_codes[11].params[0].type == "real"
    assert schema.turbulence["ToyOne"].codes == (1, 4)
    assert {rule.id for rule in schema.rules} == {
        "ramp-order",
        "ramp-steps-order",
        "set-needs-count",
        "set-count",
        "label-or-legacy",
    }
    assert [entry.name for entry in schema.entries["tools"]] == ["toyrun", "toystop"]
    assert schema.entries["outputs"][0].attrs["columns"] == ["step", "residual"]


def test_cfdpp_schema_view_is_per_version():
    first = schema_for("1.0", TOY)
    second = schema_for("2.0", TOY)
    assert first.keywords["tsteps"].status is Status.HARVESTED
    assert second.keywords["tsteps"].status is Status.VERIFIED
    assert second.keywords["tlegacy"].status is Status.BROKEN
    assert second.keywords["tlegacy"].report == REPORT_2
    assert second.keywords["tdoc"].report is None
    assert second.entries["tools"][0].status is Status.DOCUMENTED


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("1.0", "1.0"),
        (" 2.0 ", "2.0"),
        ("1.0.0", "1.0"),
        ("1.0-site", "1.0"),
        ("1.0.0-site-b", "1.0"),
        ("2.0-linux", "2.0"),
    ],
)
def test_cfdpp_schema_canonical_name_and_aliases(given, expected):
    assert canonical(given, TOY) == expected
    assert schema_for(given, TOY).version == expected


def test_cfdpp_schema_unknown_version_refused_naming_the_supported_ones():
    with pytest.raises(ValueError) as caught:
        schema_for("9.9", TOY)
    assert str(caught.value) == (
        "ToySolver 9.9: cfdbender writes toy.inp for 1.0, 2.0 only; no evidence exists "
        "for this version (see the version-onboarding page)"
    )


@pytest.mark.parametrize("given", [None, "", "  "])
def test_cfdpp_schema_missing_version_refused(given):
    with pytest.raises(ValueError) as caught:
        canonical(given, TOY)
    assert str(caught.value) == (
        "SOLVER_BUILD is required for ToySolver: state the solver version of this row "
        "(supported: 1.0, 2.0)"
    )


def test_cfdpp_schema_version_without_inheritance_statement_refuses_to_load(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "_meta.toml", lambda data: data["versions"][1].pop("inherits_base"))
    with pytest.raises(ValueError) as caught:
        schema_for("2.0", root)
    message = str(caught.value)
    assert "2.0" in message and "inherits_base" in message and "_meta.toml" in message
    assert schema_for("1.0", root).version == "1.0"


def test_cfdpp_schema_a_locator_is_optional_and_must_be_text(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "inp_control.toml", lambda data: _keyword(data, "tname").pop("locator"))
    assert schema_for("1.0", root).keywords["tname"].locator == ""
    root = _copy(tmp_path / "again")
    _edit(root, "inp_control.toml", lambda data: _keyword(data, "tname").update(locator=3))
    with pytest.raises(ValueError, match="tname.*locator"):
        schema_for("1.0", root)


def test_cfdpp_schema_every_entry_needs_a_status_row_per_registered_version(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "inp_blocks.toml", lambda data: data["infosets"][0]["status"].pop("2.0"))
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    assert "prim" in str(caught.value) and "2.0" in str(caught.value)


def test_cfdpp_schema_status_row_for_an_unregistered_version_refused(tmp_path):
    root = _copy(tmp_path)

    def change(data: dict[str, Any]) -> None:
        _keyword(data, "tsteps")["status"]["7.0"] = {"status": "documented"}

    _edit(root, "inp_control.toml", change)
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    assert "tsteps" in str(caught.value) and "7.0" in str(caught.value)


@pytest.mark.parametrize("status", ["harvested", "verified", "broken", "removed"])
def test_cfdpp_schema_measured_status_without_report_fails_the_load(tmp_path, status):
    root = _copy(tmp_path)

    def change(data: dict[str, Any]) -> None:
        _keyword(data, "tname")["status"]["2.0"] = {"status": status}

    _edit(root, "inp_control.toml", change)
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    message = str(caught.value)
    assert "tname" in message and status in message and "report" in message


def test_cfdpp_schema_unknown_status_refused(tmp_path):
    root = _copy(tmp_path)

    def change(data: dict[str, Any]) -> None:
        _keyword(data, "tname")["status"]["1.0"] = {"status": "guessed"}

    _edit(root, "inp_control.toml", change)
    with pytest.raises(ValueError, match="guessed"):
        schema_for("1.0", root)


@pytest.mark.parametrize(
    ("name", "table", "key"),
    [
        ("inp_control.toml", "keywords", "default"),
        ("inp_control.toml", "keywords", "value"),
        ("inp_blocks.toml", "infosets", "default"),
        ("inp_blocks.toml", "boundary_codes", "defaults"),
        ("rules.toml", "rules", "default"),
        ("tools.toml", "tools", "default"),
        ("inp_sections.toml", "sections", "value"),
    ],
)
def test_cfdpp_schema_no_entry_may_hold_a_default(tmp_path, name, table, key):
    root = _copy(tmp_path)
    _edit(root, name, lambda data: data[table][0].__setitem__(key, 1))
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    assert key in str(caught.value) and "default" in str(caught.value)


def test_cfdpp_schema_duplicate_entries_listed_in_one_refusal(tmp_path):
    root = _copy(tmp_path)

    def change(data: dict[str, Any]) -> None:
        data["keywords"].append(dict(_keyword(data, "tname")))
        data["keywords"].append(dict(_keyword(data, "tset"), order=99))

    _edit(root, "inp_control.toml", change)
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    message = str(caught.value)
    assert "duplicate" in message and "tname" in message and "tset" in message


def test_cfdpp_schema_unknown_field_refused(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "inp_control.toml", lambda data: _keyword(data, "tname").__setitem__("rnage", 1))
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    assert "rnage" in str(caught.value) and "tname" in str(caught.value)


def test_cfdpp_schema_measured_keyword_needs_its_place_and_types(tmp_path):
    root = _copy(tmp_path)

    def change(data: dict[str, Any]) -> None:
        _keyword(data, "tvague")["status"]["1.0"] = {
            "status": "harvested",
            "report": "reports/toy/CMP-1.0_2026-01-01_toy.yaml",
        }

    _edit(root, "inp_control.toml", change)
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    assert "tvague" in str(caught.value) and "section" in str(caught.value)


def test_cfdpp_schema_keyword_in_an_unknown_section_refused(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "inp_control.toml", lambda data: _keyword(data, "tname").update(section="nowhere"))
    with pytest.raises(ValueError) as caught:
        schema_for("1.0", root)
    assert "nowhere" in str(caught.value) and "tname" in str(caught.value)


def test_cfdpp_schema_types_must_match_arity(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "inp_control.toml", lambda data: _keyword(data, "tcfl").update(arity=3))
    with pytest.raises(ValueError, match="tcfl"):
        schema_for("1.0", root)


def test_cfdpp_schema_emission_rule():
    first = schema_for("1.0", TOY)
    second = schema_for("2.0", TOY)
    assert first.emittable("tsteps") is True
    assert second.emittable("tsteps") is True
    assert first.emittable("tdoc") is False
    assert first.emittable("tlegacy") is True
    with pytest.raises(ValueError) as broken:
        second.emittable("tlegacy")
    assert "tlegacy" in str(broken.value) and "broken" in str(broken.value)
    assert REPORT_2 in str(broken.value)
    with pytest.raises(ValueError) as removed:
        second.emittable("told")
    assert "told" in str(removed.value) and "removed" in str(removed.value)
    assert REPORT_2 in str(removed.value)
    with pytest.raises(ValueError, match="nokey"):
        first.emittable("nokey")


def test_cfdpp_schema_locator_index_lists_shared_locators(tmp_path):
    root = _copy(tmp_path)
    _edit(
        root,
        "inp_control.toml",
        lambda data: _keyword(data, "tdoc").update(locator="TOY :: control, tsteps"),
    )
    shared = locator_index(schema_for("1.0", root))
    assert shared["TOY :: control, tsteps"] == ("keywords tdoc", "keywords tsteps")
    assert all(len(names) > 1 for names in shared.values())


# TOML has no null: an unknown is an omitted key, and a wrong value is still refused.


def test_cfdpp_schema_a_wrong_inherits_base_differs_from_an_omitted_one(tmp_path):
    root = _copy(tmp_path)
    _edit(root, "_meta.toml", lambda data: data["versions"][1].__setitem__("inherits_base", "yes"))
    with pytest.raises(ValueError, match="inherits_base of 2.0 must be true or false"):
        schema_for("2.0", root)


@pytest.mark.parametrize("given", [[1, 2], {"low": 1}, 3])
def test_cfdpp_schema_range_is_a_table_of_min_and_max(tmp_path, given):
    root = _copy(tmp_path)
    _edit(
        root, "inp_control.toml", lambda data: _keyword(data, "tsteps").__setitem__("range", given)
    )
    with pytest.raises(ValueError, match="range is a table of min and max"):
        schema_for("1.0", root)


def test_cfdpp_schema_range_open_ends_are_omitted_keys(tmp_path):
    root = _copy(tmp_path)
    _edit(
        root,
        "inp_control.toml",
        lambda data: _keyword(data, "tsteps").__setitem__("range", {"max": 9}),
    )
    assert schema_for("1.0", root).keywords["tsteps"].range == (None, 9)


def test_cfdpp_schema_a_file_that_is_not_toml_is_refused_by_name(tmp_path):
    root = _copy(tmp_path)
    (root / "inp_control.toml").write_text("keywords = [", encoding="utf-8")
    with pytest.raises(ValueError, match="inp_control.toml: not valid TOML"):
        schema_for("1.0", root)

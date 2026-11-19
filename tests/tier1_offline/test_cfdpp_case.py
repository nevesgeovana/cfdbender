"""The CFD++ case builder: a checked case to a model of the input.

Every expectation is written from the toy database's bindings (tests/fixtures/cfdpp/
toy_schema/setup.toml and inp_case.toml), never from the builder's output.
"""

from __future__ import annotations

import shutil
import tomllib
import warnings
from dataclasses import replace

import pytest
from cfdpp_workspace import POST, SETUP, TOY, matrix, use_toy, with_keywords, workspace

from cfdbender.core.adapter import load_point
from cfdbender.solvers.cfdpp import _case, _config, translate
from cfdbender.solvers.cfdpp._schema import Status, schema_for
from cfdbender.solvers.cfdpp._schema._model import Boundary, InfoSet, Pair, parse
from cfdbender.solvers.cfdpp._schema._model import render as render_model
from cfdbender.workspace.planning import plan_workspace

NAMES = {1: "airfoil", 2: "farfield"}
NAMES3 = {1: "wing", 2: "farfield", 3: "sym"}
REFERENCE = {"area": 1.0, "length": 1.0, "moment_origin": [0.25, 0.0, 0.0]}
CONDITION = {"MACH": 0.15, "REmi": 6.0, "TK": 300.0, "ALPHA": 10.0}


def case(text=SETUP, *, condition=None, names=None, dimension=2, restart=False, data=TOY):
    return _config.config(
        schema_for("1.0", data),
        tomllib.loads(text),
        REFERENCE,
        tomllib.loads(POST),
        CONDITION if condition is None else condition,
        mesh="inputs/mesh/naca.cfdpp",
        names=NAMES if names is None else names,
        dimension=dimension,
        restart=restart,
    )


def pairs(built):
    return {
        item.name: item.values
        for block in built.model.blocks
        for item in block.items
        if isinstance(item, Pair)
    }


def block(built, section):
    found = built.model.block(section)
    return () if found is None else found.items


def refused(text=SETUP, **kwargs) -> str:
    with pytest.raises(ValueError) as caught:
        _case.build(case(text, **kwargs))
    return str(caught.value)


def test_cfdpp_case_every_bound_key_writes_its_keyword():
    written = pairs(_case.build(case()))
    assert written["tsteps"] == (200,)
    assert written["tdrop"] == (6.0,)
    assert written["tcfl"] == (1.0, 50.0)  # cfl and cfl_max share one keyword
    assert written["tramp"] == (1, 100)
    assert written["tturb"] == (9, 1)  # the model's codes
    assert written["tmeth"] == (1,)  # preconditioned
    assert written["tprev"] == (50.0,)
    assert written["tvisc"] == (1,) and written["tlim"] == (1,)
    assert written["tblend"] == (1.0, 1.0) and written["tbsteps"] == (1, 1)
    assert written["tout"] == (100,) and written["trst"] == (100,)
    assert written["tmode"] == (1,)  # fixed by the toy scope
    assert written["ninfo"] == (1,) and written["tset"] == (1,)
    assert written["trestart"] == (0,)
    assert "tname" not in written and "tdoc" not in written


def test_cfdpp_case_a_word_writes_its_bound_value():
    text = SETUP.replace('method = "preconditioned"\npreconditioning_velocity = 50.0',
                         'method = "density-based"')  # fmt: skip
    written = pairs(_case.build(case(text)))
    assert written["tmeth"] == (0,) and "tprev" not in written
    laminar = pairs(_case.build(case(SETUP.replace('"SpalartAllmaras"', '"laminar"'))))
    assert laminar["tturb"] == (9, 0)


def test_cfdpp_case_one_information_set_is_the_freestream_and_every_far_field_state():
    built = _case.build(case())
    stream = case().freestream
    (infoset,) = block(built, "sets")
    assert isinstance(infoset, InfoSet) and infoset.tag == "primn"
    assert infoset.values == pytest.approx(
        (stream.pressure, stream.temperature, *stream.velocity[:2], stream.turbulence["nut"])
    )
    bcs = block(built, "bcs")
    assert bcs == (Boundary(1, 12, ()), Boundary(2, 20, (1,)))
    assert pairs(built)["tset"] == (1,)


def test_cfdpp_case_three_dimensions_and_symmetry_choose_their_tag_and_codes():
    text = SETUP.replace("[boundaries.airfoil]", "[boundaries.wing]") + (
        '\n[boundaries.sym]\ntype = "symmetry"\n'
    )
    built = _case.build(case(text, names=NAMES3, dimension=3))
    (infoset,) = block(built, "sets")
    assert infoset.tag == "primn3" and len(infoset.values) == 6
    assert block(built, "bcs") == (
        Boundary(1, 12, ()),
        Boundary(2, 20, (1,)),
        Boundary(3, 30, ()),
    )


def test_cfdpp_case_each_wall_option_takes_its_code():
    def code(airfoil: str) -> Boundary:
        text = SETUP.replace(
            '[boundaries.airfoil]\ntype = "wall"\ntreatment = "solve-to-wall"\n'
            'thermal = "adiabatic"\n',
            f"[boundaries.airfoil]\n{airfoil}",
        )
        return block(_case.build(case(text)), "bcs")[0]

    iso = (
        'type = "wall"\ntreatment = "solve-to-wall"\nthermal = "isothermal"\ntemperature = 290.0\n'
    )
    assert code(iso) == Boundary(1, 13, (290.0,))
    function = 'type = "wall"\ntreatment = "wall-function"\nthermal = "adiabatic"\n'
    assert code(function) == Boundary(1, 14, ())
    assert code('type = "wall"\nslip = true\n') == Boundary(1, 15, ())


def test_cfdpp_case_a_boundary_code_without_evidence_is_refused():
    text = SETUP.replace(
        'treatment = "solve-to-wall"\nthermal = "adiabatic"',
        'treatment = "wall-function"\nthermal = "isothermal"\ntemperature = 290.0',
    )
    message = refused(text)
    assert message.startswith("setup file [boundaries.airfoil]: type = wall, ")
    assert "needs boundary code 16, which has no harvested or verified evidence" in message


def test_cfdpp_case_a_turbulence_model_without_a_matching_set_is_refused(tmp_path):
    data = tmp_path / "toy"
    shutil.copytree(TOY, data)
    path = data / "inp_case.toml"
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace('tag = "primko"', 'tag = "primkx"')
        .replace('variables = ["p", "t", "u", "v", "k", "omega"]', 'variables = ["p", "t", "k"]'),
        encoding="utf-8",
    )
    sst = SETUP.replace('"SpalartAllmaras"', '"SST"').replace(
        "viscosity_ratio = 3.0\n", "viscosity_ratio = 3.0\nturbulence_intensity = 0.01\n"
    )
    message = refused(sst, data=data)
    assert "no information-set tag holds the variables p, t, u, v, k, omega" in message


def test_cfdpp_case_a_harvested_combination_is_silent_and_sets_schema_status():
    verified = _case.build(case())  # the toy's verified combination
    assert verified.schema_status == "verified"
    text = SETUP.replace('method = "preconditioned"\npreconditioning_velocity = 50.0',
                         'method = "density-based"')  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        harvested = _case.build(case(text))
    assert harvested.schema_status == "harvested"
    assert harvested.advisories == ()


def test_cfdpp_case_restart_sets_the_restart_keyword():
    assert pairs(_case.build(case(restart=True)))["trestart"] == (1,)


def test_cfdpp_case_a_hard_rule_of_the_generated_input_is_refused():
    message = refused(SETUP.replace("cfl_max = 50.0", "cfl_max = 0.5"))
    assert "rule ramp-order: the ramp must not end below its start" in message
    message = refused(SETUP.replace("cfl_ramp_steps = [1, 100]", "cfl_ramp_steps = [100, 1]"))
    assert "rule ramp-steps-order" in message


# --- [keywords] -----------------------------------------------------------------------


def test_cfdpp_case_a_keyword_with_evidence_is_placed_and_typed():
    built = _case.build(case(with_keywords('tname = "polar"\n')))
    assert pairs(built)["tname"] == ("polar",)
    assert built.lines[-1] == "keywords tname = polar -> section control; harvested for 1.0"
    message = refused(with_keywords("tname = 3\n"))
    assert message == ("setup file [keywords] tname takes 1 value(s): word; found 3")
    message = refused(with_keywords("xgain = [1.0, 2.0]\n"))
    assert message == "setup file [keywords] xgain takes 1 value(s): real; found 1 2"
    assert pairs(_case.build(case(with_keywords("xgain = 2\n"))))["xgain"] == (2,)


def test_cfdpp_case_a_keyword_equal_to_its_shared_key_is_accepted_and_a_conflict_refused():
    built = _case.build(case(with_keywords("tsteps = 200\n")))
    assert pairs(built)["tsteps"] == (200,)
    assert refused(with_keywords("tsteps = 300\n")) == (
        "setup file: keywords tsteps conflicts with iterations; state it in one place"
    )


def test_cfdpp_case_a_fixed_keyword_is_refused():
    assert refused(with_keywords("tmode = 0\n")) == (
        "setup file [keywords] tmode: tmode is fixed by cfdbender's supported scope (1); "
        "it cannot be set"
    )


def test_cfdpp_case_broken_and_removed_keywords_are_refused_with_their_report():
    def on_2(text):
        with pytest.raises(ValueError) as caught:
            _case.build(
                _config.config(
                    schema_for("2.0", TOY),
                    tomllib.loads(text),
                    REFERENCE,
                    tomllib.loads(POST),
                    CONDITION,
                    mesh="inputs/mesh/naca.cfdpp",
                    names=NAMES,
                    dimension=2,
                )
            )
        return str(caught.value)

    broken = on_2(with_keywords("tlegacy = 1\n"))
    assert broken.startswith("setup file [keywords] tlegacy: ToySolver 2.0: keyword tlegacy is ")
    assert "broken for this version (report reports/toy/CMP-2.0_2026-01-01_toy.yaml)" in broken
    removed = on_2(with_keywords("told = 1.0\n"))
    assert "keyword told is removed in this version (report reports/toy/" in removed


def test_cfdpp_case_an_advisory_rule_is_returned_for_a_warning():
    built = _case.build(case(with_keywords('tname = "a"\ntlegacy = 1\n')))
    assert built.advisories == (
        "ToySolver 1.0: rule label-or-legacy: a label and the legacy switch are not used "
        "together (tname and tlegacy are both stated)",
    )


def test_cfdpp_case_a_documented_keyword_goes_to_its_database_section():
    built = _case.build(case(with_keywords("tdoc = 7\n")))
    assert pairs(built)["tdoc"] == (7,)
    assert any(item == Pair("tdoc", (7,)) for item in block(built, "control"))
    assert built.lines[-1] == (
        "keywords tdoc = 7 -> section control; documented only, no evidence for 1.0"
    )


def test_cfdpp_case_unknown_keywords_go_to_the_raw_section_in_setup_order():
    built = _case.build(case(with_keywords('zfirst = 1\nzsecond = [2, 3.5]\ntvague = "x"\n')))
    extras = block(built, "extras")
    assert extras == (Pair("zfirst", (1,)), Pair("zsecond", (2, 3.5)), Pair("tvague", ("x",)))
    assert built.lines[-3:] == (
        "keywords zfirst = 1 -> section extras; unknown to cfdbender",
        "keywords zsecond = 2 3.5 -> section extras; unknown to cfdbender",
        "keywords tvague = x -> section extras; documented only, no evidence for 1.0",
    )
    assert built.raw_keywords[1] == {
        "keyword": "zsecond",
        "value": [2, 3.5],
        "section": "extras",
        "evidence": "unknown",
    }


def test_cfdpp_case_the_section_hint_is_honoured_and_an_unknown_section_refused():
    built = _case.build(case(with_keywords('znew = { value = 2, section = "control" }\n')))
    assert Pair("znew", (2,)) in block(built, "control")
    assert built.lines[-1] == "keywords znew = 2 -> section control (hint); unknown to cfdbender"
    text = _case.render(case(with_keywords('znew = { value = 2, section = "control" }\n')))
    assert "znew 2\nend control" in text
    message = refused(with_keywords('znew = { value = 2, section = "physics" }\n'))
    assert message == (
        "setup file [keywords] znew: section 'physics' is not a keyword section of "
        "ToySolver 1.0; the sections are: control, extras"
    )


def test_cfdpp_case_raw_keywords_raise_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        built = _case.build(case(with_keywords("zfirst = 1\ntdoc = 2\n")))
    assert built.advisories == ()


# --- the rendered text ----------------------------------------------------------------


def test_cfdpp_case_render_is_deterministic_ascii_and_parses_back():
    text = _case.render(case())
    assert text == _case.render(case())
    text.encode("ascii")
    schema = schema_for("1.0", TOY)
    assert render_model(parse(text, schema), schema) == text
    assert text.startswith("begin control\ntsteps 200\ntmode 1\ntcfl 1.0 50.0\n")


def test_cfdpp_translate_is_deterministic_through_a_planned_point(tmp_path, monkeypatch):
    use_toy(monkeypatch)
    root = workspace(tmp_path / "ws", table=matrix(sweep="0,10"))
    plan = plan_workspace(root, "study")
    first, second = (load_point(root, point, solver="cfdpp") for point in plan.points)
    assert translate(first) == translate(first)
    assert translate(first) != translate(second)  # alpha changes the freestream
    assert translate(first).encode("ascii")


@pytest.mark.parametrize("name", ['"tmode "', '"zfree\\ntmode"'])
def test_cfdpp_raw_keyword_name_cannot_bypass_fixed_keyword_checks(name):
    with pytest.raises(ValueError, match="keyword name"):
        _case.render(case(with_keywords(f"{name} = 0\n")))


def test_cfdpp_documented_raw_keyword_does_not_activate_rules():
    declared = case(with_keywords('tname = "a"\ntlegacy = 1\n'))
    keywords = dict(declared.schema.keywords)
    keywords["tlegacy"] = replace(keywords["tlegacy"], status=Status.DOCUMENTED, report=None)
    declared = replace(declared, schema=replace(declared.schema, keywords=keywords))
    assert _case.build(declared).advisories == ()

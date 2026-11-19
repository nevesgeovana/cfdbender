"""The inputs cfdbender writes for CFD++ 22.5.U3 from a setup file.

Written from the design's requirements (cfdbender generates only harvested or verified
entries; one information set holds the freestream, an isothermal wall points at its own
temperature set; a choice without evidence is refused; the shipped setup template plans).
Every input here is rendered by the adapter itself from an invented setup.
"""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

import pytest
from cfdpp_workspace import POST, matrix, workspace

from cfdbender.core import setup_templates
from cfdbender.core.adapter import load_point
from cfdbender.solvers.cfdpp import _case, _config, translate
from cfdbender.solvers.cfdpp._schema import EMITTED, schema_for
from cfdbender.solvers.cfdpp._schema._model import Pair, parse, render
from cfdbender.workspace.planning import plan_workspace

VERSION = "22.5.U3"
REPO = Path(__file__).resolve().parents[2]
REFERENCE = {"area": 1.0, "length": 1.0, "moment_origin": [0.25, 0.0, 0.0]}
CONDITION = {"MACH": 0.2, "REmi": 4.0, "TK": 290.0, "ALPHA": 3.0}
NAMES = {1: "wing", 2: "outer"}


@pytest.fixture
def schema():
    return schema_for(VERSION)


def _case_of(setup: dict, *, names=None, dimension=2, condition=None):
    return _config.config(
        schema_for(VERSION),
        setup,
        REFERENCE,
        tomllib.loads(POST),
        CONDITION if condition is None else condition,
        mesh="inputs/mesh/naca.cfdpp",
        names=NAMES if names is None else names,
        dimension=dimension,
    )


def _template(**changes) -> dict:
    setup = tomllib.loads(setup_templates.read("cfdpp", "subsonic"))
    setup["boundaries"] = {
        "wing": {"type": "wall", "treatment": "solve-to-wall", "thermal": "adiabatic"},
        "outer": {"type": "farfield"},
    }
    for name, value in changes.items():
        table, _, leaf = name.rpartition("__")
        (setup.setdefault(table, {}) if table else setup)[leaf] = value
    return setup


# --- what cfdbender writes --------------------------------------------------------------


def test_cfdpp_g3_the_written_input_holds_harvested_entries_only(schema):
    text = _case.render(_case_of(_template()))
    model = parse(text, schema)
    for block in model.blocks:
        for item in block.items:
            if isinstance(item, Pair):
                assert schema.keywords[item.name].status in EMITTED, item.name
    assert all(not block.items for block in model.blocks if block.section == "guiopts")
    assert render(model, schema) == text
    assert "\n" not in text.replace("\r\n", "")


def test_cfdpp_g3_the_equation_set_and_freestream_open_the_information_sets(schema):
    text = _case.render(_case_of(_template()))
    lines = text.split("\r\n")
    start = lines.index("infsets 2")
    assert lines[start + 1] == "seq.# 1 #vals 31 title eqnset_define"
    assert lines[start + 2] == "values 101 1 1 1 4 "  # Spalart-Allmaras codes 1 and 4
    assert lines[start + 9] == "seq.# 2 #vals 6 title primitive_variables_2"  # 31 values: 7 lines
    assert "icslct 2" in lines and "eqslct 101" in lines
    assert "   1    7    0    0 wing" in lines
    assert "   2   16    0    2 outer" in lines  # the far field points at the freestream set
    assert lines[:3] == ["system begin", "mc_filecopy cdepsout.bin cdepsin.bin", "system end"]
    assert "xyzpol xy" in lines and "trzdir 0" in lines


def test_cfdpp_g3_an_isothermal_wall_points_at_its_own_temperature_set(schema):
    wall = {"type": "wall", "treatment": "solve-to-wall", "thermal": "isothermal"}
    setup = _template()
    setup["boundaries"]["wing"] = {**wall, "temperature": 321.5}
    lines = _case.render(_case_of(setup)).split("\r\n")
    assert "infsets 3" in lines
    start = lines.index("seq.# 3 #vals 1 title temperature")
    assert lines[start + 1] == "values 321.5 "
    assert "   1   13    0    3 wing" in lines


def test_cfdpp_g3_a_two_equation_model_writes_its_codes_and_variables(schema):
    setup = _template(turbulence="SST", turbulence_intensity=0.01)
    lines = _case.render(_case_of(setup)).split("\r\n")
    assert lines[lines.index("seq.# 1 #vals 31 title eqnset_define") + 1] == "values 101 1 1 2 3 "
    assert "seq.# 2 #vals 7 title primitive_variables_2" in lines


def test_cfdpp_g3_a_three_dimensional_grid_writes_no_two_dimensional_keyword(schema):
    text = _case.render(_case_of(_template(), dimension=3, condition={**CONDITION, "BETA": 1.0}))
    assert "xyzpol" not in text and "trzdir" not in text


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"physics__method": "density-based"}, "accepted values are: preconditioned"),
        ({"turbulence": "laminar"}, "accepted values are"),
        ({"physics__preconditioning_velocity": 40.0}, "unknown key 'preconditioning_velocity'"),
    ],
)
def test_cfdpp_g3_a_choice_without_evidence_is_refused(changes, fragment):
    with pytest.raises(ValueError, match=fragment):
        _case_of(_template(**changes))


def test_cfdpp_g3_a_boundary_without_a_harvested_code_is_refused():
    setup = _template()
    setup["boundaries"]["mid"] = {"type": "symmetry"}
    with pytest.raises(ValueError, match="has no boundary code"):
        _case.render(_case_of(setup, names={**NAMES, 3: "mid"}))


def test_cfdpp_g3_a_raw_keyword_with_evidence_goes_to_its_section(schema):
    setup = _template()
    setup["keywords"] = {"cflbot": 0.0002}
    lines = _case.render(_case_of(setup)).split("\r\n")
    assert lines.index("cflbot 0.0002") < lines.index("tsteps end")


def test_cfdpp_g3_the_template_keywords_are_harvested_and_written_in_their_sections(schema):
    setup = _template()
    table = setup["keywords"]
    assert table and all(set(entry) == {"value", "section"} for entry in table.values())
    built = _case.build(_case_of(setup))
    assert {record["evidence"] for record in built.raw_keywords} == {"harvested"}
    assert {r["keyword"]: r["section"] for r in built.raw_keywords} == {
        name: entry["section"] for name, entry in table.items()
    }
    bound = {key.keyword for key in _config.setup_keys(schema)}
    assert not set(table) & bound  # a raw keyword never repeats a setup key
    model = parse(_case.render(_case_of(setup)), schema)
    for name, entry in table.items():
        block = model.block(entry["section"])
        assert block is not None and any(getattr(i, "name", None) == name for i in block.items)


def test_cfdpp_g3_an_unknown_raw_keyword_needs_a_section_hint():
    setup = _template()
    setup["keywords"] = {"zz_invented": 1}
    with pytest.raises(ValueError, match="no section for raw keywords"):
        _case.render(_case_of(setup))


# --- planning the real version ----------------------------------------------------------


def test_cfdpp_g3_the_subsonic_template_plans_and_translates_on_the_real_version(tmp_path):
    # The synthetic 2D grid names its families airfoil and farfield, as the template does.
    setup = setup_templates.read("cfdpp", "subsonic")
    root = workspace(tmp_path / "ws", setup=setup, table=matrix(build=VERSION), executables=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "study")
    texts = [translate(load_point(root, point, solver="cfdpp")) for point in plan.points]
    assert len(texts) == 2 and texts[0] != texts[1]  # the sweep moves the freestream
    for text in texts:
        assert parse(text, schema_for(VERSION)) is not None
        assert text.startswith("system begin\r\nmc_filecopy cdepsout.bin cdepsin.bin\r\n")


def test_workspace_fixture_preserves_supplied_crlf_template(tmp_path):
    setup = setup_templates.read("cfdpp", "subsonic").replace("\r\n", "\n")
    setup = setup.replace("\n", "\r\n")
    root = workspace(tmp_path / "ws", setup=setup, executables=False)
    assert (root / "inputs/setups/s001.toml").read_bytes() == setup.encode("ascii")

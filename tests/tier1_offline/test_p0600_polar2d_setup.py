"""P2D-4: the SU2 polar setup of the docs states the JST deck in full; defaults stay.

The docs page "2D polars" carries an SU2 setup file. It is the same text as the fixture
fixtures/polar2d/su2-polar-setup.toml, it states the JST scheme with sensor coefficients
0.5 and 0.005 and a CFL number starting at 5, rising by 1.05 per iteration to 1e5, halved
on a residual rise and never below 5, and the SU2 adapter writes those values into
su2.cfg. A setup that states none of them still gets the adapter's own CFL ramp.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from cfdbender.solvers.su2._config import CFL_RAMP, config, render

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/polar2d/su2-polar-setup.toml"
REFERENCE = {"area": 1.0, "length": 1.0, "moment_origin": [0.25, 0.0, 0.0]}
CONDITION = {"MACH": 0.15, "REmi": 6.0, "ALPHA": 4.0}
POST = {"history": True}


def docs_setup() -> str:
    page = (ROOT / "docs/polars-2d.md").read_text(encoding="utf-8")
    section = page.split("## SU2 setup", 1)[1].split("## ", 1)[0]
    found = re.search("```toml" + chr(10) + "(.*?)```", section, re.S)
    assert found, "the SU2 setup section holds one toml block"
    return found[1]


def test_p0600_docs_setup_is_the_fixture():
    assert docs_setup() == FIXTURE.read_text(encoding="utf-8")


def test_p0600_setup_states_the_jst_deck():
    setup = tomllib.loads(FIXTURE.read_text(encoding="utf-8"))
    assert setup["CONV_NUM_METHOD_FLOW"] == "JST"
    assert setup["JST_SENSOR_COEFF"] == [0.5, 0.005]
    assert setup["CFL_NUMBER"] == 5.0
    assert setup["CFL_ADAPT_PARAM"][:4] == [0.5, 1.05, 5.0, 100000.0]
    assert setup["CONV_RESIDUAL_MINVAL"] == -12
    assert setup["SOLVER"] == "RANS" and setup["KIND_TURB_MODEL"] == "SA"


def test_p0600_adapter_writes_the_deck_and_keeps_its_defaults():
    setup = tomllib.loads(FIXTURE.read_text(encoding="utf-8"))
    cfg = render(config(setup, REFERENCE, POST, CONDITION, geometry="inputs/mesh/naca.su2"))
    assert "CONV_NUM_METHOD_FLOW= JST" in cfg
    assert "JST_SENSOR_COEFF= ( 0.5, 0.005 )" in cfg
    assert "CFL_NUMBER= 5" in cfg
    assert "CFL_ADAPT_PARAM= ( 0.5, 1.05, 5, 100000, 1, 0 )" in cfg
    plain = {"iterations": 20000, "SOLVER": "RANS", "KIND_TURB_MODEL": "SA"}
    default = render(config(plain, REFERENCE, POST, CONDITION, geometry="inputs/mesh/naca.su2"))
    assert "CONV_NUM_METHOD_FLOW= JST" not in default
    assert CFL_RAMP["CFL_ADAPT_PARAM"] == [0.5, 1.2, 5, 25]
    assert "CFL_ADAPT_PARAM= ( 0.5, 1.2, 5, 25 )" in default

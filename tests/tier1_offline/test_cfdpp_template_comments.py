"""Template guidance distinguishes required settings from optional raw overrides."""

from test_cfdpp_real_case import _case_of, _template

from cfdbender.core import setup_templates
from cfdbender.solvers.cfdpp import _case


def test_template_does_not_call_optional_raw_keywords_required():
    text = setup_templates.read("cfdpp", "subsonic")
    setup = _template()
    assert setup.pop("keywords")
    assert _case.render(_case_of(setup))
    assert "Every key is required" not in text

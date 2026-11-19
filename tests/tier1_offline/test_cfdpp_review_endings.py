"""Machine options cannot override an established real-file line ending."""

import pytest
from cfdpp_workspace import matrix, workspace

from cfdbender.core import setup_templates
from cfdbender.core.plan import PlanError
from cfdbender.workspace.planning import plan_workspace


def test_lf_override_of_real_schema_is_refused_at_plan(tmp_path):
    setup = setup_templates.read("cfdpp", "subsonic").replace("\r\n", "\n")
    root = workspace(
        tmp_path / "ws",
        setup=setup,
        table=matrix(build="22.5.U3"),
        executables=False,
        init_options='line_endings = "lf"\n',
    )
    with pytest.raises(PlanError, match="line_endings.*crlf"):
        plan_workspace(root, "study")

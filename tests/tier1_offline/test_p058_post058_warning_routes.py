"""P058-14: every warning obeys the shared warning-refusals setting.

Legacy mesh lookup and automatic SU2 route fallback must stop when warnings are
refused. A refused mesh warning must remain visible on a later ordinary lookup.
"""

import pytest

from cfdbender.core import mesh
from cfdbender.core.refusals import warning_refusals
from cfdbender.core.settings import WorkspaceError
from cfdbender.solvers.su2.routes import select_route


def missing_module(name):
    raise ImportError(name)


@pytest.mark.parametrize("kind", ["legacy", "both", "route"])
def test_p058_warnings_refuse_through_the_shared_policy(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(mesh, "_warned", set())
    old = tmp_path / "inputs/geometry/m.obj"
    old.parent.mkdir(parents=True)
    old.write_text("v 0 0 0\n", encoding="ascii")
    if kind == "both":
        current = tmp_path / "inputs/mesh/m.obj"
        current.parent.mkdir(parents=True)
        current.write_bytes(old.read_bytes())

    def action():
        if kind == "route":
            return select_route(importer=missing_module)
        return mesh.resolve_mesh_path(tmp_path, "inputs/geometry/m.obj")

    with warning_refusals(True), pytest.raises(WorkspaceError, match="warning-refusals"):
        action()
    category = RuntimeWarning if kind == "route" else UserWarning
    with pytest.warns(category):
        action()

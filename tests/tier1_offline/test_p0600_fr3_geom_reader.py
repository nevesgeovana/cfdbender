"""A reader refusal names the crossing components and the required trim change."""

from types import SimpleNamespace

import numpy as np
import pytest

from cfdbender.geometry import _geom_reader as reader
from cfdbender.mesh import iges_grid


@pytest.mark.parametrize("expected", [1], ids=["P0600-crossing-components"])
def test_p0600_fr3_grid_refusal_names_crossing_faces_and_dominance_change(
    tmp_path, monkeypatch, expected
):
    # Three faces meet at edge (0, 1); the rest of their boundaries stay open.
    mesh = SimpleNamespace(
        verts=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]),
        families={"body_0": [[0, 1, 2], [1, 0, 3]], "wing_0": [[0, 1, 4]]},
    )
    monkeypatch.setattr(reader, "read_obj", lambda path: mesh)
    monkeypatch.setattr(
        iges_grid,
        "grid_iges",
        lambda *args: {"conformal": {"conformal_mismatches": 0}},
    )
    manifest = {
        "pieces": [
            {
                "component": name,
                "face_records": [{"name": f"{name}_0", "iges": f"{name}.face-0.igs"}],
            }
            for name in ("body", "wing")
        ]
    }
    record = reader.grid(tmp_path, manifest)
    check = record["checks"][0]
    assert (check["status"], check["measured"], check["limit"]) == ("failed", expected, 0)
    assert check["where"].get("components") == ["body", "wing"]
    for text in ("gate.grid", "body_0", "wing_0", str(expected), "0", "dominance", "rebuild"):
        assert text in check["note"]

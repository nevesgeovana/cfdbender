"""The iges-grid decisions: the symmetry plane is stated, the encoding is looked up at entry."""

from __future__ import annotations

import pytest
from iges_support import cli, grid_file, plate_and_flap

from cfdbender._errors import InputValueError


def _symmetry_grid(tmp_path, line):
    path = tmp_path / "grid.toml"
    path.write_text(f"[defaults]\n{line}\n", encoding="utf-8")
    return path


def test_iges_decisions_symmetry_is_none_unless_the_grid_states_a_plane(tmp_path):
    """No half-model mirroring is inferred; the plane is stated."""
    from cfdbender.mesh._iges_grid_config import read_grid

    assert read_grid(_symmetry_grid(tmp_path, "size = 1.0")).symmetry == "none"
    assert read_grid(_symmetry_grid(tmp_path, 'symmetry = "y"')).symmetry == "y"


def test_iges_decisions_an_unknown_symmetry_is_refused(tmp_path):
    from cfdbender.mesh._iges_grid_config import read_grid

    with pytest.raises(InputValueError, match="symmetry"):
        read_grid(_symmetry_grid(tmp_path, 'symmetry = "auto"'))


def test_iges_decisions_an_unknown_encoding_is_refused_by_name_before_any_work(tmp_path, capsys):
    """A file with no level NAME never decodes one, so the codec is looked up at entry."""
    source = plate_and_flap(tmp_path)
    out = tmp_path / "model.obj"
    grid = grid_file(tmp_path, "[defaults]\nsize = 0.25\n")
    before = sorted(p.name for p in tmp_path.iterdir())
    code, _, stderr = cli(
        [
            "geometry",
            "iges-grid",
            str(source),
            "--grid",
            str(grid),
            "--out",
            str(out),
            "--name-encoding",
            "no-such-codec",
        ],
        capsys,
    )
    assert code != 0
    assert "no-such-codec" in stderr
    assert "pip install" not in stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_iges_decisions_the_python_entry_point_refuses_an_unknown_encoding_by_name(tmp_path):
    from cfdbender.mesh.iges_grid import grid_iges

    with pytest.raises(InputValueError, match="no-such-codec"):
        grid_iges([], tmp_path / "g.toml", tmp_path / "m.obj", name_encoding="no-such-codec")


class _ReachedError(Exception):
    """Raised by the patched ``oriented`` once it has seen its argument."""


def _crossing_skin_mirror(monkeypatch, run):
    """Run ``run`` on three faces sharing an edge; return what ``oriented`` got."""
    import numpy as np

    from cfdbender.geometry import _iges_skin

    seen: list[int | None] = []

    def oriented(self, mirror):
        seen.append(mirror)
        raise _ReachedError

    monkeypatch.setattr(_iges_skin._Assembly, "oriented", oriented)
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]])
    families = {"a": [[0, 1, 2]], "b": [[0, 1, 3]], "c": [[0, 1, 4]]}
    with pytest.raises(_ReachedError):
        run(verts, families)
    return seen


@pytest.mark.parametrize(("plane", "axis"), [("x", 0), ("y", 1), ("z", 2), ("none", None)])
def test_iges_decisions_the_grid_hands_its_configured_symmetry_to_the_skin(
    monkeypatch, plane, axis
):
    """``_inner`` passes ``config.symmetry`` on; a dropped argument mirrors nothing."""
    from types import SimpleNamespace

    from cfdbender.mesh.iges_grid import _inner

    config = SimpleNamespace(remove_inner_faces=True, symmetry=plane)

    def run(verts, families):
        meshed = SimpleNamespace(nodes=verts, faces=families, record={})
        _inner(config, meshed, list(families), [])

    assert _crossing_skin_mirror(monkeypatch, run) == [axis]

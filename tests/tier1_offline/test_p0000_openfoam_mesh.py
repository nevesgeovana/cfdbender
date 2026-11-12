"""P0000: a deterministic one-layer public NACA 0012 gmsh mesh."""

import importlib.util
from pathlib import Path

import pytest


def test_p0000_openfoam_mesh_deterministic(tmp_path):
    pytest.importorskip("gmsh", reason="gmsh Python API is not importable in this venv")
    script = Path(__file__).resolve().parents[2] / "tools/kit/openfoam_naca0012_mesh.py"
    assert script.is_file(), "OpenFOAM needs a one-layer gmsh 2.2 mesh builder"
    spec = importlib.util.spec_from_file_location("openfoam_mesh_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a, b = tmp_path / "a.msh", tmp_path / "b.msh"
    cells = module.generate(a)
    assert 100 < cells < 50000
    assert module.generate(b) == cells
    assert a.read_bytes() == b.read_bytes()
    text = a.read_text(encoding="ascii")
    assert "$MeshFormat\n2.2 0 8\n" in text
    for patch in ("airfoil", "farfield", "front", "back", "fluid"):
        assert f'"{patch}"' in text
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(a))
        _, coords, _ = gmsh.model.mesh.getNodes()
        assert set(round(z, 8) for z in coords[2::3]) == {0, 0.1}
        assert sum(len(tags) for tags in gmsh.model.mesh.getElements(3)[1]) == cells
        assert set(gmsh.model.mesh.getElements(3)[0]) <= {5, 6}
    finally:
        gmsh.finalize()

"""P0000-KIT: convert an AFLR3 prism to SU2's oriented wedge."""

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="the test builds arrays with numpy")


def test_p0000_kit_hpc_su2_prism_orientation(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/kit/ugrid_to_su2.py"
    spec = importlib.util.spec_from_file_location("converter", path)
    assert spec and spec.loader
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]])
    empty = np.empty((0, 3), dtype=int)
    out = tmp_path / "mesh.su2"
    converter.write_su2(
        out, xyz, empty, empty, [], empty, empty, np.array([[0, 1, 2, 3, 4, 5]]), empty, {}
    )
    row = out.read_text().splitlines()[2].split()
    assert row == ["13", "0", "2", "1", "3", "5", "4"]


def test_p0000_kit_hpc_marker_names_do_not_collide(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/kit/ugrid_to_su2.py"
    spec = importlib.util.spec_from_file_location("converter", path)
    assert spec and spec.loader
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    mapbc = tmp_path / "mesh.mapbc"
    mapbc.write_text("3\n1 3000 wing\n2 3000 wing\n3 3000 wing_1\n")
    names = converter.read_mapbc(mapbc)
    assert len(set(names.values())) == 3

    out = tmp_path / "mesh.su2"
    empty = np.empty((0, 3), dtype=int)
    converter.write_su2(
        out,
        empty,
        np.array([[0, 1, 2], [0, 1, 2]]),
        empty,
        [1, 4],
        empty,
        empty,
        empty,
        empty,
        {1: "tag_4"},
    )
    markers = [line for line in out.read_text().splitlines() if line.startswith("MARKER_TAG=")]
    assert len(set(markers)) == 2

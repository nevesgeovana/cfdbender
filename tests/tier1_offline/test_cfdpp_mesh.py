"""Native grid folders: headers, dimension, selectors, family names and the refusals."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from cfdpp_support import builder

from cfdbender.solvers.cfdpp import _mesh


def _files(kind: str) -> dict[str, bytes]:
    return getattr(builder(), f"grid{kind}_files")()


def test_grid_files_are_the_three_fixed_names() -> None:
    assert _mesh.GRID_FILES == ("nodesin.bin", "cellsin.bin", "exbcsin.bin")
    assert _mesh.BC_FILE == "mcfd.bc"


def test_grid2d_headers_dimension_and_selectors() -> None:
    files = _files("2d")
    facts = _mesh.grid_facts(files["nodesin.bin"], files["cellsin.bin"], files["exbcsin.bin"])
    assert facts.dimension == 2
    assert facts.node_count == 9
    assert facts.cell_count == 4
    assert facts.face_count == 8
    assert facts.selectors == (1, 2)


def test_grid3d_headers_dimension_and_selectors() -> None:
    files = _files("3d")
    facts = _mesh.grid_facts(files["nodesin.bin"], files["cellsin.bin"], files["exbcsin.bin"])
    assert facts.dimension == 3
    assert (facts.node_count, facts.cell_count, facts.face_count) == (8, 1, 6)
    assert facts.selectors == (1, 2, 3)


def test_read_bc_maps_selector_to_name() -> None:
    text = builder().bc_text(["airfoil", "farfield"])
    assert _mesh.read_bc(text) == {1: "airfoil", 2: "farfield"}


def test_read_bc_tolerates_trailing_blank_lines_and_crlf() -> None:
    text = builder().bc_text(["wing", "farfield", "sym"], trailing_blank_lines=3)
    assert _mesh.read_bc(text.replace("\n", "\r\n")) == {1: "wing", 2: "farfield", 3: "sym"}


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("seq# type modi info\n#a\n1 0 0 0\n", "first line"),
        ("# c\n#a\n1 0 0 0\n", "header line"),
        ("# c\nseq# type modi info\n#a\n", "pair"),
        ("# c\nseq# type modi info\nname\n1 0 0 0\n", "start with '#'"),
        ("# c\nseq# type modi info\n#a\n1 0 0\n", "four integers"),
        ("# c\nseq# type modi info\n#a\nx 0 0 0\n", "four integers"),
        ("# c\nseq# type modi info\n#1a\n1 0 0 0\n", "must start with a letter"),
        ("# c\nseq# type modi info\n#a b\n1 0 0 0\n", "not allowed"),
        ("# c\nseq# type modi info\n#a$\n1 0 0 0\n", "not allowed"),
        ("# c\nseq# type modi info\n#a\n1 0 0 0\n#b\n1 0 0 0\n", "twice"),
        ("# c\nseq# type modi info\n#a\n1 0 0 0\n#a\n2 0 0 0\n", "twice"),
    ],
)
def test_read_bc_refuses_malformed_files(text: str, fragment: str) -> None:
    with pytest.raises(_mesh.GridError, match=fragment):
        _mesh.read_bc(text)


def test_selectors_maps_names_in_the_order_given() -> None:
    bc = {1: "airfoil", 2: "farfield"}
    assert _mesh.selectors(["farfield", "airfoil"], bc) == (2, 1)


def test_selectors_refuses_an_unknown_name_listing_the_known_ones() -> None:
    with pytest.raises(_mesh.GridError, match="'wing'.*airfoil.*farfield"):
        _mesh.selectors(["wing"], {1: "airfoil", 2: "farfield"})


def test_truncated_header_is_refused() -> None:
    files = _files("2d")
    with pytest.raises(_mesh.GridError, match="nodesin.bin.*truncated"):
        _mesh.grid_facts(files["nodesin.bin"][:8], files["cellsin.bin"], files["exbcsin.bin"])


def test_truncated_body_is_refused() -> None:
    files = _files("2d")
    with pytest.raises(_mesh.GridError, match="exbcsin.bin.*truncated"):
        _mesh.grid_facts(files["nodesin.bin"], files["cellsin.bin"], files["exbcsin.bin"][:-6])


def test_little_endian_file_is_refused_by_name() -> None:
    files = _files("2d")
    little = builder()._pack_nodes(2, [(float(i), 0.0) for i in range(9)], "<")
    with pytest.raises(_mesh.GridError, match="nodesin.bin.*little-endian"):
        _mesh.grid_facts(little, files["cellsin.bin"], files["exbcsin.bin"])


def test_one_dimensional_and_other_plane_node_types_are_refused() -> None:
    files = _files("2d")
    nodes = builder()._pack_nodes(4, [(0.0, 0.0)] * 9)
    with pytest.raises(_mesh.GridError, match="node type 4"):
        _mesh.grid_facts(nodes, files["cellsin.bin"], files["exbcsin.bin"])


def test_read_grid_returns_facts_names_and_files(tmp_path: Path) -> None:
    folder = builder().write_grid2d(tmp_path / "naca.cfdpp")
    grid = _mesh.read_grid(folder, require_names=True)
    assert grid.facts.dimension == 2
    assert grid.names == {1: "airfoil", 2: "farfield"}
    assert [path.name for path in grid.files] == list(_mesh.GRID_FILES) + ["mcfd.bc"]
    assert grid.partitions == {}


def test_read_grid_without_names_keys_by_selector_number(tmp_path: Path) -> None:
    folder = builder().write_grid_bad(tmp_path / "g", "no_bc")
    grid = _mesh.read_grid(folder)
    assert grid.names == {1: "1", 2: "2"}


def test_missing_bc_is_refused_when_names_are_used(tmp_path: Path) -> None:
    folder = builder().write_grid_bad(tmp_path / "g", "no_bc")
    with pytest.raises(_mesh.GridError, match="mcfd.bc.*names"):
        _mesh.read_grid(folder, require_names=True)


def test_a_grid_selector_without_a_name_is_refused_when_names_are_used(tmp_path: Path) -> None:
    folder = builder().write_grid3d(tmp_path / "g")
    (folder / "mcfd.bc").write_text(builder().bc_text(["wing", "farfield"]), encoding="ascii")
    with pytest.raises(_mesh.GridError, match="selector 3"):
        _mesh.read_grid(folder, require_names=True)


@pytest.mark.parametrize("case", ["truncated", "little_endian", "bad_node_type"])
def test_bad_grid_folders_are_refused(tmp_path: Path, case: str) -> None:
    folder = builder().write_grid_bad(tmp_path / case, case)
    with pytest.raises(_mesh.GridError):
        _mesh.read_grid(folder)


def test_multi_group_file_is_refused(tmp_path: Path) -> None:
    folder = builder().write_grid_bad(tmp_path / "g", "multi_group")
    with pytest.raises(_mesh.GridError, match="cgrpsin.bin.1.*one cell group"):
        _mesh.read_grid(folder)


def test_single_group_file_is_accepted(tmp_path: Path) -> None:
    folder = builder().write_grid2d(tmp_path / "g")
    (folder / "cgrpsin.bin.1").write_bytes(struct.pack(">4i", 1, 1, 1, 1))
    assert _mesh.read_grid(folder).facts.cell_count == 4


def test_missing_grid_file_is_refused_by_name(tmp_path: Path) -> None:
    folder = builder().write_grid2d(tmp_path / "g")
    (folder / "cellsin.bin").unlink()
    with pytest.raises(_mesh.GridError, match="cellsin.bin"):
        _mesh.read_grid(folder)


def test_missing_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(_mesh.GridError, match="not a folder"):
        _mesh.read_grid(tmp_path / "absent.cfdpp")


@pytest.mark.parametrize("name", ["naca.su2", "NACA.SU2"])
def test_su2_name_is_refused_with_the_conversion_sentence(tmp_path: Path, name: str) -> None:
    with pytest.raises(_mesh.GridError, match="convert25.*native grid folder"):
        _mesh.read_grid(tmp_path / name)


def test_partition_files_are_found_by_process_count(tmp_path: Path) -> None:
    folder = builder().write_grid2d(tmp_path / "g")
    (folder / "mcpusin.bin.4").write_bytes(b"x")
    (folder / "mcpusin.bin.x").write_bytes(b"x")
    grid = _mesh.read_grid(folder)
    assert set(grid.partitions) == {4}
    assert grid.partitions[4].name == "mcpusin.bin.4"


def test_stage_grid_links_every_file_under_its_fixed_name(tmp_path: Path) -> None:
    source = builder().write_grid2d(tmp_path / "inputs" / "g")
    point = tmp_path / "point"
    point.mkdir()
    staged = _mesh.stage_grid(source, point)
    assert sorted(path.name for path in staged) == sorted([*_mesh.GRID_FILES, "mcfd.bc"])
    for path in staged:
        assert path.parent == point
        assert path.read_bytes() == (source / path.name).read_bytes()
        assert path.is_symlink() or path.samefile(source / path.name)


def test_stage_grid_refuses_a_file_changed_after_planning(tmp_path: Path) -> None:
    source = builder().write_grid2d(tmp_path / "g")
    point = tmp_path / "point"
    point.mkdir()
    digests = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in _mesh.GRID_FILES
    }
    digests["nodesin.bin"] = "0" * 64
    with pytest.raises(ValueError, match="changed after planning"):
        _mesh.stage_grid(source, point, digests)
    assert not (point / "nodesin.bin").exists()


def test_stage_grid_links_the_partition_for_the_process_count(tmp_path: Path) -> None:
    source = builder().write_grid2d(tmp_path / "g")
    (source / "mcpusin.bin.2").write_bytes(b"x")
    point = tmp_path / "point"
    point.mkdir()
    staged = _mesh.stage_grid(source, point, ncpus=2)
    assert "mcpusin.bin.2" in {path.name for path in staged}
    other = tmp_path / "other"
    other.mkdir()
    assert "mcpusin.bin.2" not in {path.name for path in _mesh.stage_grid(source, other)}


def test_optional_native_grid_inputs_are_staged_and_tracked(tmp_path: Path) -> None:
    source = builder().write_grid2d(tmp_path / "g")
    extras = {
        "cgrpsin.bin.1": struct.pack(">4i", 1, 1, 1, 1),
        "dfcelin.bin": b"synthetic deformation data",
        "inbcsin.bin": b"synthetic internal boundary data",
    }
    for name, data in extras.items():
        (source / name).write_bytes(data)
    assert set(extras) <= {path.name for path in _mesh.read_grid(source).files}
    target = tmp_path / "point"
    target.mkdir()
    _mesh.stage_grid(source, target)
    assert all((target / name).samefile(source / name) for name in extras)


def test_second_cell_group_file_is_refused_even_with_uniform_ids(tmp_path: Path) -> None:
    source = builder().write_grid2d(tmp_path / "g")
    (source / "cgrpsin.bin.2").write_bytes(struct.pack(">4i", 2, 2, 2, 2))
    with pytest.raises(_mesh.GridError, match="one cell group"):
        _mesh.read_grid(source)


@pytest.mark.parametrize("name", ["dfcelin.bin", "inbcsin.bin", "cgrpsin.bin.1", "mcpusin.bin.2"])
def test_stage_grid_rejects_files_added_after_planning(tmp_path: Path, name: str) -> None:
    source = builder().write_grid2d(tmp_path / "g")
    digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _mesh.read_grid(source).files
    }
    payload = struct.pack(">4i", 1, 1, 1, 1) if name == "cgrpsin.bin.1" else b"new data"
    (source / name).write_bytes(payload)
    point = tmp_path / "point"
    point.mkdir()
    with pytest.raises(ValueError, match="changed after planning"):
        _mesh.stage_grid(source, point, digests, ncpus=2)

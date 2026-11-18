"""P0000 BIGMESH: planning reads and hashes a large mesh once per plan, not once per point."""

import functools
import hashlib
import re
import tracemalloc

import pytest
from test_p0000_item3d_config import api, plan, workspace

from cfdbender.core import plan as plan_module
from cfdbender.core.plan import PlanError, plan_matrix
from cfdbender.solvers._host._groups import _Su2Markers, marker_names

MESH = "inputs/meshes/tiny.su2"
MATRIX = (
    "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
    "1001|1|001|001|001|MACH:0.3,ALPHA:sweep|0,1,2\n"
    "1002|1|001|001|001|MACH:0.3,ALPHA:sweep|0,1,2\n"
    "1003|1|001|001|001|MACH:0.3,ALPHA:sweep|0,1,2\n"
)


class _Counting:
    """Count opens of the mesh and the bytes hashed per digest state."""

    def __init__(self, monkeypatch, mesh):
        self.opens = 0
        self.read_bytes = 0
        self.states = []
        original_open = plan_module._open_input
        original_hasher = plan_module._hasher
        counting = self

        class Stream:
            def __init__(self, path):
                self.inner = original_open(path)
                self.watched = path == mesh

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.inner.close()

            def read(self, *args):
                data = self.inner.read(*args)
                if self.watched:
                    counting.read_bytes += len(data)
                return data

        def open_input(path):
            if path == mesh:
                counting.opens += 1
            return Stream(path)

        class Hasher:
            def __init__(self):
                self.inner = original_hasher()
                self.fed = 0
                counting.states.append(self)

            def update(self, data):
                self.fed += len(data)
                self.inner.update(data)

            def hexdigest(self):
                return self.inner.hexdigest()

        monkeypatch.setattr(plan_module, "_open_input", open_input)
        monkeypatch.setattr(plan_module, "_hasher", Hasher)

    def hashed_meshes(self, size):
        return sum(1 for state in self.states if state.fed >= size)


def padded(tmp_path, megabytes, groups=False):
    workspace(tmp_path, extra='[groups]\nwing = ["wall"]' if groups else "")
    mesh = tmp_path / MESH
    data = (
        b"NDIME= 2\nNMARK= 2\nMARKER_TAG= wall\nMARKER_ELEMS= 0\nMARKER_TAG= farfield\n"
        + b"% padding line of the mesh\n" * (megabytes * 1024 * 1024 // 27)
    )
    mesh.write_bytes(data)
    (tmp_path / "inputs/matrices/study.fs").write_text(MATRIX)
    return mesh.resolve(), data


@pytest.mark.parametrize("threshold", [64 * 1024, plan_module.LARGE_INPUT_BYTES])
def test_p0000_bigmesh_reads_and_hashes_once(tmp_path, monkeypatch, threshold):
    mesh, data = padded(tmp_path, 2, groups=True)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", threshold)
    counting = _Counting(monkeypatch, mesh)
    result = plan(tmp_path)
    assert len(result.points) == 9
    assert counting.opens == 1
    assert counting.read_bytes <= len(data)
    assert counting.hashed_meshes(len(data)) == 1
    expected = hashlib.sha256(data).hexdigest()
    assert {p.inputs_sha256[MESH] for p in result.points} == {expected}


def test_p0000_bigmesh_provenance_keys_and_values_are_unchanged(tmp_path, monkeypatch):
    padded(tmp_path, 1)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 64 * 1024)
    result = plan(tmp_path)
    for point in result.points:
        assert list(point.inputs_sha256) == sorted(
            [
                "init.toml",
                "inputs/matrices/study.fs",
                MESH,
                "inputs/pproc/p001.toml",
                "inputs/references/r001.toml",
                "inputs/setups/s001.toml",
            ]
        )
        for name, digest in point.inputs_sha256.items():
            assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


def test_p0000_bigmesh_large_mesh_is_not_held_in_memory(tmp_path, monkeypatch):
    _, data = padded(tmp_path, 16, groups=True)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024 * 1024)
    tracemalloc.start()
    try:
        plan(tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < len(data) // 4


def test_p0000_bigmesh_unknown_marker_refusal_is_unchanged(tmp_path, monkeypatch):
    padded(tmp_path, 1)
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text() + '\n[groups]\nwing = ["nowhere"]\n')
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 64 * 1024)
    with pytest.raises(PlanError) as streamed:
        plan(tmp_path)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1 << 40)
    with pytest.raises(PlanError) as held:
        plan(tmp_path)
    assert "unknown markers nowhere" in str(streamed.value)
    assert "mesh inputs/meshes/tiny.su2 has markers: farfield, wall" in str(streamed.value)
    assert str(streamed.value) == str(held.value)


@pytest.mark.parametrize("block", [1, 7, 64, 1 << 20])
@pytest.mark.parametrize(
    "text",
    [
        b"NDIME= 2\nMARKER_TAG= a\nMARKER_TAG = b % comment\n  MARKER_TAG=c\r\nx MARKER_TAG= d\n",
        b"MARKER_TAG= last-without-newline",
        b"no markers here\n",
        b"MARKER_TAG= caf\xe9\nMARKER_TAG=\xff\n",
    ],
)
def test_p0000_bigmesh_streamed_markers_match_the_whole_text_scan(text, block):
    scanner = _Su2Markers()
    for start in range(0, len(text), block):
        scanner.feed(text[start : start + block])
    assert scanner.result() == marker_names(text, "x.su2")


def test_p0000_bigmesh_a_mesh_changed_during_planning_is_refused(tmp_path, monkeypatch):
    mesh, data = padded(tmp_path, 1)
    inner = api().validate_row
    calls = []

    @functools.wraps(inner)
    def validate(row, inputs):
        calls.append(row.pol)
        if len(calls) == 2:
            mesh.write_bytes(data + b"% edited while planning\n")
        return inner(row, inputs)

    for threshold in (64 * 1024, 1 << 40):
        monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", threshold)
        mesh.write_bytes(data)
        calls.clear()
        with pytest.raises(PlanError, match=re.escape(MESH) + ".*changed while planning"):
            plan_matrix(tmp_path, "study", validators={"su2": validate})


def test_p0000_bigmesh_loading_a_point_streams_the_large_mesh_once(tmp_path, monkeypatch):
    from cfdbender.core.adapter import load_point

    mesh, data = padded(tmp_path, 16)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024 * 1024)
    point = plan(tmp_path).points[0]
    counting = _Counting(monkeypatch, mesh)
    tracemalloc.start()
    try:
        load_point(tmp_path, point, solver="su2")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert counting.opens == 1
    assert counting.hashed_meshes(len(data)) == 1
    assert peak < len(data) // 4


def test_streamed_markers_after_long_line():
    data = b"x" * ((1 << 20) + 1) + b"\nMARKER_TAG= wall\n"
    scanner = _Su2Markers()
    scanner.feed(data[: (1 << 20) + 1])
    scanner.feed(data[(1 << 20) + 1 :])
    assert scanner.result() == marker_names(data, "x.su2")

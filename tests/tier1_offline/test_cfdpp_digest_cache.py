"""CFD++ grids follow the planned-input rule: never held whole, read once when first seen.

A threshold of 64 bytes stands in for the 8 MiB one, so the grid files of the toy workspace are
"large". Every open of a grid file (the tracked reader, ``Path.open`` for reading and
``Path.read_bytes``) is counted. The grid facts (dimension, counts, selectors) are scanned in the
pass that digests each file and cached with its digest, like the SU2 marker scan.
"""

from __future__ import annotations

import hashlib
import struct
import time
import tracemalloc
import warnings
from pathlib import Path

import pytest
from cfdpp_support import builder
from cfdpp_workspace import GRID, matrix, use_toy, workspace

import cfdbender.solvers.cfdpp as cfdpp
from cfdbender.core import plan as plan_module
from cfdbender.core.adapter import load_point
from cfdbender.core.executables import bind_executable
from cfdbender.core.plan import DigestCache, InputFiles, InputSession, workspace_path
from cfdbender.core.registry import RunRegistry
from cfdbender.core.run import execute_plan
from cfdbender.matrix.cli import main
from cfdbender.solvers.cfdpp import _mesh
from cfdbender.workspace.planning import plan_workspace

ONE = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:10", "-")
TWO = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:10", "-", ncpus=2)
MESH = f"inputs/mesh/{GRID}"
#: The grid files that are planned inputs and large under the test threshold.
LARGE = {
    "nodesin.bin",
    "cellsin.bin",
    "exbcsin.bin",
    "mcpusin.bin.2",
    "cgrpsin.bin.1",
    "dfcelin.bin",
}


@pytest.fixture(autouse=True)
def small_threshold_and_settled_clock(monkeypatch):
    use_toy(monkeypatch)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 64)
    monkeypatch.setattr(plan_module, "_BLOCK", 32)
    monkeypatch.setattr(
        plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9, raising=False
    )


@pytest.fixture
def opens(monkeypatch):
    """Count every read of a grid file: tracked opens, ``Path.open`` for reading, read_bytes."""
    seen: list[str] = []
    real_open = plan_module._open_input
    real_path_open = Path.open
    real_read = Path.read_bytes

    def counted_open(path):
        if Path(path).name in LARGE:
            seen.append("open")
        return real_open(path)

    def counted_path_open(self, mode="r", *args, **kwargs):
        if self.name in LARGE and "r" in mode and "x" not in mode:
            seen.append("path.open")
        return real_path_open(self, mode, *args, **kwargs)

    def counted_read(self):
        if self.name in LARGE:
            seen.append("read_bytes")
        return real_read(self)

    monkeypatch.setattr(plan_module, "_open_input", counted_open)
    monkeypatch.setattr(Path, "open", counted_path_open)
    monkeypatch.setattr(Path, "read_bytes", counted_read)
    return seen


def add_large_files(root: Path) -> None:
    """Add the partition, cell-group and one more optional file, each above the threshold."""
    folder = root / MESH
    (folder / "mcpusin.bin.2").write_bytes(b"\x00\x00\x00\x01" * 40)
    (folder / "cgrpsin.bin.1").write_bytes(struct.pack(">1i", 1) * 40)
    (folder / "dfcelin.bin").write_bytes(b"dfcel" * 20)


def build(root: Path, *, table: str = ONE, extra: bool = False) -> Path:
    made = workspace(root, table=table).resolve()
    if extra:
        add_large_files(made)
    return made


def plan(root: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return plan_workspace(root, "study")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def per_name(seen: list[str]) -> list[str]:
    return sorted(seen)


# ---- provenance is identical -----------------------------------------------------------------


def provenance(root: Path) -> dict[str, object]:
    """The planned digests, in order, and the bytes of what a staged point holds."""
    planned = plan(root)
    point = planned.points[0]
    context = bind_executable(load_point(root, point, solver="cfdpp"), cfdpp)
    cfdpp.stage(context)
    folder = workspace_path(context.root, context.point.directory)
    return {
        "keys": list(point.inputs_sha256.items()),
        "inp": sha(folder / "mcfd.inp"),
        "files": sorted(path.name for path in folder.iterdir()),
    }


#: Recorded from the adapter before the grid was read by streaming: the planned digests in
#: order, the SHA-256 of the generated mcfd.inp and the files of a staged point.
GOLDEN = {
    "a": {
        "keys": [
            ("init.toml", "c8ff5905c334f84b738495861e808dc81d2847065fe587e400a7d02047c46f8a"),
            (
                "inputs/matrices/study.fs",
                "1d5628150c249c5848ca42364ea024b931bf6d1a75b270ef2d57e153f90cd799",
            ),
            (
                "inputs/mesh/naca.cfdpp/cellsin.bin",
                "81730ec59889f0a3d10e68eaa7c0d7e0820e00154159e460c72ed3d6a34c96ab",
            ),
            (
                "inputs/mesh/naca.cfdpp/exbcsin.bin",
                "214485d306a90584438560d6348a1fdec4ff73c068feb91fceb13553c6b1910f",
            ),
            (
                "inputs/mesh/naca.cfdpp/mcfd.bc",
                "903f2b7a03884b3ff758623308de728df382c8be6b5c6588c18b1c40787a0367",
            ),
            (
                "inputs/mesh/naca.cfdpp/nodesin.bin",
                "ae02ee58503230e4f3d8786f6829341486e59129b94d956fd22ea3292abd4f08",
            ),
            (
                "inputs/pproc/p001.toml",
                "770ae56f90a033dd018b27e6f48ae3bec8a7492fbece686af693c296dac02a18",
            ),
            (
                "inputs/references/r001.toml",
                "cbbd8b5b73f10b43335c92845c054d70882b93b1b1945d3fae10425bfada5700",
            ),
            (
                "inputs/setups/s001.toml",
                "8e9faf10eb8aa4abc126cffea9638189f5c15cb0af7071bd83a1b274871011f8",
            ),
        ],
        "inp": "366b344eb69857105feda383867f10024a77de9f47031734938ea0b06d08a5b7",
        "files": [
            "cellsin.bin",
            "cfdpp_job.sh",
            "exbcsin.bin",
            "mcfd.bc",
            "mcfd.inp",
            "nodesin.bin",
        ],
    },
    "b": {
        "keys": [
            ("init.toml", "c8ff5905c334f84b738495861e808dc81d2847065fe587e400a7d02047c46f8a"),
            (
                "inputs/matrices/study.fs",
                "4e40c83cbef9e6b91837beb9a19b9e9c71538f10c020ea61db311e10c872c297",
            ),
            (
                "inputs/mesh/naca.cfdpp/cellsin.bin",
                "81730ec59889f0a3d10e68eaa7c0d7e0820e00154159e460c72ed3d6a34c96ab",
            ),
            (
                "inputs/mesh/naca.cfdpp/cgrpsin.bin.1",
                "dee643676347a1302103c5a2b5af163dd092a852a93ff32face96ffb882875a9",
            ),
            (
                "inputs/mesh/naca.cfdpp/dfcelin.bin",
                "9abcceabb329e9f81de4686ecf5c43857918a359a7374eefb703b74d23e088d8",
            ),
            (
                "inputs/mesh/naca.cfdpp/exbcsin.bin",
                "214485d306a90584438560d6348a1fdec4ff73c068feb91fceb13553c6b1910f",
            ),
            (
                "inputs/mesh/naca.cfdpp/mcfd.bc",
                "903f2b7a03884b3ff758623308de728df382c8be6b5c6588c18b1c40787a0367",
            ),
            (
                "inputs/mesh/naca.cfdpp/mcpusin.bin.2",
                "dee643676347a1302103c5a2b5af163dd092a852a93ff32face96ffb882875a9",
            ),
            (
                "inputs/mesh/naca.cfdpp/nodesin.bin",
                "ae02ee58503230e4f3d8786f6829341486e59129b94d956fd22ea3292abd4f08",
            ),
            (
                "inputs/pproc/p001.toml",
                "770ae56f90a033dd018b27e6f48ae3bec8a7492fbece686af693c296dac02a18",
            ),
            (
                "inputs/references/r001.toml",
                "cbbd8b5b73f10b43335c92845c054d70882b93b1b1945d3fae10425bfada5700",
            ),
            (
                "inputs/setups/s001.toml",
                "8e9faf10eb8aa4abc126cffea9638189f5c15cb0af7071bd83a1b274871011f8",
            ),
        ],
        "inp": "366b344eb69857105feda383867f10024a77de9f47031734938ea0b06d08a5b7",
        "files": [
            "cellsin.bin",
            "cfdpp_job.sh",
            "cgrpsin.bin.1",
            "dfcelin.bin",
            "exbcsin.bin",
            "mcfd.bc",
            "mcfd.inp",
            "mcpusin.bin.2",
            "nodesin.bin",
        ],
    },
}


@pytest.mark.parametrize(("name", "table", "extra"), [("a", ONE, False), ("b", TWO, True)])
def test_provenance_and_the_generated_input_are_unchanged(tmp_path, name, table, extra):
    root = build(tmp_path / "ws", table=table, extra=extra)
    found = provenance(root)
    assert found["files"] == GOLDEN[name]["files"]
    assert found["inp"] == GOLDEN[name]["inp"]
    assert found["keys"] == GOLDEN[name]["keys"]


# ---- once per command, never again -----------------------------------------------------------


def test_plan_opens_each_large_grid_file_once_cold_and_never_warm(tmp_path, opens):
    root = build(tmp_path / "ws", table=TWO, extra=True)
    plan(root)
    assert per_name(opens) == ["open"] * len(LARGE)  # one pass per file, digest and scan
    opens.clear()
    plan(root)
    assert opens == []


def test_a_plan_of_many_rows_reads_the_grid_once(tmp_path, opens):
    table = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:sweep", "0,2,4,6,8")
    root = build(tmp_path / "ws", table=table)
    plan(root)
    assert per_name(opens) == ["open"] * 3


def test_a_run_and_a_collect_do_not_open_a_grid_file_after_the_plan(tmp_path, opens):
    root = build(tmp_path / "ws", table=matrix(sweep="0,10"))
    planned = plan(root)
    opens.clear()
    records = execute_plan(planned, cfdpp)
    assert [record.classification for record in records] == ["converged", "converged"]
    assert opens == []
    assert all(r.status == "COMPLETED" for r in RunRegistry(root).read())
    opens.clear()
    assert main(["collect", "study", "--workspace", str(root)]) == 0
    assert opens == []


def test_a_cold_run_reads_a_grid_file_a_bounded_number_of_times(tmp_path, opens):
    root = build(tmp_path / "ws", table=matrix(sweep="0,2,4,6"))
    planned = plan(root)
    (root / ".input-digests.cache").unlink()
    opens.clear()
    execute_plan(planned, cfdpp)
    assert len(opens) <= 2 * 3 and "read_bytes" not in opens and "path.open" not in opens


def test_staging_links_the_grid_and_reads_nothing_warm(tmp_path, opens):
    root = build(tmp_path / "ws", table=TWO, extra=True)
    planned = plan(root)
    context = bind_executable(
        load_point(root, planned.points[0], solver="cfdpp", session=InputSession(root)), cfdpp
    )
    opens.clear()
    cfdpp.stage(context)
    assert opens == []
    folder = workspace_path(root, context.point.directory)
    assert (folder / "nodesin.bin").samefile(root / MESH / "nodesin.bin")


def test_the_flag_reads_every_large_file_again(tmp_path, opens):
    root = build(tmp_path / "ws", table=TWO, extra=True)
    assert main(["plan", "study", "--workspace", str(root)]) == 0
    opens.clear()
    assert main(["plan", "study", "--workspace", str(root)]) == 0
    assert opens == []
    assert main(["plan", "study", "--workspace", str(root), "--rehash-inputs"]) == 0
    assert per_name(opens) == ["open"] * len(LARGE)


# ---- a changed grid is refused with the standard texts ----------------------------------------


def test_a_changed_grid_is_refused_when_a_point_is_loaded(tmp_path):
    root = build(tmp_path / "ws")
    planned = plan(root)
    path = root / MESH / "nodesin.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    stat = path.stat()
    path.write_bytes(bytes(data))
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5 * 10**9))
    for session in (None, InputSession(root)):
        with pytest.raises(
            ValueError, match=f"input changed after planning: {MESH}/nodesin.bin; plan again"
        ):
            load_point(root, planned.points[0], solver="cfdpp", session=session)


def test_a_grid_changed_after_the_point_was_loaded_is_refused_at_staging(tmp_path):
    root = build(tmp_path / "ws")
    planned = plan(root)
    context = bind_executable(load_point(root, planned.points[0], solver="cfdpp"), cfdpp)
    path = root / MESH / "nodesin.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="^mesh changed after planning; plan again$"):
        cfdpp.stage(context)
    folder = workspace_path(root, context.point.directory)
    assert not (folder / "nodesin.bin").exists() and not (folder / "nodesin.bin").is_symlink()


# ---- the facts of a grid are scanned, block by block ------------------------------------------


def facts_of(folder: Path):
    return _mesh.read_grid(folder).facts


@pytest.mark.parametrize("kind", ["2d", "3d"])
@pytest.mark.parametrize("block", [1, 3, 7, 12, 13, 64])
def test_the_scan_gives_the_same_facts_whatever_the_block_size(tmp_path, monkeypatch, kind, block):
    folder = tmp_path / "grid"
    getattr(builder(), f"write_grid{kind}")(folder)
    expected = facts_of(folder)
    monkeypatch.setattr(_mesh, "_STREAM_BLOCK", block)
    assert facts_of(folder) == expected
    root = tmp_path / "ws"
    target = root / "inputs" / "mesh" / "g"
    getattr(builder(), f"write_grid{kind}")(target)
    monkeypatch.setattr(plan_module, "_BLOCK", block)
    got = _mesh.read_grid(target, files=InputFiles(root.resolve()), mesh="inputs/mesh/g")
    assert got.facts == expected


@pytest.mark.parametrize("case", builder().GRID_BAD_CASES)
@pytest.mark.parametrize("block", [1, 5, 1 << 20])
def test_a_defective_grid_is_refused_with_the_same_text_whatever_the_block_size(
    tmp_path, monkeypatch, case, block
):
    folder = tmp_path / "g"
    builder().write_grid_bad(folder, case)
    monkeypatch.setattr(_mesh, "_STREAM_BLOCK", 1 << 20)
    try:
        _mesh.read_grid(folder, require_names=True)
        message = None
    except _mesh.GridError as error:
        message = str(error)
    monkeypatch.setattr(_mesh, "_STREAM_BLOCK", block)
    try:
        _mesh.read_grid(folder, require_names=True)
        again = None
    except _mesh.GridError as error:
        again = str(error)
    assert again == message


def face_file(faces: list[tuple[int, tuple[int, ...]]]) -> bytes:
    return builder()._pack_faces(faces)


@pytest.mark.parametrize(
    "data",
    [
        face_file([(1, (0, 1)), (2, (1, 2))])[:-4],
        face_file([(1, (0, 1)), (1, (0, 1)), (2, ())]),
    ],
)
def test_a_defective_face_file_is_refused_whatever_the_block_size(tmp_path, monkeypatch, data):
    files = builder().grid2d_files()
    files["exbcsin.bin"] = data
    folder = tmp_path / "g"
    builder().write_grid(folder, files, ["airfoil", "farfield"])
    messages = set()
    for block in (1, 4, 11, 1 << 20):
        monkeypatch.setattr(_mesh, "_STREAM_BLOCK", block)
        with pytest.raises(_mesh.GridError) as caught:
            _mesh.read_grid(folder)
        messages.add(str(caught.value))
    assert len(messages) == 1


# ---- memory is bounded ------------------------------------------------------------------------

FACES = 150_000


def big_grid(root: Path) -> int:
    """Replace the grid by one whose three files are megabytes; return the largest size."""
    files = builder().grid2d_files()
    folder = root / MESH
    nodes = files["nodesin.bin"] + b"\x00" * (6 << 20)
    cells = files["cellsin.bin"] + b"\x00" * (6 << 20)
    faces = builder()._pack_faces([(1 + n % 2, (n % 9, (n + 1) % 9)) for n in range(FACES)])
    for name, data in (("nodesin.bin", nodes), ("cellsin.bin", cells), ("exbcsin.bin", faces)):
        (folder / name).write_bytes(data)
    return max(len(nodes), len(cells), len(faces))


def peak_of(action) -> int:
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        action()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_planning_and_running_a_large_grid_hold_a_bounded_amount(tmp_path, monkeypatch):
    root = build(tmp_path / "ws")
    size = big_grid(root)
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1 << 20)
    monkeypatch.setattr(plan_module, "_BLOCK", 1 << 16)
    assert size > 6 << 20
    planned = []
    assert peak_of(lambda: planned.append(plan(root))) < 1 << 20
    # A warm command reads the digest cache file, which allocates its own 8 MiB buffer once;
    # holding any grid file whole (6 MiB or more) would show on top of that.
    assert peak_of(lambda: execute_plan(planned[0], cfdpp)) < DigestCache.MAX_BYTES + (1 << 20)


@pytest.mark.parametrize("scan", [[], ["0", ""]])
def test_malformed_cached_grid_scan_is_ignored(tmp_path, scan):
    import json

    root = build(tmp_path / "ws")
    plan(root)
    cache = root / ".input-digests.cache"
    document = json.loads(cache.read_text())
    document["entries"][f"{MESH}/nodesin.bin"]["scans"]["cfdpp-grid-nodes-1"] = scan
    cache.write_text(json.dumps(document))
    assert plan(root).points

"""One read of a large input per command, and a persistent digest cache between commands.

A threshold of one KiB stands in for the 8 MiB one, so the large file is a few KiB. Every open of
the mesh through the tracked reader, and every whole-file ``read_bytes`` of it, is counted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pytest
from test_matrix_columns_all_solvers import HEADER

from cfdbender.core import plan as plan_module
from cfdbender.core.adapter import load_point
from cfdbender.core.plan import InputFiles
from cfdbender.core.registry import RunRegistry
from cfdbender.core.workspace import init_workspace
from cfdbender.matrix.cli import main
from cfdbender.matrix.receipt import write_receipt
from cfdbender.solvers import available_solvers
from cfdbender.solvers._host._groups import tracked_markers
from cfdbender.workspace.planning import plan_workspace

FAKE = Path(__file__).resolve().parents[1] / "fixtures" / "su2" / "fake_su2.py"
MESH = "inputs/geometries/large.su2"
MESH_NAMES = {"large.su2", "mesh.su2"}  # the mesh and its link in a point folder
MESH_TEXT = (
    b"NDIME= 3\nNMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 0\n" + (b"%" + b"x" * 62 + b"\n") * 80
)


def build(root: Path, *, sweep: str = "0,1,2", groups: bool = True) -> Path:
    init_workspace(root, solver="su2", available_solvers=available_solvers())
    row = (
        f" 7001 | 1 | 1 | Model | clean | row | MACH:0.5, ALTFT:10000, ALPHA:sweep, BETA:0 | "
        f"{sweep} | large.su2 | 001 | 001 | 001 | - | - | 4 | 2h | - | - | -\n"
    )
    files = {
        "inputs/setups/s001.toml": 'iterations = 40\nSOLVER = "EULER"\n'
        + ('[groups]\nwing = ["wall"]\n' if groups else "")
        + f'[solver]\nexecutable = "{FAKE.as_posix()}"\n',
        "inputs/references/r001.toml": "area = 1.0\nlength = 1.0\nmoment_origin = [0, 0, 0]\n",
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": f"{HEADER}\n{'-' * 60}\n{row}",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    (root / MESH).parent.mkdir(parents=True, exist_ok=True)
    (root / MESH).write_bytes(MESH_TEXT)
    return root


@pytest.fixture(autouse=True)
def small_threshold_and_settled_clock(monkeypatch):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024)
    monkeypatch.setattr(plan_module, "_BLOCK", 512)
    monkeypatch.setattr(
        plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9, raising=False
    )


@pytest.fixture
def opens(monkeypatch):
    """Count every read of the mesh: tracked opens and whole-file read_bytes."""
    seen: list[str] = []
    real_open = plan_module._open_input
    real_read = Path.read_bytes

    def counted_open(path):
        if Path(path).name in MESH_NAMES:
            seen.append("open")
        return real_open(path)

    def counted_read(self):
        if self.name in MESH_NAMES:
            seen.append("read_bytes")
        return real_read(self)

    monkeypatch.setattr(plan_module, "_open_input", counted_open)
    monkeypatch.setattr(Path, "read_bytes", counted_read)
    return seen


@pytest.fixture
def case(tmp_path):
    root = build(tmp_path / "ws").resolve()
    plan = plan_workspace(root, "study")
    return root, plan


def cache_file(root: Path) -> Path:
    return root / ".input-digests.cache"


def cache_entries(root: Path) -> dict:
    return json.loads(cache_file(root).read_text(encoding="utf-8"))["entries"]


def digest_of(root: Path, name: str = MESH) -> str:
    with (root / name).open("rb") as stream:  # not Path.read_bytes, which is counted
        return hashlib.sha256(stream.read()).hexdigest()


def tweak_entry(root: Path, **changes):
    document = json.loads(cache_file(root).read_text(encoding="utf-8"))
    document["entries"][MESH].update(changes)
    cache_file(root).write_text(json.dumps(document), encoding="utf-8")


# ---- once per command ---------------------------------------------------------------------


def test_one_session_reads_the_mesh_once_for_every_point(case, opens):
    root, plan = case
    assert len(plan.points) == 3
    session = plan_module.InputSession(root)
    with plan_module.rehash_inputs():
        for point in plan.points:
            load_point(root, point, solver="su2", session=session).verify()
    assert opens.count("open") == 1 and "read_bytes" not in opens


def test_the_default_api_call_is_unchanged_and_reads_per_call(case, opens):
    root, plan = case
    with plan_module.rehash_inputs():
        for point in plan.points:
            context = load_point(root, point, solver="su2")
            assert context.session is None
            context.verify()
    assert opens.count("open") == 2 * len(plan.points)


def test_a_run_reads_the_mesh_a_constant_number_of_times_whatever_the_points(tmp_path, opens):
    counts = []
    for index, sweep in enumerate(("0,1", "0,1,2,3,4,5")):
        root = build(tmp_path / f"w{index}", sweep=sweep).resolve()
        write_receipt(plan_workspace(root, "study"))
        opens.clear()
        code = main(["run", "study", "--workspace", str(root), "--local", "--rehash-inputs"])
        assert code == 0
        counts.append(len(opens))
        assert "read_bytes" not in opens
    assert counts[0] == counts[1] and counts[0] <= 4, counts


def test_a_changed_mesh_between_plan_and_run_is_refused_with_the_same_text(case):
    root, plan = case
    (root / MESH).write_bytes(MESH_TEXT + b"% more\n")
    session = plan_module.InputSession(root)
    with pytest.raises(ValueError, match=f"input changed after planning: {MESH}; plan again"):
        load_point(root, plan.points[0], solver="su2", session=session)


def test_a_mesh_changed_during_the_command_is_refused_for_the_next_point(case):
    root, plan = case
    session = plan_module.InputSession(root)
    load_point(root, plan.points[0], solver="su2", session=session)
    (root / MESH).write_bytes(MESH_TEXT.replace(b"wall", b"wale"))
    with pytest.raises(ValueError, match=f"input changed after planning: {MESH}; plan again"):
        load_point(root, plan.points[1], solver="su2", session=session)


def test_a_mesh_rewritten_with_another_timestamp_is_refused_for_the_next_point(case):
    root, plan = case
    session = plan_module.InputSession(root)
    context = load_point(root, plan.points[0], solver="su2", session=session)
    stat = (root / MESH).stat()
    (root / MESH).write_bytes(MESH_TEXT.replace(b"wall", b"wale"))  # same size
    os.utime(root / MESH, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5 * 10**9))
    with pytest.raises(ValueError, match=f"input changed after planning: {MESH}; plan again"):
        context.verify()


def test_a_missing_input_is_refused_as_before(case):
    root, plan = case
    (root / MESH).unlink()
    for session in (None, plan_module.InputSession(root)):
        with pytest.raises(ValueError, match=rf"{MESH}: cannot read"):
            load_point(root, plan.points[0], solver="su2", session=session)


def test_a_changed_small_input_is_refused_inside_a_session(case):
    root, plan = case
    session = plan_module.InputSession(root)
    load_point(root, plan.points[0], solver="su2", session=session)
    reference = root / "inputs/references/r001.toml"
    stat = reference.stat()
    reference.write_bytes(reference.read_bytes().replace(b"1.0", b"2.0"))  # same size
    os.utime(reference, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # and the same timestamps
    with pytest.raises(ValueError, match="input changed after planning: inputs/references/r001"):
        load_point(root, plan.points[1], solver="su2", session=session)


# ---- between commands ---------------------------------------------------------------------


def test_a_second_command_does_not_open_the_large_file(case, opens):
    root, plan = case
    cache_file(root).unlink()  # planning cached the mesh; this test starts from a first sight
    InputFiles(root).track(MESH)
    assert opens.count("open") >= 1
    opens.clear()
    files = InputFiles(root)
    files.track(MESH)
    assert files.digests()[MESH] == digest_of(root) and opens == []
    load_point(root, plan.points[0], solver="su2")
    assert opens == []


@pytest.mark.parametrize("what", ["size", "mtime", "ctime"])
def test_a_changed_size_mtime_or_ctime_forces_a_rehash_and_the_refusal(case, opens, what):
    root, plan = case
    InputFiles(root).track(MESH)
    path = root / MESH
    stat = path.stat()
    # Each case moves one identity value only: the others are put back to the cached ones.
    if what == "size":
        path.write_bytes(MESH_TEXT.replace(b"wall", b"wal"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        tweak_entry(root, ctime_ns=path.stat().st_ctime_ns)
    elif what == "mtime":
        path.write_bytes(MESH_TEXT.replace(b"wall", b"wale"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5 * 10**9))
        tweak_entry(root, ctime_ns=path.stat().st_ctime_ns)
    else:
        path.write_bytes(MESH_TEXT.replace(b"wall", b"wale"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        tweak_entry(root, ctime_ns=stat.st_ctime_ns + 1)
    opens.clear()
    with pytest.raises(ValueError, match=f"input changed after planning: {MESH}; plan again"):
        load_point(root, plan.points[0], solver="su2")
    assert "open" in opens


@pytest.mark.parametrize(
    "text",
    [
        "not json {",
        "[]",
        json.dumps({"format": 99, "entries": {}}),
        json.dumps({"format": 1, "entries": "x"}),
        json.dumps({"format": 1, "entries": {MESH: {"size": "big"}}}),
        json.dumps({"format": 1, "entries": {MESH: {"size": 1, "mtime_ns": 1, "ctime_ns": 1}}}),
    ],
)
def test_a_corrupt_or_wrong_version_cache_is_an_empty_cache(case, opens, text):
    root, plan = case
    cache_file(root).write_text(text, encoding="utf-8")
    opens.clear()
    context = load_point(root, plan.points[0], solver="su2")
    context.verify()
    assert "open" in opens
    assert cache_entries(root)[MESH]["sha256"] == digest_of(root)  # and it is repaired


def test_an_entry_of_another_format_version_is_not_served(case):
    root, plan = case
    InputFiles(root).track(MESH)
    document = json.loads(cache_file(root).read_text(encoding="utf-8"))
    document["format"] = 99
    document["entries"][MESH]["sha256"] = "0" * 64  # served only if the version were ignored
    cache_file(root).write_text(json.dumps(document), encoding="utf-8")
    files = InputFiles(root)
    files.track(MESH)
    assert files.digests()[MESH] == digest_of(root)


def test_an_entry_of_another_name_is_not_used(case, opens):
    root, plan = case
    InputFiles(root).track(MESH)
    document = json.loads(cache_file(root).read_text(encoding="utf-8"))
    document["entries"] = {"inputs/geometries/other.su2": document["entries"][MESH]}
    cache_file(root).write_text(json.dumps(document), encoding="utf-8")
    opens.clear()
    InputFiles(root).track(MESH)
    assert "open" in opens


def test_a_just_written_file_is_not_cached(case, opens, monkeypatch):
    root, plan = case
    cache_file(root).unlink(missing_ok=True)
    newest = max((root / MESH).stat().st_mtime_ns, (root / MESH).stat().st_ctime_ns)
    monkeypatch.setattr(plan_module, "_clock_ns", lambda: newest + 10**9)  # 1 s: too young
    InputFiles(root).track(MESH)
    assert not cache_file(root).exists() or MESH not in cache_entries(root)
    opens.clear()
    InputFiles(root).track(MESH)
    assert "open" in opens
    monkeypatch.setattr(plan_module, "_clock_ns", lambda: newest + 3 * 10**9)  # settled
    InputFiles(root).track(MESH)
    assert MESH in cache_entries(root)


def test_the_flag_forces_the_reread_and_the_limit_is_the_documented_one(case, opens):
    root, plan = case
    InputFiles(root).track(MESH)
    path = root / MESH
    stat = path.stat()
    old = digest_of(root)
    path.write_bytes(MESH_TEXT.replace(b"wall", b"wale"))  # same size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    tweak_entry(root, ctime_ns=path.stat().st_ctime_ns)  # as on a file system that kept it
    opens.clear()
    files = InputFiles(root)
    files.track(MESH)
    assert files.digests()[MESH] == old and opens == []  # not detected without the flag
    with plan_module.rehash_inputs():
        files = InputFiles(root)
        files.track(MESH)
    assert files.digests()[MESH] == digest_of(root) != old and "open" in opens


def test_the_cli_flag_reaches_every_verifying_command(case, opens):
    root, plan = case
    assert main(["plan", "study", "--workspace", str(root)]) == 0
    opens.clear()
    assert main(["plan", "study", "--workspace", str(root)]) == 0
    assert opens == []
    assert main(["plan", "study", "--workspace", str(root), "--rehash-inputs"]) == 0
    assert "open" in opens
    for command in ("run", "collect"):
        with pytest.raises(SystemExit) as info:
            main([command, "--help"])
        assert info.value.code == 0
    from cfdbender.matrix.cli import build_parser

    parser, _ = build_parser()
    for words in (["plan", "m"], ["run", "m"], ["collect"]):
        assert parser.parse_args([*words, "--rehash-inputs"]).rehash_inputs is True
        assert parser.parse_args(words).rehash_inputs is False


def test_markers_served_from_the_cache_equal_the_scan(case, opens):
    root, plan = case
    first = tracked_markers(InputFiles(root), MESH)
    assert first == ("wall",)
    assert cache_entries(root)[MESH]["scans"] == {"su2-markers": ["wall"]}
    opens.clear()
    assert tracked_markers(InputFiles(root), MESH) == first and opens == []


def test_small_files_are_never_cached(case):
    root, plan = case
    assert set(cache_entries(root)) <= {MESH}
    small = root / "inputs/setups/s001.toml"
    assert small.stat().st_size <= 1024
    assert "s001" not in cache_file(root).read_text(encoding="utf-8")


def test_provenance_is_identical_cold_warm_and_rehashed(tmp_path):
    root = build(tmp_path / "ws").resolve()
    cold = plan_workspace(root, "study")
    warm = plan_workspace(root, "study")
    with plan_module.rehash_inputs():
        forced = plan_workspace(root, "study")
    shutil.copy(cache_file(root), tmp_path / "kept")
    expected = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in cold.points[0].inputs_sha256
    }
    for plan in (cold, warm, forced):
        for point in plan.points:
            assert list(point.inputs_sha256.items()) == list(cold.points[0].inputs_sha256.items())
            assert dict(point.inputs_sha256) == expected
    assert cold == warm == forced


def test_a_read_only_cache_location_is_ignored(case, monkeypatch):
    root, plan = case
    cache_file(root).unlink(missing_ok=True)

    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(plan_module.os, "replace", refuse)
    files = InputFiles(root)
    files.track(MESH)
    assert files.digests()[MESH] == digest_of(root)
    assert not list(root.glob(".input-digests-*.tmp"))


def test_the_registry_hashes_a_large_input_streamed_and_once_per_registry(case, opens):
    root, plan = case
    registry = RunRegistry(root)
    with plan_module.rehash_inputs():
        registry.record(plan)
        for point in plan.points:
            registry.transition(point.point_id, "RUNNING")
    assert opens.count("open") == 1 and "read_bytes" not in opens


# ---- the link in the point folder and the recorded output digests --------------------------


@pytest.fixture
def file_digests(monkeypatch):
    """Record every hashlib.file_digest call: the link check once streamed the mesh with it."""
    import hashlib as hashlib_module

    calls: list[object] = []
    real = hashlib_module.file_digest

    def counted(stream, *args, **kwargs):
        calls.append(getattr(stream, "name", None))
        return real(stream, *args, **kwargs)

    monkeypatch.setattr(hashlib_module, "file_digest", counted)
    return calls


def test_a_warm_run_and_collect_never_read_the_mesh(tmp_path, opens, file_digests):
    root = build(tmp_path / "ws", groups=False).resolve()
    write_receipt(plan_workspace(root, "study"))
    opens.clear()
    assert main(["run", "study", "--workspace", str(root), "--local"]) == 0
    assert main(["collect", "study", "--workspace", str(root)]) == 0
    assert opens == [] and file_digests == []
    records = RunRegistry(root).read()
    assert {r.status for r in records} == {"COMPLETED"}
    staged = [n for r in records for n in r.outputs_sha256 if n.endswith("mesh.su2")]
    assert len(staged) == len(records)
    for r in records:
        for name, digest in r.outputs_sha256.items():
            assert digest == digest_of(root, name)  # recorded exactly, link included


def test_a_cold_run_does_not_stream_the_mesh_through_each_link(tmp_path, opens, file_digests):
    root = build(tmp_path / "ws").resolve()
    write_receipt(plan_workspace(root, "study"))
    opens.clear()
    code = main(["run", "study", "--workspace", str(root), "--local", "--rehash-inputs"])
    assert code == 0 and file_digests == [] and "read_bytes" not in opens


def test_an_output_that_is_the_mesh_takes_its_digest_and_others_are_streamed(case, opens):
    root, plan = case
    link = root / "sims_link.su2"
    os.link(root / MESH, link)
    other = root / "other.bin"
    other.write_bytes(MESH_TEXT.replace(b"wall", b"wale"))
    files = InputFiles(root)
    opens.clear()
    names = list(plan.points[0].inputs_sha256)
    assert files.digest_output(link, names) == digest_of(root)
    assert files.digest_output(other, names) == digest_of(root, "other.bin") != digest_of(root)
    assert "read_bytes" not in opens


def test_a_changed_mesh_behind_the_link_is_hashed_and_refused(tmp_path):
    from cfdbender.solvers._host._mesh import link_mesh

    root = build(tmp_path / "ws").resolve()
    plan = plan_workspace(root, "study")
    planned = plan.points[0].inputs_sha256[MESH]
    path = root / MESH
    stat = path.stat()
    path.write_bytes(MESH_TEXT.replace(b"wall", b"wale"))  # same size, other bytes
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 7 * 10**9))
    folder = root / "sims" / "p"
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match="mesh changed after planning; plan again"):
        link_mesh(path, folder / "large.su2", planned, inputs=InputFiles(root), name=MESH)
    assert not (folder / "large.su2").exists()

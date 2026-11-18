"""OpenFOAM and PyFR read a large planned mesh once when first seen and never again after.

A threshold of one KiB stands in for the 8 MiB one. Every open of the mesh by any code path is
counted through the audit hook, and every whole-file ``read_bytes`` of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import pytest
from test_p0000_item3f_plan import profile

from cfdbender.core import plan as plan_module
from cfdbender.core.adapter import load_point
from cfdbender.core.workspace import init_workspace
from cfdbender.matrix.cli import main
from cfdbender.solvers import available_solvers
from cfdbender.workspace.planning import plan_workspace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FOAM_FAKE = FIXTURES / "openfoam" / "fake_openfoam.py"
MESH = "inputs/meshes/large.msh"
MESH_NAMES = {"large.msh", "mesh.msh"}  # the mesh and its link in a point folder
FOAM_STANDIN = (
    "import pathlib, runpy, shutil\n"
    "try:\n"
    f"    runpy.run_path({FOAM_FAKE.as_posix()!r}, run_name='__main__')\n"
    "except SystemExit:\n"
    "    pass\n"
    "source = pathlib.Path('postProcessing/forceCoeffs/0/coefficient.dat')\n"
    "target = pathlib.Path('postProcessing/forceCoeffs_main/0/coefficient.dat')\n"
    "target.parent.mkdir(parents=True)\n"
    "shutil.copy(source, target)\n"
)
PYFR_STANDIN = (
    "import pathlib,sys\n"
    "if 'import' in sys.argv: pathlib.Path('mesh.pyfrm').write_text('mesh')\n"
    "else: pathlib.Path('forces.csv').write_text('t,px,py,vx,vy\\n'"
    " + '0,0.1,0.3,0,0\\n1,0.1,0.3,0,0\\n2,0.1,0.3,0,0\\n3,0.1,0.3,0,0\\n')\n"
)
NAMES = '$PhysicalNames\n3\n2 1 "wing"\n2 2 "tail"\n2 3 "farfield"\n$EndPhysicalNames\n'


def mesh_text(version: str = "2.2", names: str = NAMES, lines: int = 80) -> bytes:
    head = f"$MeshFormat\n{version} 0 8\n$EndMeshFormat\n{names}$Nodes\n"
    return head.encode("ascii") + (b"1 0.5 0.25 0.125" + b" " * 46 + b"\n") * lines


def build(root: Path, adapter: str, *, sweep: str = "0,1", mesh: bytes | None = None) -> Path:
    init_workspace(root, solver=adapter, available_solvers=available_solvers())
    matrix = (
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        f"1001|1|001|001|001|ALPHA:sweep|{sweep}\n"
    )
    if adapter == "openfoam":
        (root / "standin.py").write_text(FOAM_STANDIN, encoding="ascii")
        setup = (
            f'mesh = "{MESH}"\napplication = "simpleFoam"\nspeed = 10.0\nnu = 1.5e-5\n'
            'turbulence = "SpalartAllmaras"\niterations = 4\n[groups]\nmain = ["wing"]\n'
            f"[solver]\nexecutable = {json.dumps((root / 'standin.py').as_posix())}\n"
        )
        reference = "area = 0.1\nlength = 1.0\nmoment_origin = [0.25, 0.0, 0.0]\n"
    else:
        (root / "standin.py").write_text(PYFR_STANDIN, encoding="ascii")
        setup = (
            f'mesh = "{MESH}"\nmach = 0.2\nreynolds = 10000\norder = 3\ntend = 3\n'
            "window_start = 0\n[solver]\n"
            f"executable = {json.dumps((root / 'standin.py').as_posix())}\n"
        )
        reference = ""
    files = {
        "inputs/setups/s001.toml": setup,
        "inputs/references/r001.toml": reference,
        "inputs/pproc/p001.toml": "history = true\n",
        "inputs/matrices/study.fs": matrix,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("ascii"))
    default = mesh_text("2.2" if adapter == "openfoam" else "4.1")
    (root / MESH).parent.mkdir(parents=True, exist_ok=True)
    (root / MESH).write_bytes(default if mesh is None else mesh)
    return root.resolve()


@pytest.fixture(autouse=True)
def small_threshold_and_settled_clock(monkeypatch):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 1024)
    monkeypatch.setattr(plan_module, "_BLOCK", 512)
    monkeypatch.setattr(
        plan_module, "_clock_ns", lambda: time.time_ns() + 60 * 10**9, raising=False
    )


_COUNTING = {"on": False, "opens": []}


def _hook(event: str, info: tuple) -> None:
    if _COUNTING["on"] and event == "open" and Path(str(info[0])).name in MESH_NAMES:
        _COUNTING["opens"].append("open")


sys.addaudithook(_hook)


@pytest.fixture
def opens(monkeypatch):
    """Count every open of the mesh by any code path, and every whole-file read_bytes."""
    seen: list[str] = _COUNTING["opens"]
    seen.clear()
    _COUNTING["on"] = True
    real_read = Path.read_bytes

    def counted_read(self):
        if self.name in MESH_NAMES:
            seen.append("read_bytes")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    yield seen
    _COUNTING["on"] = False


@pytest.fixture(params=["openfoam", "pyfr"])
def adapter(request):
    return request.param


def cluster(root: Path, adapter: str) -> None:
    """Add a SOLVER_BUILD column and a scheduler profile that maps it."""
    build_name = {"openfoam": "v2412", "pyfr": "3.1.0"}[adapter]
    matrix = root / "inputs/matrices/study.fs"
    header, row = matrix.read_text(encoding="ascii").strip().splitlines()
    matrix.write_text(f"{header}|SOLVER_BUILD\n{row}|{build_name}\n", encoding="ascii")
    fields = {"name": "{job_name}", "working_directory": "{workdir}", "config": "{config_path}"}
    path = profile(root, fields=fields)
    path.write_text(path.read_text().replace('"mock-1" =', f'"{build_name}" ='))


def command(root: Path, *words: str) -> int:
    return main([*words, "study", "--workspace", str(root)])


def only_opens(seen: list[str]) -> int:
    return seen.count("open")


# ---- one read when first seen, none after -------------------------------------------------


def test_a_first_plan_reads_the_mesh_once_and_a_later_plan_not_at_all(tmp_path, opens, adapter):
    root = build(tmp_path / "ws", adapter)
    opens.clear()
    assert command(root, "plan") == 0
    assert only_opens(opens) == 1 and "read_bytes" not in opens
    opens.clear()
    assert command(root, "plan") == 0
    assert opens == []


def test_a_first_local_run_reads_the_mesh_once_and_a_later_command_not_at_all(
    tmp_path, opens, adapter
):
    root = build(tmp_path / "ws", adapter)
    opens.clear()
    assert main(["run", "study", "--workspace", str(root), "--local"]) == 0
    assert only_opens(opens) == 1 and "read_bytes" not in opens
    opens.clear()
    assert command(root, "collect") == 0
    assert opens == []


def test_a_run_after_a_plan_does_not_open_the_mesh(tmp_path, opens, adapter):
    root = build(tmp_path / "ws", adapter)
    assert command(root, "plan") == 0
    opens.clear()
    assert main(["run", "study", "--workspace", str(root), "--local"]) == 0
    assert opens == []
    assert command(root, "collect") == 0
    assert opens == []


def test_a_scheduler_run_reads_the_mesh_once_when_first_seen_and_not_after(
    tmp_path, opens, adapter
):
    root = build(tmp_path / "ws", adapter)
    cluster(root, adapter)
    opens.clear()
    assert command(root, "run") == 0
    assert only_opens(opens) == 1 and "read_bytes" not in opens
    plan_workspace(root, "study")
    assert only_opens(opens) == 1


def test_rehash_inputs_forces_the_read(tmp_path, opens, adapter):
    root = build(tmp_path / "ws", adapter)
    assert command(root, "plan") == 0
    opens.clear()
    assert main(["plan", "study", "--workspace", str(root), "--rehash-inputs"]) == 0
    assert only_opens(opens) == 1 and "read_bytes" not in opens


def test_a_cache_entry_without_the_mesh_scan_is_still_read_and_completed(tmp_path, opens, adapter):
    root = build(tmp_path / "ws", adapter)
    assert command(root, "plan") == 0
    cache = root / plan_module.DigestCache.NAME
    document = json.loads(cache.read_text(encoding="utf-8"))
    assert "gmsh-info" in document["entries"][MESH]["scans"]
    document["entries"][MESH]["scans"] = {}
    cache.write_text(json.dumps(document), encoding="utf-8")
    opens.clear()
    assert command(root, "plan") == 0
    assert only_opens(opens) == 1
    opens.clear()
    assert command(root, "plan") == 0
    assert opens == []


# ---- bounded memory -----------------------------------------------------------------------


@pytest.mark.parametrize("step", ["plan", "run"])
def test_a_large_mesh_is_never_held_whole(tmp_path, monkeypatch, adapter, step):
    monkeypatch.setattr(plan_module, "LARGE_INPUT_BYTES", 256 * 1024)
    monkeypatch.setattr(plan_module, "_BLOCK", 64 * 1024)
    big = mesh_text("2.2" if adapter == "openfoam" else "4.1", lines=400_000)
    root = build(tmp_path / "ws", adapter, mesh=big)
    size = (root / MESH).stat().st_size
    assert size > 24 * 1024 * 1024
    argv = [step, "study", "--workspace", str(root)] + (["--local"] if step == "run" else [])
    tracemalloc.start()
    try:
        assert main(argv) == 0  # a cold command, which reads the mesh once
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < size // 2, (peak, size)


# ---- unchanged refusals, unchanged provenance ---------------------------------------------


def test_a_changed_mesh_is_refused_after_planning_with_the_same_text(tmp_path, adapter):
    root = build(tmp_path / "ws", adapter)
    plan = plan_workspace(root, "study")
    (root / MESH).write_bytes((root / MESH).read_bytes() + b"% more\n")
    session = plan_module.InputSession(root)
    with pytest.raises(ValueError, match=f"input changed after planning: {MESH}; plan again"):
        load_point(root, plan.points[0], solver=adapter, session=session).verify()
    with pytest.raises(ValueError, match=f"input changed after planning: {MESH}; plan again"):
        load_point(root, plan.points[0], solver=adapter)


def test_the_refusal_texts_are_unchanged_for_a_large_mesh(tmp_path, adapter):
    bad = b"$NotAMesh\n" + b"x" * 4000
    root = build(tmp_path / "ws", adapter, mesh=bad)
    expected = (
        "mesh must be gmsh 2.2 ASCII" if adapter == "openfoam" else "mesh must be gmsh format"
    )
    if adapter == "openfoam":
        (root / "inputs/setups/s001.toml").write_text(
            (root / "inputs/setups/s001.toml")
            .read_text()
            .replace('[groups]\nmain = ["wing"]\n', "")
        )
    with pytest.raises(ValueError) as caught:
        plan_workspace(root, "study")
    assert expected in str(caught.value)


def test_an_openfoam_2_1_header_is_refused_and_unknown_markers_name_the_inventory(tmp_path):
    root = build(tmp_path / "ws", "openfoam", mesh=mesh_text("4.1"))
    with pytest.raises(ValueError, match="mesh must be gmsh 2.2 ASCII"):
        plan_workspace(root, "study")
    root = build(tmp_path / "w2", "openfoam")
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace('["wing"]', '["wing", "missing"]'))
    with pytest.raises(ValueError) as caught:
        plan_workspace(root, "study")
    assert str(caught.value).endswith(
        "unknown markers missing; mesh inputs/meshes/large.msh has markers: farfield, tail, wing"
    ) or (
        "unknown markers missing" in str(caught.value)
        and "farfield, tail, wing" in str(caught.value)
    )


def test_the_recorded_digests_are_the_independent_digests_in_the_same_order(tmp_path, adapter):
    root = build(tmp_path / "ws", adapter)
    plan = plan_workspace(root, "study")
    expected = sorted(
        [
            "init.toml",
            MESH,
            "inputs/setups/s001.toml",
            "inputs/references/r001.toml",
            "inputs/pproc/p001.toml",
            "inputs/matrices/study.fs",
        ]
    )
    for point in plan.points:
        recorded = dict(point.inputs_sha256)
        assert list(recorded) == expected
        for name, digest in recorded.items():
            assert digest == hashlib.sha256((root / name).read_bytes()).hexdigest(), name


def test_a_mesh_replaced_between_commands_is_scanned_again(tmp_path, adapter):
    root = build(tmp_path / "ws", adapter)
    assert command(root, "plan") == 0
    if adapter != "openfoam":
        return
    swapped = mesh_text(names=NAMES.replace('"wing"', '"wingx"'))
    (root / MESH).unlink()
    (root / MESH).write_bytes(swapped)
    with pytest.raises(ValueError, match="unknown markers wing;"):
        plan_workspace(root, "study")
    assert os.path.exists(root / MESH)

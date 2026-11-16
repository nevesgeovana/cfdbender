"""(0.57.0): what the geometry worker raises, and what it leaves on disk.

The kernel runs in a worker process. An exception of the package arrives as itself; any
other exception is a defect, not a refusal of the input, and arrives as its own type with the
worker's traceback.
"""

from __future__ import annotations

import errno
import os
import shutil
import signal
from pathlib import Path

import pytest

from cfdbender._errors import InputArtifactError
from cfdbender.mesh._gmsh_worker import publish_stage, run_isolated

_HERE = "tests.tier1_offline.test_mesh_gmsh_worker"


class KernelOddError(Exception):
    """An exception type only the worker's own code defines."""


def worker_value_error():
    raise ValueError("a programming error in the worker")


def worker_odd():
    raise KernelOddError("the kernel said something odd")


def worker_missing_extra():
    from cfdbender.mesh._extras import MissingExtraError

    raise MissingExtraError(extra="fsi", package="PyNiteFEA", purpose="the test")


def worker_kernel_unavailable():
    from cfdbender.mesh._extras import KernelUnavailableError

    raise KernelUnavailableError(purpose="the test", reason="libGLU.so.1: cannot open")


def worker_input_artifact():
    raise InputArtifactError("bad surface. Nothing was written.", kind="geometry")


def _direct(*_args, **_kwargs):
    raise AssertionError("the worker is not the calling process")


def test_mesh_gmsh_worker_a_defect_in_the_worker_is_not_an_input_refusal():
    """a ValueError of the worker arrives as a ValueError, with its traceback."""
    with pytest.raises(ValueError, match="a programming error in the worker") as raised:
        run_isolated(f"{_HERE}:worker_value_error", {}, _direct)
    assert not isinstance(raised.value, InputArtifactError)
    notes = "\n".join(getattr(raised.value, "__notes__", []))
    assert "worker_value_error" in notes and "Traceback" in notes


def test_mesh_gmsh_worker_an_unknown_worker_exception_names_its_type():
    """a type the parent does not know returns as a RuntimeError that names it."""
    with pytest.raises(
        RuntimeError, match="KernelOddError.*the kernel said something odd"
    ) as raised:
        run_isolated(f"{_HERE}:worker_odd", {}, _direct)
    assert not isinstance(raised.value, InputArtifactError)
    assert "worker_odd" in "\n".join(raised.value.__notes__)


def worker_kernel_failure():
    raise Exception("Could not read the surfaces")  # noqa: TRY002, the binding's own form


def test_mesh_gmsh_worker_a_kernel_failure_stays_the_documented_refusal(capsys):
    """the kernel binding raises a bare Exception; the verb reports it, exit 2.

    convert_iges and grid_iges document a kernel failure as InputArtifactError, and the command
    line catches the package's errors, so it must not reach the person as a traceback.
    """
    from cfdbender.mesh._cli_support import print_checks

    with pytest.raises(InputArtifactError, match="kernel failed.*Could not read") as raised:
        run_isolated(f"{_HERE}:worker_kernel_failure", {}, _direct)
    assert "worker_kernel_failure" in "\n".join(raised.value.__notes__)
    code = print_checks(lambda: run_isolated(f"{_HERE}:worker_kernel_failure", {}, _direct))
    assert code == 2 and "Could not read" in capsys.readouterr().err


def test_mesh_gmsh_worker_a_refusal_of_the_package_keeps_its_type_and_kind():
    """the package's own refusal is still the same type and message."""
    with pytest.raises(InputArtifactError, match="bad surface. Nothing was written.") as raised:
        run_isolated(f"{_HERE}:worker_input_artifact", {}, _direct)
    assert raised.value.kind == "geometry"


def test_mesh_gmsh_worker_a_missing_extra_in_the_worker_keeps_its_fields():
    """a MissingExtraError from the worker still names the extra and the remedy."""
    from cfdbender.mesh._extras import MissingExtraError

    with pytest.raises(MissingExtraError) as raised:
        run_isolated(f"{_HERE}:worker_missing_extra", {}, _direct)
    assert raised.value.extra == "fsi"
    assert raised.value.package == "PyNiteFEA"
    assert raised.value.remedy == 'pip install "fsi"'


def test_mesh_gmsh_worker_a_kernel_unavailable_error_in_the_worker_keeps_its_type_and_message():
    """Route A: a library that fails to load inside the worker reaches the verb as itself."""
    from cfdbender.mesh._extras import KernelUnavailableError, MissingExtraError

    with pytest.raises(KernelUnavailableError) as raised:
        run_isolated(f"{_HERE}:worker_kernel_unavailable", {}, _direct)
    assert isinstance(raised.value, MissingExtraError)
    assert raised.value.reason == "libGLU.so.1: cannot open"
    assert raised.value.remedy == 'pip install "gmsh>=4.15.1,<5"'
    assert "libGLU.so.1: cannot open" in str(raised.value)
    assert "libglu1-mesa" in str(raised.value)


# ------------------------------------------------ nothing is published on a failure


def _crash():
    if os.name == "nt":
        os._exit(-1073741819)
    os.kill(os.getpid(), signal.SIGSEGV)


def _write_outputs(out: str) -> list[Path]:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    side = [target, Path(str(target) + ".boundaries.toml"), Path(str(target) + ".json")]
    for path in side:
        path.write_text("written\n", encoding="utf-8")
    return side


def worker_writes_then_crashes(out):
    _write_outputs(out)
    _crash()


def worker_writes_then_refuses(out):
    _write_outputs(out)
    raise InputArtifactError("late refusal. Nothing was written.", kind="geometry")


def worker_writes_then_hangs(out):
    import time

    _write_outputs(out)
    time.sleep(60)


def worker_writes(out):
    _write_outputs(out)
    return {"output": Path(out).name, "echo": out}


@pytest.mark.parametrize(
    ("worker", "error", "match"),
    [
        ("worker_writes_then_crashes", InputArtifactError, "kernel failed"),
        ("worker_writes_then_refuses", InputArtifactError, "late refusal"),
    ],
)
def test_mesh_gmsh_worker_a_failed_worker_leaves_no_output_behind(tmp_path, worker, error, match):
    """a conversion that fails after it wrote publishes nothing."""
    out = tmp_path / "result" / "wing.obj"
    with pytest.raises(error, match=match):
        run_isolated(f"{_HERE}:{worker}", {"out": str(out)}, _direct)
    assert not out.parent.exists() or not list(out.parent.iterdir())
    assert not list(tmp_path.rglob("*.json")) and not list(tmp_path.rglob("*.toml"))


def test_mesh_gmsh_worker_a_hung_worker_leaves_no_output_behind(tmp_path, monkeypatch):
    """a worker ended at its time limit after it wrote publishes nothing."""
    monkeypatch.setenv("CFDBENDER_GMSH_TIMEOUT", "3")
    out = tmp_path / "wing.obj"
    with pytest.raises(InputArtifactError, match="longer than 3 s"):
        run_isolated(f"{_HERE}:worker_writes_then_hangs", {"out": str(out)}, _direct)
    assert list(tmp_path.iterdir()) == []


def test_mesh_gmsh_worker_an_answered_worker_publishes_every_output_at_its_place(tmp_path):
    """after the answer, the files are at the requested path, under their names."""
    out = tmp_path / "deep" / "wing.obj"
    record = run_isolated(f"{_HERE}:worker_writes", {"out": str(out)}, _direct)
    assert record["output"] == "wing.obj"
    assert sorted(path.name for path in out.parent.iterdir()) == [
        "wing.obj",
        "wing.obj.boundaries.toml",
        "wing.obj.json",
    ]
    assert out.read_text(encoding="utf-8") == "written\n"
    out.write_text("older\n", encoding="utf-8")
    run_isolated(f"{_HERE}:worker_writes", {"out": str(out)}, _direct)
    assert out.read_text(encoding="utf-8") == "written\n"


def _staged(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("wing.obj", "wing.obj.json"):
        (stage / name).write_text("new\n", encoding="utf-8")
    folder = tmp_path / "out"
    folder.mkdir()
    for name in ("wing.obj", "wing.obj.json"):
        (folder / name).write_text("older\n", encoding="utf-8")
    return stage, folder


def _other_file_system(monkeypatch, stage):
    """``os.replace`` out of the staging folder fails, as across two file systems."""
    real = os.replace

    def replace(src, dst):
        if Path(src).parent == stage:
            raise OSError(errno.EXDEV, "cross-device link", str(src))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)


def test_mesh_gmsh_worker_a_publish_that_fails_keeps_no_torn_file_and_no_old_record(
    tmp_path, monkeypatch
):
    """a copy that fails leaves the old mesh whole, with no record to vouch for it."""
    stage, folder = _staged(tmp_path)
    _other_file_system(monkeypatch, stage)

    def torn(_src, dst, **_kwargs):
        Path(dst).write_text("ne", encoding="utf-8")
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(shutil, "copy2", torn)
    monkeypatch.setattr(shutil, "move", torn)
    with pytest.raises(OSError, match="no space"):
        publish_stage(stage, folder)
    assert (folder / "wing.obj").read_text(encoding="utf-8") == "older\n"
    assert sorted(path.name for path in folder.iterdir()) == ["wing.obj"]


def test_mesh_gmsh_worker_a_publish_across_file_systems_arrives_whole(tmp_path, monkeypatch):
    """when a move cannot be a rename, every file still arrives, the record with them."""
    stage, folder = _staged(tmp_path)
    _other_file_system(monkeypatch, stage)
    publish_stage(stage, folder)
    assert sorted(path.name for path in folder.iterdir()) == ["wing.obj", "wing.obj.json"]
    assert (folder / "wing.obj").read_text(encoding="utf-8") == "new\n"
    assert (folder / "wing.obj.json").read_text(encoding="utf-8") == "new\n"


def worker_refuses_naming_its_output(out):
    _write_outputs(out)
    raise InputArtifactError(f"{out}: late refusal. Nothing was written.", kind="geometry")


def test_mesh_gmsh_worker_a_refusal_names_the_requested_path_not_the_staging_folder(tmp_path):
    """the staging folder is not shown to the caller; the path asked for is."""
    out = tmp_path / "wing.obj"
    with pytest.raises(InputArtifactError) as raised:
        run_isolated(f"{_HERE}:worker_refuses_naming_its_output", {"out": str(out)}, _direct)
    assert str(out.parent) in str(raised.value) and "cfdb-gmsh-" not in str(raised.value)


def worker_warns_builtin():
    import warnings

    warnings.warn("overflow in a kernel array", RuntimeWarning, stacklevel=1)
    return {}


def worker_warns_package():
    from cfdbender._errors import warn

    warn("the package's own warning")
    return {}


def test_mesh_gmsh_worker_011_a_builtin_worker_warning_carries_its_category_in_the_text():
    """it stays GeometryWarning (the guard, R4-002) and says what it was."""
    from cfdbender._errors import GeometryWarning

    with pytest.warns(GeometryWarning, match="RuntimeWarning: overflow in a kernel array"):
        run_isolated(f"{_HERE}:worker_warns_builtin", {}, _direct)


def test_mesh_gmsh_worker_011_a_package_worker_warning_keeps_its_text():
    from cfdbender._errors import GeometryWarning

    with pytest.warns(GeometryWarning) as found:
        run_isolated(f"{_HERE}:worker_warns_package", {}, _direct)
    assert str(found[0].message) == "the package's own warning"

"""R1-IGES-010 (0.57.0): IGES names are read byte for byte and decoded with a stated encoding."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

# The worker process imports this module by its dotted name, without the test folder on its path.
sys.path.insert(0, str(Path(__file__).parent))
from iges_support import Iges, square  # noqa: E402

from cfdbender._errors import GeometryWarning, InputArtifactError


def _raw(tmp_path, payload: bytes):
    """A two-face file whose first level carries a NAME with the exact ``payload`` bytes."""
    iges = Iges()
    iges.patch(square(0, 0, 1, 1), level=1)
    iges.patch(square(0, 0, 1, 1, z=1), level=2)
    iges.name(1, payload.decode("latin-1"))
    iges.name(2, "Plain")
    return iges.write(tmp_path / "raw.igs")


def test_geom_iges_decisions_name_decodes_with_the_stated_encoding(tmp_path):
    from cfdbender.geometry._iges_names import read_iges_names

    path = _raw(tmp_path, "Asa_ção".encode())
    assert read_iges_names(path, name_encoding="utf-8").faces == ("Asa_ção", "Plain")


def test_geom_iges_decisions_default_encoding_is_latin1_and_never_guesses(tmp_path):
    from cfdbender.geometry._iges_names import read_iges_names

    payload = "Asa_ção".encode()
    path = _raw(tmp_path, payload)
    assert read_iges_names(path).faces[0] == payload.decode("latin-1")


def test_geom_iges_decisions_invalid_bytes_for_the_encoding_are_refused_not_replaced(tmp_path):
    from cfdbender.geometry._iges_names import read_iges_names

    path = _raw(tmp_path, b"Asa_\xe7")
    with pytest.raises(InputArtifactError, match="utf-8"):
        read_iges_names(path, name_encoding="utf-8")


def test_geom_iges_decisions_a_name_no_level_takes_is_not_decoded(tmp_path):
    """RL57B-2: a level-0 feature name, or a second NAME of a level, never refuses the file."""
    from cfdbender.geometry._iges_names import read_iges_names

    iges = Iges()
    iges.patch(square(0, 0, 1, 1), level=1)
    iges.name(0, b"Pi\xe8ce".decode("latin-1"))  # a feature name in Latin-1, on no level
    iges.name(1, "Asa_ção".encode().decode("latin-1"))
    iges.name(1, b"Asa_\xe7".decode("latin-1"))  # a later NAME of level 1: not the level's name
    path = iges.write(tmp_path / "mixed.igs")
    assert read_iges_names(path, name_encoding="utf-8").faces == ("Asa_ção",)


def test_geom_iges_decisions_unknown_encoding_is_refused(tmp_path):
    from cfdbender.geometry._iges_names import read_iges_names

    with pytest.raises(InputArtifactError, match="not a known"):
        read_iges_names(_raw(tmp_path, b"Asa"), name_encoding="no-such-codec")


def test_geom_iges_decisions_written_names_round_trip_in_utf8_with_byte_counts(tmp_path):
    from cfdbender.geometry._iges_names import (
        flag_curves_on_surfaces,
        name_faces,
        read_iges_names,
    )

    iges = Iges()
    iges.patch(square(0, 0, 1, 1), level=0)
    iges.patch(square(0, 0, 1, 1, z=1), level=0)
    source = iges.write(tmp_path / "raw.igs")
    names = ["Asa_ção", "Leme_€"]
    named = name_faces(source, tmp_path / "named.igs", names, name_encoding="utf-8")
    flagged = flag_curves_on_surfaces(named, tmp_path / "flagged.igs")
    for path in (named, flagged):
        assert read_iges_names(path, name_encoding="utf-8").faces == tuple(names)
        assert {len(line) for line in path.read_bytes().splitlines()} == {80}
    assert b"9HAsa_" in named.read_bytes()  # 9 bytes in UTF-8, 7 characters


def test_geom_iges_decisions_a_name_the_encoding_cannot_hold_is_refused_before_writing(tmp_path):
    from cfdbender.geometry._iges_names import name_faces

    iges = Iges()
    iges.patch(square(0, 0, 1, 1), level=0)
    source = iges.write(tmp_path / "raw.igs")
    with pytest.raises(InputArtifactError, match="cannot be written"):
        name_faces(source, tmp_path / "n.igs", ["Asa_€"], name_encoding="latin-1")
    assert not (tmp_path / "n.igs").exists()


def _symmetry_grid(tmp_path, line):
    path = tmp_path / "grid.toml"
    path.write_text(f"[defaults]\n{line}\n", encoding="utf-8")
    return path


def test_geom_iges_decisions_the_skin_takes_its_plane_of_symmetry_only_from_the_caller():
    import inspect

    from cfdbender.geometry._iges_skin import outer_skin

    assert inspect.signature(outer_skin).parameters["symmetry"].default == "none"


_FAULT_TEXT = "0xc0000005" if os.name == "nt" else "signal 11"


def worker_crash():
    """Stand in for a native kernel fault: the process ends at once, with no Python unwinding."""
    if os.name == "nt":
        os._exit(-1073741819)  # STATUS_ACCESS_VIOLATION, 0xC0000005
    os.kill(os.getpid(), signal.SIGSEGV)


def worker_ok(value=1):
    import warnings

    from cfdbender._errors import NotApplicableWarning

    warnings.warn("careful", NotApplicableWarning, stacklevel=1)
    return {"value": value}


def worker_refuses():
    raise InputArtifactError("bad surface. Nothing was written.", kind="geometry")


def worker_hangs():
    import time

    time.sleep(60)


_HERE = "tests.tier1_offline.test_geom_iges_decisions"


def test_geom_iges_decisions_a_kernel_fault_ends_the_worker_and_reports_its_cause_then_a_fresh():
    from cfdbender.mesh._gmsh_worker import run_isolated

    with pytest.raises(InputArtifactError) as raised:
        run_isolated(f"{_HERE}:worker_crash", {}, worker_crash)
    text = str(raised.value)
    assert _FAULT_TEXT in text.lower()
    assert "Nothing was written" in text
    assert "not reused" in text
    with pytest.warns(GeometryWarning, match="careful"):
        assert run_isolated(f"{_HERE}:worker_ok", {"value": 7}, worker_ok) == {"value": 7}


def test_geom_iges_decisions_a_refusal_in_the_worker_arrives_as_the_same_type_and_message():
    from cfdbender.mesh._gmsh_worker import run_isolated

    with pytest.raises(InputArtifactError, match="bad surface. Nothing was written."):
        run_isolated(f"{_HERE}:worker_refuses", {}, worker_refuses)


def test_geom_iges_decisions_a_hung_worker_is_ended_and_reported(monkeypatch):
    from cfdbender.mesh._gmsh_worker import run_isolated

    monkeypatch.setenv("CFDBENDER_GMSH_TIMEOUT", "2")
    with pytest.raises(InputArtifactError, match="longer than 2 s"):
        run_isolated(f"{_HERE}:worker_hangs", {}, worker_hangs)


@pytest.mark.parametrize("value", ["soon", "0", "-5", "nan", "inf"])
def test_geom_iges_decisions_a_time_limit_that_is_not_a_positive_number_is_refused(
    monkeypatch, value
):
    """RL57B-4: the documented switch is refused by name, not as a parse error or instant end."""
    from cfdbender._errors import InputValueError
    from cfdbender.mesh._gmsh_worker import run_isolated

    monkeypatch.setenv("CFDBENDER_GMSH_TIMEOUT", value)
    with pytest.raises(InputValueError, match="CFDBENDER_GMSH_TIMEOUT.*positive number"):
        run_isolated(f"{_HERE}:worker_ok", {}, worker_ok)


def test_geom_iges_decisions_convert_never_initializes_the_kernel_in_the_calling_process(
    tmp_path, monkeypatch
):
    from iges_support import gmsh_or_skip, plate_and_flap

    from cfdbender.geometry.iges_convert import convert_iges

    gmsh = gmsh_or_skip()

    def refuse(*args, **kwargs):
        raise AssertionError("the calling process initialized the kernel")

    monkeypatch.setattr(gmsh, "initialize", refuse)
    record = convert_iges([plate_and_flap(tmp_path)], tmp_path / "out.igs")
    assert record["faces"] == 2
    assert (tmp_path / "out.igs").is_file()


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
def test_geom_iges_decisions_the_skin_mirrors_across_the_axis_its_plane_names(
    monkeypatch, plane, axis
):
    """RL57B-3: each stated plane reaches the assembly as its own axis (x=0, y=1, z=2)."""
    from cfdbender.geometry._iges_skin import outer_skin

    seen = _crossing_skin_mirror(monkeypatch, lambda v, f: outer_skin(v, f, symmetry=plane))
    assert seen == [axis]


def test_geom_iges_decisions_a_recorded_warning_replays_by_name_through_the_package_route():
    """warn_by_name keeps a package category, else falls back, at the caller's line."""
    import warnings

    from cfdbender import _errors

    category = next(
        c
        for c in vars(_errors).values()
        if isinstance(c, type)
        and issubclass(c, _errors.GeometryWarning)
        and c is not _errors.GeometryWarning
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _errors.warn_by_name("named", category.__name__)
        _errors.warn_by_name("unnamed", "NoSuchWarning")
        _errors.warn_by_name("an exception, not a warning", "InputValueError")
    assert [(str(w.message), w.category) for w in caught] == [
        ("named", category),
        ("unnamed", _errors.GeometryWarning),
        ("an exception, not a warning", _errors.GeometryWarning),
    ]
    assert all(w.filename == __file__ for w in caught)

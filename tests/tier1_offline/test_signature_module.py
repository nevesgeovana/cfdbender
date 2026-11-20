"""The signature module draws the same boxes as pyflightstream's: drawings, phrases, outcomes."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from cfdbender import _signature

SEES_YOU = "geoversegoddess sees you"
BORDER = "#" * 81

#: The koala with its one phrase.
KOALA = """\
#################################################################################
#                                                                               #
#                                          .-"-.               .-"-.            #
# Let now COALA do its magic              / .-. \\   _.---._   / .-. \\           #
#                                        | (   ) |.'       '.| (   ) |          #
#                                         \\ '-' /  o       o  \\ '-' /           #
#                                          '-._|      ___      |_.-'            #
#                                               \\    (___)    /                 #
# geoversegoddess sees you                       '.    V    .'                  #
#                                                  '-.___.-'                    #
#                                                                               #
#################################################################################"""


def drawn(err: str) -> tuple[str, str] | None:
    """Return the (drawing, phrase) whose box ``err`` carries, or None."""
    for name, phrases in _signature.PHRASES.items():
        for phrase in phrases:
            if _signature.box(name, phrase) in err:
                return name, phrase
    return None


def outcome_of(err: str) -> str:
    """Return the outcome whose pool the box in ``err`` was drawn from."""
    found = drawn(err)
    assert found is not None, err
    return next(k for k, v in _signature.OUTCOME_DRAWINGS.items() if found[0] in v)


def test_constants_are_the_shared_geometry():
    assert _signature.WIDTH == 81
    assert _signature.TEXT_COLUMN == 38
    assert _signature.SEES_YOU == SEES_YOU


def test_every_box_is_81_columns_of_ascii_and_every_phrase_fits():
    assert set(_signature.DRAWINGS) == set(_signature.PHRASES)
    assert {n for names in _signature.OUTCOME_DRAWINGS.values() for n in names} == set(
        _signature.DRAWINGS
    )
    for name, phrases in _signature.PHRASES.items():
        for phrase in phrases:
            lines = _signature.box(name, phrase).split("\n")
            assert all(len(line) == 81 for line in lines), (name, phrase)
            assert all(line.isascii() for line in lines), (name, phrase)
            assert lines[0] == lines[-1] == BORDER
            assert all(line[0] == line[-1] == "#" for line in lines)
            art = lines[2:-2]
            assert art[1].startswith(f"# {phrase} "), (name, phrase)
            assert art[-2].startswith(f"# {SEES_YOU} "), name


def test_the_outcome_sets_are_those_of_the_sister_library():
    assert set(_signature.OUTCOME_DRAWINGS) == {"post", "success", "failed", "cancelled"}
    assert _signature.OUTCOME_DRAWINGS["post"] == ("coala",)
    assert len(_signature.OUTCOME_DRAWINGS["success"]) == 8
    assert len(_signature.OUTCOME_DRAWINGS["failed"]) == 6
    assert len(_signature.OUTCOME_DRAWINGS["cancelled"]) == 5
    assert len(_signature.DRAWINGS) == 20
    assert sum(len(p) for p in _signature.PHRASES.values()) == 58


def test_the_koala_is_drawn_exactly():
    assert _signature.box("coala", "Let now COALA do its magic") == KOALA


def test_a_phrase_wider_than_the_text_column_is_refused_not_truncated():
    with pytest.raises(ValueError, match="does not widen"):
        _signature.box("coala", "x" * (_signature.TEXT_COLUMN + 1))


def test_pick_draws_a_drawing_of_the_outcome_with_one_of_its_phrases():
    chooser = random.Random(3)
    for outcome, names in _signature.OUTCOME_DRAWINGS.items():
        for _ in range(30):
            name, phrase = _signature.pick(outcome, chooser)
            assert name in names
            assert phrase in _signature.PHRASES[name]


def test_the_signature_module_imports_nothing_from_the_package():
    tree = ast.parse(Path(_signature.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("cfdbender")
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("cfdbender") for a in node.names)

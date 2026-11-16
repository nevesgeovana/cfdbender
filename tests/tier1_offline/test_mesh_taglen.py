"""a long automatic level tag is a summary and a short hash.

Every source is synthetic (tests/p0380_mesh_fixtures.py): twelve disjoint curved
plates, each its own grid family with a long name, so the 0.38.0 tag of the
whole selection is far past 32 characters.
"""
# ruff: noqa: E501

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from mesh_fixtures import (
    Composite,
    plate_cells,
    plate_nodes,
    sheet_with_strips,
    write_source,
)

from cfdbender._errors import InputArtifactError
from cfdbender.mesh._refine import _config, refine_mesh

NAMES = [f"LongFamilyName{i:02d}" for i in range(12)]


def _source(tmp_path: Path, names=NAMES, toml: str | None = None, name: str = "src") -> Path:
    nodes, families = [], {}
    base = np.zeros(3)
    for k, fam in enumerate(names):
        n = plate_nodes(2, 2) + base + np.array([0.0, 0.0, 3.0 * k])
        families[fam] = plate_cells(2, [(i, j) for j in range(2) for i in range(2)])
        families[fam] = [[v + 9 * k for v in f] for f in families[fam]]
        nodes.append(n)
    mesh = Composite(np.vstack(nodes), families, np.zeros((0, 3)), {})
    return write_source(tmp_path / name, "wing", mesh, refine_toml=toml)


def _factor(i: int, factor: float = 2.0) -> float:
    """Alternate two factors so the selection is not one factor on every family (that is R<f>)."""
    return factor if i % 2 == 0 else factor + 1.0


def _parts(names, factor: float = 2.0) -> list[str]:
    return [f"{n}{_factor(i, factor):g}" for i, n in enumerate(names)]


def _toml(names, factor: float = 2.0, keep: set[str] | frozenset[str] = frozenset()) -> str:
    lines = []
    for i, n in enumerate(names):
        lines += [f"[families.{n}]", f"factor = {_factor(i, factor)}"]
        if n in keep:
            lines.append("keep_first_row = true")
    return "\n".join(lines) + "\n"


def _request(tmp_path: Path, names, factor=2.0, keep=frozenset(), over=None):
    src = _source(tmp_path, names, _toml(names, factor, keep), name="req")
    request = _config.resolve_request(src, list(names), factor=None, families=None)
    if over:
        request.specs.update(over)
    return request


def _record(level) -> dict:
    return json.loads(
        next(p for p in level.files if p.name.endswith(".refine.json")).read_text("utf-8")
    )


def _hash(names) -> str:
    """The digest as the box writes it: one line per family, sorted key=value."""
    lines = [f"{n} factor={_factor(i):g}" for i, n in enumerate(names)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:8]


def test_twelve_long_families_write_a_short_tag_and_a_level(tmp_path):
    """twelve long names, a tag of at most 20 characters."""
    src = _source(tmp_path, toml=_toml(NAMES))
    old = "R-" + "-".join(_parts(NAMES))
    assert len(old) > 32  # the 0.38.0 tag, measured: over the limit on every operating system
    request = _config.resolve_request(src, NAMES, factor=None, families=None)
    assert len(_config.level_tag(request, NAMES)) <= 20  # the measured tag length, not an OS error
    level = refine_mesh(src, overwrite=True, out_dir=tmp_path / "out")
    tag = level.folder.name.removeprefix("wing_")
    assert len(tag) <= 20 and level.folder.is_dir() and level.obj.is_file()
    assert tag == f"R-12fam-{_hash(NAMES)}"
    record = _record(level)
    assert record["tag_selection"] == NAMES
    assert sorted(record["specs"]) == sorted(NAMES)


def test_two_families_keep_the_tag_of_0_38_0(tmp_path):
    """a built tag of at most 32 characters is unchanged."""
    two = ["Wing", "Body"]
    src = _source(tmp_path, two, _toml(two))
    level = refine_mesh(src, overwrite=True, out_dir=tmp_path / "out")
    assert level.folder.name == f"wing_R-{two[0]}2-{two[1]}3"
    assert len(level.folder.name.removeprefix("wing_")) <= 32
    assert "tag_selection" not in _record(level)


def test_a_user_tag_is_used_as_given(tmp_path):
    """a [refine] tag is used as given, however long."""
    src = _source(tmp_path, toml='[refine]\ntag = "' + "x" * 33 + '"\n' + _toml(NAMES))
    level = refine_mesh(src, overwrite=True, out_dir=tmp_path / "out")
    assert level.folder.name == "wing_" + "x" * 33
    assert "tag_selection" not in _record(level)


def test_same_selection_same_tag_and_any_change_another(tmp_path):
    """stable tag; a factor changes it."""
    base = _config.level_tag(_request(tmp_path, NAMES), NAMES)
    assert base == _config.level_tag(_request(tmp_path, NAMES), NAMES)
    assert base == f"R-12fam-{_hash(NAMES)}"
    spec = _request(tmp_path, NAMES).specs[NAMES[5]]
    elsewhere = _request(tmp_path, NAMES)
    elsewhere.specs[NAMES[2]] = dataclasses.replace(elsewhere.specs[NAMES[2]], factor=7.0)
    other = _config.level_tag(elsewhere, NAMES)
    assert other != base
    changed = _request(tmp_path, NAMES)
    changed.specs[NAMES[5]] = type(spec)(**{**vars(spec), "factor": 5.0})
    assert _config.level_tag(changed, NAMES) not in (base, other)
    # a mutant that hashes the selection unordered would give the same tag for a reordered source
    assert _config.level_tag(_request(tmp_path, NAMES), NAMES[::-1]) != base


def test_folder_name_refusal_only_for_the_built_form(tmp_path):
    """a bad name is refused in a built tag, not the hash."""
    bad = ["A/B", "C"]
    request = _request(tmp_path, ["A", "C"])
    request = dataclasses.replace(
        request, specs={bad[0]: request.specs["A"], bad[1]: request.specs["C"]}
    )
    with pytest.raises(InputArtifactError) as refused:
        _config.level_tag(request, bad, "src.obj")
    assert str(refused.value) == (
        "src.obj: the family 'A/B' holds '/', so the level's tag 'R-A/B2-C3' would not be one "
        "folder name; rename the family, or give the level its name in the refinement file "
        'config (CLI: --config) as [refine] tag = "fine". Nothing was written.'
    )
    many = [f"Fam/{i:02d}" + "x" * 5 for i in range(12)]
    request = _request(tmp_path, [f"F{i}" for i in range(12)])
    request = dataclasses.replace(
        request, specs=dict(zip(many, request.specs.values(), strict=True))
    )
    assert _config.level_tag(request, many).startswith("R-12fam-")


def test_a_method_or_element_change_alone_changes_the_summary_tag(tmp_path):
    """every option of the canonical selection reaches the hash."""
    base = _config.level_tag(_request(tmp_path, NAMES), NAMES)
    spec = _request(tmp_path, NAMES).specs[NAMES[3]]
    remeshed = _request(tmp_path, NAMES)
    remeshed.specs[NAMES[3]] = dataclasses.replace(spec, method="remesh")
    assert _config.level_tag(remeshed, NAMES) != base
    # the short tag of does not distinguish the method, as on 0.38.0
    two = ["Wing", "Body"]
    short = _request(tmp_path, two)
    plain = _config.level_tag(short, two)
    short.specs["Wing"] = dataclasses.replace(short.specs["Wing"], method="remesh")
    assert _config.level_tag(short, two) == plain


def _grid_beside_a_strip(tmp_path: Path, grid: str) -> Path:
    """A grid with an unselected five-row neighbour, so cuts a band out of it."""
    mesh = sheet_with_strips((("T", "tri", (0.03, 0.04, 0.05, 0.065, 0.08)),), grid=grid)
    return write_source(tmp_path / grid[:6], "wing", mesh)


def test_tag_selection_is_taken_with_the_tag_not_after_the_bands(tmp_path):
    """a band cut out of a neighbour never enters tag_selection."""
    long_grid = "AVeryLongGridFamilyNameForTheTag"
    level = refine_mesh(
        _grid_beside_a_strip(tmp_path, long_grid),
        2.0,
        families=[long_grid],
        out_dir=tmp_path / "o1",
    )
    record = _record(level)
    assert level.folder.name.startswith("wing_R-1fam-")
    assert record["tag_selection"] == [long_grid]
    short = refine_mesh(
        _grid_beside_a_strip(tmp_path, "WingGridTwentyFiveChars"),
        2.0,
        families=["WingGridTwentyFiveChars"],
        out_dir=tmp_path / "o2",
    )
    assert "tag_selection" not in _record(short)


def test_a_uniform_whole_mesh_tag_is_not_refused_for_a_family_name(tmp_path):
    """one factor on every family is R<f>, holding no name.

    The 0.38.0 compatibility control: the sole family ``Wing/upper`` at
    factor 2 gives the tag ````, a folder name, and is not refused, nor
    is a mesh whose two families, one of them ``Wing/upper``, take factor 2.
    Two factors on the same families build the tag from the names and are
    refused naming the family.
    """
    request = _request(tmp_path, ["A"])
    sole = dataclasses.replace(request, specs={"Wing/upper": _config.FamilySpec(factor=2.0)})
    assert _config.level_tag(sole, ["Wing/upper"]) == "R2"
    assert _config.tag_selection(sole, ["Wing/upper"]) is None
    names = ["Wing/upper", "Body"]
    same = {n: _config.FamilySpec(factor=2.0) for n in names}
    assert _config.level_tag(dataclasses.replace(request, specs=same), names) == "R2"
    mixed = dataclasses.replace(request, specs={**same, "Body": _config.FamilySpec(factor=3.0)})
    with pytest.raises(InputArtifactError, match="the family 'Wing/upper' holds '/'"):
        _config.level_tag(mixed, names)


ALPHA, BETA = "LongFamilyNameAlpha", "LongFamilyNameBeta"


def _two(tmp_path: Path, alpha: float, beta: float = 5.0):
    request = _request(tmp_path, ["A"])
    specs = {ALPHA: _config.FamilySpec(factor=alpha), BETA: _config.FamilySpec(factor=beta)}
    return dataclasses.replace(request, specs=specs)


@pytest.mark.parametrize(
    "pair",
    [
        ((1.000001, "1.000001"), (1.000002, "1.000002")),
        ((2.0, "2"), (2.0000000000000004, "2.0000000000000004")),
        ((0.1 + 0.2, "0.30000000000000004"), (0.3, "0.3")),
    ],
    ids=["sixth-digit", "last-bit", "sum"],
)
def test_closely_spaced_factors_give_distinct_selections_and_tags(tmp_path, pair):
    """a factor is written in full, so close factors differ.

    Two selections whose factors differ past the sixth significant digit
    give two canonical selections, each holding its factor in the shortest
    text that reads back to it (a whole factor as ``2``), and two summary
    tags, each the hash of its own text.
    """
    texts, tags = [], []
    for alpha, written in pair:
        request = _two(tmp_path, alpha)
        picked = [ALPHA, BETA]
        text = _config.canonical_selection(request, picked)
        assert text == f"{ALPHA} factor={written}\n{BETA} factor=5"
        assert float(written) == alpha
        tag = _config.level_tag(request, picked)
        assert tag == "R-2fam-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        texts.append(text)
        tags.append(tag)
    assert texts[0] != texts[1] and tags[0] != tags[1]


def test_f323_pfs_2124_056_a_grid_ignored_mode_in_the_summary(tmp_path):
    """F-323,: a remeshed family's mode reaches the hash, a grid's not."""
    names = [ALPHA, BETA]
    grid = _two(tmp_path, 2.0)
    grid.specs[ALPHA] = _config.FamilySpec(factor=2.0, method="grid")
    base = _config.canonical_selection(grid, names)
    quad = dataclasses.replace(grid, elements={ALPHA: "quad-dominant"})
    assert _config.canonical_selection(quad, names) == base
    remeshed = dataclasses.replace(_two(tmp_path, 2.0), elements={ALPHA: "quad-dominant"})
    assert "elements=quad-dominant" in _config.canonical_selection(remeshed, names)

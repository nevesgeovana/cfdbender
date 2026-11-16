"""The layout sidecar of a generated wing, as ``cfd-mesh iges-grid`` reads it.

``<piece>.layout.json`` sits beside the IGES: each named face is split at its lines before the
four-sided test, and the trailing edge comes from the ``trailing_edge`` role. The reader is
pure and runs everywhere.
"""

from __future__ import annotations

import json

import pytest

from cfdbender.mesh import _iges_grid_config as cfg


def _sidecar(**changes):
    """A sidecar in the written format: two skin faces, the trailing-edge face and a cap."""
    skin = [
        {"face": "lower", "base": "skin", "role": "lower", "loop": _loop(0.0, 0.5, 0.0, 1.0)},
        {"face": "upper", "base": "skin", "role": "upper", "loop": _loop(0.5, 1.0, 0.0, 1.0)},
    ]
    te = [{"face": "trailing_edge", "base": "trailing_edge", "role": "trailing_edge",
           "loop": _loop(0.0, 1.0, 0.0, 1.0)}]  # fmt: skip
    cap = [{"face": "cap", "base": "tip_cap", "role": "cap", "loop": _loop(0.0, 1.0, 0.0, 1.0)}]
    faces = [
        {"name": "panel_0", "base": "skin", "role": "lower", "area_m2": 2.0, "source_faces": skin},
        {"name": "panel_1", "base": "skin", "role": "upper", "area_m2": 2.1, "source_faces": skin},
        {"name": "panel_2", "base": "trailing_edge", "role": "trailing_edge", "area_m2": 0.005,
         "source_faces": te},
        {"name": "panel_3", "base": "tip_cap", "role": "cap", "area_m2": 0.03, "source_faces": cap},
    ]  # fmt: skip
    data = {
        "lines": {
            "u": {
                "lower_te": 0.0,
                "lower_max": 0.35,
                "le": 0.5,
                "upper_max": 0.65,
                "upper_te": 1.0,
            },
            "v": {"root": 0.0, "section_1": 0.4, "tip": 1.0},
        },
        "blocks": [
            {"base": "skin", "family": "lower_aft", "u": [0.0, 0.35], "v": [0.0, 0.4]},
            {"base": "skin", "family": "lower_fore", "u": [0.35, 0.5], "v": [0.4, 1.0]},
            {"base": "skin", "family": "upper_fore", "u": [0.5, 0.65], "v": [0.0, 0.4]},
            {"base": "skin", "family": "upper_aft", "u": [0.65, 1.0], "v": [0.4, 1.0]},
            {"base": "trailing_edge", "family": "trailing_lower", "u": [0.0, 0.5], "v": [0, 1]},
            {"base": "trailing_edge", "family": "trailing_upper", "u": [0.5, 1.0], "v": [0, 1]},
        ],
        "faces": faces,
    }
    data.update(changes)
    return data


def _loop(u0, u1, v0, v1):
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return [[list(corners[i]), list(corners[(i + 1) % 4])] for i in range(4)]


def _write(tmp_path, data, name="panel"):
    path = tmp_path / f"{name}.layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path / f"{name}.igs"


def test_iges_layout_the_sidecar_sits_beside_the_iges_under_its_stem(tmp_path):
    assert cfg.sidecar_path(tmp_path / "wing.igs") == tmp_path / "wing.layout.json"
    assert cfg.sidecar_path(tmp_path / "wing.IGES") == tmp_path / "wing.layout.json"


def test_iges_layout_no_sidecar_is_a_foreign_file_and_says_nothing(tmp_path):
    notes: list[str] = []
    assert cfg.read_layout(tmp_path / "panel.igs", notes) is None
    assert notes == []


def test_iges_layout_faces_carry_base_role_and_the_lines_inside_them_with_both_ends(tmp_path):
    notes: list[str] = []
    layout = cfg.read_layout(_write(tmp_path, _sidecar()), notes)
    assert notes == [] and layout is not None
    assert layout.component == "panel"
    faces = layout.faces_by_name()
    assert list(faces) == ["panel_0", "panel_1", "panel_2", "panel_3"]
    lower, upper = faces["panel_0"], faces["panel_1"]
    # Own bounds: the lower face spans u 0 to 0.5 and the upper face u 0.5 to 1; the base's
    # block ends inside each are the lines, the ends are always present.
    assert (lower["base"], lower["role"]) == ("skin", "lower")
    assert lower["u"] == [0.0, 0.35, 0.5]
    assert upper["u"] == [0.5, 0.65, 1.0]
    assert lower["v"] == upper["v"] == [0.0, 0.4, 1.0]
    # The trailing-edge base is cut at its own half, not at the skin's chordwise lines.
    assert faces["panel_2"]["u"] == [0.0, 0.5, 1.0]
    assert faces["panel_3"]["u"] == [0.0, 1.0] and faces["panel_3"]["v"] == [0.0, 1.0]


def test_iges_layout_the_face_mapping_is_the_shape_the_junction_classifier_reads(tmp_path):
    """G11's ``junction_blocks`` takes ``{face: {"u": [...], "v": [...]}}`` with both ends."""
    layout = cfg.read_layout(_write(tmp_path, _sidecar()), [])
    for entry in layout.faces_by_name().values():
        for key in ("u", "v"):
            values = entry[key]
            assert values == sorted(values) and len(values) >= 2
            assert values[0] == min(values) and values[-1] == max(values)


def test_iges_layout_names_of_the_lines_ride_along(tmp_path):
    layout = cfg.read_layout(_write(tmp_path, _sidecar()), [])
    lower = layout.faces_by_name()["panel_0"]
    assert lower["u_names"] == {"lower_max": 0.35}
    assert lower["v_names"] == {"section_1": 0.4}


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.pop("faces"),
        lambda d: d.update(faces=[]),
        lambda d: d["faces"][0].pop("source_faces"),
        lambda d: d["faces"][1].update(name="other_1"),
        lambda d: d.update(blocks="none"),
    ],
    ids=["no-faces", "empty-faces", "no-trims", "two-components", "bad-blocks"],
)
def test_iges_layout_an_unusable_sidecar_is_warned_and_the_file_is_foreign(tmp_path, damage):
    data = _sidecar()
    damage(data)
    notes: list[str] = []
    assert cfg.read_layout(_write(tmp_path, data), notes) is None
    assert len(notes) == 1 and "panel.layout.json" in notes[0] and "no layout" in notes[0]


def test_iges_layout_text_that_is_not_json_is_warned_not_refused(tmp_path):
    path = tmp_path / "panel.layout.json"
    path.write_text("{not json", encoding="utf-8")
    notes: list[str] = []
    assert cfg.read_layout(tmp_path / "panel.igs", notes) is None
    assert len(notes) == 1 and "not JSON" in notes[0]


@pytest.mark.parametrize("damage", ["area", "coordinate", "short-point", "nan-area"])
def test_iges_layout_unusable_numeric_layout_falls_back_to_foreign(tmp_path, damage):
    data = _sidecar()
    if damage == "area":
        data["faces"][0]["area_m2"] = "not-a-number"
    elif damage == "coordinate":
        data["blocks"][0]["u"][1] = "not-a-number"
    elif damage == "short-point":
        data["faces"][0]["source_faces"][0]["loop"][0][0] = []
    else:
        data["faces"][0]["area_m2"] = float("nan")
    notes = []
    assert cfg.read_layout(_write(tmp_path, data), notes) is None
    assert len(notes) == 1 and "foreign" in notes[0]

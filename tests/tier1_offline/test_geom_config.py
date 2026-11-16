"""Geometry: the TOML configuration boundary (read, validate, resolve, write)."""

from pathlib import Path

import pytest

from cfdbender._errors import InputValueError
from cfdbender.geometry import load_config, resolve_config
from cfdbender.geometry._boundary import unpack
from cfdbender.geometry.stages import write_config

FIXTURE = Path(__file__).parent / "fixtures" / "geom_geometry_config.toml"


@pytest.mark.parametrize("kind", ["lifting_surface", "body", "wing_fuselage_fairing", "propeller"])
def test_minimal_and_round_trip(kind, tmp_path):
    path = tmp_path / "minimal.toml"
    hosts = 'hosts = ["body", "wing"]\n' if kind == "wing_fuselage_fairing" else ""
    path.write_text(f'type = "{kind}"\n{hosts}', encoding="utf-8")
    result = load_config(path)
    assert result["components"]["COMPONENT"]["type"]["source"] == "given"
    assert result["components"]["COMPONENT"]["version"]["source"] == "default"
    output = tmp_path / "resolved.toml"
    write_config(result, output)
    assert load_config(output) == result


def test_generic_multi_component_file_round_trips(tmp_path):
    result = load_config(FIXTURE)
    assert set(result["components"]) == {"FUS", "WING"}
    assert result["configurations"] == {"demo": ["FUS", "WING"]}
    wing = unpack(result)["components"]["WING"]
    assert wing["tip_cap"]["kind"] == "rounded"
    assert wing["attach"] == {"parent": "FUS", "station": 0.4}
    write_config(result, tmp_path / "again.toml")
    assert load_config(tmp_path / "again.toml") == result


def test_resolved_input_with_a_changed_derived_value_is_refused(tmp_path):
    result = load_config(FIXTURE)
    result["derived"]["WING"]["area"] += 1.0
    with pytest.raises(InputValueError, match="derived.WING.area"):
        resolve_config(result)


def test_inconsistent_root_chord():
    with pytest.raises(ValueError, match="root_chord"):
        resolve_config(
            {
                "type": "lifting_surface",
                "planform": {
                    "sref_m2": 50.0,
                    "aspect_ratio": 8.0,
                    "taper": 0.7,
                    "root_chord_mm": 4000.0,
                },
            }
        )


def test_independent_set_derives_area_and_mean_chord(tmp_path):
    result = resolve_config(
        {
            "type": "lifting_surface",
            "planform": {"sref_m2": 50.0, "aspect_ratio": 8.0, "taper": 0.7, "incidence": 3.0},
            "stations": [
                {"eta": 0.0, "section": {"kind": "naca", "code": "23018"}},
                {"eta": 1.0, "section": {"kind": "naca", "code": "23012"}},
            ],
        }
    )
    derived = result["derived"]["COMPONENT"]
    assert derived["area"] == pytest.approx(50.0)
    # mac = 2/3 * root * (1 + t + t^2) / (1 + t) with root = 2 * area / (span * (1 + t))
    root = 2 * 50.0 / (20.0 * 1.7)
    assert derived["mac"] == pytest.approx(2 / 3 * root * (1 + 0.7 + 0.49) / 1.7)
    write_config(result, tmp_path / "paper.toml")
    assert load_config(tmp_path / "paper.toml") == result


@pytest.mark.parametrize(
    "raw,parameter",
    [
        ({"type": "body", "parts": "0.2, 0.8"}, "parts"),
        ({"type": "body", "azimuths": [0, 90, 180]}, "azimuths"),
        ({"type": "body", "parts": [0.0, 0.2, "x", 0.8, 1.0]}, "parts"),
        ({"type": "lifting_surface", "planform": {"semispan_mm": -500.0}}, "span"),
        ({"type": "lifting_surface", "planform": {"aspect_ratio": float("nan")}}, "aspect_ratio"),
        (
            {
                "type": "lifting_surface",
                "stations": [{"eta": 0.5, "section": {"kind": "naca", "code": "0012"}}] * 2,
            },
            "stations",
        ),
        ({"type": "propeller", "hub_radius": 2}, "hub_radius"),
        ({"units": {"length": "ft"}, "components": {"A": {"type": "body"}}}, "units"),
        (
            {"type": "body", "length_mm": {"value": 1, "unit": "ft", "source": "given"}},
            "length_mm.unit",
        ),
        ({"type": "lifting_surface", "cap": {"type": "spiky"}}, "cap.type"),
        (
            {
                "type": "lifting_surface",
                "stations": [
                    {"eta": 0.0, "section": {"kind": "naca", "code": "CLARKY"}},
                    {"eta": 1.0, "section": {"kind": "naca", "code": "CLARKY"}},
                ],
            },
            "NACA",
        ),
    ],
)
def test_invalid_parameters_are_refused_with_the_parameter_named(raw, parameter):
    with pytest.raises(InputValueError, match=parameter):
        resolve_config(raw)


def test_a_refusal_is_a_value_error_for_callers_that_catch_the_standard_type():
    with pytest.raises(ValueError):
        resolve_config({"components": {}})


def test_parent_cycle():
    with pytest.raises(ValueError, match="attach.parent"):
        resolve_config(
            {
                "components": {
                    "A": {"type": "body", "attach": {"parent": "B"}},
                    "B": {"type": "body", "attach": {"parent": "A"}},
                }
            }
        )


def test_alternate_independent_sets():
    c = resolve_config(
        {
            "type": "lifting_surface",
            "planform": {"semispan_mm": 10000.0, "root_chord_mm": 3000.0, "tip_chord_mm": 2000.0},
        }
    )
    p = unpack(c)["components"]["COMPONENT"]["planform"]
    assert p["area"] == 50
    assert p["aspect_ratio"] == 8
    c = resolve_config(
        {"type": "lifting_surface", "planform": {"sref_m2": 50, "aspect_ratio": 8, "taper": 0.7}}
    )
    assert unpack(c)["components"]["COMPONENT"]["planform"]["span"] == 20


def test_ridge_profile_validation():
    profile = {
        "stations": [0.0, 0.5, 1.0],
        "start": [0.0, 0.01, 0.02],
        "ridge": [0.01, 0.05, 0.06],
        "exponent": [2.5, 2.5, 3.0],
    }
    resolve_config({"type": "lifting_surface", "cap": {"type": "ridged", "profile": profile}})
    for key, value, word in [
        ("stations", [0.0, 0.5, 0.9], "stations"),
        ("exponent", [1.0, 2.5, 3.0], "exponent"),
        ("ridge", [0.0, 0.05, 0.06], "ridge"),
    ]:
        bad = {**profile, key: value}
        with pytest.raises(InputValueError, match=word):
            resolve_config({"type": "lifting_surface", "cap": {"type": "ridged", "profile": bad}})


def test_parts_too_few_are_warned_and_skipped():
    """Fewer than three interior boundaries is a body the generator cannot split: warn, skip."""
    cfg = resolve_config({"type": "body", "parts": [0.0, 0.2, 0.8, 1.0]})
    assert any(w.startswith("COMPONENT.parts:") for w in cfg["warnings"])


@pytest.mark.parametrize(
    "parts", [[0.0, 0.8, 0.2, 0.5, 1.0], [0.0, 0.2, 0.2, 0.5, 0.9, 1.0], [0.0, 0.2, 0.5, 0.9, 1.4]]
)
def test_parts_out_of_order_or_beyond_the_body_are_refused(parts):
    """A malformed list is an input error, not an entry that does not apply: it is refused."""
    with pytest.raises(InputValueError, match=r"COMPONENT\.parts"):
        resolve_config({"type": "body", "parts": parts})

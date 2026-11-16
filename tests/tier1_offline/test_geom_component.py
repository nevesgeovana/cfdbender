"""Geometry: the component parameter record, its exported schema and its pins."""

import copy
import hashlib
import json
from importlib import resources

import pytest
from pydantic import ValidationError

from cfdbender._errors import InputValueError
from cfdbender.geometry import ComponentRecord
from cfdbender.geometry._boundary import (
    deviation_limit_mm,
    export_schema,
    schema_digest,
    sha256_file,
)
from cfdbender.geometry.component import DEVIATION_PROFILES
from cfdbender.geometry.stages import verify_pins

H = "a" * 64


def curve(seq, name="sections/wing.igs", digest=H):
    return {"format": "iges_type_126", "file": {"path": name, "sha256": digest}, "de_sequence": seq}


def deviation(length, size="chord"):
    return {
        "forward_scope": "reconstruction_to_original_everywhere",
        "reverse_scope": "original_to_reconstruction_inside_trimmed_boundary",
        "local_size": size,
        "limit": {
            "rule": "max(floor_mm, local_fraction * local_size_mm)",
            "floor_mm": 0.5,
            "local_fraction": 0.001,
        },
        "stopping_rule": {
            "interval_floor_fraction": 0.01,
            "component_length_mm": length,
            "unresolved_intervals": [],
        },
        "forward": {"max_mm": 0.2, "max_local_limit_ratio": 0.4},
        "reverse": {"max_mm": 0.3, "max_local_limit_ratio": 0.6},
        "evidence": {"path": "fidelity/wing.json", "sha256": "d" * 64},
    }


def lifting():
    splits = {
        "LE": 0.5,
        "max_thickness": {"upper": 0.32, "lower": 0.68},
        "TE": {"upper": 0, "lower": 1},
    }
    return {
        "component_code": "WING_A",
        "version": "1.0.0",
        "units": "mm",
        "source_iges": {"path": "original/wing.igs", "sha256": "b" * 64},
        "kind": "lifting",
        "construction": {
            "method": "tensor_loft",
            "across_station_interpolation": "pchip",
            "surface_degree": {"u": 3, "v": 3},
        },
        "stations": [
            {
                "id": "root",
                "origin": "junction",
                "axis": "span",
                "coordinate": 80,
                "section_curve": curve(1),
                "split_points": splits,
            },
            {
                "id": "mid",
                "origin": "inserted",
                "axis": "span",
                "coordinate": 300,
                "section_curve": curve(2),
                "split_points": splits,
            },
            {
                "id": "tip",
                "origin": "measured",
                "axis": "span",
                "coordinate": 600,
                "section_curve": curve(3),
                "split_points": splits,
            },
        ],
        "break_stations": [],
        "junction_stations": ["root"],
        "te_base": {
            "type": "finite",
            "surface": {
                "format": "iges_type_128",
                "file": {"path": "surfaces/te.igs", "sha256": "c" * 64},
                "de_sequence": 1,
            },
        },
        "measured_deviation": deviation(520),
    }


def revolution(symmetry="full"):
    if symmetry == "full":
        sectors = [
            {"azimuth_deg": a, "t": t} for a, t in [(0, 0), (90, 0.25), (180, 0.5), (270, 0.75)]
        ]
    else:
        sectors = [{"azimuth_deg": a, "t": t} for a, t in [(270, 0), (360, 0.5), (450, 1)]]
    return {
        "component_code": "BODY_A",
        "version": "2.0.0",
        "units": "mm",
        "source_iges": {"path": "original/body.igs", "sha256": "e" * 64},
        "kind": "revolution",
        "construction": {
            "method": "revolution",
            "across_station_interpolation": "cubic",
            "surface_degree": {"u": 2, "v": 3},
        },
        "symmetry": symmetry,
        "stations": [
            {
                "id": "inlet",
                "origin": "measured",
                "axis": "x",
                "coordinate": 120,
                "section_curve": curve(1, "sections/body.igs"),
                "split_points": {"azimuth_sectors": sectors},
            },
            {
                "id": "outlet",
                "origin": "measured",
                "axis": "x",
                "coordinate": 460,
                "section_curve": curve(3, "sections/body.igs"),
                "split_points": {"azimuth_sectors": sectors},
            },
        ],
        "break_stations": [],
        "junction_stations": [],
        "te_base": None,
        "measured_deviation": deviation(340, "diameter"),
    }


def test_lifting_and_revolution_records_validate_and_round_trip():
    for data in (lifting(), revolution(), revolution("half_y0")):
        record = ComponentRecord.model_validate(data)
        again = ComponentRecord.from_json(record.to_json())
        assert again == record
        assert json.loads(record.to_json())["component_code"] == data["component_code"]


def mutated(base, edit):
    data = copy.deepcopy(base())
    edit(data)
    return data


REFUSED = [
    ("undeclared top-level field", lifting, lambda d: d.update(extra=1), "extra"),
    ("undeclared nested field", lifting, lambda d: d["construction"].update(extra=1), "extra"),
    ("lifting with symmetry", lifting, lambda d: d.update(symmetry="full"), "symmetry"),
    ("body without symmetry", revolution, lambda d: d.pop("symmetry"), "symmetry"),
    ("lifting with null te_base", lifting, lambda d: d.update(te_base=None), "te_base"),
    ("body with te_base", revolution, lambda d: d.update(te_base={"type": "sharp"}), "te_base"),
    ("units not mm", lifting, lambda d: d.update(units="m"), "units"),
    ("bad hash", lifting, lambda d: d["source_iges"].update(sha256="xyz"), "sha256"),
    ("duplicate station id", lifting, lambda d: d["stations"][1].update(id="root"), "unique"),
    (
        "coordinates not increasing",
        lifting,
        lambda d: d["stations"][1].update(coordinate=80),
        "increasing",
    ),
    ("wrong axis", lifting, lambda d: d["stations"][0].update(axis="x"), "axis"),
    (
        "unknown junction id",
        lifting,
        lambda d: d.update(junction_stations=["nowhere"]),
        "junction_stations",
    ),
    (
        "splits out of traversal order",
        lifting,
        lambda d: d["stations"][0]["split_points"].update(LE=0.1),
        "traversal",
    ),
    (
        "sector angles not increasing",
        revolution,
        lambda d: d["stations"][0]["split_points"]["azimuth_sectors"].reverse(),
        "strictly increasing",
    ),
    (
        "half body not spanning 270 to 450",
        lambda: revolution("half_y0"),
        lambda d: d["stations"][0]["split_points"]["azimuth_sectors"][0].update(azimuth_deg=260),
        "half_y0",
    ),
    (
        "full body angle at 360",
        revolution,
        lambda d: d["stations"][0]["split_points"]["azimuth_sectors"][-1].update(azimuth_deg=360),
        "full-body",
    ),
    (
        "regions on a non hybrid",
        lifting,
        lambda d: d["construction"].update(regions=[{"x_range": [0, 1]}]),
        "regions",
    ),
    (
        "revolution method on a lifting",
        lifting,
        lambda d: d["construction"].update(method="revolution"),
        "revolution",
    ),
    (
        "wrong local size",
        lifting,
        lambda d: d["measured_deviation"].update(local_size="diameter"),
        "local_size",
    ),
    (
        "component length differs from stations",
        lifting,
        lambda d: d["measured_deviation"]["stopping_rule"].update(component_length_mm=500),
        "station extent",
    ),
    (
        "unresolved interval not adjacent",
        lifting,
        lambda d: d["measured_deviation"]["stopping_rule"].update(
            unresolved_intervals=[{"station_ids": ["root", "tip"], "max_local_limit_ratio": 2.0}]
        ),
        "adjacent",
    ),
    (
        "limit constants differ from the profile",
        lifting,
        lambda d: d["measured_deviation"]["limit"].update(floor_mm=1.0),
        "named profile",
    ),
    (
        "unknown deviation profile",
        lifting,
        lambda d: d["measured_deviation"]["limit"].update(profile="loose"),
        "unknown",
    ),
]


@pytest.mark.parametrize(("label", "base", "edit", "word"), REFUSED, ids=[r[0] for r in REFUSED])
def test_structural_rules_are_refused(label, base, edit, word):
    with pytest.raises(ValidationError, match=word):
        ComponentRecord.model_validate(mutated(base, edit))


def hybrid():
    data = lifting()
    data["construction"].update(
        method="hybrid",
        regions=[{"x_range": [80, 300]}, {"x_range": [300, 600], "join_section_curve": "mid"}],
    )
    return data


def test_hybrid_regions_must_be_contiguous_cover_the_extent_and_name_a_join():
    ComponentRecord.model_validate(hybrid())
    for edit, word in [
        (lambda d: d["construction"]["regions"][1].update(x_range=[310, 600]), "contiguous"),
        (lambda d: d["construction"]["regions"][1].update(x_range=[300, 500]), "cover"),
        (lambda d: d["construction"]["regions"][1].pop("join_section_curve"), "join_section_curve"),
        (
            lambda d: d["construction"]["regions"][1].update(join_section_curve="nowhere"),
            "station id",
        ),
        (lambda d: d["construction"]["regions"][0].update(join_section_curve="root"), "no join"),
        (
            lambda d: d["construction"].update(regions=d["construction"]["regions"][:1]),
            "at least two",
        ),
    ]:
        data = hybrid()
        edit(data)
        with pytest.raises(ValidationError, match=word):
            ComponentRecord.model_validate(data)


def test_from_json_refuses_with_the_library_error():
    with pytest.raises(InputValueError, match="component record"):
        ComponentRecord.from_json('{"component_code": "X"}')


def test_deviation_limit_is_a_named_profile():
    assert DEVIATION_PROFILES["study_default"] == {"floor_mm": 0.5, "local_fraction": 0.001}
    assert deviation_limit_mm(100.0) == 0.5
    assert deviation_limit_mm(2000.0) == pytest.approx(2.0)
    with pytest.raises(InputValueError, match="loose"):
        deviation_limit_mm(100.0, "loose")


def test_pins_pass_for_matching_bytes_and_report_each_failure(tmp_path):
    files = {
        "original/wing.igs": b"source",
        "sections/wing.igs": b"curves",
        "surfaces/te.igs": b"base",
        "fidelity/wing.json": b"{}",
    }
    for name, payload in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    data = lifting()
    data["source_iges"]["sha256"] = hashlib.sha256(b"source").hexdigest()
    for station in data["stations"]:
        station["section_curve"]["file"]["sha256"] = hashlib.sha256(b"curves").hexdigest()
    data["te_base"]["surface"]["file"]["sha256"] = hashlib.sha256(b"base").hexdigest()
    data["measured_deviation"]["evidence"]["sha256"] = hashlib.sha256(b"{}").hexdigest()
    record = ComponentRecord.model_validate(data)
    assert sha256_file(tmp_path / "sections/wing.igs") == hashlib.sha256(b"curves").hexdigest()
    assert len(record.referenced_files()) == 4
    assert verify_pins(record, tmp_path) == []
    (tmp_path / "sections/wing.igs").write_bytes(b"changed")
    (tmp_path / "surfaces/te.igs").unlink()
    problems = verify_pins(record, tmp_path)
    assert problems == [
        "sections/wing.igs: SHA-256 differs from the pin",
        "surfaces/te.igs: missing",
    ]


def test_a_pin_path_may_not_leave_the_asset_root(tmp_path):
    data = lifting()
    data["source_iges"]["path"] = "../outside.igs"
    record = ComponentRecord.model_validate(data)
    assert "../outside.igs: outside the asset root" in verify_pins(record, tmp_path / "root")


def test_packaged_schema_equals_the_export_and_has_a_stable_digest():
    packaged = resources.files("cfdbender.geometry").joinpath("schemas/component.schema.json")
    on_disk = json.loads(packaged.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(export_schema()))
    assert on_disk["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert on_disk["additionalProperties"] is False
    assert schema_digest() == schema_digest()
    assert len(schema_digest()) == 64


def test_the_record_docstring_names_no_function_the_package_no_longer_exports():
    """a role in the public docstring (and so the schema) must resolve in its module."""
    import re

    from cfdbender.geometry import component

    roles = re.findall(r":func:`([\w.]+)`", ComponentRecord.__doc__)
    assert [name for name in roles if not hasattr(component, name)] == []

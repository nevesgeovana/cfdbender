"""Read-back must detect spatially swapped areas in a component IGES.

Reproduction: export faces of areas two and one at the positions of retained
faces of areas one and two, leaving the isolated exports intact.
"""

import json

import pytest

from cfdbender.geometry import _geom_export as export
from cfdbender.mesh._gmsh import open_kernel


def stamp_faces(path, identities):
    """Retain explicit directory identities while replacing geometry in a test."""
    from cfdbender.geometry._iges_write import read_iges, serialize_iges

    document = read_iges(path)
    entities = [e for e in document["entities"] if e["type"] == 144]
    for entity, identity in zip(entities, identities, strict=True):
        entity["d2"][7] = identity
    path.write_text(serialize_iges(document), encoding="ascii")


@pytest.mark.parametrize(
    "case", ["swap", "centred"], ids=["P0600-RRF-2-component-swap", "P0600-RRF-2-centred-swap"]
)
def test_p0600_rrf_component_area_swapped(tmp_path, case):
    kernel = open_kernel("component face swap")
    try:

        def rect(x, width):
            return kernel.add_planar_face(
                [(x, 0, 0), (x + width, 0, 0), (x + width, 1, 0), (x, 1, 0)]
            )

        a, b = rect(0, 1), rect(10, 2)
        changed_a, changed_b = (
            (rect(0, 2), rect(10, 1)) if case == "swap" else (rect(-0.5, 2), rect(10.5, 1))
        )
        kernel.export_iges({"panel": [changed_a, changed_b]}, tmp_path / "panel.igs")
        kernel.export_iges({"panel_0": [a]}, tmp_path / "a.igs")
        kernel.export_iges({"panel_1": [b]}, tmp_path / "b.igs")
    finally:
        kernel.close()
    for file, labels in [
        ("panel.igs", ["F0000001", "F0000002"]),
        ("a.igs", ["F0000001"]),
        ("b.igs", ["F0000002"]),
    ]:
        stamp_faces(tmp_path / file, labels)
    manifest = {
        "iges": "panel.igs",
        "face_orders": {
            "panel.igs": ["F0000001", "F0000002"],
            "a.igs": ["F0000001"],
            "b.igs": ["F0000002"],
        },
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "source_area_m2": 3.0,
                "free_edges": 8,
                "face_records": [
                    {
                        "name": "panel_0",
                        "iges": "a.igs",
                        "area_m2": 1.0,
                        "area_error_m2": 0.0,
                        "iges_identity": "F0000001",
                        "centroid_m": [0.5, 0.5, 0],
                    },
                    {
                        "name": "panel_1",
                        "iges": "b.igs",
                        "area_m2": 2.0,
                        "area_error_m2": 0.0,
                        "iges_identity": "F0000002",
                        "centroid_m": [11, 0.5, 0],
                    },
                ],
            }
        ],
    }
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["status"] == "failed", check
    assert check["where"]["file"] == "panel.igs"
    if case == "centred":
        assert check["where"]["verification"] == "bounded, correspondence verified"
        assert check["measured"] == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "ambiguous-box",
        "missing",
        "duplicate",
        "order",
        "centroid",
        "ambiguous",
    ],
    ids=lambda s: f"P0600-RRF-2-identity-{s}",
)
def test_p0600_rrf_writer_preserves_distinct_trims_and_refuses_identity_damage(tmp_path, damage):
    from test_geom_kernel_export import _written_halves

    from cfdbender.geometry._iges_names import read_iges_names

    _written_halves(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    piece = manifest["pieces"][0]
    faces = piece["face_records"]
    labels = [face["iges_identity"] for face in faces]
    assert len(set(labels)) == 2
    assert len({face["base"] for face in faces}) == 1
    path = tmp_path / piece["file"]
    assert read_iges_names(path).faces == ("plate", "plate")
    if damage == "ambiguous-box":
        faces[0]["centroid_m"][0] += 1e-4
    elif damage == "missing":
        stamp_faces(path, ["", labels[1]])
    elif damage == "duplicate":
        stamp_faces(path, [labels[0], labels[0]])
    elif damage == "order":
        manifest["face_orders"][piece["file"]].reverse()
    elif damage == "centroid":
        faces[0]["centroid_m"][0] += 1
        faces[0]["bbox_m"][0][0] += 1
    elif damage == "ambiguous":
        faces[1]["centroid_m"] = list(faces[0]["centroid_m"])
        faces[1]["bbox_m"] = [list(corner) for corner in faces[0]["bbox_m"]]
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    good = damage == "none"
    assert check["status"] == ("passed" if good else "failed"), check
    if not good:
        for text in ("plate", "face", str(check["measured"]), "0.001", "rebuild"):
            assert text in check["note"].lower()


@pytest.mark.parametrize(
    "mixed", [False, True], ids=["P0600-RRF-2-box-witness", "P0600-RRF-2-mixed-witnesses"]
)
def test_p0600_rrf_each_face_can_have_an_independent_geometric_witness(mixed):
    faces = [
        {
            "iges_identity": "F0000001",
            "centroid_m": [0.2501, 0.5, 0],
            "bbox_m": [[0, 0, 0], [0.5, 1, 0]],
        },
        {
            "iges_identity": "F0000002",
            "centroid_m": [0.75, 0.5, 0],
            "bbox_m": [[0.5001 if mixed else 0.5, 0, 0], [1, 1, 0]],
        },
    ]
    labels = ["F0000001", "F0000002"]
    record = {
        "identities": labels,
        "areas": [0.5, 0.5],
        "centroids": [[0.25, 0.5, 0], [0.75, 0.5, 0]],
        "boxes": [[[0, 0, 0], [0.5, 1, 0]], [[0.5, 0, 0], [1, 1, 0]]],
        "geometry": ["a" * 64, "b" * 64],
    }
    paired, reason = export._corresponding_faces(record, faces, labels, record["geometry"])
    assert paired == faces, reason
    for geometry in (None, ["c" * 64, "b" * 64]):
        paired, reason = export._corresponding_faces(record, faces, labels, geometry)
        assert paired is None
        assert "unchanged recorded trim geometry" in reason

"""Warnings and invalid numerical estimates must fail closed."""

import warnings

import pytest
from scipy.integrate import IntegrationWarning
from test_p0600_rf3fix_trim import Rectangle

from cfdbender._errors import InputArtifactError
from cfdbender.geometry import _geom_area as area
from cfdbender.mesh._cad import Face


@pytest.mark.parametrize("case", ["warning"], ids=["P0600-RF3-4-warning-is-not-proof"])
def test_p0600_rf3fix_integration_warning_refuses(monkeypatch, case):
    def warned(*args, **kwargs):
        warnings.warn("maximum subdivisions reached", IntegrationWarning, stacklevel=2)
        return 1.0, 0.0

    monkeypatch.setattr(area, "dblquad", warned, raising=False)
    monkeypatch.setattr(area, "bounded_area", warned, raising=False)
    with pytest.raises(InputArtifactError, match="(?i)(warning|subdivision|unverified)"):
        area.export_area(Rectangle(), Face(1))


@pytest.mark.parametrize(
    "value,error",
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (1.0, 0.01),
        (1.0, float("nan")),
        (1.0, -1.0),
        (1.0, "missing"),
        (1.0, "source-missing"),
    ],
    ids=[
        "P0600-RF3-4-nan",
        "P0600-RF3-4-inf",
        "P0600-RF3-4-error",
        "P0600-RF3-4-nan-error",
        "P0600-RF3-4-negative-error",
        "P0600-RF3-4-missing-error",
        "P0600-RF3-4-missing-source-error",
    ],
)
def test_p0600_rf3fix_unverified_readback_fails(tmp_path, monkeypatch, value, error):
    from cfdbender.geometry import _geom_export as export

    manifest = {
        "iges": "panel.igs",
        "face_orders": {file: ["F0000001"] for file in ("panel.igs", "panel.face-0.igs")},
        "pieces": [
            {
                "component": "panel",
                "file": "panel.igs",
                "source_area_m2": 1.0,
                "free_edges": 0,
                "face_records": [
                    {
                        "name": "panel_0",
                        "iges": "panel.face-0.igs",
                        "area_m2": 1.0,
                        "area_error_m2": 0.0,
                        "iges_identity": "F0000001",
                        "centroid_m": [0.5, 0.5, 0],
                    }
                ],
            }
        ],
    }
    record = {
        "areas": [value],
        "area_errors": [error],
        "edges": [],
        "pid": 123,
        "identities": ["F0000001"],
        "centroids": [[0.5, 0.5, 0]],
    }
    if error == "missing":
        del record["area_errors"]
    elif error == "source-missing":
        del manifest["pieces"][0]["face_records"][0]["area_error_m2"]
        record["area_errors"] = [0.0]
    monkeypatch.setattr(
        export,
        "_inspect",
        lambda *args: {
            **record,
            "names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
            "property_names": ["panel_0" if args[0].name == "panel.face-0.igs" else "panel"],
        },
    )
    result = export.read_back(tmp_path, manifest, 0.1)
    check = next(c for c in result["checks"] if c["id"] == "area.read_back")
    assert check["status"] == "failed", check

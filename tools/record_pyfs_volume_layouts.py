"""Record the point layouts of another library on declared volume cuts and probe shapes.

Run once with ``PYTHONPATH=src`` and the source folder of the library that defines the layouts on
the path as well. The outputs go to ``tests/data/volume_layouts_reference.json``, which the test
suite compares with :mod:`cfdbender.post.volume_layouts`: rectangular cuts exactly (plain
arithmetic), circular cuts to 1e-12 relative (a sine and a cosine are not bit reproducible
across platforms) and circular probe shapes to one unit of the five-decimal rounding.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pyflightstream.cases.pproc import ProbeCircle, ProbeLine, ProbeRectangle, VolumeSectionSpec
from pyflightstream.cases.workflows import _probes  # type: ignore[import-not-found]
from pyflightstream.probes import layouts  # type: ignore[import-not-found]

SECTIONS: list[dict[str, Any]] = [
    {"shape": "rectangle", "plane": "XY", "corners_m": [0.0, -1.0, 2.0, 1.5], "points": [4, 5]},
    {"shape": "rectangle", "plane": "XZ", "corners_m": [-1.0, 0.5, 1.0, 2.5], "offset_m": 0.75},
    {
        "shape": "rectangle",
        "plane": "YZ",
        "corners_m": [0.1, 0.2, 0.9, 1.7],
        "points": [3, 4],
        "refinement_layers": 3,
        "offset_m": -0.4,
    },
    {"shape": "circle", "plane": "YZ", "radii_m": [0.0, 1.5], "points": [4, 7], "offset_m": 2.0},
    {"shape": "circle", "plane": "XY", "radii_m": [0.3, 1.2], "points": [3, 5]},
    {"shape": "circle", "plane": "XZ", "radii_m": [0.25, 0.75], "points": [5, 9], "offset_m": 0.1},
]

LINES = [{"start": [0.0, 0.0, 0.0], "end": [1.0, 2.0, 3.0], "points": 7, "scale": 1.0}]
RECTANGLES = [
    {
        "origin": [0.0, -1.0, -2.0],
        "along_u": [0.0, 1.0, -2.0],
        "along_v": [0.0, -1.0, 2.0],
        "points_u": 3,
        "points_v": 4,
        "scale": 1.0,
    },
    {
        "origin": [0.3, 0.1, 0.2],
        "along_u": [1.3, 0.1, 0.4],
        "along_v": [0.3, 1.9, 0.2],
        "points_u": 4,
        "points_v": 3,
        "scale": 0.5,
    },
]
CIRCLES = [
    {
        "center": [0.0, 0.0, 0.0],
        "normal": [1.0, 0.0, 0.0],
        "radius": 1.0,
        "points_radial": 3,
        "points_azimuth": 8,
        "scale": 1.0,
    },
    {
        "center": [0.4, -0.2, 0.9],
        "normal": [0.3, 0.5, -0.8],
        "radius": 0.7,
        "points_radial": 4,
        "points_azimuth": 6,
        "scale": 2.0,
    },
]


class _Script:
    """Records the probe points the library places, as its script object does."""

    def __init__(self) -> None:
        self.probe_points: list[tuple[Any, ...]] = []
        self.probe_field_layout: list[dict[str, Any]] = []
        self.simulation_length_unit = "METER"

    def emit(self, *args: Any, **kwargs: Any) -> None:
        """Take a script line and ignore it."""


def _section_points(declaration: dict[str, Any]) -> list[list[float]]:
    spec = VolumeSectionSpec(frame="REFERENCE", kind="instantaneous", **declaration)
    case = SimpleNamespace(pproc=SimpleNamespace(volume_section=spec, probes=[]))
    script = _Script()
    _probes._a_rotor_frame_the_row_leaves_out = lambda *a, **k: False
    _probes._from_metres = lambda *a, **k: 1.0
    _probes._pproc_sampled_volume(case, script, None, 0, unsteady=True)
    return [[float(x) for x in point[1:4]] for point in script.probe_points]


def main() -> None:
    """Write the reference file."""
    out: dict[str, Any] = {"sections": [], "lines": [], "rectangles": [], "circles": []}
    for declaration in SECTIONS:
        out["sections"].append({"declaration": declaration, "points": _section_points(declaration)})
    for line in LINES:
        spec = ProbeLine(start=line["start"], end=line["end"])
        points = layouts.line_points(
            spec.start, end=spec.end, points=line["points"], scale=line["scale"]
        )
        out["lines"].append({"declaration": line, "points": points})
    for rect in RECTANGLES:
        spec_r = ProbeRectangle(
            origin=rect["origin"],
            along_u=rect["along_u"],
            along_v=rect["along_v"],
            points_u=rect["points_u"],
            points_v=rect["points_v"],
        )
        out["rectangles"].append(
            {"declaration": rect, "points": layouts.rectangle_points(spec_r, scale=rect["scale"])}
        )
    for circ in CIRCLES:
        spec_c = ProbeCircle(
            center=circ["center"],
            normal=circ["normal"],
            radius=circ["radius"],
            points_radial=circ["points_radial"],
            points_azimuth=circ["points_azimuth"],
        )
        out["circles"].append(
            {"declaration": circ, "points": layouts.circle_points(spec_c, scale=circ["scale"])}
        )
    target = (
        Path(__file__).resolve().parents[1] / "tests" / "data" / "volume_layouts_reference.json"
    )
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    shapes = len(LINES) + len(RECTANGLES) + len(CIRCLES)
    print(f"wrote {target}: {len(out['sections'])} cuts, {shapes} probe shapes")


if __name__ == "__main__":
    main()

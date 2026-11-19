"""The shared names of the blade-section cuts: choices, the section record, OBJ text input."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cfdbender._errors import InputValueError

Vec = tuple[float, float, float]
AXES: dict[str, Vec] = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
SPACINGS = ("cosine", "uniform")
#: ``plane`` follows the blade's skew with the propeller slicer; ``cylinder`` cuts r = const and
#: unrolls the cut.
CUTS = ("plane", "cylinder")
ROTATIONS = ("ccw_from_front", "cw_from_front")


class BladeSliceError(InputValueError):
    """The blade cannot be sliced with the requested convention."""


@dataclass(frozen=True)
class BladeSection:
    """One radial station: radius and chord in model units, beta in degrees, unit-chord shape."""

    radius: float
    chord: float
    beta_deg: float
    shape: tuple[tuple[float, float], ...]


def parse_obj(text: str) -> tuple[list[Vec], list[tuple[int, int, int]]]:
    """Read vertices and triangulated faces from Wavefront OBJ text.

    Parameters
    ----------
    text : str
        OBJ content. Only ``v`` and ``f`` records are read; polygons are fanned into
        triangles and negative indices count back from the last vertex.

    Returns
    -------
    tuple
        Vertex list and zero-based triangle index triples, in file order.

    Raises
    ------
    BladeSliceError
        A ``v`` or ``f`` record that cannot be read.
    """
    vertices: list[Vec] = []
    faces: list[tuple[int, int, int]] = []
    for number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "v":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                ids = []
                for token in parts[1:]:
                    index = int(token.split("/")[0])
                    ids.append(index - 1 if index > 0 else len(vertices) + index)
                if len(ids) < 3 or any(not 0 <= i < len(vertices) for i in ids):
                    raise ValueError("face needs three valid vertex references")
                faces.extend((ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1))
        except (ValueError, IndexError) as error:
            raise BladeSliceError(f"OBJ line {number}: cannot read {line.strip()!r}") from error
    return vertices, faces


def station_radii(r_min: float, r_max: float, count: int, spacing: str) -> list[float]:
    """Return ``count`` cell-centered radii between the extremes, never on the ends.

    Parameters
    ----------
    r_min, r_max : float
        Inner and outer radius of the blade.
    count : int
        Number of stations.
    spacing : str
        ``"uniform"`` or ``"cosine"`` (clustered toward both ends).

    Returns
    -------
    list of float
        The radii in ascending order.
    """
    if spacing not in SPACINGS:
        raise BladeSliceError(f"spacing must be one of: {', '.join(SPACINGS)}")
    if count < 1:
        raise BladeSliceError("n_sections must be a positive integer")
    result = []
    for i in range(count):
        fraction = (i + 0.5) / count
        if spacing == "cosine":
            fraction = 0.5 * (1.0 - math.cos(math.pi * fraction))
        result.append(r_min + (r_max - r_min) * fraction)
    return result

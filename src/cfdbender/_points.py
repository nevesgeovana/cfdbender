"""Point files shared by the geometry and mesh layers."""

from __future__ import annotations

from pathlib import Path

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._decimal import plain_decimal
from cfdbender._errors import InputArtifactError
from cfdbender._lengths import scale

__all__ = ["write_trailing_edge_points"]


def _coordinates(points: object) -> NDArray[numpy.float64]:
    """Read an (n, 3) array of finite coordinates, at least one row, or refuse."""
    try:
        array = numpy.asarray(points, dtype=float)
    except (TypeError, ValueError) as error:
        raise InputArtifactError(
            "the point list could not be read as an array of coordinates: "
            f"{error}. It is an (n, 3) array of numbers, or any nested sequence numpy can "
            "read as one",
            kind="trailing_edges",
        ) from error
    if array.ndim != 2 or array.shape[1] != 3:
        raise InputArtifactError(
            f"the point list has shape {tuple(array.shape)}; it carries one row of three "
            "coordinates per point, so the array is (n, 3)",
            kind="trailing_edges",
        )
    if array.shape[0] == 0:
        raise InputArtifactError(
            "the point list carries no points; a trailing-edge points file needs at least one",
            kind="trailing_edges",
        )
    if not numpy.isfinite(array).all():
        row = int(numpy.argmax(~numpy.isfinite(array).all(axis=1))) + 1
        raise InputArtifactError(
            f"the point list carries a coordinate that is not a finite number, first at "
            f"row {row} of {array.shape[0]}",
            kind="trailing_edges",
        )
    return array


def write_trailing_edge_points(
    path: str | Path, points: object, *, unit: str, overwrite: bool = False
) -> Path:
    """Write a trailing-edge points file: its unit, then one mid-point per line.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file.
    points : array_like
        The mid-points of the mesh edges on the trailing edge, shape (n, 3), in ``unit``.
    unit : str
        The unit the points are in, a length unit word known to :func:`cfdbender._lengths.scale`,
        written alone on the first line.
    overwrite : bool
        Replace an existing destination.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    InputArtifactError
        No points, a shape other than (n, 3), a coordinate that is not finite, a unit with
        no scale, or an existing file without ``overwrite``. Each fires before the file is
        opened.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     written = write_trailing_edge_points(
    ...         Path(folder) / "edge.te.txt", [(1.0, -3.75, 0.0)], unit="METER"
    ...     )
    ...     print(written.read_text().split())
    ['METER', '1.0,-3.75,0.0']
    """
    destination = Path(path)
    array = _coordinates(points)
    scale(unit, unit)
    if destination.exists() and not overwrite:
        raise InputArtifactError(
            f"{destination} already exists. Pass overwrite to replace it deliberately, or "
            "write under a name that carries the geometry: one path is one file.",
            kind="trailing_edges",
        )
    rows = [",".join(plain_decimal(float(value)) for value in point) for point in array]
    _textio.write_text(destination, "\n".join([unit, *rows]) + "\n")
    return destination

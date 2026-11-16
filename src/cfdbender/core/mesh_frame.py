"""The mesh frame: every mesh is expected in x aft, y right, z up; another is declared.

A mesh built in another axis-aligned frame states what its +x, +y and +z point to, as
three words out of ``aft``, ``forward``, ``right``, ``left``, ``up`` and ``down``
(case-insensitive), for example ``["forward", "left", "up"]``. The three must name three
different axes and form a right-handed set: a left-handed declaration (a mirrored mesh)
is refused by name, because converting it would reverse every face's orientation. An
absent declaration is the expected frame and converts nothing.

The conversion maps the vertices by the signed permutation ``C_E/M`` (rows: the expected
axes in terms of the declared ones) before any operation reads the mesh; face order,
group names and group order are kept. The references (moment point, rotor hubs and axes)
are always stated in the expected frame. Only an OBJ mesh is converted offline; another
format declared in another frame is refused, because its file is the solver's own.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cfdbender.core.axes import Matrix

#: The frame every mesh is expected in: what its +x, +y and +z point to.
EXPECTED_FRAME = ("aft", "right", "up")
_WORDS = {
    "aft": (1.0, 0.0, 0.0),
    "forward": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}
_HINT = "accepted words: aft, forward, right, left, up, down"


def _determinant(m: Matrix) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def declared_frame(declared: Sequence[str] | None) -> tuple[str, str, str]:
    """Return the declaration in lower case, or the expected frame when none is given.

    Raises
    ------
    ValueError
        If it is not three accepted words naming three different axes, right-handed.
    """
    if declared is None:
        return EXPECTED_FRAME
    if isinstance(declared, str) or len(declared) != 3:
        raise ValueError(
            f"mesh_frame {declared!r} must list three words, what the mesh's +x, +y and +z "
            f"point to ({_HINT})"
        )
    words = tuple(str(word).strip().casefold() for word in declared)
    unknown = [word for word in words if word not in _WORDS]
    if unknown:
        raise ValueError(f"mesh_frame names {', '.join(unknown)}; {_HINT}")
    axes = [next(i for i, c in enumerate(_WORDS[word]) if c != 0.0) for word in words]
    if len(set(axes)) != 3:
        raise ValueError(
            f"mesh_frame {list(words)} names one axis twice; name one of aft or forward, one "
            "of right or left and one of up or down"
        )
    if _determinant(_matrix(words)) < 0.0:
        raise ValueError(
            f"mesh_frame {list(words)} is left-handed (a mirrored mesh); converting it would "
            "reverse every face's orientation, so mirror the mesh in the modeller instead"
        )
    return (words[0], words[1], words[2])


def _matrix(words: Sequence[str]) -> Matrix:
    columns = [_WORDS[word] for word in words]
    rows = [tuple(columns[j][i] for j in range(3)) for i in range(3)]
    return (rows[0], rows[1], rows[2])  # type: ignore[return-value]


def mesh_frame_matrix(declared: Sequence[str] | None) -> Matrix:
    """Return ``C_E/M``: a vertex in the declared frame to the expected frame."""
    return _matrix(declared_frame(declared))


def convert_point(point: Sequence[float], declared: Sequence[str] | None) -> tuple[float, ...]:
    """Return one point of the declared frame in the expected frame."""
    m = mesh_frame_matrix(declared)
    return tuple(sum(m[i][j] * float(point[j]) for j in range(3)) + 0.0 for i in range(3))


def convert_mesh_text(name: str, text: str, declared: Sequence[str] | None) -> str:
    """Return the mesh text converted to the expected frame, faces and groups kept.

    Parameters
    ----------
    name : str
        The mesh file name; only an OBJ file is converted.
    text : str
        The file's text.
    declared : sequence of str or None
        What the mesh's +x, +y and +z point to; None is the expected frame.

    Returns
    -------
    str
        The text with every ``v`` and ``vn`` line mapped; every other line unchanged.

    Raises
    ------
    ValueError
        If the declaration is refused, or the file is not an OBJ and is declared in
        another frame.
    """
    words = declared_frame(declared)
    if words == EXPECTED_FRAME:
        return text
    if not name.casefold().endswith(".obj"):
        raise ValueError(
            f"{name} is declared in mesh_frame {list(words)}; only an OBJ mesh is converted "
            "offline, so build this mesh in x aft, y right, z up"
        )
    lines = []
    for line in text.splitlines(keepends=True):
        fields = line.split()
        if fields[:1] in (["v"], ["vn"]) and len(fields) >= 4:
            body = line.rstrip(chr(13) + chr(10))
            end = line[len(body) :]
            point = convert_point([float(c) for c in fields[1:4]], words)
            rest = fields[4:]
            lines.append(" ".join([fields[0], *(repr(c) for c in point), *rest]) + end)
        else:
            lines.append(line)
    return "".join(lines)


def mesh_frame_record(
    declared: Sequence[str] | None, source: bytes, converted: bytes
) -> dict[str, object]:
    """Return the run record's ``mesh_frame`` entry: declaration, matrix and both digests."""
    return {
        "declared": list(declared_frame(declared)),
        "matrix": [list(row) for row in mesh_frame_matrix(declared)],
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "converted_sha256": hashlib.sha256(converted).hexdigest(),
    }

"""Convert an AFLR3 UGRID mesh (.b8.ugrid or .b8.ugrid.gz) and its .mapbc into a native SU2 mesh.

Usage: python ugrid_to_su2.py <mesh.b8.ugrid[.gz]> <mesh.mapbc> <out.su2>

The .b8 variant is big-endian with 4-byte integers and 8-byte reals. Surface triangles and
quadrilaterals are grouped into markers by their surface tag, named from the .mapbc label
(lower case, spaces as underscores, prefixed with the tag when two tags share a label).
Element types follow VTK numbering as SU2 expects: triangle 5, quadrilateral 9, tetrahedron 10,
hexahedron 12, prism 13, pyramid 14. UGRID node order is the SU2 order for every type except
the pyramid and the prism, which are reordered (see write_su2).
"""

from __future__ import annotations

import gzip
import sys
from array import array
from pathlib import Path


def _read(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def _rows(flat: array, width: int, shift: int = 0) -> list[list[int]]:
    return [[v - shift for v in flat[i : i + width]] for i in range(0, len(flat), width)]


def read_ugrid(path: Path):
    """Return nodes, surface triangles, quads, surface tags and the four volume blocks.

    AFLR3 .b8: big-endian, 4-byte integers, 8-byte reals; read with the standard library only.
    """
    raw = _read(path)
    off = 0
    swap = sys.byteorder == "little"

    def take(code: str, count: int) -> array:
        nonlocal off
        arr = array(code)
        size = arr.itemsize * count
        arr.frombytes(raw[off : off + size])
        if swap:
            arr.byteswap()
        off += size
        return arr

    nn, ntri, nquad, ntet, npyr, npri, nhex = take("i", 7)
    xyz = _rows(take("d", 3 * nn), 3)
    tri = _rows(take("i", 3 * ntri), 3, 1)
    quad = _rows(take("i", 4 * nquad), 4, 1)
    tags = list(take("i", ntri + nquad))
    tet = _rows(take("i", 4 * ntet), 4, 1)
    pyr = _rows(take("i", 5 * npyr), 5, 1)
    pri = _rows(take("i", 6 * npri), 6, 1)
    hexa = _rows(take("i", 8 * nhex), 8, 1)
    return xyz, tri, quad, tags, tet, pyr, pri, hexa


def read_mapbc(path: Path) -> dict[int, str]:
    """Map each surface tag to a marker name made unique when two tags share a label."""
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    names: dict[int, str] = {}
    for parts in lines[1:]:
        tag = int(parts[0])
        names[tag] = "_".join(parts[2:]).lower().replace("-", "_")
    seen: dict[str, int] = {}
    for name in names.values():
        seen[name] = seen.get(name, 0) + 1
    candidates = {t: (f"{n}_{t}" if seen[n] > 1 else n) for t, n in names.items()}
    return _unique_names(candidates)


def _unique_names(candidates: dict[int, str]) -> dict[int, str]:
    candidates = candidates.copy()
    used: set[str] = set()
    for tag in sorted(candidates):
        name = candidates[tag]
        while name in used:
            name = f"{name}_{tag}"
        candidates[tag] = name
        used.add(name)
    return candidates


def write_su2(out: Path, xyz, tri, quad, tags, tet, pyr, pri, hexa, names) -> None:
    """Write the native SU2 mesh with one marker per surface tag."""
    # Prism reorder verified: SU2 8.6.0 reports every volume element correctly oriented on
    # the CRM-HL 2D grid. The pyramid reorder is UNVERIFIED (no pyramid in that grid).
    pyr_su2 = [[r[0], r[3], r[4], r[1], r[2]] for r in pyr]
    pri_su2 = [[r[0], r[2], r[1], r[3], r[5], r[4]] for r in pri]
    with out.open("w", newline="\n") as f:
        f.write("NDIME= 3\n")
        nel = len(tet) + len(pyr) + len(pri) + len(hexa)
        f.write(f"NELEM= {nel}\n")
        for vtk, block in ((10, tet), (14, pyr_su2), (13, pri_su2), (12, hexa)):
            for row in block:
                f.write(f"{vtk} " + " ".join(str(int(v)) for v in row) + "\n")
        f.write(f"NPOIN= {len(xyz)}\n")
        for x, y, z in xyz:
            f.write(f"{float(x):.17g} {float(y):.17g} {float(z):.17g}\n")
        surf = [(5, r) for r in tri] + [(9, r) for r in quad]
        marks = sorted(set(int(t) for t in tags))
        marker_names = _unique_names({m: names.get(m, f"tag_{m}") for m in marks})
        f.write(f"NMARK= {len(marks)}\n")
        for m in marks:
            rows = [s for s, t in zip(surf, tags, strict=True) if int(t) == m]
            f.write(f"MARKER_TAG= {marker_names[m]}\nMARKER_ELEMS= {len(rows)}\n")
            for vtk, r in rows:
                f.write(f"{vtk} " + " ".join(str(int(v)) for v in r) + "\n")


def main() -> None:
    """Convert the mesh named on the command line and print its element counts."""
    mesh, mapbc, out = (Path(a) for a in sys.argv[1:4])
    parts = read_ugrid(mesh)
    names = read_mapbc(mapbc)
    write_su2(out, *parts, names)
    xyz, tri, quad, tags, tet, pyr, pri, hexa = parts
    marks = {int(t) for t in tags}
    marker_names = _unique_names({m: names.get(m, f"tag_{m}") for m in marks})
    print(
        f"nodes {len(xyz)}, tet {len(tet)}, pyr {len(pyr)}, prism {len(pri)}, hex {len(hexa)}, "
        f"surface tri {len(tri)} quad {len(quad)}, "
        f"markers {sorted(marker_names.values())}"
    )


if __name__ == "__main__":
    main()

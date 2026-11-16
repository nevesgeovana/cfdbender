"""Shared test support: read a 2D .su2 and a gmsh 2.2 .msh back, independently of the writer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


def read_su2(path: Path) -> dict:
    """Return the dimension, the quadrilaterals, the points and the markers of a 2D .su2 file."""
    lines = path.read_text(encoding="ascii").splitlines()
    position = 0

    def take(key: str) -> int:
        nonlocal position
        head, _, value = lines[position].partition("=")
        assert head.strip() == key, (key, lines[position])
        position += 1
        return int(value)

    dimension = take("NDIME")
    cells = []
    for _ in range(take("NELEM")):
        parts = lines[position].split()
        position += 1
        assert parts[0] == "9", parts
        cells.append([int(p) for p in parts[1:5]])
    points = []
    for _ in range(take("NPOIN")):
        parts = lines[position].split()
        position += 1
        points.append([float(parts[0]), float(parts[1])])
    markers: dict[str, list[tuple[int, int]]] = {}
    for _ in range(take("NMARK")):
        tag = lines[position].partition("=")[2].strip()
        position += 1
        edges = []
        for _ in range(take("MARKER_ELEMS")):
            parts = lines[position].split()
            position += 1
            assert parts[0] == "3", parts
            edges.append((int(parts[1]), int(parts[2])))
        markers[tag] = edges
    assert position == len(lines)
    return {
        "dimension": dimension,
        "cells": np.array(cells),
        "points": np.array(points),
        "markers": markers,
    }


def edge_use(cells: np.ndarray) -> Counter:
    """Return how many cells use each undirected edge."""
    used: Counter = Counter()
    for cell in cells:
        for a, b in zip(cell, np.roll(cell, -1), strict=True):
            used[(min(a, b), max(a, b))] += 1
    return used


def signed_areas(cells: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return the shoelace area of every quadrilateral, positive when counter-clockwise."""
    xy = points[cells]
    x, y = xy[..., 0], xy[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def off_wall_lengths(cells: np.ndarray, points: np.ndarray, wall: list) -> np.ndarray:
    """Return the length of the two edges of every wall cell that leave its wall edge."""
    marked = {(min(a, b), max(a, b)) for a, b in wall}
    lengths = {}
    for cell in cells:
        ring = [int(v) for v in cell]
        for k in range(4):
            a, b = ring[k], ring[(k + 1) % 4]
            if (min(a, b), max(a, b)) in marked:
                for p, q in ((a, ring[k - 1]), (b, ring[(k + 2) % 4])):
                    lengths[(min(p, q), max(p, q))] = float(np.linalg.norm(points[p] - points[q]))
    return np.array(sorted(lengths.values()))


def read_msh(path: Path) -> dict:
    """Return the physical names, the node z values and the element type counts of a .msh 2.2."""
    lines = path.read_text(encoding="ascii").splitlines()
    names = {}
    start = lines.index("$PhysicalNames")
    for line in lines[start + 2 : start + 2 + int(lines[start + 1])]:
        dim, tag, name = line.split()
        names[name.strip('"')] = (int(dim), int(tag))
    start = lines.index("$Nodes")
    rows = [line.split()[1:] for line in lines[start + 2 : start + 2 + int(lines[start + 1])]]
    zs = {round(float(row[2]), 12) for row in rows}
    distinct = len({tuple(round(float(v), 12) for v in row) for row in rows})
    start = lines.index("$Elements")
    types: Counter = Counter()
    groups: Counter = Counter()
    for line in lines[start + 2 : start + 2 + int(lines[start + 1])]:
        parts = line.split()
        types[int(parts[1])] += 1
        groups[int(parts[3])] += 1
    return {
        "names": names,
        "z": zs,
        "nodes": len(rows),
        "distinct": distinct,
        "types": types,
        "groups": groups,
        "head": lines[:3],
    }

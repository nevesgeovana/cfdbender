"""Winslow smoothing of an O-grid, the smoother of the hub ring rebuild."""

from __future__ import annotations

import numpy
from numpy.typing import NDArray

__all__ = ["winslow"]

Array = NDArray[numpy.float64]


def winslow(grid: Array, sweeps: int = 3000, relax: float = 0.4) -> Array:
    """Winslow smoothing of an O-grid (layers, loop, 2), periodic along the loop.

    The inner and outer loops stay fixed. Plain Laplace smoothing in index space folded on the
    same loop, which is why this one is used.
    """
    g = grid.copy()
    for _ in range(sweeps):
        up, down, mid = g[2:], g[:-2], g[1:-1]
        ahead, back = numpy.roll(mid, -1, axis=1), numpy.roll(mid, 1, axis=1)
        d_xi, d_eta = 0.5 * (ahead - back), 0.5 * (up - down)
        alpha = numpy.sum(d_eta * d_eta, axis=2, keepdims=True)
        beta = numpy.sum(d_xi * d_eta, axis=2, keepdims=True)
        gamma = numpy.sum(d_xi * d_xi, axis=2, keepdims=True)
        cross = 0.25 * (
            numpy.roll(up, -1, axis=1)
            - numpy.roll(up, 1, axis=1)
            - numpy.roll(down, -1, axis=1)
            + numpy.roll(down, 1, axis=1)
        )
        new = (alpha * (ahead + back) + gamma * (up + down) - 2.0 * beta * cross) / (
            2.0 * (alpha + gamma)
        )
        g[1:-1] = mid + relax * (new - mid)
    return g

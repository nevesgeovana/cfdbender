"""Section polars: XFOIL keystrokes, polar rows and the per-station QPROP fit."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Fit:
    """Per-station QPROP section parameters; the lift slope is per radian."""

    cl0: float
    cla: float
    clmin: float
    clmax: float
    cd0: float
    cd2u: float
    cd2l: float
    clcd0: float


def polar_script(
    *,
    reynolds: float,
    mach: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    ncrit: float,
    iterations: int,
) -> str:
    """Return deterministic XFOIL keystrokes for one viscous alpha sweep.

    The section outline is read from ``airfoil.dat`` and the polar is saved to
    ``polar.txt`` in the working directory.
    """
    lines = ["PLOP", "G", "", "LOAD airfoil.dat", "", "PANE", "OPER"]
    lines += [f"VISC {reynolds:.0f}", f"MACH {mach:.3f}", f"ITER {iterations}"]
    if ncrit != 9:
        lines += ["VPAR", f"N {ncrit:.15g}", ""]
    lines += ["PACC", "polar.txt", ""]
    lines += [f"ASEQ {alpha_start:.15g} {alpha_end:.15g} {alpha_step:.15g}", "", "QUIT", ""]
    return "\n".join(lines)


def read_polar_rows(text: str) -> list[tuple[float, float, float]]:
    """Return (alpha, CL, CD) from every numeric row of an XFOIL polar text."""
    rows = []
    for line in text.splitlines():
        cells = line.split()
        if len(cells) < 3:
            continue
        try:
            row = (float(cells[0]), float(cells[1]), float(cells[2]))
        except ValueError:
            continue
        if all(map(math.isfinite, row)):
            rows.append(row)
    return rows


def _drag_curvature(points: list[tuple[float, float]], cd0: float, clcd0: float) -> float:
    numerator = sum((d - cd0) * (c - clcd0) ** 2 for c, d in points)
    denominator = sum((c - clcd0) ** 4 for c, _ in points)
    return max(numerator / denominator, 0.0) if denominator else 0.02


def fit_polar(rows: list[tuple[float, float, float]]) -> Fit | None:
    """Fit CL0, CL_a, CLmin, CLmax, CD0, CD2u, CD2l and CLCD0 to a polar.

    The lift line uses alpha between -4 and 6 degrees (at least three points);
    CD0 and CLCD0 are the minimum-drag row; CD2u and CD2l are least-squares
    curvatures of drag above and below it, ignoring lift within 0.05 of CLCD0.
    Returns None when the linear range has too few points.
    """
    linear = [(a, cl) for a, cl, _ in rows if -4.0 <= a <= 6.0]
    if len(linear) < 3:
        return None
    mean_a = sum(a for a, _ in linear) / len(linear)
    mean_cl = sum(c for _, c in linear) / len(linear)
    spread = sum((a - mean_a) ** 2 for a, _ in linear)
    slope = sum((a - mean_a) * (c - mean_cl) for a, c in linear) / spread
    lifts = [cl for _, cl, _ in rows]
    best = min(rows, key=lambda row: row[2])
    clcd0, cd0 = best[1], best[2]
    upper = [(c, d) for _, c, d in rows if c > clcd0 + 0.05]
    lower = [(c, d) for _, c, d in rows if c < clcd0 - 0.05]
    return Fit(
        cl0=mean_cl - slope * mean_a,
        cla=slope * 180.0 / math.pi,
        clmin=min(lifts),
        clmax=max(lifts),
        cd0=cd0,
        cd2u=_drag_curvature(upper, cd0, clcd0),
        cd2l=_drag_curvature(lower, cd0, clcd0),
        clcd0=clcd0,
    )

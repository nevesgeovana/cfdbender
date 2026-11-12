"""Render the QPROP propeller and flight-condition files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cfdbender.solvers.qprop._polar import Fit


@dataclass(frozen=True)
class Station:
    """One blade station: radius and chord in metres, beta in degrees."""

    radius: float
    chord: float
    beta: float
    fit: Fit
    reynolds: float


def render_prop(
    title: str,
    blades: int,
    radius: float,
    stations: Sequence[Station],
    *,
    beta_add: float,
    re_exponent: float = -0.5,
) -> str:
    """Return the propeller file text.

    The global section data come from the middle station, then each station
    line carries its own section parameters.

    Parameters
    ----------
    title : str
        First line of the file.
    blades : int
        Number of blades.
    radius : float
        Tip radius in metres.
    stations : sequence of Station
        Stations in ascending radius.
    beta_add : float
        Pitch offset in degrees added to every station (QPROP ``Badd``).
    re_exponent : float
        Reynolds exponent of the drag law.
    """
    mid = stations[len(stations) // 2].fit
    mid_re = stations[len(stations) // 2].reynolds
    lines = [
        title,
        "",
        f" {blades}  {radius:.4f}   ! Nblades  R",
        "",
        f" {mid.cl0:.4f} {mid.cla:.4f}  ! CL0 CL_a",
        f" {mid.clmin:.3f} {mid.clmax:.3f}  ! CLmin CLmax",
        "",
        f" {mid.cd0:.5f} {mid.cd2u:.4f} {mid.cd2l:.4f} {mid.clcd0:.3f}  ! CD0 CD2u CD2l CLCD0",
        f" {mid_re:.0f} {re_exponent:g}  ! REref REexp",
        "",
        " 1.0 1.0 1.0  ! Rfac Cfac Bfac",
        f" 0.0 0.0 {beta_add:.2f}  ! Radd Cadd Badd",
        "",
        "# r chord beta CL0 CL_a CLmin CLmax CD0 CD2u CD2l CLCD0 REref REexp",
    ]
    for s in stations:
        f = s.fit
        lines.append(
            f"{s.radius:.5f} {s.chord:.5f} {s.beta:.3f} {f.cl0:.4f} {f.cla:.4f} "
            f"{f.clmin:.3f} {f.clmax:.3f} {f.cd0:.5f} {f.cd2u:.4f} {f.cd2l:.4f} "
            f"{f.clcd0:.3f} {s.reynolds:.0f} {re_exponent:g}"
        )
    return "\n".join(lines) + "\n"


def render_qcon(rho: float, mu: float, a: float) -> str:
    """Return the flight-condition file read by QPROP as ``qcon.def``."""
    return f" {rho:.6g}  ! rho\n {mu:.5e}  ! mu\n {a:.6g}  ! a\n"

"""A small survey-lattice builder for the far-field core tests.

The core integrates the samples of a cylindrical survey lattice and reads only a few
attributes from it (see ``cfdbender.core.farfield.fields.ProbeLattice``). This module
builds such a lattice for the tests: annular stations, ring edges clustered near the hub
and at the tip, uniform azimuths and an optional lateral cylinder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurveyLattice:
    """The attributes of a survey lattice the far-field core reads."""

    tip_radius: float
    stations: tuple[float, ...]
    ring_edges: tuple[float, ...]
    n_psi: int
    lateral_radius: float | None
    lateral_stations: tuple[float, ...]

    @property
    def n_r(self) -> int:
        return len(self.ring_edges) - 1

    @property
    def ring_centers(self) -> np.ndarray:
        edges = np.asarray(self.ring_edges)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def psi(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.n_psi) / self.n_psi


def clustered_edges(
    r_inner: float, r_outer: float, n_r: int, hub_width: float, tip_width: float
) -> np.ndarray:
    """Ring edges from the inverse of a density with a hub bump and a tip bump."""
    fine = np.linspace(r_inner, r_outer, 4001)
    density = (
        1.0
        + 2.0 * np.exp(-(((fine - r_inner) / hub_width) ** 2))
        + 3.0 * np.exp(-(((fine - 1.0) / tip_width) ** 2))
    )
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))])
    edges = np.interp(np.linspace(0.0, cdf[-1], n_r + 1), cdf, fine)
    edges[0], edges[-1] = r_inner, r_outer
    return edges


def build_lattice(
    *,
    tip_radius: float,
    stations: tuple[float, ...] = (-2.0, -0.2, 0.2, 0.5, 1.0, 1.5, 2.0),
    n_r: int = 40,
    r_inner: float = 0.05,
    r_outer: float = 2.5,
    n_psi: int = 72,
    lateral_radius: float | None = 2.5,
    lateral_stations: tuple[float, ...] | None = None,
    hub_width: float = 0.15,
    tip_width: float = 0.10,
) -> SurveyLattice:
    """Build a survey lattice with the dense defaults of the reference survey."""
    lateral: tuple[float, ...] = ()
    if lateral_radius is not None:
        lateral = tuple(stations) if lateral_stations is None else tuple(lateral_stations)
    return SurveyLattice(
        tip_radius=tip_radius,
        stations=tuple(stations),
        ring_edges=tuple(clustered_edges(r_inner, r_outer, n_r, hub_width, tip_width)),
        n_psi=n_psi,
        lateral_radius=lateral_radius,
        lateral_stations=lateral,
    )

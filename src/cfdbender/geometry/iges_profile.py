"""The IGES profile a CAD reader is known to accept, and what is not known.

A plain description with no kernel and no file access: :mod:`cfdbender.geometry._geom_gate`
measures a file against it. Every value here is either observed (a file of that shape was read
by a CAD reader) or ``None``, which means no source states a limit and a file beyond the
observed shape is warned about, never refused. Only a solid entity is refused, because the
package builds surfaces only.

Observed: non-rational, clamped, bicubic parametric surfaces (entity 128), trimmed by an outer
boundary (entities 142 and 102) and, where a face has holes, inner boundaries, each face named
by a 406 form 15 property. The observed degrees are a set per direction (1, 3 and 5 in either
direction were read) and the observed net size a limit. A reader importing the file directly
reads a face only when its outer boundary has parameter-space curves flagged 2D parametric
(entity use 05) and a PREF other than 2; the writer states both, and a PREF 2 asked of it is
warned about and written as 1. No net limit is stated because the largest net probed was not a
failing one. The profile is the one a panel-method solver's CAD import was measured against;
the measurements stay with that solver's library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "OBSERVED",
    "SOLID_ENTITIES",
    "GateFinding",
    "ReaderProfile",
    "surface_findings",
]

#: Entity types that describe a solid or a boundary representation of one: CSG primitives (150 to
#: 168), the Boolean tree (180), the manifold solid B-rep object (186), the solid instance (430)
#: and the B-rep topology entities (502 to 514). The package writes surfaces only.
SOLID_ENTITIES: tuple[int, ...] = (
    150,
    152,
    154,
    156,
    158,
    160,
    162,
    164,
    168,
    180,
    184,
    186,
    430,
    502,
    504,
    508,
    510,
    514,
)


@dataclass(frozen=True)
class ReaderProfile:
    """What the reader is known to accept.

    ``degrees`` are the degrees a file was read with, in both directions; ``degrees_u`` and
    ``degrees_v`` replace them for one direction once the probe observes the two apart.
    ``max_degree`` is a stated limit, ``None`` while no source gives one: a degree outside the
    observed ones is then warned about; once a source sets it, a degree above it fails.
    ``max_net`` is the largest net, as points in u and in v, a probe read: ``None`` states no
    limit, and a larger net then fails. ``bad_side_counts`` are the numbers of curves of an outer
    boundary that fail: a three-sided face collapses a side and is not among the observed shapes.
    ``loop_gap`` is the gap, in parameter units, within which a trim loop is closed.
    ``inner_loops`` is true once a probe read a face with an inner boundary with the hole
    subtracted; while it is false, a face with inner boundaries is warned about.
    """

    rational: bool = False
    clamped: bool = True
    degrees: tuple[int, ...] = (3,)
    max_degree: int | None = None
    bad_side_counts: tuple[int, ...] = (3,)
    degrees_u: tuple[int, ...] | None = None
    degrees_v: tuple[int, ...] | None = None
    max_net: tuple[int, int] | None = None
    loop_gap: float = 1e-7
    inner_loops: bool = False

    def observed(self, direction: str) -> tuple[int, ...]:
        """Return the degrees observed read in ``direction`` (``"u"`` or ``"v"``)."""
        own = self.degrees_u if direction == "u" else self.degrees_v
        return self.degrees if own is None else own


#: The profile every file is measured against. Inner loops are observed read: the probe face P5
#: (one face with an inner loop) read with the hole subtracted, its area within 0.03 percent of the
#: written face, both after ``iges-convert`` and imported directly.
OBSERVED = ReaderProfile(degrees_u=(1, 3, 5), degrees_v=(1, 3, 5), inner_loops=True)


@dataclass(frozen=True)
class GateFinding:
    """One defect: the check that found it, ``fail`` or ``warn``, the entity and the detail."""

    check: str
    severity: str
    entity: int | None
    detail: str


def _number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def surface_findings(entity: dict[str, Any], profile: ReaderProfile) -> list[GateFinding]:
    """Return the findings of one surface entity (128) measured against ``profile``.

    Parameters
    ----------
    entity : dict
        One parsed entity of type 128: its ``id`` and its ``parameters`` as read from the
        parameter data section.
    profile : ReaderProfile
        The profile the surface is measured against, usually :data:`OBSERVED`.

    Returns
    -------
    list of GateFinding
        One finding per defect: rational weights, unclamped knots or a degree outside the
        profile. Empty when the surface matches the profile.
    """
    values = entity["parameters"]
    ident = entity["id"]
    k1, k2, m1, m2 = (int(values[i]) for i in range(1, 5))
    start = 10
    t1 = [_number(v) for v in values[start : start + k1 + m1 + 2]]
    start += k1 + m1 + 2
    t2 = [_number(v) for v in values[start : start + k2 + m2 + 2]]
    start += k2 + m2 + 2
    weights = [_number(v) for v in values[start : start + (k1 + 1) * (k2 + 1)]]
    found: list[GateFinding] = []
    if not profile.rational and max(weights) - min(weights) > 1e-12:
        found.append(GateFinding("profile.rational", "fail", ident, "weights are not all one"))
    if profile.clamped:
        for label, knots, degree in (("u", t1, m1), ("v", t2, m2)):
            ends = knots[: degree + 1] + knots[-(degree + 1) :]
            if len(set(knots[: degree + 1])) != 1 or len(set(knots[-(degree + 1) :])) != 1:
                found.append(
                    GateFinding(
                        "profile.unclamped", "fail", ident, f"{label} knots are not clamped: {ends}"
                    )
                )
    if profile.max_net is not None and (k1 + 1 > profile.max_net[0] or k2 + 1 > profile.max_net[1]):
        found.append(
            GateFinding(
                "profile.net_size",
                "fail",
                ident,
                f"net {k1 + 1} by {k2 + 1} is larger than the {profile.max_net[0]} by "
                f"{profile.max_net[1]} a probe read",
            )
        )
    for direction, degree in ("u", m1), ("v", m2):
        if profile.max_degree is not None and degree > profile.max_degree:
            found.append(
                GateFinding("profile.degree", "fail", ident, f"degree {degree} above the limit")
            )
        elif degree not in profile.observed(direction):
            found.append(
                GateFinding(
                    "profile.degree",
                    "warn",
                    ident,
                    f"degree {degree} in {direction} was not observed read and no limit is "
                    "stated for it",
                )
            )
    return found

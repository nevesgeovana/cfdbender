"""The run types a matrix row selects in its WORKFLOW cell, and the row variables each reads.

The names are pyflightstream's: ``steady``, ``unsteady``, ``unsteady_rotor`` and
``qsteady_rotor`` (a rotor solved steady in the frame that turns with it: the blades held
still and the free stream turning about the shaft). An empty cell or ``-`` is ``steady``.
pyflightstream's ``LEGACY`` (a recipe) has no meaning here and is refused.

Each run type CARRIES its declarations, so a setup file states only what is particular to
the case. The agnostic part of a declaration is the row variables the run type reads, which
it requires and which it refuses (this module, one table); the solver part is the
configuration an adapter writes for the run type (for SU2,
:mod:`cfdbender.solvers.su2._run_types`). A row variable that another run type reads is
warned and skipped on this one, as pyflightstream does, and refused under warning refusals.

The turbulence model is the row variable :data:`TURBULENCE_MODEL` (a name of this package:
pyflightstream has none). It names a fully turbulent model; no run type ever selects a
transition model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from cfdbender.core.refusals import warn_or_refuse

STEADY = "steady"
UNSTEADY = "unsteady"
UNSTEADY_ROTOR = "unsteady_rotor"
QSTEADY_ROTOR = "qsteady_rotor"
#: Every run type, in pyflightstream's spelling.
RUN_TYPES = (STEADY, UNSTEADY, UNSTEADY_ROTOR, QSTEADY_ROTOR)
#: The time-accurate run types.
TIME_ACCURATE = (UNSTEADY, UNSTEADY_ROTOR)
#: The run types that turn a rotor.
ROTOR_RUN_TYPES = (UNSTEADY_ROTOR, QSTEADY_ROTOR)

#: The row variable naming the turbulence model (PROVISIONAL name: pyflightstream has none).
TURBULENCE_MODEL = "TURBULENCE_MODEL"

CLOCK_KEYS = ("DELTA_THETA", "REVOLUTIONS", "DELTA_TIME", "TIME_ITERATIONS")
WINDOW_KEYS = ("LAST_REVS_AVG", "LAST_ITERS_AVG")
EXPORT_KEYS = (
    "EXPORT_UNSTEADY_LAST_REV",
    "EXPORT_UNSTEADY_LAST_ITER",
    "EXPORT_UNSTEADY_AFTER_REV",
    "EXPORT_UNSTEADY_AFTER_ITER",
)
#: The row variable that writes the restart file periodically (PROVISIONAL name).
RESTART_KEYS = ("RESTART_EVERY_REV",)
#: Row variables of the run types that every run type reads.
COMMON_KEYS = ("SYMMETRY", "SYMMETRY_LOADS", "symmetry_loads", "PERIODIC_COPIES", TURBULENCE_MODEL)


@dataclass(frozen=True)
class RunTypeDeclaration:
    """The row variables one run type reads.

    Attributes
    ----------
    name : str
        The run type.
    keys : tuple of str
        Every row variable of the controlled vocabulary the run type reads.
    """

    name: str
    keys: tuple[str, ...]


DECLARATIONS: Mapping[str, RunTypeDeclaration] = {
    STEADY: RunTypeDeclaration(STEADY, COMMON_KEYS),
    QSTEADY_ROTOR: RunTypeDeclaration(QSTEADY_ROTOR, (*COMMON_KEYS, "PASSAGE_POSITIONS")),
    UNSTEADY: RunTypeDeclaration(
        UNSTEADY,
        (*COMMON_KEYS, *CLOCK_KEYS, *WINDOW_KEYS, "CLOCK_MOTION", *EXPORT_KEYS, *RESTART_KEYS),
    ),
    UNSTEADY_ROTOR: RunTypeDeclaration(
        UNSTEADY_ROTOR,
        (
            *COMMON_KEYS,
            *CLOCK_KEYS,
            *WINDOW_KEYS,
            "CLOCK_MOTION",
            "MOTIONS",
            *EXPORT_KEYS,
            *RESTART_KEYS,
        ),
    ),
}
#: Row variables of pyflightstream's vocabulary this package does not offer yet, and why.
NOT_OFFERED: Mapping[str, str] = {
    "CLOCK_MOTION": "a row turns one rotor here, whose speed is the clock",
    "MOTIONS": "a row turns the one rotor of its reference file here",
    "PASSAGE_POSITIONS": "the clockings of a whole wheel are not offered; solve a periodic "
    "sector, or the rotor as unsteady_rotor",
}
#: Every row variable some run type reads: never forwarded to a solver as a native key.
CONTROLLED_KEYS = tuple(dict.fromkeys(k for d in DECLARATIONS.values() for k in d.keys))


def run_type(workflow: str, *, pol: str, offered: Collection[str]) -> str:
    """Return the row's run type, refusing one that is not known or not offered.

    Parameters
    ----------
    workflow : str
        The WORKFLOW cell (``steady`` where the row states none).
    pol : str
        The row's POL, named by the refusal.
    offered : collection of str
        The run types the adapter declares.

    Raises
    ------
    ValueError
        The value is not one of :data:`RUN_TYPES`, or the adapter does not offer it.
    """
    if workflow not in RUN_TYPES:
        raise ValueError(
            f"WORKFLOW value {workflow!r} of POL {pol} names no known run type; the run "
            f"types are {', '.join(RUN_TYPES)}. The column names WHICH run type the row "
            "solves, so an unrecognized value would run the wrong one silently"
        )
    if workflow not in offered:
        raise ValueError(
            f"WORKFLOW {workflow} of POL {pol} is not a run type this solver offers; it "
            f"offers {', '.join(name for name in RUN_TYPES if name in offered)}"
        )
    return workflow


def readers(key: str) -> tuple[str, ...]:
    """Return the run types that read a controlled row variable."""
    return tuple(name for name, d in DECLARATIONS.items() if key in d.keys)


def skip_unread_variables(
    name: str, variables: Mapping[str, object], *, pol: str
) -> dict[str, object]:
    """Return the row variables without the controlled ones run type ``name`` does not read.

    Each skipped variable is warned once, naming the run types that read it (refused under
    warning refusals), as pyflightstream does for a key outside the row's run type.
    """
    declared = DECLARATIONS[name].keys
    kept: dict[str, object] = {}
    for key, value in variables.items():
        if key in NOT_OFFERED and key in declared:
            warn_or_refuse(
                f"POL {pol}: {key} is skipped: {NOT_OFFERED[key]}",
                UserWarning,
                stacklevel=2,
            )
            continue
        if key.startswith("EXPORT_") and key not in CONTROLLED_KEYS:
            warn_or_refuse(
                f"POL {pol}: {key} is not registered for run type {name!r}. No run type reads "
                "it. Remove it from this row; it is skipped",
                UserWarning,
                stacklevel=2,
            )
            continue
        if key in CONTROLLED_KEYS and key not in declared:
            warn_or_refuse(
                f"POL {pol}: {key} is not registered for run type {name!r}. It is read by "
                f"{', '.join(readers(key))}. Remove it from this row; it is skipped",
                UserWarning,
                stacklevel=2,
            )
            continue
        kept[key] = value
    return kept


def is_controlled(key: str) -> bool:
    """Whether a row variable belongs to the run types and is never a solver key."""
    return key in CONTROLLED_KEYS

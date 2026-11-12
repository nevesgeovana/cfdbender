"""Discover and load registered adapters outside the solver-independent core."""

from importlib import import_module
from types import ModuleType

_SOLVERS = {
    "flightstream": "cfdbender.solvers.flightstream",
    "mock": "cfdbender.solvers.mock",
    "openfoam": "cfdbender.solvers.openfoam",
    "pyfr": "cfdbender.solvers.pyfr",
    "qprop": "cfdbender.solvers.qprop",
    "su2": "cfdbender.solvers.su2",
    "xfoil": "cfdbender.solvers.xfoil",
}


def available_solvers() -> tuple[str, ...]:
    """Return registered solver names in deterministic order without loading them."""
    return tuple(sorted(_SOLVERS))


def load_solver(name: str) -> ModuleType:
    """Load an adapter by its registered name.

    Parameters
    ----------
    name : str
        Exact registered name; arbitrary import paths are refused.

    Returns
    -------
    ModuleType
        The adapter package. This skeleton does not define an execution protocol.

    Raises
    ------
    ValueError
        If the name is unknown, naming the available solvers.
    """
    if name not in _SOLVERS:
        raise ValueError(
            f"unknown solver {name!r}; available solvers: {', '.join(available_solvers())}"
        )
    adapter = import_module(_SOLVERS[name])
    if name == "flightstream":
        adapter.require_support()
    return adapter

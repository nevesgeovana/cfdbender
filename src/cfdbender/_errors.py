"""The exceptions of the geometry and mesh layers.

Pipeline role: the one home of the refusals that the geometry, mesh and slicing modules raise,
so a caller catches one family. Each class keeps a standard base (``ValueError`` or
``RuntimeError``), so ``except ValueError`` still catches an input refusal.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

__all__ = [
    "CheckFailedError",
    "GeometryError",
    "GeometryWarning",
    "InputArtifactError",
    "InputTypeError",
    "InputValueError",
    "NotApplicableWarning",
    "warn",
    "warn_by_name",
]


class GeometryError(Exception):
    """Base of every refusal raised by the geometry and mesh layers."""


class InputValueError(GeometryError, ValueError):
    """A value given by the caller cannot be used; the message names the cause and the remedy."""


class InputTypeError(GeometryError, TypeError):
    """An argument given by the caller has the wrong type; the message names the remedy."""


class InputArtifactError(GeometryError, RuntimeError):
    """An input file cannot be read or validated.

    Parameters
    ----------
    message : str
        What is wrong and how to correct it.
    kind : str, optional
        The kind of artifact that failed to resolve.
    identifier : str, optional
        The id that failed to resolve.
    available : tuple of str, optional
        The ids that would have resolved.
    where : dict, optional
        Diagnostic locations: names, zero-based vertex indices, coordinates.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        identifier: str | None = None,
        available: tuple[str, ...] = (),
        where: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.identifier = identifier
        self.available = available
        self.where = {} if where is None else where


class CheckFailedError(InputValueError):
    """A check measured as failed where the construction cannot go on; it carries the check.

    Parameters
    ----------
    check : Mapping
        The check record, with at least ``id`` and ``note``.
    """

    def __init__(self, check: Mapping[str, Any]) -> None:
        super().__init__(f"{check['id']}: {check['note']}")
        self.check = dict(check)


class GeometryWarning(UserWarning):
    """A geometry or mesh input was used with a stated adjustment."""


class NotApplicableWarning(GeometryWarning):
    """An entry that does not apply to the input was skipped, with the reason."""


def warn(message: str, category: type[Warning] = GeometryWarning, stacklevel: int = 1) -> None:
    """Warn as :func:`warnings.warn` does, attributed to the caller's line at ``stacklevel`` 1.

    Parameters
    ----------
    message : str
        The warning text.
    category : type of Warning
        The category, :class:`GeometryWarning` or a subclass at every site of the package.
    stacklevel : int
        As for :func:`warnings.warn`: 1 is the line that calls this function.
    """
    warnings.warn(message, category, stacklevel=stacklevel + 1)


def warn_by_name(message: str, name: str, stacklevel: int = 1) -> None:
    """Replay a warning recorded elsewhere by its category's name.

    A worker process sends back the class name of each warning it caught; this raises it again
    as that category of this module, or as :class:`GeometryWarning` for any other name.

    Parameters
    ----------
    message : str
        The warning text.
    name : str
        The class name of the recorded category.
    stacklevel : int
        As for :func:`warn`, counted from the caller of this function.
    """
    found = globals().get(name)
    category = (
        found if isinstance(found, type) and issubclass(found, GeometryWarning) else GeometryWarning
    )
    warn(message, category, stacklevel + 1)

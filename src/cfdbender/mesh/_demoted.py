"""Warnings and refusals the mesh verbs share for inputs they skip or refuse.

A refinement or audit file may carry a key, a table or a selection that does not apply. The
verbs warn and skip such an entry, with the reason, rather than refuse the run; the
categories below name what was skipped so a caller can filter them.
"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import get_close_matches

from cfdbender._errors import GeometryError, GeometryWarning, warn

__all__ = [
    "EmptySelectionWarning",
    "InputTypeError",
    "UnknownKeyWarning",
    "demote",
    "suggest_name",
]


class UnknownKeyWarning(GeometryWarning):
    """An unknown key, option or table was ignored."""


class EmptySelectionWarning(GeometryWarning):
    """A selection names no applicable entries and was skipped."""


class InputTypeError(GeometryError, TypeError):
    """An input has the wrong type."""


def demote(exc: Exception, *, warning: type[Warning]) -> None:
    """Warn about an input the caller skips, giving the refusal it would otherwise raise.

    Parameters
    ----------
    exc : Exception
        The refusal that describes the skipped input, with its remedy.
    warning : type of Warning
        The category identifying the skipped input's family.
    """
    warn(str(exc), warning, stacklevel=2)


def suggest_name(name: str, known: Iterable[str]) -> str:
    """Return one close-name suggestion, or an empty string.

    Parameters
    ----------
    name : str
        Unknown input name.
    known : iterable of str
        Accepted names, compared with a similarity cutoff of 0.75.

    Returns
    -------
    str
        A sentence suitable for appending to a cause and remedy.
    """
    matches = get_close_matches(name, sorted(known), n=1, cutoff=0.75)
    return f" Did you mean {matches[0]!r}?" if matches else ""

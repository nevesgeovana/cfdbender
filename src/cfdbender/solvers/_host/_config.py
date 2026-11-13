"""Validate common numeric inputs without loading a solver adapter."""

import difflib
import math
from collections.abc import Collection


def number(value: object, name: str, where: str | None = None) -> float:
    """Validate a finite numeric input, refusing booleans and nonnumbers.

    ``where`` names the file or cell the value comes from; an absent value then reads
    ``<where> needs <name>``, while a value that is present and wrong reads as invalid.
    """
    if value is None and where is not None:
        raise ValueError(f"{where} needs {name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where + ' ' if where else ''}{name} must be a finite number")
    return float(value)


def unknown_keys(where: str, keys: Collection[str], known: Collection[str]) -> ValueError:
    """Return the refusal of an unknown key: where, the key as written, the closest known one."""
    names = ", ".join(repr(key) for key in sorted(keys))
    close = difflib.get_close_matches(sorted(keys)[0], sorted(known), n=1)
    hint = f" (closest known key: {close[0]})" if close else ""
    return ValueError(f"{where}: unknown key {names}{hint}; known keys: {', '.join(sorted(known))}")


def steady_only(solver: str) -> ValueError:
    """Return the one refusal of a WORKFLOW other than steady."""
    return ValueError(f"WORKFLOW must be steady: {solver} runs steady cases only")

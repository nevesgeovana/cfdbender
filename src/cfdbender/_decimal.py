"""A float spelled as a plain decimal, below every layer.

Pipeline role: below every layer, imported by the script layer (the wake-edge
node file) and the workspace layer (the trailing-edge points file) alike, and
the sign flip of a table cell the post layer's inflow tables share. It
imports nothing from this package, for the reason :mod:`cfdbender._digest`
does: a helper two layers need is published beneath both, never reached for
across a boundary.
"""

from __future__ import annotations

import math
from decimal import Decimal

__all__ = ["negated_text", "plain_decimal"]


def plain_decimal(value: float) -> str:
    """Spell a finite float as a plain decimal that reads back to the same float.

    The shortest round-trip digits, never an exponent: a number such as
    1.665e-17, which a trailing-edge vertex of a committed wing mesh
    carries, would otherwise be written with the letter e, and a letter on
    any line of the node file makes the import mark nothing (RPT-061).

    Examples
    --------
    >>> plain_decimal(1.665e-17)
    '0.00000000000000001665'
    >>> plain_decimal(-3.75)
    '-3.75'
    """
    return format(Decimal(repr(value)), "f")


def negated_text(text: str) -> str:
    """Return a table cell with its sign flipped, character for character.

    A cell that is not a number, zero or not finite comes back unchanged.
    The inflow tools and their installed-frame copies both flip a column
    this way, so the rule has one home below both.

    Parameters
    ----------
    text : str
        The cell as written in the table.

    Returns
    -------
    str
        The cell with its sign flipped, or ``text`` itself.
    """
    stripped = text.strip()
    try:
        value = float(stripped)
    except ValueError:
        return text
    if value == 0.0 or not math.isfinite(value):
        return text
    if stripped.startswith("-"):
        return stripped[1:]
    return "-" + stripped.removeprefix("+")

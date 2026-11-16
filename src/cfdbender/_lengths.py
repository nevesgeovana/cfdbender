"""Length units: metres per unit, exact as decimals, and the factor between two units.

The unit words are the upper-case names a geometry or mesh file declares (METER, MILLIMETER,
INCH and the others below).
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from types import MappingProxyType

__all__ = ["METRES_PER_UNIT", "scale"]

#: Metres in one of each length unit, exact as decimals.
METRES_PER_UNIT: Mapping[str, Fraction] = MappingProxyType(
    {
        "METER": Fraction(1),
        "CENTIMETER": Fraction("0.01"),
        "MILLIMETER": Fraction("0.001"),
        "MICRON": Fraction("0.000001"),
        "KILOMETER": Fraction(1000),
        "INCH": Fraction("0.0254"),
        "FEET": Fraction("0.3048"),
        "MILE": Fraction("1609.344"),
        "MILS": Fraction("0.0000254"),
        "MICROINCH": Fraction("0.0000000254"),
    }
)


def scale(from_unit: str, to_unit: str) -> float | None:
    """Return the factor that turns a length in ``from_unit`` into ``to_unit``.

    The ratio is taken exactly and rounded once, so metres to millimetres multiplies by exactly
    1000.0.

    Parameters
    ----------
    from_unit, to_unit : str
        Unit words of :data:`METRES_PER_UNIT`.

    Returns
    -------
    float or None
        The factor, or None when either word names no known length; the caller refuses,
        because only it knows which length was being converted.

    Examples
    --------
    >>> scale("METER", "MILLIMETER")
    1000.0
    >>> scale("METER", "OTHER") is None
    True
    """
    if from_unit not in METRES_PER_UNIT or to_unit not in METRES_PER_UNIT:
        return None
    return float(METRES_PER_UNIT[from_unit] / METRES_PER_UNIT[to_unit])

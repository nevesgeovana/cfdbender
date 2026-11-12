"""Name a datapoint folder from the point's flight condition.

The rule is the one pyflightstream writes, so a cfdbender workspace and a
pyflightstream workspace name the same operating point the same way: the
folder is ``DP-<point name>`` and the point name is every FLIGHT_CONDITION
variable of the row, in the row's order, each written as a code, the value
times a scale, rounded, and zero padded to a fixed width.

========  ======  =====  ======  =======================================
Key       Code    Scale  Width   Example
========  ======  =====  ======  =======================================
MACH      M       1000   3       0.144 -> M144
TASmps    V       10     4       34 -> V0340
REmi      RE      100    3       4.38 -> RE438
ALTFT     ALT     1      5       5000 -> ALT05000
dISA      DT      10     4 sign  10 -> DT+100
RHOkgm3   RHO     10000  5       1.225 -> RHO12250
MUPas     MU      1e9    5       1.8e-5 -> MU18000
ASMPS     A       10     4       340 -> A3400
TK        T       10     4       290 -> T2900
PPA       PS      1      6       100000 -> PS100000
ALPHA     AL      10     4 sign  -2 -> AL-020
BETA      BE      10     4 sign  4 -> BE+040
ADVANCE_RATIO J   100    4 sign  0.8 -> J+080 (``J`` is accepted)
RPM       RPM     1      5       800 -> RPM00800 (a magnitude)
roll_rate P       10     4 sign  as ALPHA
pitch_rate Q      10     4 sign  as ALPHA
yaw_rate  R       10     4 sign  as ALPHA
========  ======  =====  ======  =======================================

A signed field counts its sign in the width. An unsigned field refuses a
negative value, and RPM, a magnitude, refuses a negative value too. Rounding
is Python's ``round`` on the scaled value, as pyflightstream does.

A key that pyflightstream has no code for (the native keys of a solver row,
for example QPROP's ``V``, ``BETA07`` or ``RHO``) is written as its upper case
name followed by the value in Python's general format, such as ``V20`` or
``RHO1.2``, so every row stays nameable. Two points of one row that write the
same name are refused when the plan is built, naming both points.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cfdbender.core.settings import WorkspaceError

DATAPOINT_PREFIX = "DP-"

#: The name of a point whose row states no FLIGHT_CONDITION.
NO_CONDITION = "NOCOND"


class PointNameError(WorkspaceError):
    """A flight condition cannot be written as a point name."""


@dataclass(frozen=True)
class NameField:
    """How one variable is written: code, then the scaled value, padded to width."""

    code: str
    scale: float
    width: int
    signed: bool
    magnitude: bool = False


POINT_NAME_FIELDS: dict[str, NameField] = {
    "MACH": NameField("M", 1000.0, 3, False),
    "TASmps": NameField("V", 10.0, 4, False),
    "REmi": NameField("RE", 100.0, 3, False),
    "ALTFT": NameField("ALT", 1.0, 5, False),
    "dISA": NameField("DT", 10.0, 4, True),
    "RHOkgm3": NameField("RHO", 10000.0, 5, False),
    "MUPas": NameField("MU", 1e9, 5, False),
    "ASMPS": NameField("A", 10.0, 4, False),
    "TK": NameField("T", 10.0, 4, False),
    "PPA": NameField("PS", 1.0, 6, False),
    "ALPHA": NameField("AL", 10.0, 4, True),
    "BETA": NameField("BE", 10.0, 4, True),
    "ADVANCE_RATIO": NameField("J", 100.0, 4, True),
    "RPM": NameField("RPM", 1.0, 5, False, magnitude=True),
    "roll_rate": NameField("P", 10.0, 4, True),
    "pitch_rate": NameField("Q", 10.0, 4, True),
    "yaw_rate": NameField("R", 10.0, 4, True),
}

#: Spellings that name a coded key, so QPROP's own ``J`` is written like ADVANCE_RATIO.
_ALIASES = {"J": "ADVANCE_RATIO"}

_PORTABLE = re.compile(r"[A-Za-z0-9._+-]+")


def name_field(key: str, value: float) -> str:
    """Write one variable the way a point name carries it, for example ``M144``.

    Parameters
    ----------
    key : str
        A FLIGHT_CONDITION key. A key with no code is written as its upper case
        name and the value in general format.
    value : float
        The variable's value on the point.

    Returns
    -------
    str
        The field text.

    Raises
    ------
    PointNameError
        If the value is not finite, or a negative value meets an unsigned or
        magnitude field.
    """
    number = float(value)
    if not math.isfinite(number):
        raise PointNameError(f"{key} is {value!r}, which a point name cannot write")
    field = POINT_NAME_FIELDS.get(_ALIASES.get(key, key))
    if field is None:
        return f"{key.upper()}{format(0.0 if number == 0 else number, 'g')}"
    if field.magnitude and number < 0.0:
        raise PointNameError(
            f"{key} is {value!r}; a rotor speed is a magnitude, so write it positive"
        )
    scaled = round(number * field.scale)
    if field.signed:
        return f"{field.code}{scaled:+0{field.width}d}"
    if number < 0.0:
        raise PointNameError(
            f"{key} is {value!r}, and a point name writes {key} without a sign; "
            "the name cannot carry a negative value"
        )
    return f"{field.code}{scaled:0{field.width}d}"


def point_name(order: Sequence[str], condition: Mapping[str, float]) -> str:
    """Return a point's name: its keys in the row's order, each as :func:`name_field`.

    Parameters
    ----------
    order : sequence of str
        The FLIGHT_CONDITION keys in the order the row declares them. Keys of
        ``condition`` the order does not list follow, in the condition's order.
    condition : mapping of str to float
        The point's resolved values, the swept key at the point's own value.

    Returns
    -------
    str
        The point name, such as ``M120RE009AL+000BE+000J+100``.

    Raises
    ------
    PointNameError
        If a field cannot be written or the name is not portable.
    """
    keys = [key for key in order if key in condition]
    keys.extend(key for key in condition if key not in keys)
    name = "".join(name_field(key, condition[key]) for key in keys) or NO_CONDITION
    if _PORTABLE.fullmatch(name) is None:
        raise PointNameError(f"{name!r} is not a portable point name")
    return name


def datapoint_dir_name(name: str) -> str:
    """Return the folder a point's outputs go to: ``DP-<point name>``."""
    return f"{DATAPOINT_PREFIX}{name}"

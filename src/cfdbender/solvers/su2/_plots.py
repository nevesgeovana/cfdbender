"""The ``[plots]`` groups of a time-accurate SU2 point: which loads, summed over which markers.

pyflightstream's post file plots a force or moment ``parameter`` of a ``group`` of families
and names the plot ``<parameter>_<group>`` (``CL_MRP_TOTAL``). This module reads those groups
for SU2 from the history:

* ``AERO_COEFF_SURF`` gives each monitored marker its own load columns, ``CFx(blade)`` (a
  multizone history appends the zone first, ``CFx[1](blade)``); a group's value is the SUM of
  its markers' coefficients, which SU2 forms with the one reference area, length and moment
  origin of the point. The total of ``MRP_TOTAL`` needs no per-surface column: it is the
  history's total, summed over the zones of a multizone point.
* A *family* is a named set of wall markers the post file declares in ``[plots.families]``; a
  marker no family claims is a family under its own name. ``families = "each"`` expands over
  those families, ``"all"`` is every monitored marker, and a word of a group may be a family
  name, a wall marker, a reference ``[aliases]`` entry or the rotor's name. Everything is
  composed at post time from the per-marker columns of the history, so a new grouping needs
  only a new post.
* ``CL`` and ``CD`` are the wind-axis coefficients and ``FX`` to ``MZ`` the force (N) and
  moment (N m) in SU2's body axes about the reference moment origin; ``CDI``, ``CDO``, ``CDP``
  and ``CDV`` are absent (SU2's history has no such split), each with its reason.
* A group in a rotor's own frame (``SMRP``, ``RMRP``, ``LOCAL_AXIS``) is not supplied and is
  skipped with its reason.
* A multizone point without a ``[plots]`` table gets the two groups its per-zone totals can
  fill: ``MRP_TOTAL`` (the sum of the zones) and ``ROTOR_<alias>`` (the turning zone, the name
  pyflightstream gives a rotor's own group); a point with one zone gets none.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from cfdbender.core.pproc import (
    AXES_PLOT_COMPONENTS,
    AXES_PLOT_GROUP,
    DEFAULT_FORCE_PLOTS,
    PlotGroup,
    Plots,
    read_plots,
)
from cfdbender.core.rotor import rotor_block

#: The history column each plot parameter is formed from, and the factor kind of its value.
COEFFICIENT = {
    "CL": "CL",
    "CD": "CD",
    "FX": "CFx",
    "FY": "CFy",
    "FZ": "CFz",
    "MX": "CMx",
    "MY": "CMy",
    "MZ": "CMz",
}
UNFILLABLE = {
    "CDI": "SU2 does not split drag into induced and profile parts",
    "CDO": "SU2 does not split drag into induced and profile parts",
    "CDP": "SU2's history holds no pressure-drag column",
    "CDV": "SU2's history holds no viscous-drag column",
}
ROTOR_GROUP_PREFIX = "ROTOR_"


@dataclass(frozen=True)
class Group:
    """One plot group: its name, the parameters it plots and the markers it sums.

    ``markers`` is None for every monitored marker (the history's totals); ``turning`` says the
    group is the turning zone of a multizone point.
    """

    name: str
    parameters: tuple[str, ...]
    markers: tuple[str, ...] | None = None
    turning: bool = False


def _refuse(text: str) -> ValueError:
    return ValueError(f"[plots]: {text}")


def _family_map(
    declared: Mapping[str, tuple[str, ...]], monitored: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Return the families: each declared one, then each marker no family claims.

    A declared marker the history holds no column for is refused, naming the markers it has.
    """
    for name, markers in declared.items():
        unknown = [m for m in markers if m not in monitored]
        if unknown:
            raise _refuse(
                f"family {name!r} names marker(s) {', '.join(unknown)} that the point's history "
                f"holds no per-surface column of (it holds: {', '.join(monitored) or 'none'})"
            )
    claimed = {m for markers in declared.values() for m in markers}
    for name in declared:
        if name in monitored and name not in claimed:
            raise _refuse(
                f"family {name!r} has the name of the wall marker {name!r}, which no family "
                'claims, so "each" would hide that marker; rename the family or put the '
                "marker in a family"
            )
    families = dict(declared)
    for marker in monitored:
        if marker not in claimed and marker not in families:
            families[marker] = (marker,)
    return families


def history_markers(headers: Sequence[str]) -> list[str]:
    """Return the markers a history holds per-surface load columns of, in column order.

    A column is ``CFx(blade)``, or ``CFx[1](blade)`` on a multizone point.
    """
    found: dict[str, None] = {}
    for header in headers:
        match = re.fullmatch(r"(?:CL|CD|CSF|CFx|CFy|CFz|CMx|CMy|CMz)(?:\[\d+\])?\((.+)\)", header)
        if match:
            found[match[1]] = None
    return list(found)


def _word(
    word: str,
    families: Mapping[str, tuple[str, ...]],
    reference: Mapping[str, object],
    seen: tuple[str, ...] = (),
    markers: Sequence[str] = (),
) -> list[str] | None:
    """Return the family names a word stands for; None when it stands for nothing."""
    if word in families:
        return [word]
    if word in markers:
        return [word]
    aliases = reference.get("aliases")
    if isinstance(aliases, Mapping) and word in aliases and word not in seen:
        members = aliases[word]
        found: list[str] = []
        for member in members if isinstance(members, list) else [members]:
            found += _word(str(member), families, reference, (*seen, word), markers) or []
        return found or None
    block = rotor_block(reference)
    table = reference.get(block.name) if block is not None else None
    if block is not None and word == block.name and isinstance(table, Mapping):
        owned = [*(table.get("families_blades") or []), *(table.get("families_general") or [])]
        return [name for name in owned if name in families] or None
    return None


def plot_groups(
    reference: Mapping[str, object],
    post: Mapping[str, object],
    monitored: Sequence[str],
    *,
    multizone: bool,
) -> tuple[tuple[Group, ...], tuple[tuple[str, str], ...]]:
    """Return the groups a time-accurate point plots and each entry skipped with its reason.

    A point whose post file declares no ``[plots]`` gets the two groups of a multizone point
    (see the module) and none otherwise. Pyflightstream's default applies: when no group of
    the global ``MRP`` frame plots the six load components, ``MRP_TOTAL`` over every marker is
    added with them.

    Raises
    ------
    ValueError
        A marker claimed by two families, or two groups with one name.
    """
    plots: Plots | None = read_plots(post)
    block = rotor_block(reference)
    if plots is None:
        if not multizone:
            return (), ()
        found = [Group(AXES_PLOT_GROUP, DEFAULT_FORCE_PLOTS)]
        if block is not None:
            found.append(Group(ROTOR_GROUP_PREFIX + block.alias, DEFAULT_FORCE_PLOTS, turning=True))
        return tuple(found), ()
    families = _family_map(dict(plots.families), monitored)
    found = []
    skipped: list[tuple[str, str]] = []
    for entry in plots.groups:
        if entry.frame.strip().upper() != "MRP":
            skipped.append(
                (
                    entry.name,
                    f"the frame {entry.frame} is not supplied: SU2 measures loads in the "
                    "MRP frame only",
                )
            )
            continue
        found += _expand(entry, plots.parameters, families, reference, skipped, monitored)
    names = [group.name for group in found]
    if len(set(names)) != len(names):
        twice = sorted({n for n in names if names.count(n) > 1})
        raise _refuse(f"plot name(s) {', '.join(twice)} stated twice")
    covered = any(entry.frame.strip().upper() == "MRP" for entry in plots.groups) and set(
        AXES_PLOT_COMPONENTS
    ) <= set(plots.parameters)
    if not covered and AXES_PLOT_GROUP not in names:
        parameters = tuple(dict.fromkeys([*plots.parameters, *AXES_PLOT_COMPONENTS]))
        found.insert(0, Group(AXES_PLOT_GROUP, parameters))
    return tuple(found), tuple(skipped)


def _expand(
    entry: PlotGroup,
    parameters: tuple[str, ...],
    families: Mapping[str, tuple[str, ...]],
    reference: Mapping[str, object],
    skipped: list[tuple[str, str]],
    monitored: Sequence[str],
) -> list[Group]:
    if entry.families == "each":
        if not families:
            skipped.append(
                (
                    entry.name,
                    "families = 'each' has no family to expand over: the history holds no "
                    "per-surface columns (a point planned before per-surface output)",
                )
            )
            return []
        groups: list[Group] = []
        for name, held in families.items():
            if re.fullmatch(r"[A-Za-z0-9_]+", name) is None:
                skipped.append(
                    (
                        entry.name.replace("{family}", name),
                        f"the marker {name!r} is in no family and its name is not letters, "
                        "digits and underscores, so it cannot name a plot: put it in a family "
                        "of [plots.families]",
                    )
                )
                continue
            groups.append(Group(entry.name.replace("{family}", name), parameters, held))
        return groups
    words = [entry.families] if isinstance(entry.families, str) else list(entry.families)
    if words == ["all"]:
        return [Group(entry.name, parameters, None)]
    markers: list[str] = []
    unknown: list[str] = []
    for word in words:
        if word == "all":
            resolved = list(monitored)
        else:
            resolved = [
                m
                for name in (_word(word, families, reference, (), monitored) or [])
                for m in (families[name] if name in families else (name,))
            ]
        if not resolved:
            unknown.append(word)
            continue
        markers += [m for m in resolved if m not in markers]
    if unknown:
        skipped.append(
            (
                entry.name,
                f"families {', '.join(map(repr, unknown))} name no family or wall marker of this "
                f"point (families: {', '.join(families) or 'none'}; wall markers: "
                f"{', '.join(monitored) or 'none'})",
            )
        )
        return []
    return [Group(entry.name, parameters, tuple(markers))]


def needs_surface_columns(groups: Sequence[Group]) -> bool:
    """Whether any group sums markers, so SU2 must write its per-surface history group."""
    return any(group.markers is not None for group in groups)


def column_names(groups: Sequence[Group]) -> tuple[str, ...]:
    """Return ``<parameter>_<group>`` for every parameter SU2 can fill, groups in order."""
    return tuple(
        f"{parameter}_{group.name}"
        for group in groups
        for parameter in group.parameters
        if parameter in COEFFICIENT
    )


def unfillable(groups: Sequence[Group]) -> dict[str, str]:
    """Return each parameter SU2 cannot fill, with its reason."""
    return {
        parameter: UNFILLABLE[parameter]
        for group in groups
        for parameter in group.parameters
        if parameter in UNFILLABLE
    }


@dataclass(frozen=True)
class Scales:
    """The dynamic pressure SU2 scales each zone's coefficients by, and the references.

    ``zone`` maps a zone (None for a single-zone point) to its ``q`` in Pa: the free-stream
    one for a fixed zone and the one of the zone file's ``MACH_MOTION`` for the turning zone.
    ``free`` is the free-stream ``q`` (0 in a hover, None when the deck fixes none).
    """

    zone: Mapping[int | None, float | None]
    free: float | None
    area: float
    length: float

    def q(self, zone: int | None) -> float | None:
        """Return the dynamic pressure of a zone, None when unknown."""
        return self.zone.get(zone)

    def same_as_free(self, zone: int | None) -> bool:
        """Whether the zone's coefficients are scaled by the free-stream pressure."""
        q = self.q(zone)
        return (
            q is not None
            and self.free is not None
            and abs(q - self.free) <= 1e-9 * max(abs(self.free), 1e-300)
        )


def _no_pressure(scales: Scales | None, bad: Sequence[int | None]) -> str:
    if scales is None:
        return "the run states no free stream to turn a coefficient into a load"
    where = ", ".join("the point" if z is None else f"zone {z}" for z in bad)
    return (
        f"the dynamic pressure of {where} is 0 or unknown (a zone moves at its MACH_MOTION, "
        "a fixed zone at the free stream), so the coefficient cannot be turned into a load"
    )


def zone_scales(
    values: Mapping[str, object], folder: Path, *, turning_zone: int | None, zones: int
) -> Scales | None:
    """Return the dynamic pressures of a point's zones, None without a free stream.

    SU2 normalises a moving zone's coefficients by ``MACH_MOTION`` and a fixed zone's by the
    free stream (its log says so per zone). The turning zone's ``MACH_MOTION`` is the one its
    zone file states, else the deck's, else the free stream's Mach number.
    """
    import math

    from cfdbender.solvers._host._config import number
    from cfdbender.solvers.su2._config import free_stream

    area = number(values["REF_AREA"], "REF_AREA")
    length = number(values["REF_LENGTH"], "REF_LENGTH")
    if zones <= 1:
        stream = free_stream(values)
        if stream is None:
            return None
        return Scales({None: stream.dynamic_pressure}, stream.dynamic_pressure, area, length)
    stream = free_stream({k: v for k, v in values.items() if k != "MACH_MOTION"})
    if stream is None:
        return None
    free = stream.dynamic_pressure
    pressures: dict[int | None, float | None] = dict.fromkeys(range(zones), free)
    stated: object = values.get(f"zone_{turning_zone}.cfg/MACH_MOTION")
    path = folder / f"zone_{turning_zone}.cfg"
    if path.is_file():
        found = re.search(
            r"^[ \t]*MACH_MOTION[ \t]*=[ \t]*([0-9.eE+-]+)", path.read_text(), re.MULTILINE
        )
        if found:
            stated = found[1]
    if stated is not None and turning_zone is not None:
        motion = number(float(str(stated)), "MACH_MOTION")
        if stream.density is None:
            pressures[turning_zone] = None
        elif not math.isclose(motion, stream.mach, rel_tol=1e-9):
            speed = motion * stream.sonic_velocity
            pressures[turning_zone] = 0.5 * stream.density * speed * speed
    return Scales(pressures, free, area, length)


def _steady_table(folder: Path) -> tuple[list[str], list[list[str]]]:
    """Return the header and the rows of a steady point's ``history.csv``."""
    import csv

    from cfdbender.solvers.su2._history import history_headers

    path = folder / "history.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        headers = history_headers(next(reader), path)
        rows = [cells for cells in reader if cells]
    if not rows:
        raise ValueError(f"{path}: history has no data rows")
    return headers, rows


def _steady_history(folder: Path, names: Sequence[str], marker: str | None) -> Any:
    """Return the last row of a steady history as a one-step history of ``names``."""
    import math

    import numpy as np

    from cfdbender.post.time_average import TimeHistory

    headers, rows = _steady_table(folder)
    wanted = {name: name if marker is None else f"{name}({marker})" for name in names}
    missing = [column for column in wanted.values() if column not in headers]
    if missing:
        raise ValueError(f"{folder / 'history.csv'}: no column {', '.join(missing)}")
    last = dict(zip(headers, rows[-1], strict=False))
    values = {name: float(last[column]) for name, column in wanted.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"{folder / 'history.csv'}: the last row holds a value that is not finite")
    return TimeHistory(
        np.array([1], dtype=int), {name: np.array([value]) for name, value in values.items()}
    )


def plot_series(
    folder: Path,
    groups: Sequence[Group],
    scales: Scales | None,
    *,
    turning_zone: int | None,
    zones: int,
    steady: bool = False,
) -> tuple[Any, dict[str, str]]:
    """Read the plot columns of the groups from the history of a point.

    Parameters
    ----------
    folder : Path
        The point folder.
    groups : sequence of Group
        What :func:`plot_groups` returned.
    scales : Scales or None
        The dynamic pressure of each zone, that turn a force and a moment coefficient into the
        dimensional parameter; None when the run states no free stream.
    turning_zone, zones : int
        The turning zone and the zone count of a multizone point; ``zones`` is 1 otherwise.
    steady : bool, optional
        Read ``history.csv`` of a steady point and keep its last row, the steady reduction.

    Returns
    -------
    tuple
        The :class:`~cfdbender.post.time_average.TimeHistory` of the columns found (None when
        none) and ``{column, group or parameter: reason}`` of what is absent.
    """
    import numpy as np

    from cfdbender.post.time_average import TimeHistory
    from cfdbender.solvers.su2._history import history_table
    from cfdbender.solvers.su2._time_history import time_history, zone_column

    headers = _steady_table(folder)[0] if steady else history_table(folder)[0]
    names = tuple(COEFFICIENT.values())
    steps = None
    columns: dict[str, Any] = {}
    absent: dict[str, str] = {}
    for parameter, reason in unfillable(groups).items():
        absent[parameter] = reason
    cache: dict[tuple[int | None, str | None], Any] = {}

    def series(zone: int | None, marker: str | None) -> Any:
        if (zone, marker) not in cache:
            cache[(zone, marker)] = (
                _steady_history(folder, names, marker)
                if steady
                else time_history(folder, zone=zone, columns=names, marker=marker)
            )
        return cache[(zone, marker)]

    for group in groups:
        try:
            if group.markers is None and group.turning:
                parts = [(turning_zone, series(turning_zone, None))]
            elif group.markers is None:
                parts = [(z, series(z, None)) for z in (range(zones) if zones > 1 else [None])]
            else:
                parts = []
                for marker in group.markers:
                    where = [
                        z
                        for z in (range(zones) if zones > 1 else [None])
                        if zone_column("CL", z, marker) in headers
                    ]
                    if not where:
                        raise ValueError(
                            f"the history holds no per-surface column of marker {marker!r}: "
                            "keep AERO_COEFF_SURF in HISTORY_OUTPUT and the marker in "
                            "MARKER_MONITORING"
                        )
                    parts.append((where[0], series(where[0], marker)))
        except (OSError, ValueError, StopIteration) as error:
            absent[group.name] = str(error)
            continue
        total = {n: sum(p.columns[n] for _, p in parts) for n in names}
        if all(not np.any(total[n]) for n in names):
            absent[group.name] = "its markers carry no load: every value is zero in every row"
            continue
        steps = parts[0][1].steps
        zone_q = [scales.q(z) if scales is not None else None for z, _ in parts]
        # SU2's own coefficients: the markers of one zone, or of zones it scales alike
        same = len({z for z, _ in parts}) == 1 or (
            scales is not None and all(scales.same_as_free(z) for z, _ in parts)
        )
        for parameter in group.parameters:
            if parameter not in COEFFICIENT:
                continue
            column = f"{parameter}_{group.name}"
            coefficient = COEFFICIENT[parameter]
            if parameter[0] in "FM":
                bad = [z for (z, _), q in zip(parts, zone_q, strict=True) if not q or q <= 0.0]
                if scales is None or bad:
                    absent[column] = _no_pressure(scales, bad)
                    continue
                factor = scales.area * (scales.length if parameter[0] == "M" else 1.0)
                values = sum(
                    p.columns[coefficient] * q * factor
                    for (_, p), q in zip(parts, zone_q, strict=True)
                )
            elif same:
                values = total[coefficient]
            elif scales is None or not scales.free or scales.free <= 0.0 or None in zone_q:
                absent[column] = (
                    "the group's markers lie in zones that SU2 normalises by different dynamic "
                    "pressures, and the free-stream dynamic pressure is 0 or unknown, so the "
                    "coefficient has no reference; the force and moment columns remain"
                )
                continue
            else:
                values = (
                    sum(
                        p.columns[coefficient] * cast(float, q)
                        for (_, p), q in zip(parts, zone_q, strict=True)
                    )
                    / scales.free
                )
            columns[column] = values
    history = None if steps is None or not columns else TimeHistory(steps, columns)
    return history, absent

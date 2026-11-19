"""The frames a product declaration may name, built from the reference file as pyflightstream.

* ``REFERENCE``: the reference (mesh) frame itself.
* ``MRP``: at the moment reference point, with the reference axes.
* ``<ALIAS>_SMRP``: the rotor's hub frame, at the hub. Its shaft axis (the letter the rotor's
  ``axis`` names) follows the installed shaft; the axis ``blade1.zero`` names follows that
  direction (with its sign) projected onto the disc and normalized; the third axis makes the
  frame right-handed. An untilted shaft with an unsigned zero gives exactly the reference axes.
* ``<ALIAS>_BMRP``: at the hub, with the reference axes.
* ``<ALIAS>_RMRP``: the hub frame turning with the rotor: at step ``s`` it is the hub frame
  turned about the shaft by ``sense * s * 360 / steps_per_revolution`` degrees.
* ``<ALIAS>_RMRP<k>``: turned further to blade ``k``, by ``blade1.azimuth_deg + (k - 1) 360 /
  B`` degrees, right-handed about the shaft.
* ``SMRP`` and ``RMRP``: a rotor's own frame of that kind; with one rotor declared they name
  that rotor's.

A frame of a steady point does not turn: a turning frame is placed where it is at step zero.
A frame the reference does not let cfdbender build (``LOCAL_AXIS``, ``BLADE_AXIS``, a frame
of a ``[[frames]]`` table) is not in the mapping, and the product that names it is skipped
with the frames that exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from cfdbender.core.rotor import RotorBlock, rotor_block, shaft_axis
from cfdbender.post.sections import Frame as SectionFrame
from cfdbender.post.volume_frames import Frame, RotorSpin

__all__ = ["posed", "product_frames", "rotor_table", "section_frame"]

_UNIT = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rotor_table(reference: Mapping[str, object]) -> tuple[RotorBlock, Mapping[str, object]] | None:
    """Return the reference's rotor declaration and its raw table, None without one."""
    block = rotor_block(reference)
    if block is None:
        return None
    table = reference.get(block.name)
    return (block, table) if isinstance(table, Mapping) else None


def _hub_axes(block: RotorBlock, letter: str) -> tuple[tuple[float, float, float], ...]:
    zero = block.blade1.zero
    name = zero.lstrip("+-")
    datum = np.asarray(_UNIT[name], dtype=np.float64) * (-1.0 if zero.startswith("-") else 1.0)
    shaft = np.asarray(block.axis, dtype=np.float64)
    radial = datum - (datum @ shaft) * shaft
    radial = radial / float(np.linalg.norm(radial))
    axes: dict[str, np.ndarray] = {letter: shaft, name: radial}
    missing = next(axis for axis in "XYZ" if axis not in axes)
    left, right = {"X": ("Y", "Z"), "Y": ("Z", "X"), "Z": ("X", "Y")}[missing]
    axes[missing] = np.cross(axes[left], axes[right])
    rows = [axes[axis] for axis in "XYZ"]
    return tuple((float(r[0]), float(r[1]), float(r[2])) for r in rows)


def posed(frame: Frame, step: float | None) -> Frame:
    """Return the fixed frame where ``frame`` is at ``step`` (step zero when None)."""
    if frame.spin is None:
        return frame
    at = 0.0 if step is None else float(step)
    origin = np.asarray(frame.origin, dtype=np.float64)
    placed = frame.place(np.asarray(_IDENTITY, dtype=np.float64), at) - origin
    axes = tuple(tuple(float(v) for v in row) for row in placed)
    return Frame(frame.name, frame.origin, axes)  # type: ignore[arg-type]


def section_frame(frame: Frame, step: float | None) -> SectionFrame:
    """Return the fixed frame ``frame`` is at ``step``, as the sections cut takes it."""
    fixed = posed(frame, step)
    return SectionFrame(fixed.origin, fixed.axes)


def product_frames(
    reference: Mapping[str, object],
    *,
    moment_point: Sequence[float] = (0.0, 0.0, 0.0),
    steps_per_revolution: float | None = None,
) -> dict[str, Frame]:
    """Return every frame a product of this point may name, by name.

    Parameters
    ----------
    reference : mapping
        The parsed REF file.
    moment_point : sequence of float
        The moment reference point the run used.
    steps_per_revolution : float, optional
        The rotor clock of a time-accurate point; a turning frame turns only with one.
    """
    origin = (float(moment_point[0]), float(moment_point[1]), float(moment_point[2]))
    frames = {"REFERENCE": Frame("REFERENCE"), "MRP": Frame("MRP", origin)}
    found = rotor_table(reference)
    if found is None:
        return frames
    block, table = found
    letter, _ = shaft_axis(table.get("axis"), block.name)
    axes = _hub_axes(block, letter)
    alias = block.alias
    hub = block.origin
    frames[f"{alias}_SMRP"] = Frame(f"{alias}_SMRP", hub, axes)
    frames[f"{alias}_BMRP"] = Frame(f"{alias}_BMRP", hub)
    count = block.blades or 1
    for name, datum in (
        (f"{alias}_RMRP", 0.0),
        *(
            (f"{alias}_RMRP{k}", block.blade1.azimuth_deg + (k - 1) * 360.0 / count)
            for k in range(1, count + 1)
        ),
    ):
        spin = RotorSpin(
            block.axis,
            datum,
            float(block.rpm_sign),
            steps_per_revolution if steps_per_revolution else 1.0,
        )
        frame = Frame(name, hub, axes, spin)
        frames[name] = frame if steps_per_revolution else posed(frame, 0.0)
    frames["SMRP"] = frames[f"{alias}_SMRP"]
    frames["RMRP"] = frames[f"{alias}_RMRP"]
    return frames

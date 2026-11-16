"""The machine-readable statement products.json carries beside the tree (schema version 2).

``products.json`` states the units and frames of every column so the csv files stay lean:
``tree`` (what each folder holds), ``kinds`` (the columns of every kind of file, each with
its unit, frame, quantity and sign), ``run_block`` (the block's columns with unit and
source), ``frames`` (the frames and the values used for this matrix: the R angles per
rotor, the mesh frame), ``families``, ``groups``, ``polars`` (the run type of each) and
``migrated`` (an earlier-layout file left in place, with the file that replaces it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cfdbender.matrix._family_columns import ColumnSpec
from cfdbender.matrix._family_polar import PolarFiles
from cfdbender.matrix._run_block import IDENTIFIER_PLACEHOLDER, RUN_BLOCK_SPEC

SCHEMA_VERSION = 2
#: The folders of the tree. cfdbender writes polars/ today; the time history of an
#: unsteady point and its reductions get their own folders.
TIME_HISTORY_DIR = "time_history"
REDUCTIONS_DIR = "reductions"
POLARS_DIR = "polars"
SECTIONS_DIR = "sections"
TREE: Mapping[str, Mapping[str, object]] = {
    POLARS_DIR: {
        "holds": "FAM and group files, one set per polar",
        "kinds": ["FAM-DIM", "FAM-ACOEF", "FAM-RCOEF", "DIM", "ACOEF", "RCOEF"],
    },
    SECTIONS_DIR: {"holds": "sectional loads per point", "kinds": ["sections"]},
    TIME_HISTORY_DIR: {"holds": "the history of an unsteady point, one row per step", "kinds": []},
    REDUCTIONS_DIR: {"holds": "reductions of an unsteady point", "kinds": []},
}
_PAGE = "docs/results.md"
_ANCHORS = {"DIM": "dim", "ACOEF": "acoef", "RCOEF": "rcoef", "sections": "sections"}
_ROWS = {
    "FAM-DIM": ("one per point and family", ["POL", "POINT_ID", "FAMILY_ID"]),
    "FAM-ACOEF": ("one per point and family", ["POL", "POINT_ID", "FAMILY_ID"]),
    "FAM-RCOEF": ("one per point and rotor family", ["POL", "POINT_ID", "FAMILY_ID", "ROTOR_ID"]),
    "DIM": ("one per point", ["POL", "POINT_ID", "GROUP_ID"]),
    "ACOEF": ("one per point", ["POL", "POINT_ID", "GROUP_ID"]),
    "RCOEF": (
        "one per point, for a group whose members sit inside one rotor",
        ["POL", "POINT_ID", "GROUP_ID", "ROTOR_ID"],
    ),
    "sections": ("one per station", ["POL", "POINT_ID", "FAMILY_ID"]),
}


def column_statement(spec: ColumnSpec, kind: str) -> dict[str, object]:
    """Return one column's entry under ``kinds``."""
    entry: dict[str, object] = {
        "name": spec.name,
        "unit": spec.unit,
        "frame": spec.frame,
        "quantity": spec.quantity,
        "definition": f"{_PAGE}#{_ANCHORS[kind.removeprefix('FAM-')]}",
    }
    if spec.sign is not None:
        entry["sign"] = spec.sign
    return entry


def kinds_statement(headers: Mapping[str, Sequence[ColumnSpec]]) -> dict[str, object]:
    """Return ``kinds``: every kind written, its rows, keys and columns after the run block."""
    out: dict[str, object] = {}
    for kind, columns in headers.items():
        rows, keys = _ROWS[kind]
        out[kind] = {
            "rows": rows,
            "keys": keys,
            "run_block": True,
            "columns": [column_statement(spec, kind) for spec in columns],
        }
    return out


def run_block_statement(block: Sequence[str]) -> list[dict[str, object]]:
    """Return ``run_block``: each written block column with its unit and source."""
    by_template = {spec.name: spec for spec in RUN_BLOCK_SPEC}
    out: list[dict[str, object]] = []
    for name in block:
        spec = by_template.get(name)
        if spec is None:
            template = next(
                (s for s in RUN_BLOCK_SPEC if s.per_rotor and _matches(s.name, name)), None
            )
            spec = template or by_template[IDENTIFIER_PLACEHOLDER]
        out.append({"name": name, "unit": spec.unit, "source": spec.source})
    return out


def _matches(template: str, name: str) -> bool:
    stem = template.split("<id>")[0]
    return name.startswith(stem) and name[len(stem) :].isdigit()


def frames_statement(polar: PolarFiles) -> dict[str, object]:
    """Return ``frames``: their definitions and the rotor angles used for this matrix."""
    frames: dict[str, object] = {
        "E": {"name": "MRP", "axes": "x aft, y right, z up"},
        "B": {"from": "E", "dcm": "C2(pi)"},
        "S": {"from": "B", "dcm": "C2(-alpha_s)"},
        "W": {"from": "B", "dcm": "C3(beta_w) C2(-alpha_s)"},
    }
    alias = {rotor.rotor_id: rotor.alias for rotor in polar.rotors}
    for rotor_id, frame in sorted(polar.frames.items()):
        frames[frame.name] = {
            "rotor": alias[rotor_id],
            "from": "B",
            "dcm": "C2(theta) C3(psi)",
            "pitch_deg": frame.pitch_deg,
            "toe_deg": frame.toe_deg,
            "sense": frame.sense,
        }
    frames["mesh"] = {"declared": ["aft", "right", "up"], "converted": False}
    if polar.mesh_frames:
        frames["mesh"] = {
            "expected": ["aft", "right", "up"],
            "points": dict(polar.mesh_frames),
        }
    return frames


def manifest_statement(
    polar: PolarFiles, *, migrated: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Return the schema-2 keys of products.json for one matrix's polar files."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tree": {name: dict(value) for name, value in TREE.items()},
        "kinds": kinds_statement(
            {
                **polar.headers,
                **({"sections": polar.sections_columns} if polar.sections_columns else {}),
            }
        ),
        "run_block": run_block_statement(polar.run_block),
        "frames": frames_statement(polar),
        "families": [
            {"FAMILY_ID": f.family_id, "FAMILY": f.name, "ROTOR_ID": f.rotor_id}
            for f in polar.families
        ],
        "groups": [
            {
                "GROUP_ID": g.group_id,
                "GROUP": g.name,
                "origin": g.origin,
                "members": list(g.members),
            }
            for g in polar.groups
        ],
        "polars": {pol: {"run_type": run_type} for pol, run_type in polar.run_types.items()},
        "migrated": dict(migrated or {}),
    }

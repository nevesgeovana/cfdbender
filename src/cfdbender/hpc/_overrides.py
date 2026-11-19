"""Row-level overrides of one HPC profile, written ``hpc.<name>: <value>`` in a matrix row."""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from cfdbender.core.matrix import HPC_OVERRIDE_PREFIX

from ._profiles import HpcProfile

#: Names of the matrix columns that own a value of the descriptor; an override of one is refused.
COLUMN_OWNED = {"ncpus": "NCPUS", "walltime": "WALLTIME"}
#: The recorded origin of an override.
ORIGIN = "matrix row"
_SCRIPT_SUFFIXES = (".py", ".sh", ".bash")


def row_overrides(variables: Mapping[str, object]) -> dict[str, object]:
    """Return the row's ``hpc.<name>`` entries keyed by ``<name>``, in row order."""
    return {
        key[len(HPC_OVERRIDE_PREFIX) :]: value
        for key, value in variables.items()
        if key.startswith(HPC_OVERRIDE_PREFIX)
    }


def _placeholders(profile: HpcProfile) -> set[str]:
    names: set[str] = set()
    pending = [str(part) for part in (*profile.fields.values(), *profile.submit)]
    while pending:
        for _, name, spec, _ in string.Formatter().parse(pending.pop()):
            if name is not None:
                names.add(name)
            if spec:
                pending.append(spec)
    return names


def accepted_names(profile: HpcProfile) -> tuple[str, ...]:
    """Return what a row may override: the profile's descriptor fields and its substitutions.

    The substitutions are ``application_id`` (what the job launches) and ``cluster`` when
    the profile writes ``{cluster}``. ``ncpus`` and ``walltime`` keep their matrix columns.
    """
    names = {name for name in profile.fields if name not in COLUMN_OWNED}
    names.add("application_id")
    if "cluster" in _placeholders(profile):
        names.add("cluster")
    return tuple(sorted(names))


def is_script_path(value: object) -> bool:
    """Whether an application value names a script of the workspace rather than a command."""
    text = str(value)
    return "/" in text or chr(92) in text or text.lower().endswith(_SCRIPT_SUFFIXES)


def _script(root: Path, name: str, row: str) -> Path:
    from cfdbender.core.plan import PlanError, workspace_path

    try:
        target = workspace_path(root, name.replace(chr(92), "/"))
    except PlanError as error:
        raise ValueError(f"row {row}: {HPC_OVERRIDE_PREFIX}application_id: {error}") from None
    if not target.is_file():
        raise ValueError(
            f"row {row}: {HPC_OVERRIDE_PREFIX}application_id {name!r} is not a file of the "
            "workspace; a script is written relative to the workspace folder"
        )
    return target


def apply_overrides(
    profile: HpcProfile, overrides: Mapping[str, object], *, root: Path, row: str
) -> tuple[HpcProfile, dict[str, object]]:
    """Return the profile with the row's values replacing the profile's, and resolved values.

    The profile is the base: each named value is replaced and everything else is kept. The
    second result carries the substitution values (``cluster``) the planner must set.

    Raises
    ------
    ValueError
        A name the profile does not declare, a value a matrix column owns, an empty value
        or a script that is not in the workspace.
    """
    accepted = accepted_names(profile)
    fields = dict(profile.fields)
    application = profile.application_id
    values: dict[str, object] = {}
    for name, value in overrides.items():
        column = COLUMN_OWNED.get(name)
        if column is not None:
            raise ValueError(
                f"row {row}: {HPC_OVERRIDE_PREFIX}{name} is refused; {name} is the {column} "
                f"column, so state it there"
            )
        if name not in accepted:
            raise ValueError(
                f"row {row}: {HPC_OVERRIDE_PREFIX}{name} is not declared by the HPC profile "
                f"{profile.path.name}; it accepts {', '.join(accepted)}"
            )
        if isinstance(value, (list, dict)) or str(value).strip() == "":
            raise ValueError(f"row {row}: {HPC_OVERRIDE_PREFIX}{name} needs one non-empty value")
        text = str(value).strip()
        if name == "application_id":
            if is_script_path(text):
                text = _script(root, text, row).as_posix()
            application = text
        elif name == "cluster":
            values["cluster"] = text
        if name in fields:
            fields[name] = (
                value if isinstance(value, bool) else text.replace("{", "{{").replace("}", "}}")
            )
    return (
        replace(profile, fields=fields, application_id=application, application_ids={}),
        values,
    )

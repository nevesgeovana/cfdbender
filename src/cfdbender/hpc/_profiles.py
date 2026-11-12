"""Read workspace HPC profiles and refuse ineffective or misplaced settings."""

from __future__ import annotations

import re
import tomllib
import warnings
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

HPC_FORMATS = ("yaml", "json", "toml", "text")
HPC_VERSION_ALIAS = "solver_version_alias"
_PROFILE_KEYS = frozenset(
    {
        "application_id",
        "descriptor",
        "submit",
        "defaults",
        "builds",
        "application_ids",
        "max_walltime",
        "walltime_arithmetic",
    }
)
_CLOCK = re.compile(r"(\d+):([0-5]\d):([0-5]\d)")


def _descriptor_name(name: str, source: object) -> None:
    alias = (
        name.endswith((".", " "))
        or ":" in name
        or re.search(r"~\d", name)
        or name.split(".")[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"COM{i}" for i in range(1, 10)],
            *[f"LPT{i}" for i in range(1, 10)],
        }
    )
    if (
        name.strip()
        and not any(sep in name for sep in "/\\")
        and name not in (".", "..")
        and not alias
    ):
        return
    raise ValueError(
        f"the HPC profile {source} names its descriptor {name!r}; the "
        "descriptor is written in the folder each point runs in, so its name "
        "is a plain file name, with no folder, no parent folder and no form "
        "Windows reads as another file's."
    )


@dataclass(frozen=True)
class HpcProfile:
    """The application, descriptor fields and argv declared by one cluster profile.

    ``builds`` and ``application_ids`` are keyed by exact registered solver
    versions. ``walltime_arithmetic`` selects the row's written clock (``wall``)
    or integer seconds. ``path`` identifies the source in every refusal.
    Descriptor fields keep declaration order, and booleans keep their type.
    """

    application_id: str
    descriptor_format: str
    descriptor_name: str
    fields: dict[str, str | bool]
    submit: tuple[str, ...]
    path: Path
    builds: dict[str, str] = field(default_factory=dict)
    application_ids: dict[str, str] = field(default_factory=dict)
    walltime_arithmetic: str = "wall"
    max_walltime_s: float | None = None

    def __post_init__(self) -> None:
        """Hold Python-created profiles to the same descriptor filename rule."""
        _descriptor_name(self.descriptor_name, self.path)

    def application_id_for(self, solver_version: object) -> str:
        """Return the version's application override, otherwise the required default."""
        return self.application_ids.get(str(solver_version), self.application_id)


def _unknown(
    target: Path, table: Mapping[str, object], allowed: Collection[str], where: str
) -> None:
    unknown = sorted(key for key in table if key not in allowed)
    if unknown:
        raise ValueError(
            f"the HPC profile {target} states {', '.join(unknown)} {where}, and this package "
            f"reads {', '.join(sorted(allowed))} there. A key nothing reads is a key that looks "
            "like it works. Correct the spelling, or move the key to the table that takes it."
        )


def _aliases(target: Path, table: object, name: str, versions: Collection[str]) -> dict[str, str]:
    if table is None:
        return {}
    if not isinstance(table, dict):
        raise ValueError(
            f"the HPC profile {target} has a {name} entry that is not a table. Write it as "
            f"[{name}] with one line per canonical solver version."
        )
    aliases: dict[str, str] = {}
    for key, value in table.items():
        version = str(key).strip()
        if version not in versions or version != key:
            raise ValueError(
                f"the HPC profile {target} maps {key!r} in its [{name}] table, and a key "
                "there must name ONE registered solver version, because several versions can "
                "share one scheduler name and only the solver version says which. "
                "Supply the registered solver_versions and use their exact spelling."
            )
        alias = value.strip() if isinstance(value, str) else ""
        if not alias:
            raise ValueError(
                f"the HPC profile {target} maps solver version {version!r} to {value!r} in its "
                f"[{name}] table; the value is the scheduler's name, as text, "
                "and it cannot be empty."
            )
        aliases[version] = alias
    return aliases


def _max_walltime(target: Path, stated: object) -> float | None:
    if stated is None:
        return None
    found = _CLOCK.fullmatch(stated.strip()) if isinstance(stated, str) else None
    if found is None or not any(int(part) for part in found.groups()):
        raise ValueError(
            f"the HPC profile {target} states max_walltime = {stated!r}. It is the longest "
            "wall clock the cluster grants a job, written HH:MM:SS with hours beyond 24 "
            'allowed, for example "48:00:00"; 2d and 48h are not read, and a clock of zero '
            "grants nothing."
        )
    hours, minutes, seconds = (int(part) for part in found.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def read_hpc_profile(path: str | Path, *, solver_versions: Collection[str] = ()) -> HpcProfile:
    """Read a profile, refusing keys or values this planner cannot act on.

    Parameters
    ----------
    path : str or Path
        A TOML file under the workspace's ``inputs/hpc`` directory.
    solver_versions : collection of str, optional
        Exact registered solver versions supplied by the caller. Translation
        tables require this registry; no solver adapter is imported here.

    Returns
    -------
    HpcProfile
        The validated scheduler declaration.

    Raises
    ------
    ValueError
        A missing or unreadable file, invalid TOML, ineffective key, unsupported
        format, invalid clock, missing fields or command, or unregistered version.
    """
    target = Path(path)
    try:
        with target.open("rb") as stream:
            table = tomllib.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read hpc profile {target}: {error}") from None
    if not str(table.get("application_id") or "").strip():
        raise ValueError(
            f"the HPC profile {target} states no 'application_id'. It is the scheduler's own "
            "name for the application and the one field nothing else in the "
            "workspace can supply, which is why it is the only mandatory one."
        )
    for name in ("descriptor", "submit"):
        if name in table and not isinstance(table[name], dict):
            raise ValueError(
                f"the HPC profile {target} has a {name} entry that is not a table. Write it "
                f"as [{name}], with its keys under it."
            )
    if "defaults" in table:
        warnings.warn(
            f"the HPC profile {target} carries a defaults block, which has no effect; "
            "it is ignored. Delete it.",
            UserWarning,
            stacklevel=2,
        )
    # A scalar owned by the profile cannot silently disappear inside another table.
    for name, value in table.items():
        if name == "defaults" or not isinstance(value, dict):
            continue
        for owned in ("max_walltime", "walltime_arithmetic"):
            if owned in value:
                raise ValueError(
                    f"the HPC profile {target} states {owned} under [{name}], and this "
                    "package reads it at the top level of the file. A key nothing reads is "
                    "a key that looks like it works: a limit written there is dropped and "
                    f"the profile loads with none. Move {owned} above the first table."
                )
    descriptor = table.get("descriptor") or {}
    _unknown(target, descriptor, {"format", "name", "fields"}, "under [descriptor]")
    fmt = str(descriptor.get("format") or "yaml").strip().lower()
    if fmt not in HPC_FORMATS:
        raise ValueError(
            f"the HPC profile {target} asks for a {fmt!r} descriptor; this package "
            f"writes {', '.join(HPC_FORMATS)}."
        )
    if "name" in descriptor:
        _descriptor_name(str(descriptor["name"]), target)
    fields = descriptor.get("fields") or {}
    if not isinstance(fields, dict) or not fields:
        raise ValueError(
            f"the HPC profile {target} lists no descriptor fields, so the file it "
            "writes would be empty. The fields table is what this cluster expects and "
            "what makes the profile portable."
        )
    submit_table = table.get("submit") or {}
    _unknown(target, submit_table, {"command"}, "under [submit]")
    submit = submit_table.get("command") or []
    if not isinstance(submit, list) or not submit:
        raise ValueError(
            f"the HPC profile {target} states no submit command. Write it argument by "
            'argument, for example command = ["submit-tool", "{descriptor_path}"]: one string '
            "through a shell splits a path that has a space in it."
        )
    if any(not isinstance(part, str) or not part.strip() for part in submit):
        raise ValueError(f"the HPC profile {target} submit command needs non-empty strings")
    builds = _aliases(target, table.get("builds"), "builds", solver_versions)
    applications = _aliases(
        target, table.get("application_ids"), "application_ids", solver_versions
    )
    arithmetic = str(table.get("walltime_arithmetic") or "wall").strip().lower()
    if arithmetic not in {"wall", "seconds"}:
        raise ValueError(
            f"the HPC profile {target} states walltime_arithmetic = {arithmetic!r}, and "
            "this package writes seconds, wall. 'wall' puts the row's cell in the descriptor "
            "as written (4h stays 4h); 'seconds' puts the whole clock in integer seconds."
        )
    _unknown(target, table, _PROFILE_KEYS, "at its top level")
    return HpcProfile(
        application_id=str(table["application_id"]),
        descriptor_format=fmt,
        descriptor_name=str(descriptor.get("name") or f"submit.{'txt' if fmt == 'text' else fmt}"),
        fields={str(k): v if isinstance(v, bool) else str(v) for k, v in fields.items()},
        submit=tuple(str(part) for part in submit),
        path=target,
        builds=builds,
        application_ids=applications,
        walltime_arithmetic=arithmetic,
        max_walltime_s=_max_walltime(target, table.get("max_walltime")),
    )


def resolve_hpc_profile(
    inputs_dir: str | Path, name: str | None = None, *, solver_versions: Collection[str] = ()
) -> HpcProfile | None:
    """Select a named profile, the only profile, or no profile when the folder is empty.

    Parameters
    ----------
    inputs_dir : str or Path
        The workspace's ``inputs`` directory.
    name : str, optional
        File stem or filename, such as ``h001`` or ``h001.toml``.
    solver_versions : collection of str, optional
        Registered versions for the profile's translation tables.

    Returns
    -------
    HpcProfile or None
        The selected profile or None when none exists and no name was requested.

    Raises
    ------
    ValueError
        The selector names no file, several files need a selector, or loading fails.
    """
    directory = Path(inputs_dir) / "hpc"
    found = sorted(directory.glob("*.toml")) if directory.is_dir() else []
    selected = Path(name).name if name is not None else None
    if selected is not None and selected.lower().endswith(".toml"):
        selected = selected[: -len(".toml")]
    if selected is not None:
        for path in found:
            if path.stem == selected:
                return read_hpc_profile(path, solver_versions=solver_versions)
        raise ValueError(
            f"{directory} holds no profile named {selected!r}; "
            f"available: {', '.join(path.stem for path in found) or 'none'}."
        )
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            f"{directory} holds {len(found)} profiles "
            f"({', '.join(path.name for path in found)}) and nothing says which cluster "
            "this machine is. One profile needs no selector; several need one: select "
            "the profile by its file stem, hpc (CLI: --hpc) taking <name>, or "
            "resolve_hpc_profile(inputs_dir, name); guessing spends a queue."
        )
    return read_hpc_profile(found[0], solver_versions=solver_versions)

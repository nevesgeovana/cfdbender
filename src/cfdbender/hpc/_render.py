"""Render profile-owned descriptors and submit arguments without side effects."""

from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping
from pathlib import Path

from ._profiles import HPC_FORMATS, HPC_VERSION_ALIAS, HpcProfile

_SUBMIT_VALUES = frozenset(
    {
        "application_id",
        "config_path",
        "config_paths",
        "script_path",
        "workdir",
        "work_dir",
        "descriptor_path",
        HPC_VERSION_ALIAS,
        "solver_version",
        "solver_type",
        "cluster",
        "sim",
        "point",
        "ncpus",
        "walltime",
        "walltime_s",
        "walltime_written",
        "job_name",
        "sims",
        "first_sim",
        "last_sim",
        "batch",
        "batch_id",
    }
)


def _bound_values(profile: HpcProfile, values: Mapping[str, object]) -> dict[str, object]:
    bound = {key: value for key, value in values.items() if value is not None and value != ""}
    version = bound.get("solver_version")
    bound["application_id"] = profile.application_ids.get(str(version), profile.application_id)
    token = "{" + HPC_VERSION_ALIAS + "}"
    names_alias = any(token in str(part) for part in (*profile.fields.values(), *profile.submit))
    if names_alias and str(version) not in profile.builds:
        mapped = ", ".join(f"{k} -> {v}" for k, v in sorted(profile.builds.items())) or "nothing"
        raise ValueError(
            f"the HPC profile {profile.path} writes {{{HPC_VERSION_ALIAS}}} and its [builds] "
            f"table maps no alias for solver version(s) {version or 'solver_version (missing)'} "
            f"(it maps {mapped}). Add a line per solver version under [builds]. The table is a "
            "declaration: it does not check that the scheduler starts that solver version, "
            "which only the version in a collected log shows."
        )
    if str(version) in profile.builds:
        bound[HPC_VERSION_ALIAS] = profile.builds[str(version)]
    return bound


def render_descriptor(profile: HpcProfile, values: Mapping[str, object]) -> str:
    """Render a descriptor using the profile's fields and typed substitutions.

    Parameters
    ----------
    profile : HpcProfile
        The descriptor format, fields, and version translation tables.
    values : mapping of str to object
        Resolved point or group values. None and empty strings are missing.

    Returns
    -------
    str
        Newline-terminated YAML, JSON, TOML or text, without writing any file.

    Raises
    ------
    ValueError
        Unsupported format, unmapped version, invalid template or missing value.
    """
    if profile.descriptor_format not in HPC_FORMATS:
        raise ValueError(f"the HPC profile {profile.path} asks for {profile.descriptor_format!r}")
    return _render_descriptor(profile, _bound_values(profile, values))


def build_submit_command(
    profile: HpcProfile, values: Mapping[str, object], descriptor_path: str | Path
) -> list[str]:
    """Format the profile's submit argv without invoking it or a shell.

    Parameters
    ----------
    profile : HpcProfile
        Source of every command argument; no scheduler command is supplied here.
    values : mapping of str to object
        Point or group substitutions, including the resolved solver_version.
    descriptor_path : str or Path
        The descriptor's planned path; a space remains inside one argument.

    Returns
    -------
    list of str
        The command and arguments, ready for a separately authorized executor.

    Raises
    ------
    ValueError
        A template asks for an unknown or unresolved value or an unmapped version.
    """
    bound = _bound_values(profile, values)
    bound["descriptor_path"] = Path(descriptor_path).as_posix()
    known = _SUBMIT_VALUES | set(bound)
    pending = list(profile.submit)
    unknown: list[str] = []
    while pending:
        part = pending.pop()
        try:
            fields = list(string.Formatter().parse(part))
        except ValueError:
            unknown.append(f"an unreadable brace in {part!r}")
            continue
        for _, name, spec, _ in fields:
            if name is None:
                continue
            if spec:
                pending.append(spec)
            root = re.split(r"[.\[]", name, maxsplit=1)[0]
            if root not in known:
                unknown.append("{" + name + "}")
    if unknown:
        raise ValueError(
            f"the HPC profile {profile.path} writes {', '.join(unknown)} in its submit "
            "command, and no run supplies it. What a run can supply: "
            f"{', '.join(sorted(known))}."
        )
    try:
        return [part.format(**bound) for part in profile.submit]
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as error:
        said = f"{{{error.args[0]}}}" if isinstance(error, KeyError) else str(error)
        raise ValueError(
            f"the HPC profile {profile.path} writes {said} in its submit command, "
            f"and this point cannot supply it. What it can: {', '.join(sorted(bound))}."
        ) from None


def _numeric_field(template: object, values: Mapping[str, object]) -> bool:
    """Whether a descriptor field is a bare number rather than a quoted string.

    True only when the template is exactly one substitution and the value
    behind it is an int or a float. `26.123` is a build identifier that
    reads as a float, and quoting it is the difference between a scheduler
    receiving that build and receiving 26.12.
    """
    if isinstance(template, bool):
        # A TOML boolean in the profile (``driveg = true``) is a boolean the
        # scheduler reads, written bare and lower case: quoted, the cluster
        # reads the string "true" or "True" and not the switch.
        return True
    text = str(template).strip()
    if not (text.startswith("{") and text.endswith("}") and text.count("{") == 1):
        return False
    return isinstance(values.get(text[1:-1]), (int, float)) and not isinstance(
        values.get(text[1:-1]), bool
    )


def _render_descriptor(profile: HpcProfile, values: Mapping[str, object]) -> str:
    """Render one submission descriptor from a profile and this point's values.

    THE PROFILE OWNS THE KEYS and this owns none: every line comes from
    the profile's own field table, so a cluster that spells ``cpus`` or
    ``queue`` or ``account`` is served by editing that table and nothing
    else. A substitution the values cannot supply is refused rather than
    written empty, because a descriptor with a blank where a job name goes
    is a job the scheduler names for you.

    Parameters
    ----------
    profile : HpcProfile
        The cluster profile: its field table, its descriptor format
        (``json``, ``text``, ``toml`` or ``yaml``) and the path it was
        read from.
    values : mapping of str to object
        This point's values, keyed by the placeholder names the profile's
        templates use (for example ``sim``, ``point``, ``fs_build``,
        ``ncpus``).

    Returns
    -------
    str
        The descriptor text, newline terminated, in the profile's format.

    Raises
    ------
    ValueError
        When a field of the profile asks for a placeholder that ``values``
        does not carry.
    """
    lines: list[str] = []
    rendered: dict[str, str] = {}
    for key, template in profile.fields.items():
        try:
            text = (
                str(template).lower()
                if isinstance(template, bool)
                else str(template).format(**values)
            )
        except (KeyError, IndexError, ValueError, AttributeError, TypeError) as missing:
            raise ValueError(
                f"the HPC profile {profile.path} asks for {str(missing)!r} in its "
                f"{key!r} "
                f"field, and this run cannot supply it. What it can: "
                f"{', '.join(sorted(values))}."
            ) from None
        rendered[key] = text
    if profile.descriptor_format == "json":
        # THE YAML RULE, IN JSON: a boolean is a JSON boolean and a
        # single substitution of a number is a JSON number; the rest are strings.
        typed: dict[str, object] = {}
        for key, text in rendered.items():
            template = profile.fields[key]
            if isinstance(template, bool):
                typed[key] = template
            elif _numeric_field(template, values):
                typed[key] = values[str(template).strip()[1:-1]]
            else:
                typed[key] = text
        return json.dumps(typed, indent=2) + "\n"
    for key, text in rendered.items():
        template = profile.fields[key]
        safe_key = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key))
        if profile.descriptor_format == "yaml" and key.lower() in {
            "true",
            "false",
            "null",
            "yes",
            "no",
            "on",
            "off",
        }:
            safe_key = False
        structured_key = key if safe_key else json.dumps(key, ensure_ascii=False)
        if profile.descriptor_format == "text":
            lines.append(f"{key}: {text}")
        elif profile.descriptor_format == "toml":
            # A boolean and a single substitution of a number are bare, as in yaml.
            bare = _numeric_field(template, values)
            # A quoted string escapes its quote, backslash and control characters (F-1458).
            lines.append(
                f"{structured_key} = {text}"
                if bare
                else f"{structured_key} = {json.dumps(text, ensure_ascii=False)}"
            )
        else:
            # yaml. A number stays bare and everything else is quoted, which
            # is what the predecessor's own descriptor does.
            #
            # THE SOURCE TYPE DECIDES, NOT THE TEXT. A build identifier is
            # `26.123`, which reads as a float and is not one: unquoted, a
            # scheduler is handed 26.123 and may hand back 26.12. So a field
            # is bare only when it is exactly one substitution AND the value
            # behind it was a number.
            bare = _numeric_field(template, values)
            # A double-quoted yaml scalar escapes as json does (F-1459).
            lines.append(
                f"{structured_key}: {text}"
                if bare
                else f"{structured_key}: {json.dumps(text, ensure_ascii=False)}"
            )
    return "\n".join(lines) + "\n"

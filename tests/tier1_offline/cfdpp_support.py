"""Load the CFD++ fixture builder, which lives beside the fixtures it writes, and write TOML."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

_BUILDER = Path(__file__).resolve().parents[1] / "fixtures" / "cfdpp" / "build.py"


def builder() -> ModuleType:
    """Return the fixture builder module, loaded once."""
    name = "cfdpp_fixture_build"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _BUILDER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# ---------------------------------------------------------------------------
# A minimal TOML writer. The standard library reads TOML (tomllib) and writes
# none; the tests edit a copy of a database folder and write it back, so they
# need the inverse of the loader for the shapes the database uses: tables of
# scalars, lists, and arrays of tables. A null has no TOML form, so an unknown
# is an omitted key and the writer refuses a None.
# ---------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(name: object) -> str:
    text = str(name)
    return text if _BARE_KEY.match(text) else json.dumps(text, ensure_ascii=False)


def _value(value: Any) -> str:
    if value is None:
        raise ValueError("TOML has no null: omit the key of an unknown")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items())
        return "{" + (" " + inner + " " if inner else "") + "}"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(i, dict) for i in value)


def toml_dumps(data: dict[str, Any], comments: dict[tuple[str, int], str] | None = None) -> str:
    """Write ``data`` as TOML: scalars, then ``[table]`` and ``[[array of tables]]`` blocks.

    ``comments`` maps (array name, entry index) to a comment line written above that entry.
    """
    out: list[str] = []
    tables = [(k, v) for k, v in data.items() if isinstance(v, dict) or _is_table_array(v)]
    table_keys = {k for k, _ in tables}
    for key, value in data.items():
        if key not in table_keys:
            out.append(f"{_key(key)} = {_value(value)}")
    for key, value in tables:
        if isinstance(value, dict):
            out.append("")
            out.append(f"[{_key(key)}]")
            out.extend(f"{_key(k)} = {_value(v)}" for k, v in value.items())
            continue
        for index, item in enumerate(value):
            out.append("")
            note = (comments or {}).get((key, index))
            if note:
                out.append(note)
            out.append(f"[[{_key(key)}]]")
            out.extend(f"{_key(k)} = {_value(v)}" for k, v in item.items())
    return "\n".join(out).lstrip("\n") + "\n"


def toml_load(path: Path) -> dict[str, Any]:
    """Read one TOML file."""
    with path.open("rb") as handle:
        return tomllib.load(handle)

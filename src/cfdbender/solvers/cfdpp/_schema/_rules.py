"""Evaluate the interdependency rules of the interface database on an input model.

Each rule is data: a kind, a level (``hard`` refuses, ``advisory`` warns) and a
message in our own words. The kinds are:

- ``requires``: when ``keyword`` is stated, ``other`` must be stated too;
- ``excludes``: ``keyword`` and ``other`` are not stated together;
- ``not_greater``: value ``index`` of ``keyword`` is not above value ``other_index``
  of ``other``, when both are stated;
- ``count``: value ``index`` of ``keyword`` equals the number of entries of
  ``section`` (a section the input does not hold counts as empty).

A rule measured ``broken`` or ``removed`` for the version is not applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfdbender.solvers.cfdpp._schema import Rule, Schema, Status
from cfdbender.solvers.cfdpp._schema._model import InputModel, Pair


@dataclass(frozen=True)
class Violation:
    """A rule the model breaks: the rule id, its level and a message naming the values."""

    rule: str
    level: str
    message: str


def check(model: InputModel, schema: Schema) -> list[Violation]:
    """Return every rule of ``schema`` that ``model`` breaks, in database order."""
    violations: list[Violation] = []
    for rule in schema.rules:
        if rule.status in (Status.BROKEN, Status.REMOVED):
            continue
        if any(
            schema.keywords[name].status is Status.DOCUMENTED
            for name in (rule.keyword, rule.other)
            if name is not None
        ):
            continue
        detail = _broken(rule, model)
        if detail is not None:
            violations.append(Violation(rule.id, rule.level, f"{rule.message} ({detail})"))
    return violations


def _broken(rule: Rule, model: InputModel) -> str | None:
    first = model.pair(rule.keyword)
    second = model.pair(rule.other) if rule.other is not None else None
    if rule.kind == "requires":
        if first is not None and second is None:
            return f"{rule.keyword} is stated without {rule.other}"
        return None
    if rule.kind == "excludes":
        if first is not None and second is not None:
            return f"{rule.keyword} and {rule.other} are both stated"
        return None
    if rule.kind == "not_greater":
        if first is None or second is None:
            return None
        left, right = _number(first, rule.index), _number(second, rule.other_index)
        if left is not None and right is not None and left > right:
            return (
                f"{rule.keyword} value {rule.index + 1} is {left}, above "
                f"{rule.other} value {rule.other_index + 1}, {right}"
            )
        return None
    if first is None or rule.section is None:
        return None
    stated = _number(first, rule.index)
    block = model.block(rule.section)
    held = len(block.items) if block is not None else 0
    if stated != held:
        return f"{rule.keyword} states {stated} and section {rule.section} holds {held}"
    return None


def _number(pair: Pair, index: int) -> float | None:
    if index >= len(pair.values):
        return None
    value = pair.values[index]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value

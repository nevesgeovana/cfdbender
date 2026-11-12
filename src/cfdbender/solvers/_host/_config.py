"""Validate common numeric inputs without loading a solver adapter."""

import math


def number(value: object, name: str) -> float:
    """Validate a finite numeric input, refusing booleans and nonnumbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)

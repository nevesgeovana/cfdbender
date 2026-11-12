"""Solver-independent workspace declarations and validation."""

from cfdbender.core.settings import WorkspaceConfig, WorkspaceError, read_init
from cfdbender.core.workspace import INPUT_KINDS, init_workspace, input_kinds

__all__ = [
    "INPUT_KINDS",
    "WorkspaceConfig",
    "WorkspaceError",
    "init_workspace",
    "input_kinds",
    "read_init",
]

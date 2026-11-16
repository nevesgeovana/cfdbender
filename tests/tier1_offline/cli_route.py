"""Route a verb to the command that owns it, as a user types it.

Tests that exercise a record, storage or log verb alongside matrix verbs call this
``main``: a verb that acts on what the workspace holds goes to cfd-workspace, any other to
cfd-matrix.
"""

from __future__ import annotations

from cfdbender.matrix.cli import MOVED
from cfdbender.matrix.cli import main as matrix_main
from cfdbender.workspace.cli import main as workspace_main


def main(argv: list[str] | None = None) -> int:
    """Run the argument list on the command that owns its verb and return its exit code."""
    words = list(argv or [])
    verb = next((word for word in words if not word.startswith("-")), None)
    return (workspace_main if verb in MOVED else matrix_main)(words)

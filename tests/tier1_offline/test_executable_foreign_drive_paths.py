from pathlib import PurePosixPath

import pytest

from cfdbender.core import executables
from cfdbender.core.settings import WorkspaceError


class PosixPathForTest(PurePosixPath):
    def expanduser(self):
        return self


def test_foreign_drive_paths_are_refused_on_posix(monkeypatch):
    monkeypatch.setattr(executables, "Path", PosixPathForTest)
    monkeypatch.setattr(
        executables, "_tables", lambda *args: ({"build": {"serial": "C:/vendor/solver"}}, {})
    )
    with pytest.raises(WorkspaceError, match="Windows.*path"):
        executables.resolve_fields(
            PurePosixPath("/workspace"), "cfdpp", "build", {"serial": "path"}
        )

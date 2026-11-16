"""Containment fixtures reach their assertions without PowerShell on PATH."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from test_review_paths import link_folder


@pytest.mark.skipif(os.name != "nt", reason="Windows junction fallback")
def test_junction_fallback_uses_system_shell_without_path(tmp_path, monkeypatch):
    system = tmp_path / "Windows"
    shell = system / "System32/WindowsPowerShell/v1.0/powershell.exe"
    shell.parent.mkdir(parents=True)
    shell.touch()
    monkeypatch.setenv("SystemRoot", str(system))
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def denied(*args, **kwargs):
        raise OSError("symlink privilege unavailable")

    monkeypatch.setattr(Path, "symlink_to", denied)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    link_folder(tmp_path / "linked folder", tmp_path / "target")
    args, kwargs = calls[0]
    assert args[0] == str(shell)
    assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW

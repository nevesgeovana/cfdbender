"""Audit real command paths, including restarted points and both mesh link types."""

import json
import os
import shutil
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_optcheck_identity as audit
from test_optcheck_identity import cli, small_threshold, workspace  # noqa: F401

from cfdbender.core.registry import RunRegistry
from cfdbender.solvers._host import _mesh
from cfdbender.workspace.cli import main as workspace_main


@pytest.mark.parametrize("kind", ["su2", "cgns"])
@pytest.mark.parametrize("link", ["symbolic", "hard"])
@pytest.mark.parametrize("hdf5", ["missing", "spy"])
def test_audit_restart_and_storage(tmp_path, monkeypatch, kind, link, hdf5):
    if link == "symbolic":
        source = tmp_path / "link-source"
        source.touch()
        try:
            (tmp_path / "link-probe").symlink_to(source)
        except OSError as error:
            pytest.skip(f"host cannot create symbolic links: {error}")
    root = workspace(tmp_path / "ws", kind)
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace('[groups]\nwing = ["wall"]\n', ""))
    if link == "hard":

        def refuse(*args):
            raise OSError("exercise hard-link fallback")

        monkeypatch.setattr(_mesh, "_symlink", refuse)
    calls = []

    def h5open(*args, **kwargs):
        calls.append(args)
        raise AssertionError("HDF5 opened a mesh")

    monkeypatch.setitem(
        sys.modules, "h5py", None if hdf5 == "missing" else SimpleNamespace(File=h5open)
    )
    monkeypatch.setenv("CFDBENDER_NO_READ_LARGE_INPUTS", "1")
    audit._SEEN.clear()
    audit._ACTIVE[0] = True
    try:
        assert cli(root, "plan") == 0
        assert cli(root, "run", "--local") == 0
        records = RunRegistry(root).read()
        staged = root / records[0].point.directory / f"mesh.{kind}"
        assert staged.is_symlink() == (link == "symbolic")
        assert os.path.samefile(staged, root / audit.mesh_name(kind))
        for record in records:
            (root / record.point.directory / "restart_flow.csv").write_bytes(
                b"PointID,Value\n0,1\n"
            )
        assert cli(root, "run", "--local", "--restart") == 0
        assert all(record.point.restart_from for record in RunRegistry(root).read())
        assert cli(root, "collect") == 0
        assert audit.main(["status", "--workspace", str(root)]) == 0
        assert workspace_main(["space-in-use", "--workspace", str(root)]) == 0
        recipe = root / "inputs/management/review.toml"
        recipe.parent.mkdir(exist_ok=True)
        recipe.write_text('[[delete_extensions]]\nextensions=[".unused"]\n')
        assert workspace_main(["free-space", "review", "--apply", "--workspace", str(root)]) == 0
        assert (
            workspace_main(
                [
                    "delete-sims",
                    "--sims",
                    "7001",
                    "--matrix-products",
                    "points-only",
                    "--workspace",
                    str(root),
                ]
            )
            == 0
        )
        assert audit._SEEN == []
        assert calls == []
    finally:
        audit._ACTIVE[0] = False

    # A storage sync is allowed to read files: document that separate surface explicitly.
    other = tmp_path / "other"
    shutil.copytree(root, other, symlinks=True)
    registry = RunRegistry(other)
    registry.replace_records(
        {r.point_id: replace(r, solver_version="review-spy") for r in registry.read()}
    )
    (root / "inputs/sync-workspaces.toml").write_text(
        "[workspaces]\nother=" + json.dumps(other.as_posix()) + "\n"
    )
    audit._SEEN.clear()
    audit._ACTIVE[0] = True
    try:
        result = workspace_main(
            ["sync", "all", "--prefer-other", "--overwrite", "--workspace", str(root)]
        )
        assert result == 0
        if link == "hard":
            assert len(audit._SEEN) == 6  # two workspaces, three staged hard links each
        else:
            assert audit._SEEN == []
        audit._SEEN.clear()
        assert workspace_main(["archive", str(other), "7001"]) == 0
        assert (
            workspace_main(
                [
                    "delete-sims",
                    "--sims",
                    "7001",
                    "--matrix-products",
                    "points-only",
                    "--apply",
                    "--workspace",
                    str(root),
                ]
            )
            == 0
        )
        assert audit._SEEN == []
    finally:
        audit._ACTIVE[0] = False


@pytest.mark.parametrize("kind", ["su2", "cgns"])
@pytest.mark.parametrize("hdf5", ["missing", "spy"])
def test_submission_with_hdf5_unavailable_or_spied(tmp_path, monkeypatch, kind, hdf5):
    root = workspace(tmp_path / "ws", kind, hpc=True)
    calls = []

    def h5open(*args, **kwargs):
        calls.append(args)
        raise AssertionError("HDF5 opened a mesh")

    monkeypatch.setitem(
        sys.modules, "h5py", None if hdf5 == "missing" else SimpleNamespace(File=h5open)
    )
    monkeypatch.setenv("CFDBENDER_NO_READ_LARGE_INPUTS", "1")
    audit._SEEN.clear()
    audit._ACTIVE[0] = True
    try:
        assert cli(root, "plan") == 0
        assert cli(root, "run", "--hpc", "h001") == 0
        assert cli(root, "collect") in (0, 1, 3)
        assert all(r.status == "SUBMITTED" for r in RunRegistry(root).read())
        assert audit._SEEN == []
        assert calls == []
    finally:
        audit._ACTIVE[0] = False

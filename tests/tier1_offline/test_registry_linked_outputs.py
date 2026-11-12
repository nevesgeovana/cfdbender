"""A recorded point whose staged mesh is a symbolic link out of sims/ still reads back.

On a host with symlink privilege (the cluster) the point's mesh output resolves to
inputs/geometry; the registry must judge the recorded path, not the link target. This host
stages a hard link, so the test plays the symlink by making path resolution follow the leaf.
"""

from pathlib import Path

import test_p0000_item3d_config as su2_case
from test_hpc_self_contained import submit

from cfdbender.core import registry
from cfdbender.core.registry import RunRegistry


def test_symlinked_mesh_output_keeps_the_record_readable(tmp_path, monkeypatch):
    su2_case.workspace(tmp_path)
    folder, _config = submit(tmp_path)
    target = tmp_path / "inputs/meshes/tiny.su2"
    real = registry.workspace_path

    def follows_links(root: Path, name: str) -> Path:
        if name.endswith("/mesh.su2"):
            return target.resolve()
        return real(root, name)

    monkeypatch.setattr(registry, "workspace_path", follows_links)
    records = RunRegistry(tmp_path).read()
    assert [r.status for r in records] == ["SUBMITTED"]
    assert any(name.endswith("/mesh.su2") for name in records[0].point.outputs)

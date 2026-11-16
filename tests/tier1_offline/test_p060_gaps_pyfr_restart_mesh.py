"""P060: a PyFR restart keeps the native mesh paired with its saved solution."""

import json

from test_p060_gaps_restart import call, prepared, solution

from cfdbender.core.registry import RunRegistry


def test_p060_pyfr_restart_uses_the_saved_native_mesh_without_import(tmp_path):
    source, argv = prepared(tmp_path, "pyfr")
    solution(tmp_path, source, "pyfr")
    native = (tmp_path / source.point.directory / "mesh.pyfrm").read_bytes()
    assert call([*argv, "--restart"]) == 0
    record = RunRegistry(tmp_path).read()[0]
    folder = tmp_path / record.point.directory
    commands = json.loads((folder / "execution.json").read_bytes())["argv"]
    assert len(commands) == 1 and "restart" in commands[0]
    assert (folder / "mesh.pyfrm").read_bytes() == native
    assert "mesh.pyfrm" in getattr(record.point, "restart_sha256", {})

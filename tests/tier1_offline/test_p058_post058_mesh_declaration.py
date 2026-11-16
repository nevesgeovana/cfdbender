"""P058-15: a QPROP setup declares a mesh frame before the blade is read.

Two OBJ files of the same blade, one in aft/right/up and one in forward/left/up,
must produce the same section plan. The source stays intact, and the output records
the declaration, conversion and original and converted byte digests.
"""

import hashlib
import json

import pytest
import qprop_workspace as qw

from cfdbender.matrix.cli import main


def test_p058_declared_mesh_is_converted_before_planning_and_recorded(tmp_path):
    standard = qw.workspace(tmp_path / "standard")
    other = qw.workspace(tmp_path / "other")
    expected = qw.adapter().translate(qw.prepared(standard))
    blade = other / "inputs/geometry/blade.obj"
    converted = blade.read_bytes()
    lines = []
    for line in converted.decode("ascii").splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            fields[1] = str(-float(fields[1]))
            fields[2] = str(-float(fields[2]))
            line = " ".join(fields)
        lines.append(line)
    source = ("\n".join(lines) + "\n").encode("ascii")
    blade.write_bytes(source)
    setup = other / "inputs/setups/s001.toml"
    setup.write_bytes(b'mesh_frame = ["forward", "left", "up"]\n' + setup.read_bytes())
    try:
        actual = qw.adapter().translate(qw.prepared(other))
    except ValueError as error:
        pytest.fail(f"the declared right-handed OBJ frame must be converted: {error}")
    assert actual == expected
    assert main(["run", "study", "--workspace", str(other)]) == 0
    assert main(["collect", "study", "--workspace", str(other)]) == 0
    point = next((other / "sims/sim_1001/datapoints").iterdir())
    frame = json.loads((point / "sections.json").read_bytes())["mesh_frame"]
    assert frame["declared"] == ["forward", "left", "up"]
    assert frame["matrix"] == [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert frame["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert frame["converted_sha256"] != frame["source_sha256"]
    products = json.loads((other / "post/study/products.json").read_bytes())
    assert products["frames"]["mesh"]["points"]["1001/" + point.name] == frame
    assert blade.read_bytes() == source


def test_p058_bad_mesh_declaration_is_named_before_planning(tmp_path):
    qw.workspace(tmp_path)
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_bytes(b'mesh_frame = ["aft", "left", "up"]\n' + setup.read_bytes())
    with pytest.raises(ValueError, match="mesh_frame.*left-handed"):
        qw.prepared(tmp_path)

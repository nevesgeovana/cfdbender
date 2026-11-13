"""QPROP stages its declared text outputs with LF on every platform."""

from qprop_workspace import adapter, prepared, workspace


def test_staged_text_outputs_carry_no_crlf(tmp_path):
    workspace(tmp_path)
    point = prepared(tmp_path)
    adapter().execute(point, check_ram=lambda: None)
    folder = tmp_path / point.point.directory
    for name in ("sections.json", "propeller.prop", "qcon.def", "execution.json"):
        data = (folder / name).read_bytes()
        assert data and b"\r" not in data, name

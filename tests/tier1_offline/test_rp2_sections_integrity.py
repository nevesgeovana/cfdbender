"""RP-2: rejected QPROP output must not become a sectional product."""

import json

import qprop_workspace as qw

from cfdbender.matrix.cli import main


def test_rejected_changed_stdout_is_not_exported_as_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDBENDER_QPROP_FAKE_RADIAL", "1")
    qw.workspace(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    stdout = next((tmp_path / "sims/sim_1001/datapoints").iterdir()) / "stdout.log"
    with stdout.open("a", encoding="utf-8") as stream:
        stream.write("\nchanged after execution\n")
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 1
    data = json.loads((tmp_path / "post/study/products.json").read_text(encoding="utf-8"))
    assert not any(p["kind"] == "sections" for p in data["products"])
    assert any(s["product"].startswith("sections/") for s in data["skipped"])

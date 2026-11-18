"""Identity metadata must remain distinguishable in workspace comparisons."""

import argparse
import json
from types import SimpleNamespace

from cfdbender.matrix import queries


def test_diff_detects_changed_identity_before_formatting(monkeypatch, capsys):
    a = {"inputs_sha256": {"mesh": "identity:size=10;mtime_ns=1"}}
    b = {"inputs_sha256": {"mesh": "identity:size=11;mtime_ns=1"}}
    monkeypatch.setattr(queries, "_open", lambda args: None)
    monkeypatch.setattr(
        queries,
        "load_runs",
        lambda registry: [
            SimpleNamespace(run_id="a", record=a),
            SimpleNamespace(run_id="b", record=b),
        ],
    )
    monkeypatch.setattr(queries, "encode_record", lambda record: record)
    args = argparse.Namespace(run_a="a", run_b="b", json=True, csv=False)
    assert queries.run_diff(args) == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows == [{"field": "inputs_sha256.mesh", "a": "not hashed", "b": "not hashed"}]

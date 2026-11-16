"""P060: a plan downgraded from cluster to local still checks this host's build."""

from test_p0000_item3f_plan import prepared

from cfdbender.matrix import cli
from cfdbender.solvers import mock


def test_p060_downgraded_plan_checks_builds_and_local_executable(tmp_path, monkeypatch):
    prepared(tmp_path)
    monkeypatch.setattr(mock, "CAPABILITIES", frozenset())
    calls = []
    monkeypatch.setattr(cli, "check_builds", lambda *a: calls.append("build"))
    monkeypatch.setattr(cli, "check_local_executables", lambda *a: calls.append("executable"))
    assert cli.main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert calls == ["build", "executable"]

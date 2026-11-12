"""Review regression for invalid submit arguments."""

import pytest
from test_p0000_item3f_plan import cli, prepared, tree


@pytest.mark.parametrize("command", ["[1]", '["valid", 42]'])
def test_rev7_non_string_submit_refused_before_writes(tmp_path, command):
    path = prepared(tmp_path)
    text = path.read_text(encoding="utf-8")
    lines = [
        f"command = {command}" if line.startswith("command = ") else line
        for line in text.splitlines()
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    before = tree(tmp_path)
    assert cli(tmp_path, "run", "--hpc", "h001") == 2
    assert tree(tmp_path) == before

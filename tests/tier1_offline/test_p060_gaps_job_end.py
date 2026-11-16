"""P060: a job's end markers gate pickup even when its numerical output exists."""

import json

from test_p0000_item3f_collect import outputs
from test_p0000_item3f_plan import prepared

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def test_p060_collect_requires_every_recorded_job_end_file(tmp_path):
    profile = prepared(tmp_path)
    profile.write_text('job_end_files = ["job.done", "status/finished"]\n' + profile.read_text())
    argv = ["study", "--workspace", str(tmp_path)]
    assert main(["run", *argv]) == 0
    record = RunRegistry(tmp_path).read()[0]
    folder = tmp_path / record.point.directory
    outputs(tmp_path, record.point)
    assert main(["collect", *argv]) == 3
    (folder / "job.done").touch()
    assert main(["collect", *argv]) == 3
    (folder / "status").mkdir()
    (folder / "status/finished").touch()
    assert main(["collect", *argv, "--interval", "0"]) == 0
    stored = json.loads((tmp_path / "runs.json").read_bytes())["records"][0]
    assert stored["submission"]["job_end_files"] == [
        f"{record.point.directory}/job.done",
        f"{record.point.directory}/status/finished",
    ]

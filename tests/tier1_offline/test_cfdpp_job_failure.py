"""Execute generated cluster scripts with failing stand-ins, including paths with spaces."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cfdbender.solvers.cfdpp import _execute


@pytest.mark.parametrize("failed_step", ["partition", "solver"])
def test_job_preserves_failure_and_stops_later_steps(tmp_path: Path, failed_step: str) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("a POSIX shell is required to execute the job script")
    target = tmp_path / "point with spaces"
    target.mkdir()
    steps = [
        _execute.Step("partition", ("sh", "-c", "touch mcpusin.bin.2; exit 7"), "log.partition"),
        _execute.Step("solver", ("sh", "-c", "touch solver.started; exit 9"), "stdout.log"),
    ]
    if failed_step == "solver":
        steps = steps[1:]
    script = target / "cfdpp_job.sh"
    script.write_text(_execute.job_script(steps, 2), encoding="ascii")
    result = subprocess.run([shell, str(script)], capture_output=True, check=False)
    assert result.returncode == (7 if failed_step == "partition" else 9)
    assert (target / "job.done").is_file()
    assert json.loads((target / "execution.json").read_text())["exit_code"] == result.returncode
    assert (target / "solver.started").exists() == (failed_step == "solver")


@pytest.mark.parametrize("shell_name", ["sh", "bash"])
@pytest.mark.parametrize("failed_step", ["partition", "solver", "post"])
def test_all_rendered_steps_propagate_failure_in_each_shell(tmp_path, shell_name, failed_step):
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    steps = []
    for name in ("partition", "solver", "post"):
        status = 7 if name == failed_step else 0
        command = f"touch {name}.started; touch mcpusin.bin.2; exit {status}"
        steps.append(_execute.Step(name, (shell, "-c", command), "unused"))
    script = tmp_path / "job.sh"
    script.write_text(_execute.job_script(steps, 2), encoding="ascii")
    result = subprocess.run([shell, str(script)], capture_output=True, check=False)
    assert result.returncode == 7
    assert json.loads((tmp_path / "execution.json").read_text())["exit_code"] == 7
    assert (tmp_path / f"{failed_step}.started").exists()
    names = ["partition", "solver", "post"]
    assert all(
        not (tmp_path / f"{name}.started").exists()
        for name in names[names.index(failed_step) + 1 :]
    )

"""Execute a planned profile argv once, without a shell or a Windows console."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from ._plan import SubmissionPlan

SUBMIT_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class SubmitResult:
    """Submit-client evidence; its return code says nothing about solver convergence."""

    return_code: int
    stdout: str
    stderr: str
    started_at: str
    finished_at: str
    timeout_s: float
    timed_out: bool


def submit(job: SubmissionPlan) -> SubmitResult:
    """Write a descriptor and capture the profile's bounded submit client to files.

    Parameters
    ----------
    job : SubmissionPlan
        Fully rendered descriptor and argv. The caller has claimed the point
        and created its folder; existing descriptor and capture files refuse.

    Returns
    -------
    SubmitResult
        Process result with up to 64 KiB from each captured stream. Complete
        bytes stay in submit.stdout.txt and submit.stderr.txt. Code 124 means
        timeout, with scheduler acceptance unknown; 127 means launch failed.
    """
    folder = job.descriptor_path.parent
    with job.descriptor_path.open("x", encoding="utf-8", newline="") as descriptor:
        descriptor.write(job.descriptor_text)
    started = datetime.now(UTC).isoformat()
    timed_out = False
    with (
        (folder / "submit.stdout.txt").open("xb") as stdout,
        (folder / "submit.stderr.txt").open("xb") as stderr,
    ):
        try:
            result = subprocess.run(
                job.command,
                cwd=folder,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                timeout=SUBMIT_TIMEOUT_S,
                check=False,
                env=os.environ.copy(),
                # The constant exists only on Windows; 0 elsewhere (mypy checks both platforms).
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
            stderr.write(
                (
                    f"the submit command {job.command[0]!r} did not return within "
                    f"{SUBMIT_TIMEOUT_S} s; whether the scheduler took the job is unknown; "
                    f"the descriptor is {job.descriptor_path}" + chr(10)
                ).encode("utf-8")
            )
        except OSError as error:
            code = 127
            stderr.write(
                (
                    f"the submit command {job.command[0]!r} could not be run: {error}; "
                    "the descriptor was written and nothing was submitted." + chr(10)
                ).encode("utf-8")
            )
    captures = []
    for name in ("submit.stdout.txt", "submit.stderr.txt"):
        with (folder / name).open("rb") as stream:
            captures.append(stream.read(65536).decode("utf-8", errors="replace"))
    return SubmitResult(
        code,
        captures[0],
        captures[1],
        started,
        datetime.now(UTC).isoformat(),
        SUBMIT_TIMEOUT_S,
        timed_out,
    )

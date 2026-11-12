"""Plan portable HPC jobs without writing files or launching processes.

Read ``inputs/hpc/h001.toml`` with :func:`read_hpc_profile`, passing the
workspace's registered solver versions. :func:`plan_submission` prepares a
descriptor and argument list for a later executor. Translation tables declare
scheduler names; they do not verify the version installed on a cluster.
"""

from ._plan import SubmissionPlan, plan_submission
from ._profiles import HpcProfile, read_hpc_profile, resolve_hpc_profile
from ._render import build_submit_command, render_descriptor

__all__ = [
    "HpcProfile",
    "SubmissionPlan",
    "plan_submission",
    "read_hpc_profile",
    "resolve_hpc_profile",
    "build_submit_command",
    "render_descriptor",
]

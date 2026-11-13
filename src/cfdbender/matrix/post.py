"""The post stage as a function: rebuild every product from runs.json and saved outputs.

``post_workspace`` is the reading half of ``collect``: it reads the recorded
points through the workspace's adapter, rebuilds ``campaign_summary.json``, the
CSV tables, ``products.json`` and ``post.log`` under ``post/<matrix>/``, and never
launches a solver. It changes no record: a SUBMITTED point stays outstanding
until ``collect`` picks it up. Calling it twice over unchanged records writes
the same bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cfdbender.matrix.collection import CollectReport, collect_workspace


def post_workspace(
    root: str | Path,
    matrix: str | None = None,
    *,
    sims: Sequence[str] = (),
    points: Sequence[str] = (),
    runs: str = "runs.json",
    archive: bool = False,
) -> CollectReport:
    """Rebuild every post product of the recorded matrices without running anything.

    Parameters
    ----------
    root : str or Path
        Workspace containing init.toml and runs.json.
    matrix : str, optional
        Matrix stem or filename; omit for every recorded matrix.
    sims : sequence of str, optional
        POL ids to read; empty reads every simulation.
    points : sequence of str, optional
        Point names to read; empty reads every point.
    runs : str, optional
        The run manifest to read.
    archive : bool, optional
        Keep a copy of each replaced product under archive/<stamp>/.

    Returns
    -------
    CollectReport
        Counts, the summaries, tables and the products and log written.
    """
    return collect_workspace(
        root, matrix, sims=sims, points=points, runs=runs, archive=archive, pickup=False
    )

"""The plan receipt: what a plan measured, and the check that a run is about that.

``post/<matrix>/plan.json`` holds the SHA-256 of the matrix file and, for every
planned point, its ``inputs_sha256``. A run refuses a missing receipt, a matrix
that changed since it was planned, a point nobody planned and an input whose
bytes changed, each with a message that says what to do. Planning a selection
(``--sims`` or ``--points``) adds or replaces only its own points, keeping the
receipt of the others while the matrix is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from cfdbender.core.plan import RunPlan, workspace_path

PLAN_REQUIRED_MESSAGE = (
    "no plan for this matrix, and a run needs one. Run `cfd-matrix plan <matrix>` first: "
    "it pre-flights every point without spending any solver time and writes the receipt "
    "this refusal is asking for."
)


def receipt_path(root: Path, matrix: str) -> Path:
    """Return post/<matrix>/plan.json of a workspace, for a matrix name or stem."""
    return workspace_path(Path(root), f"post/{Path(matrix).stem}/plan.json")


def _matrix_digest(plan: RunPlan) -> str:
    return hashlib.sha256(workspace_path(plan.root, plan.matrix).read_bytes()).hexdigest()


def _read(plan: RunPlan) -> tuple[Path, dict[str, object] | None, str | None]:
    path = receipt_path(plan.root, plan.matrix)
    if not path.is_file():
        return path, None, f"{PLAN_REQUIRED_MESSAGE} Expected it at {path}."
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        return path, None, f"the plan at {path} cannot be read: {error}. {PLAN_REQUIRED_MESSAGE}"
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") != 1
        or data.get("solver") != plan.solver
        or data.get("matrix") != plan.matrix
        or not isinstance(data.get("points"), list)
        or not _valid_points(data["points"])
    ):
        return (
            path,
            None,
            f"the plan at {path} is not a plan receipt. {PLAN_REQUIRED_MESSAGE}",
        )
    return path, data, None


def _valid_points(points: list[object]) -> bool:
    seen: set[str] = set()
    for entry in points:
        if not isinstance(entry, dict):
            return False
        name = entry.get("point_id")
        if not isinstance(name, str) or not name or name in seen:
            return False
        if not isinstance(entry.get("inputs_sha256"), dict):
            return False
        seen.add(name)
    return True


def write_receipt(plan: RunPlan, profiles: tuple[str, ...] = ()) -> Path:
    """Write post/<matrix>/plan.json for the plan's points and return its path.

    Points of an earlier receipt that the plan does not name are kept while the
    matrix digest is unchanged, and dropped when the matrix changed.
    """
    digest = _matrix_digest(plan)
    _, previous, _ = _read(plan)
    kept: dict[str, dict[str, object]] = {}
    if previous is not None and previous.get("matrix_sha256") == digest:
        kept = {str(p["point_id"]): p for p in previous["points"] if isinstance(p, dict)}  # type: ignore[attr-defined]
    for index, point in enumerate(plan.points):
        kept[point.point_id] = {
            "point_id": point.point_id,
            "inputs_sha256": dict(sorted(point.inputs_sha256.items())),
        }
        if profiles:
            kept[point.point_id]["profile"] = profiles[index]
            kept[point.point_id]["profile_sha256"] = hashlib.sha256(
                workspace_path(plan.root, profiles[index]).read_bytes()
            ).hexdigest()
    data = {
        "schema_version": 1,
        "matrix": plan.matrix,
        "solver": plan.solver,
        "matrix_sha256": digest,
        "points": [kept[name] for name in sorted(kept)],
    }
    target = receipt_path(plan.root, plan.matrix)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".plan-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(data, indent=2, sort_keys=True) + chr(10)).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def receipt_error(plan: RunPlan, profiles: tuple[str, ...] = ()) -> str | None:
    """Return why a run of this plan may not proceed, or None when the receipt answers for it."""
    path, data, problem = _read(plan)
    if problem is not None or data is None:
        return problem
    planned = data.get("matrix_sha256")
    now = _matrix_digest(plan)
    if planned != now:
        return (
            f"the matrix changed since it was planned: the plan at {path} measured "
            f"{str(planned)[:12]} and {Path(plan.matrix).name} is {now[:12]} now. "
            "Plan it again; a plan that measured another study is not a plan for this one."
        )
    recorded = {str(p["point_id"]): p for p in data["points"] if isinstance(p, dict)}  # type: ignore[attr-defined]
    for index, point in enumerate(plan.points):
        entry = recorded.get(point.point_id)
        if entry is None:
            return (
                f"point {point.point_id} was not planned: the plan at {path} does not "
                "name it. Plan it again."
            )
        if profiles and entry.get("profile") != profiles[index]:
            return f"profile changed since point {point.point_id} was planned. Plan it again."
        if (
            profiles
            and entry.get("profile_sha256")
            != hashlib.sha256(workspace_path(plan.root, profiles[index]).read_bytes()).hexdigest()
        ):
            return (
                f"profile contents changed since point {point.point_id} was planned. Plan it again."
            )
        before = entry.get("inputs_sha256")
        before = before if isinstance(before, dict) else {}
        for name, digest in point.inputs_sha256.items():
            if before.get(name) != digest:
                return (
                    f"input changed since it was planned: {name} (point {point.point_id}). "
                    "Plan it again."
                )
    return None


def require_receipt(plan: RunPlan, profiles: tuple[str, ...] = ()) -> None:
    """Raise ValueError with the refusal text unless the plan has a fresh receipt."""
    if (message := receipt_error(plan, profiles)) is not None:
        raise ValueError(message)

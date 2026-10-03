"""Bind native outcomes to the exact task, layout, expert and execution files."""

import math
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest, read_json, relative_path


def verify_case(case, root):
    root = Path(root).resolve()
    path = (root / relative_path(case["file"])).resolve()
    if (
        not path.is_relative_to(root)
        or digest(read_json(path)) != case["layout_sha256"]
    ):
        raise ValueError("Layout changed before or during execution")
    for name, sha in case["validation_artifacts"].items():
        artifact = (root / relative_path(name)).resolve()
        if not artifact.is_relative_to(root) or file_digest(artifact) != sha:
            raise ValueError("Layout initialization evidence changed")


def episode_receipt(
    path, *, directory, task, split, layout_sha256, expert, harness_sha256
):
    root, path = Path(directory).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Episode receipt must be inside its execution directory")
    receipt = read_json(path)
    expected = dict(
        schema="physicalrsi.robodojo.episode/v1",
        task=task,
        split=split,
        layout_sha256=layout_sha256,
        expert=expert["name"],
        revision=expert["revision"],
        harness_sha256=harness_sha256,
    )
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Native episode identity differs from the requested execution")
    if (
        receipt.get("state") != "completed"
        or receipt.get("natural_terminal") is not True
        or type(receipt.get("native_exit_code")) is not int
        or receipt["native_exit_code"] != 0
    ):
        raise ValueError("Failed or truncated native execution is not task inability")
    score = receipt.get("episode_score")
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or type(receipt.get("success")) is not bool
    ):
        raise ValueError("Native score/success is missing or invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Native outcome requires execution artifacts")
    evidence = {str(path): file_digest(path)}
    for name, sha in artifacts.items():
        artifact = (root / relative_path(name)).resolve()
        if not artifact.is_relative_to(root) or file_digest(artifact) != sha:
            raise ValueError("Native execution artifact escaped or changed")
        evidence[str(artifact)] = sha
    return dict(
        task=task,
        split=split,
        layout_sha256=layout_sha256,
        expert=expert["name"],
        revision=expert["revision"],
        state="completed",
        natural_terminal=True,
        episode_score=score,
        success=receipt["success"],
        evidence=evidence,
        diagnostics=receipt.get("diagnostics", {}),
    )


def cases(runtime, *, tasks, count, split, output, comparison=None, excluded=()):
    """The runtime generates/validates layouts; the harness checks their binding."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = runtime.cases(
        tasks=tasks, count=count, split=split, output=root, comparison=comparison
    )
    if len(rows) != len(tasks) * count:
        raise ValueError("Incomplete task/layout coverage")
    seen, counts = set(excluded), {task: 0 for task in tasks}
    for row in rows:
        if row["task"] not in counts or row["split"] != split:
            raise ValueError("Unexpected task or leaked split")
        path = (root / relative_path(row["file"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Generated layout must stay within its cohort")
        sha = digest(read_json(path))
        if sha != row["layout_sha256"] or sha in seen:
            raise ValueError("Changed, duplicate or reused development geometry")
        seen.add(sha)
        counts[row["task"]] += 1
        if row.get("validity") != "passed" or not row.get("validation_artifacts"):
            raise ValueError("Layout requires native initialization validity evidence")
        for name, expected in row["validation_artifacts"].items():
            artifact = (root / relative_path(name)).resolve()
            if not artifact.is_relative_to(root) or file_digest(artifact) != expected:
                raise ValueError("Layout validity evidence changed")
    if any(number != count for number in counts.values()):
        raise ValueError("Unbalanced task/layout coverage")
    return rows

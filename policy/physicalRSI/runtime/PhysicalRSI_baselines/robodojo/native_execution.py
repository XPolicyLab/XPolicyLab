"""Supervise native workers and bind their evidence to a requested execution.

Model-service identity/ownership is established by the calling runtime. This
module never infers a checkpoint identity from a WebSocket endpoint string.
"""

import math
from pathlib import Path

from PhysicalRSI_core.infra.batch import ProcessJob, run_batch
from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    read_json,
    relative_path,
)

from .evidence import episode_receipt
from .process_env import with_project_path


def _inside(root, name):
    path = root / relative_path(name)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("Native evidence escaped its execution directory")
    return resolved


def finalize(root, request):
    """Publish receipts only after successful cleanup and exact native coverage."""
    root = Path(root).resolve()
    process = read_json(_inside(root, "processes/simulator/process.json"))
    if (
        process.get("state") != "completed"
        or type(process.get("returncode")) is not int
        or process["returncode"] != 0
    ):
        raise ValueError("Native process failed, timed out or did not clean up")
    if (
        read_json(_inside(root, "request.json")) != request
        or read_json(_inside(root, "run/request.json")) != request
    ):
        raise ValueError("Native execution request changed")
    if (root / "run/failed.json").exists():
        raise ValueError("Native worker recorded a failure")
    completion = read_json(_inside(root, "run/completed.json"))
    observed = read_json(_inside(root, "run/observed.json"))
    outcomes = completion["outcomes"]
    if (
        completion.get("state") != "completed"
        or observed.get("failures")
        or observed.get("outcomes") != outcomes
    ):
        raise ValueError("Native observer and completion disagree")
    layouts = request["layouts"]
    ids = [row["layout_id"] for row in outcomes]
    if any(type(index) is not int for index in ids) or sorted(ids) != list(
        range(len(layouts))
    ):
        raise ValueError("Native execution did not cover the exact requested cohort")
    layout_root = Path(request["layout_root"]).resolve()
    results = {}
    for row in outcomes:
        case = layouts[row["layout_id"]]
        layout = _inside(layout_root, case["file"])
        if (
            case["task"] != request["task"]
            or case["split"] != request["split"]
            or digest(read_json(layout)) != case["layout_sha256"]
        ):
            raise ValueError("Native input layout identity changed")
        result = _inside(root, "run/" + row["result_file"])
        if file_digest(result) != row["result_sha256"]:
            raise ValueError("Native score artifact changed")
        results[result] = read_json(result)["details"]
        matches = [
            entry
            for entry in results[result].values()
            if entry["layout_id"] == row["layout_id"]
        ]
        score = row["episode_score"]
        if (
            len(matches) != 1
            or matches[0]["score"] != score
            or matches[0]["success"] != row["success"]
        ):
            raise ValueError("Native score differs from observer evidence")
        native_index = next(
            key
            for key, entry in results[result].items()
            if entry["layout_id"] == row["layout_id"]
        )
        tag = "success" if row["success"] else "fail"
        videos = list(
            result.parent.glob(f"episode_{int(native_index):07d}_*_{tag}.mp4")
        )
        if not videos or any(
            not video.is_file() or video.stat().st_size == 0 for video in videos
        ):
            raise ValueError("Native episode requires recorded camera evidence")
        if (
            row.get("natural_terminal") is not True
            or type(row["success"]) is not bool
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Invalid native terminal outcome")
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Native artifacts must not be symlinks")
        if path.is_file():
            name = str(path.relative_to(root))
            artifacts[name] = file_digest(_inside(root, name))
    receipts = []
    receipt_root = root / "episodes"
    receipt_root.mkdir(exist_ok=False)
    for row in outcomes:
        case = layouts[row["layout_id"]]
        receipt = receipt_root / f"{row['layout_id']}.json"
        atomic_json(
            receipt,
            dict(
                schema="physicalrsi.robodojo.episode/v1",
                task=request["task"],
                split=request["split"],
                layout_sha256=case["layout_sha256"],
                expert=request["expert"]["name"],
                revision=request["expert"]["revision"],
                harness_sha256=request["harness_sha256"],
                state="completed",
                natural_terminal=True,
                native_exit_code=0,
                episode_score=row["episode_score"],
                success=row["success"],
                artifacts=artifacts,
                diagnostics=dict(
                    action_count=row["action_count"], step_limit=row["step_limit"]
                ),
            ),
        )
        episode_receipt(
            receipt,
            directory=root,
            task=request["task"],
            split=request["split"],
            layout_sha256=case["layout_sha256"],
            expert=request["expert"],
            harness_sha256=request["harness_sha256"],
        )
        receipts.append(receipt)
    return receipts


def _run(request, *, python, environment, output, timeout_s=1800, cancelled=None):
    """Run one cohort without retries; failed attempts retain their entire tree."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    request_path = root / "request.json"
    atomic_json(request_path, request)
    job = ProcessJob(
        "simulator",
        (
            str(python),
            "-m",
            "PhysicalRSI_baselines.robodojo.native_worker",
            str(request_path),
            str(root / "run"),
        ),
        root,
        with_project_path(environment),
        timeout_s,
    )
    run_batch([job], root / "processes", workers=1, cancelled=cancelled)
    return root


def execute(request, **options):
    return finalize(_run(request, **options), request)


def initialize(request, **options):
    from copy import deepcopy
    from .native_initialization import finalize as finalize_initialization

    request = deepcopy(request)
    request["mode"] = "initialize"
    request["num_envs"] = len(request["layouts"])
    if not 1 <= request["num_envs"] <= 10:
        raise ValueError("Initialization supports one exact batch of up to ten layouts")
    return finalize_initialization(_run(request, **options), request)

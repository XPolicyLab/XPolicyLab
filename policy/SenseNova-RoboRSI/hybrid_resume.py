"""Bind the original own-episode recovery to the official XPolicyLab adapter.

The source recovery policy and recorded-joint replay/validation stay unchanged.
This module supplies host-owned identity, stopped-writer checks and attempt
records for the websocket adapter, which doesn't use RoboRSI Runner receipts.
No resume source or host receipt is mounted into the model sandbox.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
import time
import types
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    """Publish a host record atomically, without replacing another attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".resume-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def workspace_digest(path: Path) -> str:
    """Seal own files and link text, matching the original snapshot semantics."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Resume needs the retained own-episode workspace")
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        h.update(item.relative_to(path).as_posix().encode() + b"\0")
        if item.is_symlink():
            h.update(b"link\0" + os.readlink(item).encode() + b"\0")
        elif item.is_file():
            h.update(b"file\0" + bytes.fromhex(digest(item)))
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular recovery record: {path.name}")
    return json.loads(path.read_text())


def process_identity(pid: int) -> str | None:
    """Distinguish a writer from an unrelated process that reused its PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except (OSError, IndexError):
        return None


def assert_stopped(root: Path, owner: dict) -> None:
    pid = owner.get("pid")
    if (
        not owner.get("closed")
        and pid
        and process_identity(pid) == owner.get("process_start")
    ):
        raise ValueError("Stop the source policy writer before preparing resume")
    if owner.get("closed") is not True and not pid:
        raise ValueError("Source writer has no verified stopped-process identity")


def declared_identity(model_cfg: dict, run_id: str) -> dict:
    """Bind deployment task/seed/layout without reading evaluator hidden state."""
    task = model_cfg["task_name"]
    seed = int(model_cfg.get("seed") or 0)
    match = re.search(r"-" + re.escape(task) + rf"-s{seed}-layout(\d+)(?:-|$)", run_id)
    layout = os.environ.get("ROBORSI_NATIVE_LAYOUT_ID", model_cfg.get("resume_layout"))
    if layout is not None:
        layout = int(layout)
        if layout < 0:
            raise ValueError("Native layout must be nonnegative")
    if layout is None and match:
        layout = int(match[1])
    return {"task": task, "eval_seed": seed, "layout": layout, "run_id": run_id or None}


def validate_native_identity(native: dict, identity: dict, *, step: int | None = None,
                             step_limit: int | None = None, scene_sha256: str | None = None) -> dict:
    """Require simulator-issued scene identity before replay or continuation.

    A deployment label is insufficient. This receipt is attached on the native
    side only after the official reset has loaded and verified the saved scene.
    """
    if not isinstance(native, dict) or native.get("schema") != 1:
        raise ValueError("Missing verified native scene identity")
    for key in ("task", "eval_seed", "layout", "run_id"):
        expected = identity.get(key)
        if expected is not None and native.get(key) != expected:
            raise ValueError(f"Native scene differs from requested episode: {key}")
    sha = native.get("scene_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Native scene hash is missing or invalid")
    if scene_sha256 is not None and sha != scene_sha256:
        raise ValueError("Native scene hash changed across resume")
    if type(native.get("control_step")) is not int or native["control_step"] < 0:
        raise ValueError("Invalid native control step")
    if type(native.get("step_limit")) is not int or native["step_limit"] <= 0:
        raise ValueError("Invalid native step limit")
    if step is not None and native["control_step"] != step:
        raise ValueError("Native control step differs from acknowledged action history")
    if step_limit is not None and native["step_limit"] != step_limit:
        raise ValueError("Native step limit differs from original task budget")
    return copy.deepcopy(native)


def source_identity(root: Path, *, trace_root: Path | None = None) -> dict:
    owner_path = root / "episode-owner.json"
    if owner_path.is_file():
        return read_json(owner_path)
    # Adopt a legacy adapter source only through its real archive provenance.
    # This is explicitly different from manufacturing a RoboRSI Runner receipt.
    cfg = read_json(root / "agent/controller/config.json")
    episode = cfg["test_episodes"][0]
    task, seed = episode["task"], episode["eval_seed"]
    candidates = set()
    error = root / "trajectory_archive_error.txt"
    if error.is_file():
        pattern = (
            r"/([^/]+-"
            + re.escape(task)
            + rf"-s{seed}-layout\d+(?:-[^/]*)?)/"
            + re.escape(task)
            + r"/"
        )
        candidates.update(re.findall(pattern, error.read_text()))
    if trace_root:
        for manifest in trace_root.glob(f"*/{task}/*/manifest.json"):
            data = read_json(manifest)
            if data.get("source") == str(root) and data.get("task") == task:
                candidates.add(data["run_id"])
    if len(candidates) != 1:
        raise ValueError(
            "Legacy source lacks unique task/seed/layout archive provenance"
        )
    identity = declared_identity({"task_name": task, "seed": seed}, candidates.pop())
    return {"format": "legacy_archive_provenance", "identity": identity}


def validate_trajectory_id(value: str) -> str:
    """Keep the same safe header identity throughout an episode's attempts."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value
    ):
        raise ValueError("trajectory_id must be 1-128 safe ASCII characters")
    return value


def source_trajectory_id(
    root: Path, owner: dict | None = None, *, _seen: set[Path] | None = None
) -> tuple[str, dict[str, str]]:
    """Recover an existing header ID and seal every record used to identify it.

    Older attempts used their directory UUID as the header. Follow their own
    recovery lineage first so adopting an old resumed attempt preserves the
    original episode's ID rather than the former per-attempt replacement.
    """
    root = Path(root).absolute()
    seen = set() if _seen is None else set(_seen)
    canonical = root.resolve()
    if root.is_symlink() or canonical in seen:
        raise ValueError("Trajectory recovery lineage contains a cycle or symlink")
    seen.add(canonical)
    owner = source_identity(root) if owner is None else owner
    identity = owner["identity"]
    seals = {}
    owner_path = root / "episode-owner.json"
    if owner_path.exists() or owner_path.is_symlink():
        read_json(owner_path)
        seals[str(owner_path)] = digest(owner_path)
    explicit = []
    if "trajectory_id" in owner:
        explicit.append(validate_trajectory_id(owner["trajectory_id"]))
    legacy = []
    records = [root.parent / "trajectory.json"]
    if root.name == "runtime":
        records.append(root.parent / "manifest.json")
    for path in records:
        if not (path.exists() or path.is_symlink()):
            continue
        data = read_json(path)
        if data.get("task") != identity["task"] or (
            "seed" in data and data["seed"] != identity["eval_seed"]
        ):
            raise ValueError("Trajectory record belongs to another episode")
        if path.name == "manifest.json" and data.get("format") != "xpolicylab_episode_archive_v1":
            raise ValueError("Invalid trajectory archive identity")
        seals[str(path)] = digest(path)
        if "trajectory_id" in data:
            explicit.append(validate_trajectory_id(data["trajectory_id"]))
        elif "episode_id" in data:
            legacy.append(validate_trajectory_id(data["episode_id"]))
    if explicit:
        if len(set(explicit)) != 1:
            raise ValueError("Conflicting recorded trajectory IDs")
        return explicit[0], seals
    if owner.get("source"):
        parent = Path(owner["source"]).absolute()
        parent_owner = source_identity(parent)
        if parent_owner.get("identity") != identity:
            raise ValueError("Trajectory recovery source belongs to another episode")
        assert_stopped(parent, parent_owner)
        native = owner.get("native_identity")
        if native is not None:
            validate_native_identity(
                parent_owner.get("native_identity"), identity,
                scene_sha256=native.get("scene_sha256"),
                step_limit=native.get("step_limit"),
            )
        trajectory_id, parent_seals = source_trajectory_id(parent, parent_owner, _seen=seen)
        seals.update(parent_seals)
        return trajectory_id, seals
    if "episode_id" in owner:
        legacy.append(validate_trajectory_id(owner["episode_id"]))
    if not legacy:
        raise ValueError("Recovery source has no recorded trajectory ID")
    if len(set(legacy)) != 1:
        raise ValueError("Conflicting recorded trajectory IDs")
    return legacy[0], seals


def action_chain(run: Path, cfg: dict) -> tuple[int, dict[str, str]]:
    """Check every completed joint ACK before the original replay can move."""
    latest = read_json(run / "observations/latest.json")
    step = latest["observation"]["control_step"]
    limit = latest["observation"]["step_limit"]
    if type(step) is not int or not 0 <= step < limit or latest.get("done"):
        raise ValueError("Checkpoint must precede native termination")
    episode = cfg["test_episodes"][0]
    chains, seen, current, cutoff = [], set(), run, step
    seals = {}
    while True:
        if current in seen:
            raise ValueError("Replay history contains a cycle")
        seen.add(current)
        config = read_json(current / "config.json")
        if any(
            config.get(k) != cfg.get(k)
            for k in (
                "test_episodes",
                "model",
                "reasoning_effort",
                "environment_revision",
            )
        ):
            raise ValueError("Replay history changes episode or policy identity")
        record = current / "episodes/evaluation-0001"
        if read_json(record / "episode.json") != episode:
            raise ValueError("Recorded episode identity changed")
        result = read_json(record / "result.json")
        if result.get("native_done"):
            raise ValueError("A native terminal episode cannot be resumed")
        actions = record / "actions.jsonl"
        rows = (
            [json.loads(line) for line in actions.read_text().splitlines()]
            if actions.is_file()
            else []
        )
        chains.append((rows, cutoff))
        for path in [
            current / "config.json",
            record / "episode.json",
            record / "result.json",
            actions,
            record / "transitions.jsonl",
        ]:
            if path.is_file():
                seals[str(path)] = digest(path)
        replay = config["options"].get("replay")
        if not replay:
            break
        if replay["episode_id"] != episode["id"]:
            raise ValueError("Parent replay belongs to another episode")
        parent = Path(replay["source_record"])
        if parent.name != "evaluation-0001" or parent.is_symlink():
            raise ValueError("Parent replay must identify its own episode record")
        current, cutoff = parent.parent.parent, replay["control_step"]
    completed = 0
    for rows, cutoff in reversed(chains):
        for row in rows:
            if row.get("status") != "completed":
                continue
            if row["control_step"] > cutoff:
                break
            if row["control_step"] != completed + 1:
                raise ValueError("Completed native actions are not contiguous")
            command = row.get("applied_action", {}).get("joint_command")
            channels = {
                "left_arm_joint_state": 6,
                "left_ee_joint_state": 1,
                "right_arm_joint_state": 6,
                "right_ee_joint_state": 1,
            }
            if not isinstance(command, dict) or set(command) != set(channels):
                raise ValueError("Completed action has no applied native joint command")
            for name, size in channels.items():
                values = command[name]
                if (
                    not isinstance(values, list)
                    or len(values) != size
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in values
                    )
                    or (size == 1 and not 0 <= values[0] <= 1)
                ):
                    raise ValueError("Invalid recorded native joint command")
            completed += 1
        if completed != cutoff:
            raise ValueError("Incomplete native actions for checkpoint")
    if completed != step:
        raise ValueError("Replay does not reach the saved checkpoint")
    seals[str(run / "observations/latest.json")] = digest(
        run / "observations/latest.json"
    )
    if step:
        image = run / f"observations/evaluation-0001/{step:05d}_cam_head.png"
        seals[str(image)] = digest(image)
    return step, seals


def prepare_resume(
    source: Path,
    output: Path,
    reference,
    *,
    trace_root: Path | None = None,
    native_progress: Path | None = None,
    reason: str = "Infrastructure recovery: outer case timeout",
) -> dict:
    """Apply unchanged prepare_retry after adapter-specific ownership checks."""
    root = Path(source).absolute()
    if root.is_symlink() or not root.is_dir() or output.exists():
        raise ValueError("Use an existing retained source and a new request path")
    owner = source_identity(root, trace_root=trace_root)
    failed_root = root
    identity = copy.deepcopy(owner["identity"])
    native_binding = None
    if native_progress is not None:
        identity, native_binding = bind_native_progress(identity, native_progress)
    if identity.get("layout") is None:
        raise ValueError("Resume needs an explicit deployment layout identity")
    assert_stopped(root, owner)
    native_identity = validate_native_identity(owner.get("native_identity"), identity)
    trajectory_id, trajectory_seals = source_trajectory_id(root, owner)
    run = root / "agent/controller"
    cfg = read_json(run / "config.json")
    failed_replay = None
    # Original reset returns its first observation only after all replayed
    # actions validate. A technical interruption before that point has no
    # latest.json; the original recovery policy deliberately retries its
    # already verified parent checkpoint instead of adopting partial replay.
    if not (run / "observations/latest.json").is_file() and cfg.get("options", {}).get("replay"):
        if any(path.stat().st_size for path in (run / "episodes").glob("*/actions.jsonl")):
            raise ValueError("A replay without a checkpoint cannot discard new action records")
        failed_summary = read_json(run / "summary.json")
        error = failed_summary.get("error") or ""
        if not isinstance(error, str) or not any(
            token in error for token in ("Simulator failed", "Simulator exited", "BrokenPipe")
        ):
            raise ValueError("Interrupted replay is not eligible for original parent recovery")
        replay = cfg["options"]["replay"]
        record = Path(replay["source_record"])
        parent_run = record.parent.parent
        if (
            record.name != "evaluation-0001"
            or record.parent.name != "episodes"
            or record.is_symlink()
            or cfg.get("continue_from_run") != str(parent_run)
        ):
            raise ValueError("Failed replay does not identify its own continuation parent")
        root = parent_run.parent.parent
        parent_owner = source_identity(root, trace_root=trace_root)
        if parent_owner.get("identity") != identity:
            raise ValueError("Failed replay parent belongs to another native episode")
        parent_trajectory_id, parent_seals = source_trajectory_id(root, parent_owner)
        if parent_trajectory_id != trajectory_id:
            raise ValueError("Failed replay parent belongs to another trajectory")
        trajectory_seals.update(parent_seals)
        assert_stopped(root, parent_owner)
        parent_native = validate_native_identity(
            parent_owner.get("native_identity"), identity,
            scene_sha256=native_identity["scene_sha256"],
            step_limit=native_identity["step_limit"],
        )
        if native_identity["control_step"] > parent_native["control_step"]:
            raise ValueError("Partial replay exceeds its verified parent checkpoint")
        failed_segment_path = run / "segments/evaluation-0001/result.json"
        failed_segment = read_json(failed_segment_path)
        wall = failed_segment.get("wall_seconds_total", failed_segment.get("wall_seconds"))
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            raise ValueError("Failed replay lacks measured cumulative wall time")
        parent_cfg = read_json(parent_run / "config.json")
        validate_policy(cfg, parent_cfg)
        failed_replay = {
            "run": run, "config": cfg, "summary": failed_summary,
            "wall_seconds": wall, "segment_path": failed_segment_path,
        }
        run, cfg, native_identity = parent_run, parent_cfg, parent_native
    if (
        cfg["test_episodes"][0]["task"] != identity["task"]
        or cfg["test_episodes"][0]["eval_seed"] != identity["eval_seed"]
    ):
        raise ValueError("Source archive and controller episode differ")
    step, seals = action_chain(run, cfg)
    seals.update(trajectory_seals)
    validate_native_identity(native_identity, identity, step=step,
                             step_limit=cfg["options"]["review_window_steps"])
    if cfg["test_episodes"][0].get("layout_id") != identity["layout"]:
        raise ValueError("Controller episode differs from actual native layout")
    seals[str(root / "episode-owner.json")] = digest(root / "episode-owner.json")
    if failed_replay:
        for path in (
            failed_root / "episode-owner.json",
            failed_replay["run"] / "config.json",
            failed_replay["run"] / "summary.json",
            failed_replay["segment_path"],
        ):
            seals[str(path)] = digest(path)
    if native_binding:
        seals[native_binding["path"]] = native_binding["sha256"]
    segment_path = run / "segments/evaluation-0001/result.json"
    segment = read_json(segment_path)
    wall = segment.get("wall_seconds_total", segment.get("wall_seconds"))
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Missing measured source wall time")
    seals[str(segment_path)] = digest(segment_path)
    state_path = Path(
        owner.get("recovery_state", str(root / "adapter-recovery/state.json"))
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(state_path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = read_json(state_path) if state_path.exists() else {}
        if row.get("prepared_from") == str(failed_root):
            raise ValueError(
                "This source already has a prepared resume; reuse its request"
            )
        row.update(run=str(run), infrastructure_recovery={"reason": reason})
        if failed_replay:
            row["summary"] = failed_replay["summary"]
        elif (run / "summary.json").is_file():
            row["summary"] = read_json(run / "summary.json")
        workspace = failed_root / "agent/workspace"
        seal = workspace_digest(workspace)
        # Original recovery writes only this private configuration staging root.
        # The source cfg, source results and vendored recovery.py remain intact.
        with tempfile.TemporaryDirectory(
            prefix="prepare-", dir=state_path.parent
        ) as staging:
            stage = Path(staging)
            (stage / "config").mkdir()
            instance = identity["task"] + "-s" + str(identity["eval_seed"])
            staged = copy.deepcopy(cfg)
            staged.setdefault("local_integration", {})
            write_json(stage / "config" / f"{instance}.json", staged)
            if failed_replay:
                # The unchanged recovery.py measures failed reset wall time
                # from a timestamped Runner directory. Adapt our controller
                # directory in private staging using the measured duration;
                # never change the retained source's configuration or times.
                staged_run = stage / "20000101T000000Z-replay-interrupted"
                write_json(staged_run / "config.json", failed_replay["config"])
                summary_path = staged_run / "summary.json"
                write_json(summary_path, failed_replay["summary"])
                from datetime import datetime, timezone

                began = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
                ended = began + failed_replay["wall_seconds"]
                os.utime(summary_path, (ended, ended))
                row["run"] = str(staged_run)
            spec = importlib.util.spec_from_file_location(
                "_v4_original_recovery", reference.source_dir / "recovery.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ROOT = stage
            accepted = module.prepare_retry(instance, row)
            if not accepted:
                write_json(state_path, row)
                raise ValueError(
                    row.get(
                        "recovery_block", "Original recovery rejected this interruption"
                    )
                )
            recovered = read_json(stage / "config" / f"{instance}.json")
        if failed_replay:
            for attempt in row.get("attempt_history", []):
                if attempt.get("run") == str(staged_run):
                    attempt["run"] = str(failed_replay["run"])
        if (
            recovered.get("options", {}).get("replay", {}).get("control_step", 0)
            != step
        ):
            raise ValueError("Original recovery selected a different checkpoint")
        request = {
            "format": "xpolicylab_own_episode_resume_v1",
            "source": str(root),
            "identity": identity,
            "trajectory_id": trajectory_id,
            "step_offset": step,
            "config": recovered,
            "source_files": seals,
            "workspace_sha256": seal,
            "recovery_state": str(state_path),
            "recovery_count": row["recovery_count"],
            "created_at": time.time(),
            "native_validation_required": True,
            "native_identity": native_identity,
        }
        if failed_replay:
            request.update(
                workspace_source=str(failed_root),
                recovery_source=str(failed_root),
                failed_replay_wall_seconds=failed_replay["wall_seconds"],
            )
        if native_binding:
            request["native_layout_binding"] = native_binding
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(request, stream, indent=2, allow_nan=False)
            stream.write("\n")
        row["prepared_from"] = str(failed_root)
        row["request"] = str(output.absolute())
        write_json(state_path, row)
        return request
    finally:
        os.close(lock)


def bind_native_progress(identity: dict, path: Path) -> tuple[dict, dict]:
    """Validate scored queue progress without relabeling the interrupted scene."""
    path = Path(path).absolute()
    native = read_json(path)
    for field, expected in (
        ("task_name", identity["task"]),
        ("eval_seed", identity["eval_seed"]),
        ("run_id", identity["run_id"]),
        ("policy_name", "SenseNova-RoboRSI"),
        ("config_name", "arx_x5"),
    ):
        if native.get(field) != expected:
            raise ValueError(f"Native progress belongs to another deployment: {field}")
    details = native.get("details") or {}
    completed = {int(row["layout_id"]) for row in details.values()}
    if completed != set(native.get("completed_layout_ids", [])):
        raise ValueError("Native completed layouts lack matching scored details")
    if len(details) != native.get("success_nums", 0) + native.get("fail_nums", 0):
        raise ValueError("Native result counters differ from scored details")
    if any(type(row.get("score")) not in (int, float)
           or not math.isfinite(row["score"]) for row in details.values()):
        raise ValueError("Native completed layout lacks an official score")
    abandoned = set(native.get("abandoned_layout_ids", []))
    if any(type(value) is not int or value < 0 for value in completed | abandoned):
        raise ValueError("Invalid native layout IDs")
    layout = identity.get("layout")
    if type(layout) is not int or layout < 0 or layout in completed | abandoned:
        raise ValueError("Requested native layout is invalid or already ended")
    return copy.deepcopy(identity), {
        "path": str(path), "sha256": digest(path),
        "actual_layout": layout,
        "completed_layout_ids": sorted(completed),
        "abandoned_layout_ids": sorted(abandoned),
        "physical_validation_required": True,
    }


def load_resume(path: Path, identity: dict) -> dict:
    """Validate a request before replay or launching any model process."""
    request = read_json(path)
    if (
        request.get("format") != "xpolicylab_own_episode_resume_v1"
        or request["identity"] != identity
    ):
        raise ValueError(
            "Resume request differs from this deployment task/seed/layout/run"
        )
    validate_native_identity(request.get("native_identity"), identity,
                             step=request["step_offset"],
                             step_limit=request["config"]["options"]["review_window_steps"])
    for name, sha in request["source_files"].items():
        if digest(Path(name)) != sha:
            raise ValueError("A stopped source record changed after resume preparation")
    root = Path(request["source"])
    owner = source_identity(root)
    if owner.get("identity") != identity:
        raise ValueError("Resume trajectory belongs to another episode")
    trajectory_id, trajectory_seals = source_trajectory_id(root, owner)
    if str(root / "episode-owner.json") not in request["source_files"]:
        raise ValueError("Resume lacks sealed trajectory ownership")
    if "trajectory_id" in request:
        if validate_trajectory_id(request["trajectory_id"]) != trajectory_id:
            raise ValueError("Resume trajectory ID differs from its own source")
        if any(request["source_files"].get(name) != sha for name, sha in trajectory_seals.items()):
            raise ValueError("Resume lacks sealed trajectory identity records")
    workspace_root = Path(request.get("workspace_source", str(root)))
    if workspace_root != root:
        if request.get("recovery_source") != str(workspace_root):
            raise ValueError("Fallback workspace differs from its own recovery attempt")
        owner = source_identity(workspace_root)
        if owner.get("identity") != identity:
            raise ValueError("Fallback workspace belongs to another native episode")
        workspace_trajectory_id, workspace_seals = source_trajectory_id(workspace_root, owner)
        if workspace_trajectory_id != trajectory_id:
            raise ValueError("Fallback workspace belongs to another trajectory")
        if "trajectory_id" in request and any(
            request["source_files"].get(name) != sha for name, sha in workspace_seals.items()
        ):
            raise ValueError("Fallback workspace lacks sealed trajectory identity records")
        assert_stopped(workspace_root, owner)
        validate_native_identity(
            owner.get("native_identity"), identity,
            scene_sha256=request["native_identity"]["scene_sha256"],
            step_limit=request["native_identity"]["step_limit"],
        )
        paths = [workspace_root / "episode-owner.json", *(
            workspace_root / "agent/controller" / name
            for name in ("config.json", "summary.json", "segments/evaluation-0001/result.json")
        )]
        if any(str(path) not in request["source_files"] for path in paths):
            raise ValueError("Fallback workspace lacks sealed failed-replay evidence")
    if workspace_digest(workspace_root / "agent/workspace") != request["workspace_sha256"]:
        raise ValueError("The stopped own-episode workspace changed")
    if request["config"].get("continue_from_run") != str(root / "agent/controller"):
        raise ValueError("Resume continuation differs from its own source")
    if request["config"]["options"].get("replay_fresh_session") is not True:
        raise ValueError("External recovery must preserve original fresh-session rule")
    replay = request["config"]["options"].get("replay")
    step = request["step_offset"]
    if (
        type(step) is not int
        or step < 0
        or request.get("native_validation_required") is not True
    ):
        raise ValueError("Resume needs a validated original control-step checkpoint")
    if step and (
        not replay
        or replay.get("source_record")
        != str(root / "agent/controller/episodes/evaluation-0001")
        or replay.get("control_step") != step
        or replay.get("episode_id") != request["config"]["test_episodes"][0]["id"]
    ):
        raise ValueError("Replay differs from the own-episode checkpoint")
    if not step and replay:
        raise ValueError("A zero-step continuation cannot import replay actions")
    return {**request, "trajectory_id": trajectory_id}


def validate_policy(resumed: dict, fresh: dict) -> None:
    """Recovery may relocate execution, but cannot select another task/policy."""
    for key in (
        "model",
        "reasoning_effort",
        "environment_revision",
        "test_episodes",
        "terminal_feedback",
    ):
        if resumed.get(key) != fresh.get(key):
            raise ValueError(f"Resume changes the original policy/episode: {key}")
    deployed = {"gpu", "physical_gpu", "vla_port", "replay", "replay_fresh_session"}
    old = {k: v for k, v in resumed["options"].items() if k not in deployed}
    new = {k: v for k, v in fresh["options"].items() if k not in deployed}
    if old != new:
        raise ValueError("Resume changes original action/review/context options")
    if not 0 < resumed["episode_seconds"] <= fresh["episode_seconds"]:
        raise ValueError("Resume grants an invalid cumulative wall budget")


def original_replay_reset(reference, adapter, begin_reset, **kwargs):
    """Execute the unmodified original replay reset with a deferred native owner."""
    original = reference.HybridAdapter.__mro__[1].reset
    namespace = {
        **original.__globals__,
        "super": lambda: types.SimpleNamespace(reset=begin_reset),
    }
    reset = types.FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    return reset(adapter, **kwargs)


def main() -> None:
    """Prepare an explicit source, without simulator reset or model requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--native-progress", type=Path)
    parser.add_argument(
        "--reason", default="Infrastructure recovery: outer case timeout"
    )
    args = parser.parse_args()
    from .hybrid_backend import load_reference_source
    from .runtime_config import POLICY_DIR, submission

    pins = submission()["hybrid"]
    reference = load_reference_source(
        POLICY_DIR / pins["source_path"],
        POLICY_DIR / pins["manifest_path"],
        pins["manifest_sha256"],
    )
    request = prepare_resume(
        args.source,
        args.output,
        reference,
        trace_root=args.trace_root,
        native_progress=args.native_progress,
        reason=args.reason,
    )
    print(
        json.dumps(
            {
                "request": str(args.output),
                "step_offset": request["step_offset"],
                "remaining_seconds": request["config"]["episode_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()

"""Small file helpers for workspace-session-v1."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

from orchestrator.config import TaskConfig


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)

WRITABLE_COMPONENTS = ("skills", "lessons", "submission", "scratch")


def codex_token_usage(home: Path) -> dict:
    """Read the final cumulative token counters from Codex rollout JSONL files.

    Codex emits multiple ``token_count`` events per rollout; each event is a
    cumulative counter, so summing events would double count. We take the last
    counter from each rollout file and sum only across distinct files.
    """
    sessions = Path(home) / ".codex" / "sessions"
    total = {field: 0 for field in TOKEN_FIELDS}
    rollout_count = token_events = parse_errors = 0
    if not sessions.is_dir():
        return {
            **total,
            "rollout_files": 0,
            "token_events": 0,
            "parse_errors": 0,
        }
    for path in sorted(sessions.rglob("*.jsonl")):
        latest = None
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(event, dict):
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                if event.get("type") == "event_msg" and payload.get("type") == "token_count":
                    info = payload.get("info")
                    # Rate-limit-only events legitimately have info: null.
                    if not isinstance(info, dict):
                        continue
                    usage = info.get("total_token_usage")
                    if isinstance(usage, dict):
                        latest = usage
                        token_events += 1
        except OSError:
            parse_errors += 1
            continue
        if latest is None:
            continue
        rollout_count += 1
        for field in TOKEN_FIELDS:
            value = latest.get(field, 0)
            if isinstance(value, int | float):
                total[field] += value
    return {
        **total,
        "rollout_files": rollout_count,
        "token_events": token_events,
        "parse_errors": parse_errors,
    }


def capture_writable_changes(
    workspace: Path, destination: Path, previous: dict[str, str]
) -> dict[str, str]:
    """Record only changed agent-authored files for one execution.

    Read-only protocol files and observations are deliberately excluded.  The
    returned state is kept by the session so an unchanged file is never copied
    into every subsequent feedback directory.
    """
    current: dict[str, str] = {}
    changed: list[str] = []
    deleted = sorted(set(previous))
    workspace = Path(workspace)
    destination = Path(destination)
    for component in WRITABLE_COMPONENTS:
        root = workspace / component
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(workspace))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            current[relative] = digest
            if previous.get(relative) == digest:
                if relative in deleted:
                    deleted.remove(relative)
                continue
            changed.append(relative)
            if relative in deleted:
                deleted.remove(relative)
            target = destination / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_json(
        destination / "files.json",
        {"changed": changed, "deleted": deleted, "files": current},
    )
    return current


def hashes(root: Path) -> dict[str, str]:
    if root.is_symlink():
        raise ValueError(f"Symlink is not a frozen harness: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Harness symlinks are forbidden: {path}")
        if path.is_file() and path != root / "freeze.json":
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@contextlib.contextmanager
def run_lock(run: Path):
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".protocol.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {run}") from exc
        yield


def test_scene_records(cfg, seeds, settings=None):
    """Human-readable host log of the exact native scenes selected for Test."""
    records = []
    for index, (seed, variant) in enumerate(zip(seeds, settings or ["standard"] * len(seeds), strict=True)):
        native = cfg.simulator.env_task
        if cfg.simulator.backend == "robodojo" and variant == "random" and not native.endswith("_random"):
            native += "_random"
        row = {"scene_index": index, "env_task": native, "variant": variant, "seed": seed}
        if cfg.simulator.backend == "robodojo":
            row.update(bank=seed // 1_000_000, layout_ordinal=seed % 1_000_000)
        records.append(row)
    return records


def valid_test_seeds(cfg, seeds, settings=None):
    """A layout ordinal is unique within a native variant, not across variants."""
    if (not isinstance(seeds, list) or len(seeds) != cfg.test.scenes
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        return False
    variants = settings if settings is not None else ["standard"] * len(seeds)
    if len(variants) != len(seeds):
        return False
    native = cfg.simulator.env_task
    keys = [(native + "_random" if setting == "random" and not native.endswith("_random")
             else native, seed) for setting, seed in zip(variants, seeds)]
    if cfg.simulator.backend != "robodojo":
        keys = [(native, seed) for seed in seeds]
    excluded = {cfg.playground.seed} if cfg.playground is not None else set()
    if cfg.test.excluded_seed is not None:
        excluded.add(cfg.test.excluded_seed)
    return len(set(keys)) == len(keys) and not any((native, seed) in keys for seed in excluded)


def shared_test_seeds(cfg, *, settings=None):
    """Draw once on the host; paired groups and retries reuse the same plan."""
    source = Path(cfg.test.shared_plan)
    source.parent.mkdir(parents=True, exist_ok=True)
    identity = {"env_task": cfg.simulator.env_task,
                "settings": settings or ["standard"] * cfg.test.scenes,
                "excluded_seed": cfg.test.excluded_seed}
    with source.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source.exists():
            plan = json.loads(source.read_text())
            if any(plan.get(key) != value for key, value in identity.items()):
                raise ValueError("Shared Test plan has different task, settings or Playground exclusion")
        else:
            plan = {**identity, "seeds": test_seeds(cfg, settings=settings)}
            plan["scenes"] = test_scene_records(cfg, plan["seeds"], settings)
            write_json(source, plan)
        if not valid_test_seeds(cfg, plan.get("seeds"), identity["settings"]):
            raise ValueError("Invalid shared Test plan")
    return plan["seeds"]


def test_seeds(cfg: TaskConfig, *, settings=None) -> list[int]:
    """Choose a new host-only Test plan; callers must save it for resume."""
    excluded = {cfg.playground.seed} if cfg.playground is not None else set()
    if cfg.test.excluded_seed is not None:
        excluded.add(cfg.test.excluded_seed)
    if cfg.simulator.backend == "robodojo":
        from simulators.robodojo_layouts import available_scenes

        if settings is not None:
            if len(settings) != cfg.test.scenes or any(
                setting not in {"standard", "random"} for setting in settings
            ):
                raise ValueError("RoboDojo Test settings must match the configured scenes")
            # Numeric ordinals are local to each native task's layout bank.
            # Sampling the base task before switching to *_random can select
            # nonexistent ordinals when those banks have different sizes.
            seeds = [None] * cfg.test.scenes
            used = {cfg.simulator.env_task: set(excluded)}
            for setting in dict.fromkeys(settings):
                native = cfg.simulator.env_task
                if setting == "random" and not native.endswith("_random"):
                    native += "_random"
                indices = [i for i, value in enumerate(settings) if value == setting]
                seen = used.setdefault(native, set())
                candidates = [seed for seed in available_scenes(native) if seed not in seen]
                if len(candidates) < len(indices):
                    raise ValueError(f"RoboDojo {native} has insufficient unused Test layouts")
                selected = random.SystemRandom().sample(candidates, len(indices))
                for index, seed in zip(indices, selected, strict=True):
                    seeds[index] = seed
                seen.update(selected)
            return seeds
        candidates = [
            seed for seed in available_scenes(cfg.simulator.env_task)
            if seed not in excluded
        ]
        if len(candidates) < cfg.test.scenes:
            raise ValueError(
                f"RoboDojo has {len(candidates)} available Test layouts; "
                f"requested {cfg.test.scenes}"
            )
        return random.SystemRandom().sample(candidates, cfg.test.scenes)
    # One extra draw allows excluding Playground without replacement or retries.
    candidates = random.SystemRandom().sample(range(2**32), cfg.test.scenes + len(excluded))
    return [seed for seed in candidates if seed not in excluded][: cfg.test.scenes]

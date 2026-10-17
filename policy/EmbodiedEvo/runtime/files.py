"""Small file helpers for workspace-session-v1."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path



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

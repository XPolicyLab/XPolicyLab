import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import time


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance() -> dict:
    package = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(package.rglob("*.py"))}
    def git(*args):
        result = subprocess.run(["git", *args], cwd=package, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except FileNotFoundError:
        commit, status = None, None
    return {"python": platform.python_version(), "git_commit": commit,
            "git_dirty": bool(status) if status is not None else None,
            "package_sha256": fingerprint(files), "package_files": files}


class Trace:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.sequence = 0

    def emit(self, event: str, **data):
        record = {"seq": self.sequence, "time": time(), "event": event, **data}
        with (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        self.sequence += 1


def _text_of(content):
    if isinstance(content, str):
        return [content]
    return [part.get("text", "") for part in content or [] if part.get("type") in ("text", "input_text")]


def context_summary(messages, system=None):
    """What one model call received in full, one short row per message: role, opening
    text, size, images still attached and images replaced by an omission note. The
    trace keeps each message's content once (where it was first sent); this records
    how the whole history looked at each call, after windowing and image pruning."""
    rows = [{"role": "system", "text": system[:160], "chars": len(system), "images": 0, "omitted": 0}] if system else []
    for message in messages:
        content = message.get("content")
        texts = _text_of(content)
        parts = [] if isinstance(content, str) else list(content or [])
        calls = [f"{(c.get('function') or {}).get('name')}(...)" for c in message.get("tool_calls") or []]
        joined = " ".join(t for t in texts if t) or " ".join(calls)
        rows.append({
            "role": message.get("role"),
            "text": " ".join(joined.split())[:160],
            "chars": sum(len(t) for t in texts),
            "images": sum(part.get("type") in ("input_image", "image_url") for part in parts),
            "omitted": sum("image omitted" in t for t in texts),
        })
    return rows


def read_events(directory: Path):
    with (directory / "events.jsonl").open() as handle:
        for line in handle:
            yield json.loads(line)

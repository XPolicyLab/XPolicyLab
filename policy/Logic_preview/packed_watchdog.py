"""Client and Isaac Kit log monitoring adapted from the 746 packed scheduler."""
import re
from pathlib import Path
from typing import Any

DEFAULT_FATAL_LOG_PATTERNS = (
    "PhysX Internal CUDA error",
    "PhysX has reported too many errors",
    "Scene state is corrupted",
    "CUDA error 700",
    "cudaErrorIllegalAddress",
    "vkCreateRayTracingPipelinesKHR failed",
    "Out of GPU memory",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "no suitable CUDA GPU was found",
    "Failed to create stream on device",
    "ERROR_INCOMPATIBLE_DRIVER",
    "Failed to create any GPU devices",
    "PermissionError: [Errno 13]",
)

KIT_HANG_PATTERNS = (
    "Hang detected. Asking user if we should crash.",
    "Kit appears to be hanging",
)


def _monitor_text(path: Path, monitor: dict[str, Any], prefix: str, offset: int = 0) -> str:
    try:
        metadata = path.stat()
        identity = [metadata.st_dev, metadata.st_ino]
        cursor = int(monitor.get(f"{prefix}_cursor", offset))
        if monitor.get(f"{prefix}_identity", identity) != identity or cursor > metadata.st_size:
            cursor = 0
            monitor[f"{prefix}_pending"] = ""
        with path.open("rb") as handle:
            handle.seek(cursor)
            chunk = handle.read(1024 * 1024)
            monitor[f"{prefix}_cursor"] = handle.tell()
        monitor[f"{prefix}_identity"] = identity
    except OSError:
        return ""
    if not chunk:
        return ""
    text = monitor.get(f"{prefix}_pending", "") + chunk.decode("utf-8", errors="replace")
    boundary = max(text.rfind("\n"), text.rfind("\r"))
    monitor[f"{prefix}_pending"] = text[boundary + 1:][-4096:]
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def update_job_monitor(
    job: dict[str, Any], now: float, manifest_stamp: int | None, patterns: tuple[str, ...] = DEFAULT_FATAL_LOG_PATTERNS
) -> str | None:
    monitor = job.setdefault("monitor", {})
    if job.get("manifest_stamp") != manifest_stamp:
        job["last_progress_at"] = now
    job["manifest_stamp"] = manifest_stamp
    text = _monitor_text(Path(job["log"]), monitor, "log", int(job.get("log_offset", 0)))
    fatal = next((pattern for pattern in patterns if pattern in text), None)
    steps = monitor.setdefault("steps", {})
    for match in re.finditer(r"env(\d+) step:\s*(\d+)\s*/\s*\d+", text):
        environment, current = match.group(1), int(match.group(2))
        previous = steps.get(environment)
        if previous is None or current > previous or (current == 0 and previous > 0):
            job["last_progress_at"] = now
        steps[environment] = current
    for match in re.finditer(r"\[carb\] Logging to file:[ \t]*([^\r\n]+)[\r\n]", text):
        candidate = match.group(1).strip().strip("\"'")
        if candidate != monitor.get("kit_path"):
            for key in ("kit_cursor", "kit_pending", "kit_identity"):
                monitor.pop(key, None)
            monitor["kit_path"] = candidate
    if monitor.get("kit_path"):
        kit_path = Path(monitor["kit_path"])
        try:
            current_attempt = kit_path.stat().st_mtime >= float(job.get("started_at", now))
        except OSError:
            current_attempt = False
        if current_attempt:
            native = _monitor_text(kit_path, monitor, "kit")
            fatal = fatal or next((pattern for pattern in (*patterns, *KIT_HANG_PATTERNS)
                                   if pattern in native), None)
    return fatal

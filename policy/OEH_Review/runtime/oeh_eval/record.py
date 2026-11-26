"""Append-only episode evidence with a validated final seal."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import time
from typing import Any, Mapping

from .contracts import AttemptPlan, NativeOutcome

SCHEMA = "oeh.eval.episode.v1"
# Why a non-completed attempt ended (seal.json failure_class; absent in seals written before eval/stable).
# benchmark_void: the benchmark voided the episode itself (contracts.NATIVE_VOID); excluded for every arm, never rerun.
# harness: the evaluation harness itself failed (an evaluator exception, a policy runtime that never started an episode);
# cleanup_lost: the official episode ended but OEH's cleanup after it took the native result with it (harness audit
# 2026-10-02, ATTR-1 and ATTR-3). Both are excluded for every arm and rerun like infrastructure.
FAILURE_CLASSES = (None, "infrastructure", "policy", "benchmark_void", "harness", "cleanup_lost")
# Classes `--rerun-infrastructure` sets aside and runs again (a benchmark void is never rerun).
RERUN_CLASSES = frozenset({"infrastructure", "harness", "cleanup_lost"})
ZERO = "0" * 64


def plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"not a finite JSON value: {type(value).__name__}")


def encoded(value: Any) -> bytes:
    return (json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# Lossless image evidence (rdmem, 2026-09-26). A camera frame an observation carries (8-bit, 1, 3 or 4 channels) is
# stored as PNG instead of .npy. Its identity stays the sha256 of the array's .npy serialization, which is what the
# .npy file's hash was, so observation descriptors and digests, initial_state_sha256 and parity are unchanged. A frame
# is stored as PNG only when its PNG decodes back to exactly the same .npy bytes; any other array stays .npy. The
# observation event's file entry then names format "png" and the PNG's own sha256 (file_sha256).
PNG_COMPRESS_LEVEL = 1          # measured on RoboDojo 640x480 frames: ~2.3x smaller than .npy, ~30 ms per frame


def npy_bytes(array: Any) -> bytes:
    """The bytes np.save writes for the array (allow_pickle False): the identity an observation file carries."""
    import io
    import numpy as np
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def png_array(data: bytes) -> Any:
    """The array a PNG's bytes decode to (Pillow; image decoding, not text I/O)."""
    import io
    import numpy as np
    from PIL.Image import open as decode_image
    with decode_image(io.BytesIO(data)) as image:
        return np.array(image)


def lossless_png(array: Any, identity: bytes) -> bytes | None:
    """PNG bytes of an 8-bit image array that decode to the same .npy bytes (``identity``), else None."""
    import numpy as np
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or not (
            array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (3, 4))) or 0 in array.shape:
        return None
    try:
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        data = buffer.getvalue()
        return data if npy_bytes(png_array(data)) == identity else None
    except Exception:  # no Pillow, or an encoder refusal: the frame stays .npy
        return None


def observation_file_sha(item: Mapping[str, Any]) -> str:
    """The sha256 the stored file itself must have: the PNG's own for a PNG frame, else the array identity."""
    return item["file_sha256"] if item.get("format") == "png" else item["sha256"]


# Observation evidence formats. An observation event names its frames and arrays in "files"; how their bytes are
# stored has three forms, told apart by observation_format:
#   files            one whole file per entry at entry["path"] (rounds 1-2, the other benches' evidence writers);
#   pack             every entry a slice (offset, length) of one pack file per attempt, named in the event itself
#                    (the first pack layout, 2026-10-23; no campaign ran on it);
#   pack-retained/1  the entry carries identity only (sha256, format, file_sha256, bytes); its bytes, when kept, are a
#                    pack slice named by a later observation_stored event (RoboDojo's BridgeLog since the retention
#                    change). Frames nobody read may be dropped: their hashes stay, so descriptors, digests and
#                    initial_state_sha256 are unchanged, and a dropped frame is a known absence, never a changed file.
# A declared format this module does not know is refused: a reader never guesses.
OBSERVATION_PACK = "policy-observations/observations.pack"
OBSERVATION_FORMAT = "pack-retained/1"       # what BridgeLog writes
OBSERVATION_STORED = "observation_stored"    # the event kind that names stored slices of an earlier observation
OBSERVATION_FORMATS = ("files", "pack", OBSERVATION_FORMAT)


def observation_format(payload: Mapping[str, Any]) -> str:
    """The storage format of one observation event's files (see OBSERVATION_FORMATS); unknown declarations raise."""
    declared = payload.get("evidence_format")
    if declared is None:
        files = payload.get("files") or []
        return "pack" if any(isinstance(item, Mapping) and "offset" in item for item in files) else "files"
    if declared != OBSERVATION_FORMAT:
        raise ValueError(f"unknown observation evidence format {declared!r}")
    return declared


def stored_locations(events: Any) -> dict[int, dict[int, dict[str, Any]]]:
    """index -> slot -> {path, offset, length, reason} from the observation_stored events among ``events``.

    Each (index, slot) may be stored once; a malformed event raises (the event chain is append-only, so a reader
    that meets one has a corrupt or foreign record)."""
    located: dict[int, dict[int, dict[str, Any]]] = {}
    for event in events:
        if event.get("kind") != OBSERVATION_STORED:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or type(payload.get("index")) is not int or payload["index"] < 0 or \
                not isinstance(payload.get("reason"), str) or not isinstance(payload.get("files"), list):
            raise ValueError("observation storage event malformed")
        slots = located.setdefault(payload["index"], {})
        for item in payload["files"]:
            if not isinstance(item, Mapping) or type(item.get("slot")) is not int or item["slot"] < 0 or \
                    item.get("path") != OBSERVATION_PACK or type(item.get("offset")) is not int or \
                    type(item.get("length")) is not int or item["offset"] < 0 or item["length"] < 0:
                raise ValueError("observation storage event malformed")
            if item["slot"] in slots:
                raise ValueError("observation slice stored twice")
            slots[item["slot"]] = {"path": item["path"], "offset": item["offset"], "length": item["length"],
                                   "reason": payload["reason"]}
    return located


def observation_files(payload: Mapping[str, Any], located: Mapping[int, Mapping[int, Mapping[str, Any]]] | None = None
                      ) -> list[dict[str, Any]]:
    """One observation's file entries with their storage: each gains ``stored`` and, when stored, the path (and
    pack offset and length) observation_bytes reads. ``located`` is stored_locations() of the record's events; the
    files and pack formats need none."""
    files = payload.get("files") or []
    if observation_format(payload) != OBSERVATION_FORMAT:
        return [{**item, "stored": True} for item in files]
    slots = (located or {}).get(payload.get("index"), {})
    entries = []
    for slot, item in enumerate(files):
        entry = {key: value for key, value in item.items() if key not in ("path", "offset", "length")}
        location = slots.get(slot)
        if location is None:
            entry["stored"] = False
        else:
            entry.update(path=location["path"], offset=location["offset"], length=location["length"], stored=True)
        entries.append(entry)
    return entries


def observation_bytes(root: Path, item: Mapping[str, Any]) -> bytes:
    """The stored bytes of one observation file entry under ``root``: its whole file, or its pack slice."""
    source = (root / item["path"]).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError("observation evidence path missing or outside episode")
    if "offset" not in item:
        return source.read_bytes()
    offset, length = item.get("offset"), item.get("length")
    if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
        raise ValueError("observation pack slice is malformed")
    with source.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(length)
    if len(data) != length:
        raise ValueError("observation pack slice runs past the end of the pack")
    return data


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    # A unique temporary name: an interrupted write may leave one behind, but it never blocks the next write.
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


class EpisodeRecorder:
    def __init__(self, plan: AttemptPlan, frozen: Mapping[str, Any]):
        self.path = plan.output
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "evidence").mkdir()
        self.events = self.path / "events.jsonl"
        self.events.touch(exist_ok=False)
        self.count = 0
        self.head = ZERO
        self.closed = False
        self.attachments: dict[str, dict[str, Any]] = {}
        self.manifest = {
            "schema": SCHEMA,
            "attempt_id": plan.attempt_id,
            "split": plan.split,
            "scenario": plain(plan.scenario),
            "condition": plain(plan.condition),
            "frozen": plain(frozen),
            "status": "in_progress",
            "started_wall_ns": time.time_ns(),
        }
        atomic_json(self.path / "manifest.json", self.manifest)

    def append(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("episode is sealed")
        if not kind or not isinstance(payload, Mapping):
            raise ValueError("event needs kind and object payload")
        body = {
            "seq": self.count,
            "kind": kind,
            "payload": plain(payload),
            "wall_ns": time.time_ns(),
            "mono_ns": time.monotonic_ns(),
            "prev_sha256": self.head,
        }
        digest = sha_bytes(encoded(body))
        with self.events.open("ab") as stream:
            stream.write(encoded({**body, "sha256": digest}))
            stream.flush()
            os.fsync(stream.fileno())
        self.head = digest
        self.count += 1

    def attach(self, name: str, source: Path) -> None:
        if self.closed or not name.isidentifier() or name in self.attachments:
            raise ValueError("invalid or duplicate attachment")
        if not source.is_file() or source.is_symlink():
            raise ValueError("attachment must be a regular file")
        target = self.path / "evidence" / name
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        self.attachments[name] = {"path": str(target.relative_to(self.path)), "sha256": sha_file(target),
                                  "bytes": target.stat().st_size}

    def seal(self, outcome: NativeOutcome, status: str, *, status_reason: str | None = None,
             evidence_issues: list[str] | None = None, failure_class: str | None = None,
             runner_issue: str | None = None) -> None:
        """``failure_class`` (every status but completed): ``infrastructure`` when the attempt never started, was
        killed from outside, its simulator reported a lost GPU, a hosted model's provider refused the operator's
        account, key, model or budget, or the evaluator could not read its evidence; ``harness`` when the evaluator
        itself raised or the policy runtime failed before the episode's first native step; ``cleanup_lost`` when the
        official episode ended and OEH's cleanup after it lost the native result; else ``policy``; decided from the
        cause, never from the outcome. ``runner_issue``: what went wrong with the run, kept even when a valid native result
        makes the attempt ``completed``."""
        if self.closed or status not in {"completed", "invalid_evidence", "aborted", "preflight_failed", "unjudged"}:
            raise ValueError("invalid finalization")
        if (status == "completed") != (failure_class is None) or failure_class not in FAILURE_CLASSES:
            raise ValueError("a completed attempt has no failure class; any other status needs one")
        evidence_issues = list(evidence_issues or [])
        if status == "invalid_evidence" and (not outcome.valid or not evidence_issues or not status_reason):
            raise ValueError("invalid evidence status needs issues and reason")
        if status == "completed" and evidence_issues:
            raise ValueError("completed attempt cannot have evidence issues")
        seal = {
            "schema": SCHEMA,
            "status": status,
            "status_reason": status_reason,
            "evidence_issues": evidence_issues,
            "manifest_sha256": sha_file(self.path / "manifest.json"),
            "native_outcome": plain(outcome),
            "failure_class": failure_class,
            "runner_issue": runner_issue,
            "events": {"count": self.count, "head_sha256": self.head, "file_sha256": sha_file(self.events)},
            "attachments": self.attachments,
            "sealed_wall_ns": time.time_ns(),
        }
        atomic_json(self.path / "seal.json", seal)
        self.closed = True
        validate_record(self.path)


def validate_record(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((path / "seal.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or seal.get("schema") != SCHEMA:
        raise ValueError("unknown episode schema")
    if sha_file(path / "manifest.json") != seal.get("manifest_sha256"):
        raise ValueError("frozen manifest changed")
    if manifest.get("split") not in {"development", "sealed"}:
        raise ValueError("unknown split")
    events = path / "events.jsonl"
    if sha_file(events) != seal["events"]["file_sha256"]:
        raise ValueError("event file changed")
    head = ZERO
    count = 0
    rows = [json.loads(line) for line in events.read_bytes().splitlines()]
    try:
        located = stored_locations(rows)
    except ValueError:
        raise ValueError("observation storage event malformed") from None
    observed: dict[int, int] = {}  # observation index -> its file count, for the storage events that follow it
    for event in rows:
        digest = event.pop("sha256")
        if event["seq"] != count or event["prev_sha256"] != head or sha_bytes(encoded(event)) != digest:
            raise ValueError("event chain broken")
        if event["kind"] == "observation":
            payload = event["payload"]
            try:
                entries = observation_files(payload, located)
            except ValueError:
                raise ValueError("unknown observation evidence format") from None
            if observation_format(payload) == OBSERVATION_FORMAT:
                if type(payload.get("index")) is not int or payload["index"] in observed:
                    raise ValueError("observation index missing or repeated")
                observed[payload["index"]] = len(entries)
            for item in entries:
                if not item["stored"]:
                    continue  # a frame nobody read: its hash is kept in the event, its bytes were never stored
                try:
                    data = observation_bytes(path, item)
                except ValueError:
                    raise ValueError("observation payload missing or changed") from None
                if sha_bytes(data) != observation_file_sha(item):
                    raise ValueError("observation payload missing or changed")
        elif event["kind"] == OBSERVATION_STORED:
            # Names slices of an observation that precedes it; stored_locations checked its shape.
            index = event["payload"]["index"]
            if index not in observed or any(item["slot"] >= observed[index] for item in event["payload"]["files"]):
                raise ValueError("observation storage event names no earlier observation slot")
        head = digest
        count += 1
    if count != seal["events"]["count"] or head != seal["events"]["head_sha256"]:
        raise ValueError("event seal mismatch")
    for attachment in seal["attachments"].values():
        target = (path / attachment["path"]).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError("attachment missing or outside episode")
        if target.stat().st_size != attachment["bytes"] or sha_file(target) != attachment["sha256"]:
            raise ValueError("attachment changed")
    outcome = seal["native_outcome"]
    if seal.get("status") not in {"completed", "invalid_evidence", "aborted", "preflight_failed", "unjudged"}:
        raise ValueError("unknown attempt status")
    if outcome["valid"] and (not outcome["complete"] or type(outcome["success"]) is not bool):
        raise ValueError("valid native result needs completed Boolean judgment")
    if seal["status"] == "completed" and not outcome["valid"]:
        raise ValueError("completed episode needs valid native result")
    if seal["status"] == "completed" and seal.get("evidence_issues"):
        raise ValueError("completed episode cannot have declared evidence issues")
    if "failure_class" in seal and ((seal["status"] == "completed") != (seal["failure_class"] is None) or
                                    seal["failure_class"] not in FAILURE_CLASSES):
        raise ValueError("failure class inconsistent with status")
    if seal["status"] == "invalid_evidence" and (not outcome["valid"] or
                                                    not seal.get("evidence_issues") or
                                                    not seal.get("status_reason")):
        raise ValueError("invalid evidence episode needs explicit issues and reason")
    return {"manifest": manifest, "seal": seal}

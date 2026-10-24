"""Immutable, append-only RoboDojo index snapshots for distributed training.

Watch roots are final-publication directories: producers must close each file
before an atomic rename/link makes its final name visible. File age or a stable
mtime is deliberately not used as a substitute for this contract.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Iterator, Mapping, Sequence
import uuid

import numpy as np

from worldscape_policy.data.robodojo_chunks import task_id_from_path
from worldscape_policy.data.robodojo_index import read_index


DEFAULT_WATCH_ROOT = Path("data/robodojo_live")
DEFAULT_PUBLISH_DIR = Path("data/robodojo_hotload")
FORMAT_VERSION = 1


def validate_writable_path(path: str | Path) -> Path:
    """Resolve symlinks and enforce optional WSP_WRITABLE_ROOTS boundaries."""
    resolved = Path(path).expanduser().resolve()
    configured = os.environ.get("WSP_WRITABLE_ROOTS", "")
    roots = [Path(root).expanduser().resolve() for root in configured.split(os.pathsep) if root]
    if configured and not roots:
        raise ValueError("WSP_WRITABLE_ROOTS must contain at least one directory")
    if roots and not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"Output must be inside the user workspace: {resolved}")
    return resolved


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, data: bytes) -> None:
    with validate_writable_path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _verify_identity(record: Mapping[str, Any]) -> None:
    actual = _identity(Path(record["path"]))
    if any(actual[key] != record[key] for key in ("size", "mtime_ns")):
        raise ValueError(f"Published source changed: {record['path']}")


def _is_final_path(path: Path, root: Path) -> bool:
    """Exclude staging directories as well as temporary filenames."""
    path.relative_to(root)  # The caller must supply a descendant, not an escape.
    parts = path.parts
    return path.suffix.lower() in {".h5", ".hdf5"} and not any(
        part.startswith(".") or part.endswith((".tmp", ".partial", ".part", ".pending"))
        for part in parts
    )


def _episode_index(path: Path, episode_id: int, options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a finalized episode and index metadata through the native reader."""
    import h5py

    from worldscape_policy.data.adapters.common import (
        LanguageTemporalPacker, _effective_length, _eligible_temporal_anchors,
        _first, _temporal_labels,
    )
    from worldscape_policy.data.adapters.robodojo import (
        RoboDojoJointHDF5Dataset, _ACTION_JOINT_PATHS, _CAMERA_PATHS,
        _STATE_JOINT_PATHS, _pack_joint_array,
    )

    before = _identity(path)
    with h5py.File(path, "r") as handle:
        success = handle.attrs.get("success", False)
        if not isinstance(success, (bool, np.bool_, int, np.integer)) or success != 1:
            raise ValueError("Episode is not marked successful")
        recording_valid = handle.attrs.get("recording_valid", False)
        if not isinstance(recording_valid, (bool, np.bool_, int, np.integer)) or recording_valid != 1:
            raise ValueError("Live episode lacks recording_valid completion metadata")
        length = len(handle[_STATE_JOINT_PATHS[0]])
        if int(handle.attrs.get("frames", length)) != length:
            raise ValueError("Frame count disagrees with completion metadata")
        state = _pack_joint_array(handle, _STATE_JOINT_PATHS)
        actions = _pack_joint_array(handle, _ACTION_JOINT_PATHS)
        if state.shape != (length, 14) or actions.shape != (length, 14):
            raise ValueError("Expected aligned 14-dimensional joint state/action")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite joint state/action")
        for camera in _CAMERA_PATHS.values():
            if len(handle[camera]) != length:
                raise ValueError(f"Camera length differs from state: {camera}")
        source_key = handle.attrs.get("source_key", "")
        if isinstance(source_key, bytes):
            source_key = source_key.decode("utf-8")
    episode = {
        "episode_index": episode_id, "path": str(path),
        "source": "live",
        "action_label_source": "stored_action",
    }
    if source_key:
        episode["source_key"] = str(source_key)
    # This metadata-only reader avoids constructing a second full dataset and
    # never reads JPEG payloads or computes normalization statistics.
    reader = object.__new__(RoboDojoJointHDF5Dataset)
    reader.paths, reader.episodes, reader.tasks = [path], [episode], {}
    reader.include_joint_velocity = False
    reader._index_arrays = {}
    arrays = reader._read_index_arrays(0)
    length = _effective_length(arrays)
    chunks, horizon = int(options["max_chunk_size"]), int(options["action_horizon"])
    anchor_start = int(options["temporal_anchor_index"])
    anchors = _eligible_temporal_anchors(
        arrays, chunks, action_horizon=horizon,
        respect_trajectory_segments=bool(options["respect_trajectory_segments"]),
        anchor_start=anchor_start, step_stride=int(options["step_stride"]),
    )
    if anchor_start in anchors:
        anchors = [anchor_start, *(anchor for anchor in anchors if anchor != anchor_start)]
    if not anchors:
        raise ValueError("Episode has no eligible temporal anchors")
    mask_value = _first(arrays, ("meta.is_exec", "is_exec"))
    mask = None if mask_value is None else np.asarray(mask_value).astype(bool).reshape(-1)
    labels, segments = _temporal_labels(arrays, length, mask)
    packer = LanguageTemporalPacker(max_chunk_size=chunks, chunk_size=horizon)
    packed = [packer.indices(anchor, labels,
        trajectory_ids=segments if options["respect_trajectory_segments"] else None)
        for anchor in anchors]
    digest = file_sha256(path)
    if _identity(path) != before:
        raise ValueError("Episode changed while validating publication")
    episode["source_sha256"] = digest
    return {
        "episode": episode, "source": {**before, "sha256": digest, "source_key": str(source_key)},
        "episode_length": length,
        "index": np.column_stack((np.full(len(anchors), episode_id, dtype=np.int64),
                                  np.asarray(anchors, dtype=np.int64))),
        "sampling_progress": np.asarray([item.video[0] / max(1, length) for item in packed], dtype=np.float64),
        "lengths": np.asarray([(len(item.action), len(item.state), len(item.video)) for item in packed], dtype=np.int64),
    }


class SnapshotStore:
    """Store immutable full NPZ snapshots; readers never need the source repo.

    ``latest()`` returns an exact descriptor. Persist that descriptor in training
    checkpoints rather than resolving ``latest`` again during exact resume.
    ``resolve`` verifies the manifest and index digests, and optionally source
    size/mtime identities. New file content is SHA256 checked on ingestion.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self._lock_owner: tuple[int, int] | None = None
        self._lock_depth = 0

    @contextmanager
    def publisher_lock(self) -> Iterator[None]:
        """Exclusive writer lock, reentrant only within this process/thread."""
        owner = (os.getpid(), threading.get_ident())
        if self._lock_owner == owner:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        directory = validate_writable_path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        with validate_writable_path(directory / ".publisher.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another publisher owns {directory}") from error
            self._lock_owner, self._lock_depth = owner, 1
            try:
                yield
            finally:
                self._lock_owner, self._lock_depth = None, 0
                fcntl.flock(lock, fcntl.LOCK_UN)

    def latest(self) -> dict[str, Any] | None:
        """Read the atomic pointer without opening index or source files."""
        path = self.directory / "latest.json"
        if not path.exists():
            return None
        descriptor = json.loads(path.read_text())
        self._descriptor_path(descriptor)
        return descriptor

    def _descriptor_path(self, descriptor: Mapping[str, Any]) -> Path:
        if set(descriptor) != {"version", "manifest_sha256", "manifest_path"}:
            raise ValueError("Invalid snapshot descriptor fields")
        version = descriptor["version"]
        digest = descriptor["manifest_sha256"]
        if type(version) is not int or version < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid snapshot version or digest")
        expected = self.directory / "versions" / f"{version:08d}" / "manifest.json"
        if Path(descriptor["manifest_path"]).resolve() != expected or not expected.resolve().is_relative_to(self.directory):
            raise ValueError("Snapshot manifest must belong to this store/version")
        return expected

    def resolve(self, descriptor: Mapping[str, Any], verify_files: bool = True) -> dict[str, Any]:
        """Resolve an exact pinned version, including one older than latest."""
        path = self._descriptor_path(descriptor)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor["manifest_sha256"]:
            raise ValueError("Snapshot manifest SHA256 mismatch")
        manifest = json.loads(raw)
        if manifest.get("format_version") != FORMAT_VERSION or manifest.get("version") != descriptor["version"]:
            raise ValueError("Snapshot manifest version mismatch")
        index_path = path.parent / "index.npz"
        if Path(manifest["index_path"]).resolve() != index_path or not index_path.resolve().is_relative_to(self.directory):
            raise ValueError("Snapshot index path mismatch")
        if file_sha256(index_path) != manifest["index_sha256"]:
            raise ValueError("Snapshot index SHA256 mismatch")
        artifact = read_index(index_path, verify_files=verify_files)
        if (manifest["options"] != artifact["manifest"]["options"]
                or manifest["episode_count"] != len(artifact["manifest"]["paths"])
                or manifest["sample_count"] != len(artifact["index"])):
            raise ValueError("Snapshot index metadata mismatch")
        if manifest["previous"] is not None:
            previous = manifest["previous"]
            previous_path = self._descriptor_path(previous)
            previous_raw = previous_path.read_bytes()
            if hashlib.sha256(previous_raw).hexdigest() != previous["manifest_sha256"]:
                raise ValueError("Snapshot parent manifest SHA256 mismatch")
            parent = json.loads(previous_raw)
            if (previous["version"] >= descriptor["version"]
                    or parent["base_index_sha256"] != manifest["base_index_sha256"]
                    or parent["options"] != manifest["options"]
                    or parent["allowed_tasks"] != manifest["allowed_tasks"]
                    or parent["episode_count"] >= manifest["episode_count"]
                    or parent["sample_count"] >= manifest["sample_count"]):
                raise ValueError("Snapshot parent/append contract mismatch")
        elif descriptor["version"] != 0 or manifest["base_index_sha256"] != manifest["index_sha256"]:
            raise ValueError("Snapshot zero must preserve the original index bytes")
        return {"descriptor": dict(descriptor), "manifest": manifest, "index_path": str(index_path)}

    def initialize(self, base_index: str | Path) -> dict[str, Any]:
        """Return version zero, preserving original NPZ bytes even after updates."""
        base_index = Path(base_index).expanduser().resolve()
        digest = file_sha256(base_index)
        # Training is a reader when version zero is already published. The
        # separate --watch process holds the writer lock for its lifetime, so
        # taking that lock here would prevent training startup and exact resume.
        if self.latest() is not None:
            return self._initialized_zero(digest)
        with self.publisher_lock():
            current = self.latest()
            if current is not None:
                return self._initialized_zero(digest)
            # A crash after publishing the immutable directory but before the
            # latest pointer must not force deletion/recreation of version zero.
            zero_path = self.directory / "versions" / "00000000" / "manifest.json"
            if zero_path.exists():
                zero = {"version": 0, "manifest_sha256": file_sha256(zero_path),
                        "manifest_path": str(zero_path)}
                record = self.resolve(zero, verify_files=True)
                if record["manifest"]["base_index_sha256"] != digest:
                    raise ValueError("Unpublished version zero has a different base index")
                self._publish_pointer(zero)
                return zero
            artifact = read_index(base_index, verify_files=True)
            options = artifact["manifest"]["options"]
            # Flash and memory share the publisher. New episodes inherit the
            # base index's chunk limit; their actual packed lengths remain in
            # the index for global bucketing and exact cursor restoration.
            if (not options["temporal_packing"] or options["action_horizon"] != 24
                    or options["max_chunk_size"] not in {1, 24} or "chunk_policy" in options):
                raise ValueError(
                    "Live RoboDojo snapshots require uniform H24/C1 (flash) "
                    "or H24/C24 (memory) temporal packing"
                )
            records = {str(Path(row["path"]).resolve()): dict(row)
                       for row in artifact["manifest"]["source_files"]}
            if any(str(Path(path).resolve()) not in records for path in artifact["manifest"]["paths"]):
                raise ValueError("Base index must record the identity of every episode")
            for episode in artifact["manifest"]["episodes"]:
                row = records.get(str(Path(episode["path"]).resolve()))
                if row is not None:
                    if episode.get("source_sha256"):
                        row["sha256"] = episode["source_sha256"]
                    if episode.get("source_key"):
                        row["source_key"] = episode["source_key"]
            manifest = {
                "format_version": FORMAT_VERSION, "version": 0, "previous": None,
                "base_index_sha256": digest, "options": options,
                "allowed_tasks": sorted({task_id_from_path(Path(path)) for path in artifact["manifest"]["paths"]}),
                "episode_count": len(artifact["manifest"]["paths"]), "sample_count": len(artifact["index"]),
                "sources": list(records.values()),
            }
            return self._commit(manifest, artifact=None, copy_index=base_index, expected_digest=digest)

    def _initialized_zero(self, expected_base_digest: str) -> dict[str, Any]:
        """Read immutable version zero without participating in publisher locks."""
        zero_path = self.directory / "versions" / "00000000" / "manifest.json"
        zero = {"version": 0, "manifest_sha256": file_sha256(zero_path),
                "manifest_path": str(zero_path)}
        record = self.resolve(zero, verify_files=False)
        if record["manifest"]["base_index_sha256"] != expected_base_digest:
            raise ValueError("Snapshot store already has a different base index")
        return zero

    def _commit(self, manifest: dict[str, Any], *, artifact: dict[str, Any] | None,
                copy_index: Path | None = None, expected_digest: str | None = None) -> dict[str, Any]:
        versions = validate_writable_path(self.directory / "versions")
        versions.mkdir(parents=True, exist_ok=True)
        version = manifest["version"]
        final = validate_writable_path(versions / f"{version:08d}")
        if final.exists():
            raise FileExistsError(f"Immutable snapshot version already exists: {final}")
        temporary = validate_writable_path(versions / f".pending-{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            index_path = validate_writable_path(temporary / "index.npz")
            if copy_index is not None:
                with copy_index.open("rb") as source, index_path.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            else:
                assert artifact is not None
                with index_path.open("xb") as target:
                    np.savez_compressed(target, **{**artifact, "manifest": np.asarray(json.dumps(artifact["manifest"]))})
                    target.flush()
                    os.fsync(target.fileno())
            digest = file_sha256(index_path)
            if expected_digest is not None and digest != expected_digest:
                raise ValueError("Base index changed while copying version zero")
            read_index(index_path, verify_files=True)
            manifest = {**manifest, "index_path": str(final / "index.npz"), "index_sha256": digest}
            raw = _json_bytes(manifest)
            _write_bytes(temporary / "manifest.json", raw)
            _sync_directory(temporary)
            temporary.rename(final)
            _sync_directory(versions)
            descriptor = {"version": version, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                          "manifest_path": str(final / "manifest.json")}
            self._publish_pointer(descriptor)
            return descriptor
        finally:
            # Remove only this transaction's private unfinished directory.
            if temporary.exists():
                shutil.rmtree(validate_writable_path(temporary))

    def _publish_pointer(self, descriptor: Mapping[str, Any]) -> None:
        pointer = validate_writable_path(self.directory / f".latest-{uuid.uuid4().hex}.tmp")
        try:
            _write_bytes(pointer, _json_bytes(descriptor))
            pointer.replace(validate_writable_path(self.directory / "latest.json"))
            _sync_directory(self.directory)
        finally:
            pointer.unlink(missing_ok=True)

    def publish(self, watch_roots: Sequence[str | Path]) -> dict[str, Any]:
        """Discover completed episodes and append one atomic version if needed.

        Bad/unready new files are reported and left untouched. A changed source
        already referenced by any snapshot is fatal, rather than silently
        invalidating the training stream or its recovery history.
        """
        with self.publisher_lock():
            previous = self.latest()
            if previous is None:
                raise ValueError("Initialize a base snapshot before publishing")
            current = self.resolve(previous, verify_files=True)
            previous_manifest = current["manifest"]
            artifact = read_index(current["index_path"])
            old = artifact["manifest"]
            known = {str(Path(row["path"]).resolve()): row for row in previous_manifest["sources"]}
            digests = {row["sha256"] for row in known.values() if row.get("sha256")}
            keys = {row["source_key"] for row in known.values() if row.get("source_key")}
            allowed = set(previous_manifest["allowed_tasks"])
            candidates: set[Path] = set()
            for value in watch_roots:
                root = Path(value).expanduser().resolve()
                if not root.is_dir():
                    raise ValueError(f"Watch root is not a directory: {root}")
                for path in root.rglob("*"):
                    if _is_final_path(path, root) and path.is_file():
                        resolved = path.resolve()
                        # Symlink aliases into pending or outside this producer
                        # root are not completed publication events.
                        if resolved.is_relative_to(root) and _is_final_path(resolved, root):
                            candidates.add(resolved)
            additions, rejected = [], []
            for path in sorted(candidates):
                if str(path) in known:
                    _verify_identity(known[str(path)])
                    continue
                if task_id_from_path(path) not in allowed:
                    rejected.append({"path": str(path), "reason": "task_not_in_base_index"})
                    continue
                try:
                    addition = _episode_index(path, len(old["paths"]) + len(additions), old["options"])
                    source = addition["source"]
                    if source["sha256"] in digests or source["source_key"] and source["source_key"] in keys:
                        rejected.append({"path": str(path), "reason": "duplicate_source"})
                        continue
                    digests.add(source["sha256"])
                    if source["source_key"]:
                        keys.add(source["source_key"])
                    additions.append(addition)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    rejected.append({"path": str(path), "reason": str(error)})
            if not additions:
                return {"status": "unchanged", "descriptor": previous, "accepted": 0, "rejected": rejected}
            merged = copy.deepcopy(old)
            merged["paths"].extend(item["episode"]["path"] for item in additions)
            merged["episodes"].extend(item["episode"] for item in additions)
            merged["episode_lengths"].extend(item["episode_length"] for item in additions)
            merged["source_files"].extend({key: item["source"][key] for key in ("path", "size", "mtime_ns")}
                                          for item in additions)
            next_artifact = {"manifest": merged, **{
                key: np.concatenate([artifact[key], *(item[key] for item in additions)], axis=0)
                for key in ("index", "sampling_progress", "lengths")}}
            # Stale immutable directories from an interrupted pointer update are
            # retained. A later transaction uses a fresh monotonically larger ID.
            versions = self.directory / "versions"
            version = max([previous["version"], *(int(p.name) for p in versions.iterdir() if p.name.isdigit())]) + 1
            manifest = {**previous_manifest, "version": version, "previous": previous,
                        "episode_count": len(merged["paths"]), "sample_count": len(next_artifact["index"]),
                        "sources": [*previous_manifest["sources"], *(item["source"] for item in additions)]}
            # Detect a source producer changing an accepted episode before the
            # snapshot becomes visible, including during other episode reads.
            for item in additions:
                _verify_identity(item["source"])
            descriptor = self._commit(manifest, artifact=next_artifact)
            return {"status": "published", "descriptor": descriptor, "accepted": len(additions), "rejected": rejected}

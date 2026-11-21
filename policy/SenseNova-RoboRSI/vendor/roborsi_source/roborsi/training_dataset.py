# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned, indexed datasets of portable agent training episodes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker

from roborsi.training_data import convert_episode, load_training_episode
from roborsi.training_trace import convert_roborsi_trace

_SPLITS = frozenset({"train", "validation", "test"})
_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def training_dataset_schema() -> dict[str, Any]:
    """Return a fresh copy of the dataset manifest version 1 JSON Schema."""
    source = files("roborsi.schemas").joinpath("training_dataset.v1.json")
    return json.loads(source.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _component(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    encoded = quote(value, safe="")
    if encoded in {".", ".."} or len(encoded) > 120:
        encoded = f"id-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    return encoded


def _dataset_name(value: str, name: str) -> str:
    if not isinstance(value, str) or not _DATASET_NAME.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty path-safe identifier")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    Draft202012Validator(
        training_dataset_schema(), format_checker=FormatChecker()
    ).validate(manifest)
    if (
        root.name != manifest["dataset_version"]
        or root.parent.name != manifest["dataset_id"]
    ):
        raise ValueError("dataset directory does not match manifest identity")
    if (manifest["status"] == "sealed") != (manifest["sealed_at"] is not None):
        raise ValueError("dataset status and sealed_at disagree")
    return manifest


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    Draft202012Validator(
        training_dataset_schema(), format_checker=FormatChecker()
    ).validate(manifest)
    temporary = root / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _writer_lock(root: Path) -> Iterator[None]:
    with (root / ".writer.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _entry(root: Path, document: Path, split: str) -> dict[str, Any]:
    episode = load_training_episode(document)
    task = episode["task"]
    outcome = episode["outcome"]
    return {
        "path": document.relative_to(root).as_posix(),
        "sha256": _sha256(document),
        "episode_id": episode["episode_id"],
        "split": split,
        "benchmark": task["benchmark"],
        "benchmark_version": task.get("benchmark_version"),
        "task_id": task["task_id"],
        "seed": task.get("seed"),
        "status": outcome["status"],
        "success": outcome["success"],
        "score": outcome["score"],
    }


def _expected_path(root: Path, entry: Mapping[str, Any]) -> Path:
    return (
        root
        / "episodes"
        / entry["split"]
        / _component(entry["benchmark"], "benchmark")
        / _component(entry["task_id"], "task_id")
        / _component(entry["episode_id"], "episode_id")
        / "episode.json"
    )


def _verify(root: Path, manifest: Mapping[str, Any]) -> None:
    paths: set[str] = set()
    task_splits: dict[tuple[str, str], str] = {}
    for recorded in manifest["episodes"]:
        path = _expected_path(root, recorded)
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"dataset episode escapes its root: {recorded['path']}")
        if recorded["path"] != path.relative_to(root).as_posix():
            raise ValueError(f"noncanonical dataset episode path: {recorded['path']}")
        if recorded["path"] in paths:
            raise ValueError(f"duplicate dataset episode path: {recorded['path']}")
        paths.add(recorded["path"])
        task_key = (recorded["benchmark"], recorded["task_id"])
        prior = task_splits.setdefault(task_key, recorded["split"])
        if prior != recorded["split"]:
            raise ValueError(f"task appears in multiple splits: {task_key}")
        if not path.is_file() or _sha256(path) != recorded["sha256"]:
            raise ValueError(f"dataset episode checksum mismatch: {recorded['path']}")
        if _entry(root, path, recorded["split"]) != recorded:
            raise ValueError(f"dataset index differs from episode: {recorded['path']}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "episodes").glob("**/episode.json")
    }
    if actual_paths != paths:
        raise ValueError("dataset has unindexed or missing episode files")


class TrainingDataset:
    """One versioned dataset, written safely by independent worker processes.

    Episodes are stored under ``episodes/<split>/<benchmark>/<task>/<episode>``.
    The manifest is replaced atomically after each successful episode. An
    advisory lock serializes writers to this dataset version on Linux.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        *,
        dataset_id: str,
        dataset_version: str,
        producer: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingDataset:
        """Create a new ``<base>/<dataset_id>/<dataset_version>`` directory.

        ``producer`` identifies the exporter; episode provenance separately
        identifies the agent and benchmark versions used during collection.
        """
        dataset_id = _dataset_name(dataset_id, "dataset_id")
        dataset_version = _dataset_name(dataset_version, "dataset_version")
        root = Path(base_dir).expanduser().resolve() / dataset_id / dataset_version
        root.mkdir(parents=True, exist_ok=False)
        try:
            now = _now()
            manifest = {
                "schema_version": 1,
                "episode_schema_version": 1,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "status": "draft",
                "created_at": now,
                "updated_at": now,
                "sealed_at": None,
                "split_policy": "task_disjoint",
                "producer": {
                    "name": producer["name"],
                    "version": producer["version"],
                    "git_commit": producer.get("git_commit"),
                },
                "metadata": dict(metadata or {}),
                "episodes": [],
            }
            _write_manifest(root, manifest)
        except BaseException:
            shutil.rmtree(root)
            raise
        return cls(root)

    @classmethod
    def open(cls, root: str | Path) -> TrainingDataset:
        """Open an existing version without changing its manifest."""
        dataset = cls(Path(root))
        _read_manifest(dataset.root)
        return dataset

    def manifest(self) -> dict[str, Any]:
        """Read the latest manifest, including entries written by other workers."""
        return _read_manifest(self.root)

    def verify(self) -> dict[str, Any]:
        """Check the index, each episode JSON checksum and all image assets."""
        with _writer_lock(self.root):
            manifest = _read_manifest(self.root)
            _verify(self.root, manifest)
            return manifest

    def _destination(
        self,
        manifest: Mapping[str, Any],
        task: Mapping[str, Any],
        episode_id: str,
        split: str,
    ) -> Path:
        if manifest["status"] != "draft":
            raise ValueError("sealed dataset versions cannot accept new episodes")
        if split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        benchmark = task["benchmark"]
        task_id = task["task_id"]
        task_key = (benchmark, task_id)
        if any(
            (row["benchmark"], row["task_id"]) == task_key and row["split"] != split
            for row in manifest["episodes"]
        ):
            raise ValueError(f"task appears in multiple splits: {task_key}")
        return (
            self.root
            / "episodes"
            / split
            / _component(benchmark, "benchmark")
            / _component(task_id, "task_id")
            / _component(episode_id, "episode_id")
        )

    def _add(
        self,
        task: Mapping[str, Any],
        *,
        episode_id: str,
        split: str,
        convert: Callable[[Path], Path],
    ) -> Path:
        with _writer_lock(self.root):
            manifest = _read_manifest(self.root)
            destination = self._destination(manifest, task, episode_id, split)
            if destination.exists():
                raise FileExistsError(f"dataset episode already exists: {destination}")
            document = convert(destination)
            try:
                manifest["episodes"].append(_entry(self.root, document, split))
                manifest["updated_at"] = _now()
                _write_manifest(self.root, manifest)
            except BaseException:
                shutil.rmtree(destination)
                raise
            return document

    def add_episode(
        self,
        task: Mapping[str, Any],
        calls: Iterable[Mapping[str, Any]],
        *,
        episode_id: str,
        split: str,
        outcome: Mapping[str, Any] | None = None,
        source_dir: str | Path | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Convert benchmark-owned LLM calls and index the resulting episode."""
        return self._add(
            task,
            episode_id=episode_id,
            split=split,
            convert=lambda output_dir: convert_episode(
                task,
                calls,
                output_dir=output_dir,
                outcome=outcome,
                episode_id=episode_id,
                source_dir=source_dir,
                provenance=provenance,
            ),
        )

    def add_codex_capture(
        self,
        capture_dir: str | Path,
        task: Mapping[str, Any],
        *,
        episode_id: str,
        split: str,
        outcome: Mapping[str, Any] | None = None,
        versions: Mapping[str, str] | None = None,
        allow_partial: bool = False,
        require_encrypted_content: bool = False,
    ) -> Path:
        """Convert one explicitly selected capture under a caller-owned episode ID.

        Capture directory names such as ``call-0001`` are not globally unique.
        The caller identifies the native episode and chooses its task split.
        """
        from roborsi.training_capture import convert_codex_capture

        return self._add(
            task,
            episode_id=episode_id,
            split=split,
            convert=lambda output_dir: convert_codex_capture(
                capture_dir,
                task,
                output_dir=output_dir,
                outcome=outcome,
                episode_id=episode_id,
                versions=versions,
                allow_partial=allow_partial,
                require_encrypted_content=require_encrypted_content,
            ),
        )

    def add_roborsi_trace(
        self,
        run_dir: str | Path,
        task: Mapping[str, Any],
        *,
        split: str,
        outcome: Mapping[str, Any] | None = None,
        versions: Mapping[str, str] | None = None,
        allow_partial: bool = False,
    ) -> Path:
        """Convert a RoboRSI full trace and use its run ID as episode ID."""
        trace_manifest = json.loads(
            (Path(run_dir) / "trace/manifest.json").read_text(encoding="utf-8")
        )
        episode_id = trace_manifest.get("run_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("RoboRSI trace manifest needs a run_id")
        return self._add(
            task,
            episode_id=episode_id,
            split=split,
            convert=lambda output_dir: convert_roborsi_trace(
                run_dir,
                task,
                output_dir=output_dir,
                outcome=outcome,
                allow_partial=allow_partial,
                versions=versions,
            ),
        )

    def seal(self) -> dict[str, Any]:
        """Verify and freeze this dataset version; later changes need a new one."""
        with _writer_lock(self.root):
            manifest = _read_manifest(self.root)
            if manifest["status"] != "draft":
                raise ValueError("dataset version is already sealed")
            if not manifest["episodes"]:
                raise ValueError("cannot seal an empty dataset")
            _verify(self.root, manifest)
            manifest["status"] = "sealed"
            manifest["sealed_at"] = _now()
            manifest["updated_at"] = manifest["sealed_at"]
            _write_manifest(self.root, manifest)
            return manifest

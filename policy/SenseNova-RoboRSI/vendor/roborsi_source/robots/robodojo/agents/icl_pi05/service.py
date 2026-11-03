# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Serve declared original ICL episodes and their complete native artifacts."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roborsi.benchmarks.registry import directory_sha256
from roborsi.evaluation.http_artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactPublisher,
    artifact_path,
)
from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    BenchmarkRuntimeInfo,
    EpisodeStop,
    NativeResult,
    QueryResult,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.config import get_project_env
from robots.robodojo.agents.icl_pi05.benchmark import IclBenchmarkAdapter
from robots.robodojo.agents.icl_pi05.preparation import verify_source
from robots.robodojo.benchmark_service import native_commit, verify_native_source

_SECRET_PARTS = {".codex", ".ssh", "credentials", "private-home"}
_SECRET_FILES = {"auth.json", "credentials.json", "api_keys.yaml", ".env"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _regular(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise ValueError("ICL service paths must be absolute")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("ICL service paths cannot contain symlinks")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("ICL service input must be a regular file")
    return info


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    """Exclude atime, which legitimate verification reads may update."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
    )


def _document(receipt: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if set(receipt) != {"path", "sha256"}:
        raise ValueError("ICL document requires an explicit path and SHA-256")
    path = Path(receipt["path"])
    before = _regular(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt["sha256"]:
        raise ValueError("ICL document SHA-256 differs")
    if _file_identity(_regular(path)) != _file_identity(before):
        raise ValueError("ICL document changed during verification")
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError("ICL service document must be an object")
    return path, document


def _census(root: Path) -> dict[str, tuple[int, ...]]:
    """Check links/types and pin every file and directory's current identity."""
    if not root.is_absolute() or any(
        path.is_symlink() for path in (root, *root.parents)
    ):
        raise ValueError("ICL asset paths must be absolute without symlinks")
    paths = [root]
    if root.is_dir():
        for parent, dirs, files in os.walk(root, followlinks=False):
            paths.extend(Path(parent) / name for name in [*dirs, *files])
    result = {}
    for path in paths:
        info = path.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError("ICL asset contains a link or special file")
        result[path.relative_to(root).as_posix()] = _file_identity(info)
    return result


def _publish_file(
    source: Path, relative: str, publisher: ArtifactPublisher
) -> dict[str, Any]:
    """Preserve every byte, splitting transport objects rather than truncating."""
    before = _regular(source)
    if before.st_uid != os.getuid():
        raise ValueError("Native artifact belongs to a different owner")
    base = "native/icl-original/" + relative
    mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    chunks, digest = [], hashlib.sha256()
    with source.open("rb") as stream:
        index = 0
        while True:
            data = stream.read(MAX_ARTIFACT_BYTES)
            if not data and (index or before.st_size):
                break
            ref = (
                base
                if before.st_size <= MAX_ARTIFACT_BYTES
                else (
                    f"native/icl-chunks/{hashlib.sha256(relative.encode()).hexdigest()}/{index:06d}"
                )
            )
            target = artifact_path(publisher.root, ref, create=True)
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
            info = publisher.publish(ref, "file", mime)
            digest.update(data)
            chunks.append(info.as_dict())
            index += 1
            if not data:
                break
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Native source changed during publication")
    return {
        "path": relative,
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
        "chunks": chunks,
    }


class _PublishedIclAdapter:
    """Construct the original adapter only after the declared Task is known."""

    def __init__(
        self,
        benchmark: IclBenchmark,
        directory: Path,
        publisher: ArtifactPublisher,
        *,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.directory = directory
        self.publisher = publisher
        self.backend_factory = backend_factory
        self._adapter = None
        self._result = None
        self._published = False

    def start(self, task: Task, episode_id: str) -> StartResult:
        """Bind the exact exported task before starting the original worker."""
        if self._adapter is not None or task not in self.benchmark.tasks():
            raise ValueError("ICL task differs from the exported catalog")
        config = self.benchmark.config_for(task)
        self._adapter = IclBenchmarkAdapter(
            config,
            self.directory / "icl-private",
            self.directory / "media",
            backend_factory=self.backend_factory,
        )
        return self._adapter.start(task, episode_id)

    def step(self, action: Action) -> StepResult:
        return self._adapter.step(action)

    def query(self, request: BenchmarkQuery) -> QueryResult:
        return self._adapter.query(request)

    def stop(self, request: EpisodeStop) -> StepResult:
        return self._adapter.stop(request)

    def finish(self) -> NativeResult:
        """Return original scoring with verified byte-preserving evidence."""
        if self._result is not None:
            return self._result
        result = self._adapter.finish()
        # Flush the original worker and physics/log handles before snapshotting.
        self._adapter.close()
        self._check_native_stopped()
        metrics = artifact_path(
            self.publisher.root, "native/original-metrics.json", create=True
        )
        write_json_atomic(metrics, result.details["original_metrics"])
        metrics_info = self.publisher.publish(
            "native/original-metrics.json", "native_result", "application/json"
        )
        self._publish_records("complete")
        self._result = result.model_copy(
            update={
                "source": metrics_info.ref,
                "source_sha256": metrics_info.sha256,
                "details": {
                    **result.details,
                    "metrics_evidence": "snapshot_of_original_native_metrics_object",
                },
            }
        )
        return self._result

    def _publish_records(self, status: str) -> None:
        if self._published:
            return
        rows, exclusions = [], []
        root = self._adapter.directory
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in dirs:
                child = Path(parent) / name
                if child.is_symlink():
                    raise ValueError("Native artifact directory is a symlink")
            for name in sorted(files):
                path = Path(parent) / name
                relative = path.relative_to(root)
                if (
                    relative == Path("config.json")
                    or name in _SECRET_FILES
                    or _SECRET_PARTS.intersection(relative.parts)
                ):
                    exclusions.append(
                        {"path": relative.as_posix(), "reason": "private_configuration"}
                    )
                    continue
                rows.append(_publish_file(path, relative.as_posix(), self.publisher))
        manifest = artifact_path(
            self.publisher.root, "native/icl-reconstruction.json", create=True
        )
        write_json_atomic(
            manifest,
            {
                "schema_version": 1,
                "status": status,
                "files": rows,
                "exclusions": exclusions,
                "metrics_source": "snapshot_of_original_native_metrics_object"
                if status == "complete"
                else None,
            },
        )
        self.publisher.publish(
            "native/icl-reconstruction.json", "file", "application/json"
        )
        self._published = True

    def _check_native_stopped(self) -> None:
        if not self._adapter._closed:
            raise ValueError("Original native worker cleanup has not completed")
        backend = self._adapter._backend
        if backend is None:
            return
        process = getattr(backend, "process", None)
        if process is None:
            if self.backend_factory is None:
                raise ValueError("Original native worker has no process-exit evidence")
            return  # Explicit offline backend injection is not GPU evidence.
        if process.poll() is None:
            raise ValueError("Original native worker is still running")
        receipt = self._adapter.directory / "worker-close.json"
        _regular(receipt)
        closed = json.loads(receipt.read_text())
        if (
            closed.get("native_closed") is not True
            or closed.get("pid") != process.pid
            or closed.get("episode_id") != self.publisher.episode_id
            or closed.get("identity") != self._adapter._task.as_dict()
        ):
            raise ValueError("Original native worker exit receipt differs")

    def close(self) -> None:
        """Finalize native handles and retain partial technical evidence."""
        if self._adapter is not None:
            self._adapter.close()
            self._check_native_stopped()
            self._publish_records("partial_technical_interruption")


class IclBenchmark:
    """Serve one exported eval seed with pinned original configurations.

    Runtime platform fields come from an explicitly pinned deployment receipt;
    this service checks source and asset bytes, not a fresh GPU/model episode.
    """

    name = "robodojo"
    supports_episode_stop = True
    supports_benchmark_queries = True
    runtime_evidence_source = "provided_verified_receipt"

    def __init__(
        self, config: Path, *, backend_factory: Callable[..., Any] | None = None
    ) -> None:
        """Validate declared catalog, original inputs and deployment receipts."""
        before = _regular(config)
        data = config.read_bytes()
        if _file_identity(_regular(config)) != _file_identity(before):
            raise ValueError("ICL service configuration changed during reading")
        document = json.loads(data)
        if document.get("schema_version") != 1 or set(document) != {
            "schema_version",
            "catalog",
            "configs",
            "assets",
            "runtime",
            "native_root",
        }:
            raise ValueError(
                "ICL service needs a complete explicit deployment configuration"
            )
        self.document = document
        self._document_pins = [(config, hashlib.sha256(data).hexdigest())]
        self.backend_factory = backend_factory
        catalog_path, catalog = _document(document["catalog"])
        runtime_path, runtime = _document(document["runtime"])
        self._document_pins.extend(
            [
                (catalog_path, document["catalog"]["sha256"]),
                (runtime_path, document["runtime"]["sha256"]),
            ]
        )
        self._runtime = BenchmarkRuntimeInfo.from_dict(runtime)
        if (
            self._runtime.benchmark != self.name
            or catalog.get("schema_version") != 1
            or catalog.get("benchmark_commit") != self._runtime.benchmark_commit
            or self._runtime.benchmark_commit != native_commit()
        ):
            raise ValueError("ICL runtime/catalog/native profile differ")
        native_root = Path(document["native_root"])
        if (
            not native_root.is_absolute()
            or not native_root.is_dir()
            or any(path.is_symlink() for path in (native_root, *native_root.parents))
        ):
            raise ValueError(
                "ICL native root must be an absolute directory without symlinks"
            )
        verify_native_source(native_root)
        self._tasks = [Task.from_dict(item) for item in catalog["tasks"]]
        keys = {(task.task_id, task.seed) for task in self._tasks}
        if not self._tasks or len(keys) != len(self._tasks):
            raise ValueError("ICL catalog requires unique task/layout identities")
        if len({task.native.get("eval_seed") for task in self._tasks}) != 1:
            raise ValueError("ICL catalog must contain one eval seed")
        self._configs = {}
        for row in document["configs"]:
            if set(row) != {"task_id", "layout_id", "eval_seed", "path", "sha256"}:
                raise ValueError("ICL config binding differs")
            path, cfg = _document({key: row[key] for key in ("path", "sha256")})
            key = (row["task_id"], row["layout_id"])
            if key in self._configs or key not in keys:
                raise ValueError("ICL config binding is duplicated or undeclared")
            task = next(
                task for task in self._tasks if (task.task_id, task.seed) == key
            )
            episodes = cfg.get("test_episodes", [])
            if len(episodes) != 1:
                raise ValueError("ICL config must declare one evaluation episode")
            episode = episodes[0]
            options = cfg["options"]
            offset = options.get("replay", {}).get("control_step", 0)
            source_adapter = f"{Path(__file__).resolve().parent / 'source/hybrid_adapter.py'}:Adapter"
            if cfg.get("environment_dir") != document["native_root"] or (
                self.backend_factory is None and cfg.get("adapter") != source_adapter
            ):
                raise ValueError(
                    "ICL executed native/source paths differ from verified deployment"
                )
            if (
                task.benchmark != self.name
                or task.capabilities.get("recoverable_action_rejections") is not True
                or (
                    episode["task"],
                    episode["layout_id"],
                    episode["eval_seed"],
                    episode["robot"],
                )
                != (
                    task.task_id,
                    task.seed,
                    task.native["eval_seed"],
                    task.native["config"],
                )
                or row["eval_seed"] != episode["eval_seed"]
                or task.max_steps
                != options["review_window_steps"] * options["review_max_windows"]
                - offset
            ):
                raise ValueError("ICL catalog/config native identity or budget differs")
            self._configs[key] = (path, row["sha256"])
            self._document_pins.append((path, row["sha256"]))
        if self._configs.keys() != keys:
            raise ValueError("ICL catalog configuration coverage differs")
        self._assets = self._verify_assets()
        self._check_deployment()

    def _verify_assets(self) -> list[dict[str, Any]]:
        assets = []
        self._asset_census = []
        if not self.document["assets"]:
            raise ValueError("ICL service requires verified assets/checkpoint receipts")
        names = [row["name"] for row in self.document["assets"]]
        if len(names) != len(set(names)) or not {
            "robodojo-assets",
            "pi05-checkpoint",
        } <= set(names):
            raise ValueError(
                "ICL service requires unique simulation assets and pi0.5 checkpoint roles"
            )
        for row in self.document["assets"]:
            if set(row) != {"name", "kind", "path", "sha256"}:
                raise ValueError("ICL asset requires name/kind/path/SHA-256")
            path = Path(row["path"])
            before = _census(path)
            if row["kind"] == "directory":
                actual = directory_sha256(path)
            elif row["kind"] == "file":
                _regular(path)
                actual = _sha(path)
            else:
                raise ValueError("Unsupported ICL asset kind")
            if actual != row["sha256"]:
                raise ValueError("ICL asset SHA-256 differs")
            if _census(path) != before:
                raise ValueError("ICL asset changed during byte verification")
            self._asset_census.append((path, before))
            assets.append({key: row[key] for key in ("name", "kind", "sha256")})
        return assets

    def _check_deployment(self) -> None:
        if self.backend_factory is None:
            verify_source()
        verify_native_source(Path(self.document["native_root"]))
        if native_commit() != self._runtime.benchmark_commit:
            raise ValueError("ICL native profile changed after startup")
        for path, expected in self._document_pins:
            _regular(path)
            if _sha(path) != expected:
                raise ValueError("ICL deployment document changed after startup")
        for path, expected in self._asset_census:
            if _census(path) != expected:
                raise ValueError("ICL asset identity changed; revalidation is required")

    def tasks(self) -> list[Task]:
        """Return only supplied, verified native-exported task definitions."""
        self._check_deployment()
        return [Task.from_dict(task.as_dict()) for task in self._tasks]

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        """Return the pinned runtime receipt, without claiming a new GPU test."""
        self._check_deployment()
        return BenchmarkRuntimeInfo.from_dict(self._runtime.as_dict())

    def assets(self) -> list[dict[str, Any]]:
        """Recheck current metadata after the startup whole-byte SHA check."""
        self._check_deployment()
        return [
            dict(row, verification="initial_bytes_sha_and_currentmetadata_census")
            for row in self._assets
        ] + [
            {
                "name": "icl-runtime-receipt",
                "kind": "file",
                "sha256": self.document["runtime"]["sha256"],
                "source": self.runtime_evidence_source,
            }
        ]

    def config_for(self, task: Task) -> Path:
        """Retrieve the explicit source config while refusing content drift."""
        path, expected = self._configs[(task.task_id, task.seed)]
        if _sha(path) != expected:
            raise ValueError("ICL bound config changed after startup")
        return path

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> _PublishedIclAdapter:
        """Create a case-owned wrapper without starting a native episode."""
        return _PublishedIclAdapter(
            self, directory, artifacts, backend_factory=self.backend_factory
        )


def make_icl_benchmark() -> IclBenchmark:
    """Construct the service only from an explicit deployment config path."""
    token = get_project_env("ROBORSI_BENCHMARK_TOKEN")
    if not token or not token.strip():
        raise ValueError("Original ICL service requires ROBORSI_BENCHMARK_TOKEN")
    value = os.environ.get("ROBORSI_ICL_BENCHMARK_CONFIG")
    if not value:
        raise ValueError(
            "Set ROBORSI_ICL_BENCHMARK_CONFIG to a verified deployment JSON"
        )
    return IclBenchmark(Path(value))

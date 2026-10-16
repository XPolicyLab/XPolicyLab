# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish an exported native catalog and isolated RoboDojo HTTP episodes."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import directory_sha256
from roborsi.benchmarks.runtime_source import benchmark_source_digest
from roborsi.evaluation.http_artifacts import ArtifactPublisher
from roborsi.evaluation.messages import (
    BenchmarkRuntimeInfo,
    NativeResult,
    StartResult,
    Task,
)
from roborsi.utils.config import get_project_env
from robots.robodojo.roborsi_adapter import (
    ACTION_SCHEMA,
    EVALUATION_RULES,
    RoboDojoRoboRSIAdapter,
)

NATIVE_COMMIT = "726e9aabfaa642203722eb126f5eaf0f37f3e1ad"
NATIVE_ROOT = Path("/workspace/RoboDojo")
ASSETS_ROOT = Path("/roborsi/assets/robodojo")
CATALOG_PATH = Path("/roborsi/assets/robodojo-catalog")


def _native_inventory() -> tuple[str, dict[str, str]]:
    inventory = json.loads(Path(__file__).with_name("native_files.json").read_text())
    if inventory["commit"] != NATIVE_COMMIT:
        raise ValueError("RoboDojo source inventory and supported commit differ")
    commit = get_project_env("ROBORSI_ROBODOJO_NATIVE_COMMIT", NATIVE_COMMIT)
    expected = dict(inventory["files"])
    if commit != NATIVE_COMMIT:
        overrides = inventory.get("profiles", {}).get(commit)
        if overrides is None:
            raise ValueError(f"unsupported RoboDojo native commit: {commit}")
        if not overrides.keys() <= expected.keys():
            raise ValueError(
                "RoboDojo profile contains files outside the source inventory"
            )
        expected.update(overrides)
    return commit, expected


def native_commit() -> str:
    """Resolve an explicitly supported native revision before export or startup."""
    return _native_inventory()[0]


def verify_native_source(root: Path = NATIVE_ROOT) -> None:
    """Check every native file against the selected revision's pinned Git blobs."""
    _, expected = _native_inventory()
    for name, oid in expected.items():
        path = root / name
        if path.is_symlink():
            raise ValueError("RoboDojo native source must not contain symlinks")
        data = path.read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if actual != oid:
            raise ValueError(f"RoboDojo source differs from supported commit: {name}")
    for directory in {name.split("/")[0] for name in expected}:
        for path in (root / directory).rglob("*.py"):
            if path.relative_to(root).as_posix() not in expected:
                raise ValueError(f"unexpected RoboDojo source: {path}")


def _isaacsim_version() -> dict[str, str]:
    try:
        value = version("isaacsim")
        source = "distribution:isaacsim"
    except PackageNotFoundError:
        root = get_project_env("ISAACSIM_ROOT_PATH", "/isaac-sim")
        if not root or not Path(root).is_absolute():
            raise ValueError("ISAACSIM_ROOT_PATH must be an absolute directory")
        path = Path(root) / "VERSION"
        value = path.read_text().strip()
        source = str(path)
    if not value or not value.isprintable():
        raise ValueError("Isaac Sim version must be a nonempty printable line")
    return {"isaacsim": value, "isaacsim_version_source": source}


def runtime_info() -> BenchmarkRuntimeInfo:
    """Report checked native code and the actual simulator/runtime platform."""
    verify_native_source()
    commit = native_commit()
    import torch

    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    root = Path(__file__).resolve().parents[2]
    digest = benchmark_source_digest(root, "robodojo")
    return BenchmarkRuntimeInfo(
        benchmark="robodojo",
        benchmark_version="RoboDojo-" + commit[:12],
        benchmark_commit=commit,
        adapter_version="roborsi-source-sha256:" + digest,
        evaluation=EVALUATION_RULES,
        platform={
            "system": platform.system(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu": cpu,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            **_isaacsim_version(),
            "gpus": ", ".join(
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            )
            if torch.cuda.is_available()
            else "none",
        },
    )


def read_catalog(path: Path) -> dict:
    """Reject duplicate identities and catalogs exported for another native build."""
    document = json.loads(path.read_text())
    if (
        document.get("schema_version") != 1
        or document.get("benchmark_commit") != native_commit()
    ):
        raise ValueError("unsupported RoboDojo catalog version")
    Draft202012Validator({"type": "string", "pattern": "^[0-9a-f]{64}$"}).validate(
        document["assets_sha256"]
    )
    tasks = [Task.from_dict(item) for item in document["tasks"]]
    identities = {(task.task_id, task.seed) for task in tasks}
    if not tasks or len(identities) != len(tasks):
        raise ValueError("RoboDojo catalog needs unique task/layout pairs")
    for task in tasks:
        if (
            task.benchmark != "robodojo"
            or task.capabilities.get("action_schema") != ACTION_SCHEMA
        ):
            raise ValueError("RoboDojo catalog has an incompatible task")
        native = task.native
        Draft202012Validator(
            {
                "type": "object",
                "required": [
                    "config",
                    "eval_seed",
                    "native_step_limit",
                    "layout_sha256",
                ],
                "properties": {
                    "config": {"const": "arx_x5"},
                    "eval_seed": {"type": "integer", "minimum": 0},
                    "native_step_limit": {"type": "integer", "minimum": task.max_steps},
                    "layout_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            }
        ).validate(native)
        cameras = task.capabilities.get("observation_channels")
        if (
            not isinstance(cameras, list)
            or not cameras
            or any(not isinstance(c, str) or not c for c in cameras)
            or cameras != sorted(set(cameras))
        ):
            raise ValueError("RoboDojo catalog needs unique sorted camera names")
    if len({t.native["eval_seed"] for t in tasks}) != 1:
        raise ValueError(
            "one RoboDojo catalog must use a single native evaluation seed"
        )
    return document


class PublishedAdapter(RoboDojoRoboRSIAdapter):
    def __init__(
        self,
        directory: Path,
        publisher: ArtifactPublisher,
        *,
        tasks: list[Task],
        **kwargs,
    ) -> None:
        super().__init__(directory, **kwargs)
        self.publisher = publisher
        self.tasks = tasks

    def start(self, task: Task, episode_id: str) -> StartResult:
        if task not in self.tasks:
            raise ValueError("RoboDojo task differs from the exported catalog")
        return super().start(task, episode_id)

    def finish(self) -> NativeResult:
        result = super().finish()
        for path in sorted((self.directory / "native").rglob("*.mp4")):
            self.publisher.publish(
                path.relative_to(self.directory).as_posix(), "file", "video/mp4"
            )
        return result


class RoboDojoBenchmark:
    """Serve only cases previously exported from the pinned native runtime."""

    supports_episode_stop = True
    supports_benchmark_queries = True

    name = "robodojo"

    def runtime_info(self) -> BenchmarkRuntimeInfo:
        return runtime_info()

    def tasks(self) -> list[Task]:
        return [Task.from_dict(item) for item in read_catalog(CATALOG_PATH)["tasks"]]

    def assets(self) -> list[dict]:
        catalog = read_catalog(CATALOG_PATH)
        digest = directory_sha256(ASSETS_ROOT)
        if digest != catalog["assets_sha256"]:
            raise ValueError("RoboDojo assets differ from the exported catalog")
        return [
            {"name": "robodojo", "kind": "directory", "sha256": digest},
            {
                "name": "robodojo-catalog",
                "kind": "file",
                "sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
            },
        ]

    def create_adapter(
        self, directory: Path, artifacts: ArtifactPublisher
    ) -> PublishedAdapter:
        return PublishedAdapter(
            directory,
            artifacts,
            tasks=self.tasks(),
            operation_timeout_s=float(
                get_project_env("ROBORSI_ROBODOJO_OPERATION_TIMEOUT_S", "900")
            ),
        )


def make_robodojo_benchmark() -> RoboDojoBenchmark:
    return RoboDojoBenchmark()

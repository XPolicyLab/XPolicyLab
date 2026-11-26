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

"""Seal assets, inspect native scenes, and export a RoboRSI RoboDojo manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

from roborsi.benchmarks.registry import (
    benchmark_manifest_schema,
    directory_sha256,
    snapshot_directory,
)
from roborsi.evaluation.messages import BenchmarkRuntimeInfo, Task
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.logging import get_logger
from robots.robodojo.benchmark_service import (
    ASSETS_ROOT,
    NATIVE_ROOT,
    native_commit,
    read_catalog,
    runtime_info,
)
from robots.robodojo.roborsi_adapter import RoboDojoRoboRSIAdapter

logger = get_logger("benchmarks.robodojo.prepare")


def native_tasks(root: Path = NATIVE_ROOT) -> list[str]:
    """Discover all native task classes with matching evaluation configurations.

    Parse sources without importing simulator dependencies. A missing class or
    configuration is an error, rather than silently shrinking full coverage.
    Callers verify the native source revision before inspecting these tasks.
    """
    task_root = root / "task/RoboDojo"
    modules = {
        path.stem: path
        for path in (task_root / "tasks").glob("*.py")
        if not path.name.startswith("_")
    }
    configs = {
        path.stem
        for path in (task_root / "config").glob("*.yml")
        if not path.name.startswith("_")
    }
    if not modules or set(modules) != configs:
        raise ValueError("native task modules and configurations must match")
    for name, path in modules.items():
        classes = {
            node.name
            for node in ast.parse(path.read_text(), filename=str(path)).body
            if isinstance(node, ast.ClassDef)
        }
        if name not in classes:
            raise ValueError(f"native task is missing its exported class: {name}")
    return sorted(modules)


def build_manifest(
    image: str,
    asset: dict,
    catalog: dict,
    catalog_asset: dict,
    runtime: BenchmarkRuntimeInfo,
) -> dict:
    """Keep seed-specific native layout hashes and instructions in the manifest."""
    repository, separator, digest = image.partition("@")
    if not separator or runtime.benchmark != "robodojo":
        raise ValueError("RoboDojo needs repository@sha256:digest and its own runtime")
    tasks = [Task.from_dict(item) for item in catalog["tasks"]]
    groups = defaultdict(list)
    for task in tasks:
        groups[task.task_id].append(task)
    if not tasks or any(task.capabilities != tasks[0].capabilities for task in tasks):
        raise ValueError(
            "one local manifest needs a nonempty catalog with matching capabilities"
        )
    if any(len({t.max_steps for t in group}) != 1 for group in groups.values()):
        raise ValueError(
            "one native task must use the same action limit for all layouts"
        )
    document = {
        "schema_version": 1,
        "benchmark": "robodojo",
        "benchmark_version": runtime.benchmark_version,
        "benchmark_commit": runtime.benchmark_commit,
        "adapter_version": runtime.adapter_version,
        "image_repository": repository,
        "image_digest": digest,
        "assets_version": catalog["assets_sha256"],
        "assets": [asset, catalog_asset],
        "adapter_command": [
            "python",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "benchmark",
            "--factory",
            "robots.robodojo.benchmark_service:make_robodojo_benchmark",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "service_port": 8000,
        "resources": {
            "gpus": 1,
            "cpus": 4,
            "memory_mb": 32768,
            "startup_timeout_s": 180,
        },
        "capabilities": tasks[0].capabilities,
        "evaluation": runtime.evaluation,
        "tasks": [
            {
                "task_id": name,
                "seeds": [t.seed for t in group],
                "instruction": group[0].instruction,
                "instructions_by_seed": {str(t.seed): t.instruction for t in group},
                "native": group[0].native,
                "native_by_seed": {str(t.seed): t.native for t in group},
                "max_steps": group[0].max_steps,
            }
            for name, group in groups.items()
        ],
    }
    Draft202012Validator(benchmark_manifest_schema()).validate(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("assets")
    seal.add_argument("--source", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    catalog = commands.add_parser("catalog")
    selection = catalog.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", action="append")
    selection.add_argument(
        "--all-tasks", action="store_true", help="Inspect every native task and variant"
    )
    catalog.add_argument("--layout", type=int, action="append", required=True)
    catalog.add_argument("--eval-seed", type=int, default=0)
    catalog.add_argument("--max-steps", type=int)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument(
        "--work-dir",
        type=Path,
        help="New directory retaining native inspection logs; failures always retain logs",
    )
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--image", required=True)
    manifest.add_argument("--asset", type=Path, required=True)
    manifest.add_argument("--catalog", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assets":
        source = args.source.expanduser().resolve()
        for name in ("Robots", "Object", "Material", "Eval_Layout"):
            if not (source / name).is_dir():
                parser.error(f"RoboDojo assets are missing {name}")
        root = args.output_dir.resolve()
        snapshot = snapshot_directory(source, root / "snapshots")
        write_json_atomic(
            root / "asset.json",
            {
                "name": "robodojo",
                "kind": "directory",
                "version": snapshot.name,
                "path": snapshot.relative_to(root).as_posix(),
                "sha256": snapshot.name,
            },
        )
    elif args.command == "catalog":
        if args.all_tasks:
            runtime_info()  # Verify the tree before deriving the full task selection.
            args.task = native_tasks()
        if (
            len(set(args.task)) != len(args.task)
            or len(set(args.layout)) != len(args.layout)
            or min(args.layout) < 0
            or args.eval_seed < 0
        ):
            parser.error("provide unique tasks and nonnegative layout/evaluation seeds")
        runtime_info()  # Refuse another native source tree before initializing scenes.
        digest = directory_sha256(ASSETS_ROOT)
        workspace = (
            args.work_dir.expanduser().resolve()
            if args.work_dir is not None
            else Path(tempfile.mkdtemp(prefix="roborsi-dojo-catalog-"))
        )
        if args.work_dir is not None:
            workspace.mkdir(parents=True, exist_ok=False)
        tasks = []
        try:
            for task in args.task:
                for layout in args.layout:
                    directory = workspace / f"case-{len(tasks):04d}"
                    adapter = RoboDojoRoboRSIAdapter(directory)
                    try:
                        exported = adapter.inspect(
                            {
                                "task_id": task,
                                "layout_id": layout,
                                "eval_seed": args.eval_seed,
                                "max_steps": args.max_steps,
                            }
                        )
                        tasks.append(exported.as_dict())
                    finally:
                        adapter.close()
            if directory_sha256(ASSETS_ROOT) != digest:
                raise ValueError("RoboDojo assets changed during catalog export")
            write_json_atomic(
                args.output,
                {
                    "schema_version": 1,
                    "benchmark_commit": native_commit(),
                    "assets_sha256": digest,
                    "tasks": tasks,
                },
            )
        except BaseException:
            logger.exception(
                "RoboDojo catalog failed; inspection logs retained at %s", workspace
            )
            raise
        else:
            if args.work_dir is None:
                shutil.rmtree(workspace)
            else:
                logger.info("RoboDojo inspection logs retained at %s", workspace)
    else:
        data = read_catalog(args.catalog)
        asset = json.loads(args.asset.read_text())
        if data["assets_sha256"] != asset["sha256"]:
            raise ValueError("catalog and asset snapshot differ")
        catalog_asset = {
            "name": "robodojo-catalog",
            "kind": "file",
            "version": "1",
            "path": args.catalog.resolve()
            .relative_to(args.output.resolve().parent)
            .as_posix(),
            "sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        }
        write_json_atomic(
            args.output,
            build_manifest(args.image, asset, data, catalog_asset, runtime_info()),
        )


if __name__ == "__main__":
    main()

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

"""Authenticate original resources and prepare a new, independent ICL episode.

The vendored controller is unchanged. This module only relocates deployment
paths and creates its original preparation-import contract, without adopting
any author's evaluation state.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DeploymentPaths:
    """Explicit host paths and service placement; no policy overrides.

    Attributes:
        benchmark_python: Executable benchmark interpreter or wrapper.
        agent_python: Original agent runtime interpreter.
        codex_binary: Pinned standalone Codex executable.
        environment_dir: Private native RoboDojo checkout.
        gpu: Simulator's visible GPU ordinal.
        physical_gpu: Host GPU identity recorded by the original deployment.
        vla_port: Existing pi0.5 service port.
    """

    benchmark_python: Path
    agent_python: Path
    codex_binary: Path
    environment_dir: Path
    gpu: int = 0
    physical_gpu: int = 0
    vla_port: int = 19802


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source() -> dict[str, Any]:
    """Check every copied source/interface/template against captured SHA pins.

    Returns:
        Copied-file provenance whose bytes have been verified.
    """
    provenance = _read(ROOT / "provenance/source-files.json")
    for row in provenance["files"]:
        path = ROOT / row["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked ICL source: {row['path']}")
        if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            raise ValueError(f"Original ICL source differs: {row['path']}")
    return provenance


def _verify_bundle(bundle: Path, rows: list[dict[str, Any]]) -> None:
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Frozen bundle must be a materialized directory")
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("Frozen bundle contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
        elif not path.is_dir():
            raise ValueError("Frozen bundle contains a non-file resource")
    if actual != {row["path"] for row in rows}:
        raise ValueError("Frozen bundle inventory differs from authenticated input")
    for row in rows:
        path = bundle / row["path"]
        if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            raise ValueError(f"Frozen material differs: {row['path']}")


def generate_fresh_config(
    instance: str,
    deployment: DeploymentPaths,
    frozen_bundle: Path,
    output: Path,
) -> Path:
    """Prepare one original fresh config and its frozen preparation import.

    Args:
        instance: Original task identity, e.g. ``solve_equation-s1``.
        deployment: Host interpreter, binary, native environment and service paths.
        frozen_bundle: Complete selected frozen bundle, not author evaluation data.
        output: New case preparation directory. It must not already exist.

    Returns:
        Path to config.json consumed by the unchanged original controller.

    Raises:
        ValueError: Unknown task, changed sources/materials or non-fresh output.
    """
    verify_source()
    resources = _read(ROOT / "resources/frozen-bundles.json")
    mappings = {
        row["task_instance"]: row["bundle"] for row in resources["task_mappings"]
    }
    if instance not in mappings:
        raise ValueError(f"Unknown original task instance: {instance}")
    bundle_name = mappings[instance]
    _verify_bundle(frozen_bundle, resources["bundles"][bundle_name])
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh preparation output must not already exist")
    for protected in (
        frozen_bundle.resolve(),
        deployment.environment_dir.resolve(),
        ROOT,
    ):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(
                "Fresh output must be separate from original input resources"
            )
    for name in ("benchmark_python", "agent_python", "codex_binary"):
        if not getattr(deployment, name).is_file():
            raise ValueError(f"Missing deployment executable: {name}")
    if not deployment.environment_dir.is_dir():
        raise ValueError("Missing native environment directory")
    if any(
        type(value) is not int or value < 0
        for value in (deployment.gpu, deployment.physical_gpu)
    ):
        raise ValueError("Deployment GPU ordinals must be non-negative integers")
    if type(deployment.vla_port) is not int or not 1 <= deployment.vla_port <= 65535:
        raise ValueError("Invalid pi0.5 service port")
    cfg = _read(ROOT / "resources/fresh-configs" / f"{instance}.json")
    if cfg["episode_seconds"] != 86400 or "replay" in cfg["options"]:
        raise ValueError("Original preparation template is not a fresh baseline")
    if not cfg["options"]["evaluation_independent"]:
        raise ValueError("Original preparation template is not independent")
    # The authenticated preparation configs are first-run oracles, never the
    # author's later config/ recovery snapshots. local_integration is metadata.
    metadata = cfg.pop("local_integration", None)
    cfg["local_integration"] = {}
    cfg.pop("continue_from_run", None)
    cfg["options"].pop("replay_fresh_session", None)
    for name in ("benchmark_python", "agent_python", "codex_binary", "environment_dir"):
        cfg[name] = str(getattr(deployment, name).absolute())
    cfg["codex_home"] = ""
    cfg["adapter"] = f"{ROOT / 'source/hybrid_adapter.py'}:Adapter"
    cfg["public_instructions"] = str(ROOT / "resources/interfaces" / f"{instance}.md")
    cfg["training_set"] = str(frozen_bundle.resolve())
    cfg["agent_workspace"] = str(output / "workspace")
    cfg["continue_from_run"] = str(output / "import")
    for name in ("gpu", "physical_gpu", "vla_port"):
        cfg["options"][name] = getattr(deployment, name)
    output.mkdir(parents=True)
    imported = output / "import"
    imported.mkdir()
    shutil.copytree(frozen_bundle, imported / "prepared_workspace", symlinks=False)
    # Verify destination bytes as well, before any model or simulator starts.
    _verify_bundle(imported / "prepared_workspace", resources["bundles"][bundle_name])
    text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
    (output / "config.json").write_text(text, encoding="utf-8")
    (imported / "config.json").write_text(text, encoding="utf-8")
    receipt = {
        "instance": instance,
        "bundle": bundle_name,
        "import_only": True,
        "no_exploration": True,
        "fresh_episode": True,
        "source_template_sha256": _sha(
            ROOT / "resources/fresh-configs" / f"{instance}.json"
        ),
        "material_manifest_sha256": resources["source_manifest_sha256"],
        "original_preparation_metadata": metadata,
        "relocated_fields": [
            "benchmark_python",
            "agent_python",
            "codex_binary",
            "environment_dir",
            "adapter",
            "public_instructions",
            "training_set",
            "agent_workspace",
            "continue_from_run",
            "options.gpu",
            "options.physical_gpu",
            "options.vla_port",
        ],
        "evaluation_state_imported": False,
    }
    (output / "PREPARATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return output / "config.json"

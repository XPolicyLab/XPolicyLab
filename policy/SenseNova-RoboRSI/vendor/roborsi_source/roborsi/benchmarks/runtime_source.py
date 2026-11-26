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

"""Explicit server source boundary shared by packaging and Adapter versions."""

from __future__ import annotations

import hashlib
from importlib.metadata import distributions
from pathlib import Path

_COMMON = (
    "roborsi/__init__.py",
    "roborsi/benchmarks/__init__.py",
    "roborsi/benchmarks/registry.py",
    "roborsi/benchmarks/runtime_source.py",
    "roborsi/evaluation/__init__.py",
    "roborsi/evaluation/adapter_service.py",
    "roborsi/evaluation/benchmark_http.py",
    "roborsi/evaluation/http_artifacts.py",
    "roborsi/evaluation/media.py",
    "roborsi/evaluation/messages.py",
    "roborsi/evaluation/protocol.py",
    "roborsi/evaluation/result.py",
    "roborsi/evaluation/service_host.py",
    "roborsi/evaluation/service_protocol.py",
    "roborsi/schemas/__init__.py",
    "roborsi/schemas/roborsi_benchmark_manifest.v1.json",
    "roborsi/schemas/roborsi_interaction_envelope.v1.json",
    "roborsi/utils/__init__.py",
    "roborsi/utils/config.py",
    "roborsi/utils/daemon.py",
    "roborsi/utils/logging.py",
    "roborsi/utils/rwlock.py",
    "roborsi/utils/serialization.py",
    "roborsi/utils/templates.py",
    "roborsi/utils/rpc/__init__.py",
    "roborsi/utils/rpc/client_utils.py",
    "roborsi/utils/rpc/http_abort.py",
    "roborsi/utils/rpc/http_rpc.py",
    "roborsi/utils/rpc/main_thread_serve.py",
    "roborsi/utils/rpc/rpc_client.py",
    "roborsi/utils/rpc/rpc_facade.py",
    "roborsi/utils/rpc/socket_rpc.py",
    "robots/__init__.py",
)
_ENV_FACADE = (
    "roborsi/robots/__init__.py",
    "roborsi/robots/components/__init__.py",
    "roborsi/robots/components/env_facade_base.py",
)
_BENCHMARK = {
    "libero": (
        *_ENV_FACADE,
        "roborsi/utils/egl.py",
        "robots/libero/__init__.py",
        "robots/libero/benchmark_service.py",
        "robots/libero/roborsi_adapter.py",
        "robots/libero/env_server.py",
        "benchmarks/libero/manifest.py",
        "benchmarks/libero/download_assets.py",
    ),
    "libero_pro": (
        *_ENV_FACADE,
        "roborsi/utils/egl.py",
        "robots/libero/__init__.py",
        "robots/libero/benchmark_catalog.py",
        "robots/libero/three_benchmark_adapter.py",
        "robots/libero/three_benchmark_env.py",
        "robots/libero/three_benchmark_service.py",
        "robots/libero/three_benchmark_serve.py",
        "robots/libero/env_server.py",
        "robots/libero/lilo_task.py",
        "robots/libero/lilo_assistance.py",
        "benchmarks/libero_pro/manifest.py",
    ),
    "libero_lilo": (
        "robots/libero/__init__.py",
        "robots/libero/benchmark_catalog.py",
        "robots/libero/three_benchmark_adapter.py",
        "robots/libero/three_benchmark_env.py",
        "robots/libero/three_benchmark_service.py",
        "robots/libero/three_benchmark_serve.py",
        "robots/libero/lilo_task.py",
        "robots/libero/lilo_assistance.py",
        "benchmarks/libero_lilo/manifest.py",
    ),
    "robotwin": (
        *_ENV_FACADE,
        "robots/robotwin/__init__.py",
        "robots/robotwin/benchmark_service.py",
        "robots/robotwin/roborsi_adapter.py",
        "robots/robotwin/env_contract.py",
        "robots/robotwin/env_server.py",
        "robots/robotwin/rlinf_env.py",
        "robots/robotwin/reward_compat.py",
        "robots/robotwin/eval/demo_randomized.json",
        "benchmarks/robotwin/manifest.py",
        "benchmarks/robotwin/assets.py",
    ),
    "robodojo": (
        "robots/robodojo/benchmark_service.py",
        "robots/robodojo/roborsi_adapter.py",
        "robots/robodojo/worker.py",
        "robots/robodojo/episode.py",
        "robots/robodojo/native.py",
        "robots/robodojo/native_files.json",
        "robots/robodojo/eef_contracts.py",
        "robots/robodojo/eef_queries.py",
        "robots/robodojo/eef_native.py",
        "robots/robodojo/eef_types.py",
        "robots/robodojo/eef_trace.py",
        "robots/robodojo/eef_docs.py",
        "robots/robodojo/eef_pose.py",
        "robots/robodojo/joint_docs.py",
        "benchmarks/robodojo/prepare.py",
    ),
}


def benchmark_source_files(benchmark: str) -> tuple[str, ...]:
    """List only protocol, HTTP, artifact and selected native wrapper sources."""
    if benchmark not in _BENCHMARK:
        raise ValueError(f"unsupported Benchmark source bundle: {benchmark}")
    return tuple(sorted((*_COMMON, *_BENCHMARK[benchmark])))


def benchmark_source_digest(root: Path, benchmark: str) -> str:
    """Hash the same named files in a checkout and in its standalone image."""
    digest = hashlib.sha256(b"RoboRSI-benchmark-source-v1\0")
    for name in benchmark_source_files(benchmark):
        path = root / name
        if (
            any(parent.is_symlink() for parent in (path, *path.parents))
            or not path.is_file()
        ):
            raise ValueError(f"Benchmark source is missing or linked: {name}")
        digest.update(
            name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
        )
    return digest.hexdigest()


def verify_benchmark_runtime(root: Path, benchmark: str) -> None:
    """Reject Agent packages inherited from a runtime base or stale source tree."""
    benchmark_source_digest(root, benchmark)
    allowed = set(benchmark_source_files(benchmark))
    for directory in ("roborsi", "heart_runtime", "rpent", "robots", "benchmarks"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.relative_to(root).as_posix() not in allowed:
                raise ValueError(f"undeclared Benchmark image source: {path}")
    forbidden = {
        "roborsi",
        "heart-runtime",
        "heart",
        "rpent",  # Reject the previously published Agent distribution as well.
        "pydantic-ai",
        "pydantic-ai-slim",
        "openai",
        "anthropic",
        "claude-agent-sdk",
        "openai-codex",
        "openai-codex-cli-bin",
        "mcp",
    }
    installed = {
        package.metadata["Name"].lower().replace("_", "-")
        for package in distributions()
    }
    if overlap := forbidden & installed:
        raise ValueError(
            "Benchmark base must exclude Agent packages: " + ", ".join(sorted(overlap))
        )

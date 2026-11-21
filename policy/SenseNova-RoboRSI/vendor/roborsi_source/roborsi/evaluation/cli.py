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

"""Command-line entry point for RoboRSI batch evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.agent_process import PythonAgentLauncher
from roborsi.evaluation.batch import BatchEvaluator, expand_cases
from roborsi.evaluation.docker_agent import DockerAgentLauncher
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.http_agent import HttpAgentLauncher
from roborsi.evaluation.http_benchmark import (
    HttpBenchmarkExecutor,
    HttpBenchmarkRegistry,
)
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.store import ComparisonPolicy, ResultStore
from roborsi.utils.rpc import RpcError


def _preset_launcher(
    config: str,
    args: argparse.Namespace,
    overrides: list[str] | tuple[str, ...],
    *,
    image: str | None = None,
    python: str | None = None,
) -> PythonAgentLauncher | DockerAgentLauncher:
    """Build a preset launcher without starting an Agent or Benchmark."""
    from roborsi.agents.config import bundle_package

    image = image or args.agent_image
    python = python or args.agent_python
    if image:
        if python:
            raise ValueError("--agent-image cannot use --agent-python")
        return DockerAgentLauncher(
            image,
            f"{bundle_package(config)}:make_agent",
            agent_config=config,
            agent_overrides=overrides,
            cpus=args.agent_cpus if args.agent_cpus is not None else 1.0,
            memory_mb=args.agent_memory_mb
            if args.agent_memory_mb is not None
            else 2048,
            credential_env=args.agent_credential_env,
        )
    return PythonAgentLauncher(
        agent_config=config,
        agent_overrides=overrides,
        python_executable=python or sys.executable,
    )


def main(argv: list[str] | None = None) -> int:
    """Run benchmark cases or query their state, emitting JSON on stdout."""
    parser = argparse.ArgumentParser(
        prog="bench", description="RoboRSI benchmark runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    from roborsi.evaluation import rsi_cli

    rsi_cli.add_parser(commands)
    compare = commands.add_parser("compare", help="compare two saved cases")
    compare.add_argument("--results-root", type=Path, required=True)
    compare.add_argument("--left-run", required=True)
    compare.add_argument("--left-case", required=True)
    compare.add_argument("--right-run", required=True)
    compare.add_argument("--right-case", required=True)
    compare.add_argument("--comparison-policy", type=Path, help="JSON ComparisonPolicy")
    status = commands.add_parser(
        "status", help="query case state, including failed result writes"
    )
    status.add_argument("--results-root", type=Path, required=True)
    status.add_argument("--status-root", type=Path)
    status.add_argument("--run-id", required=True)
    status.add_argument("--case-id", help="omit to list all cases in this run")
    evaluate = commands.add_parser("eval", help="run a fresh benchmark matrix")
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry-root", type=Path)
    evaluate.add_argument(
        "--sco-config",
        type=Path,
        help="SCO JSON deployment configuration; requires --registry-root",
    )
    source.add_argument(
        "--benchmark-url",
        action="append",
        help="name=HTTP(S) origin; repeat for each remote Benchmark",
    )
    evaluate.add_argument(
        "--benchmark-token-env",
        help="environment variable holding the remote Benchmark Bearer token",
    )
    evaluate.add_argument("--results-root", type=Path, required=True)
    evaluate.add_argument(
        "--status-root",
        type=Path,
        help="separate directory for case state and storage failures",
    )
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument(
        "--resume-from", help="previous batch run ID; requires a new --run-id"
    )
    agent_source = evaluate.add_mutually_exclusive_group(required=True)
    agent_source.add_argument(
        "--agent-config", help="Agent YAML/JSON path or installed bundle:name"
    )
    agent_source.add_argument(
        "--agent-matrix",
        type=Path,
        help="named Agent presets and per-Agent overrides in YAML/JSON",
    )
    evaluate.add_argument(
        "--agent-set",
        action="append",
        default=[],
        help="PATH=JSON preset override; repeat as needed",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the complete plan without starting services",
    )
    agent_source.add_argument("--agent-factory", help="module:function")
    agent_source.add_argument(
        "--agent-url", help="external RoboRSI HTTP(S) service origin"
    )
    agent_source.add_argument(
        "--agent-artifact-store",
        type=Path,
        help="saved Agent source and environment versions",
    )
    evaluate.add_argument(
        "--agent-python", help="Python executable for a factory or saved Agent version"
    )
    evaluate.add_argument(
        "--agent-image", help="run the factory inside repository@sha256:digest"
    )
    evaluate.add_argument(
        "--agent-cpus", type=float, help="Docker Agent CPU limit (default: 1)"
    )
    evaluate.add_argument(
        "--agent-memory-mb",
        type=int,
        help="Docker Agent memory limit (default: 2048 MiB)",
    )
    evaluate.add_argument(
        "--agent-credential-env",
        action="append",
        default=[],
        help="credential variable to pass to a locked Agent; repeat for each name",
    )
    evaluate.add_argument(
        "--agent-version",
        help="required unless derived from --agent-image or --agent-config",
    )
    evaluate.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="name or comma-separated names",
    )
    evaluate.add_argument(
        "--task", action="append", help="benchmark:task_id; repeat to select tasks"
    )
    evaluate.add_argument("--seed", type=int, action="append")
    evaluate.add_argument("--max-concurrent", type=int, default=1)
    evaluate.add_argument("--timeout-s", type=float, default=30.0)
    evaluate.add_argument("--case-timeout-s", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if args.command == "rsi":
        return rsi_cli.execute(args)
    if args.command == "compare":
        try:
            policy = (
                ComparisonPolicy.from_dict(
                    json.loads(args.comparison_policy.read_text())
                )
                if args.comparison_policy
                else ComparisonPolicy()
            )
            report = ResultStore(args.results_root).compare(
                args.left_run,
                args.left_case,
                args.right_run,
                args.right_case,
                policy=policy,
            )
        except (ValueError, OSError, TypeError) as exc:
            compare.error(str(exc))
        print(json.dumps(report.as_dict(), allow_nan=False, ensure_ascii=False))
        return 0
    if args.command == "status":
        store = ResultStore(args.results_root, status_root=args.status_root)
        try:
            report = (
                store.status(args.run_id, args.case_id).as_dict()
                if args.case_id
                else {
                    case_id: store.status(args.run_id, case_id).as_dict()
                    for case_id in store.list_cases(args.run_id)
                }
            )
            if not args.case_id:
                try:
                    plan = store.load_batch_plan(args.run_id)
                except FileNotFoundError:
                    plan = {}
                if plan.get("kind") == "agent_matrix":
                    from roborsi.evaluation.matrix import matrix_status

                    report = matrix_status(store, args.run_id)
        except (ValueError, OSError) as exc:
            status.error(str(exc))
        print(json.dumps(report, allow_nan=False, ensure_ascii=False))
        return 0
    if args.sco_config and (not args.registry_root or args.agent_image):
        evaluate.error(
            "--sco-config requires --registry-root and a Python or HTTP Agent"
        )
    if args.agent_image and (
        not (args.agent_factory or args.agent_config or args.agent_matrix)
        or args.agent_python
    ):
        evaluate.error(
            "--agent-image requires --agent-factory or --agent-config bundle:name and cannot use --agent-python"
        )
    if (
        not args.agent_image
        and not args.agent_matrix
        and (args.agent_cpus is not None or args.agent_memory_mb is not None)
    ):
        evaluate.error("Agent resource limits require --agent-image")
    if (
        not args.agent_image
        and not args.agent_config
        and not args.agent_matrix
        and not args.agent_version
    ):
        evaluate.error(
            "--agent-version is required without --agent-image or --agent-config"
        )
    if args.agent_url and args.agent_python:
        evaluate.error(
            "--agent-python requires --agent-factory or --agent-artifact-store"
        )
    if args.agent_credential_env and not (
        args.agent_artifact_store or args.agent_image or args.agent_matrix
    ):
        evaluate.error(
            "--agent-credential-env requires --agent-artifact-store or --agent-image"
        )
    if args.agent_set and not (args.agent_config or args.agent_matrix):
        evaluate.error("--agent-set requires --agent-config or --agent-matrix")
    if args.agent_matrix and args.agent_version:
        evaluate.error(
            "matrix Agent versions are derived separately; omit --agent-version"
        )
    configured_launcher = None
    matrix_launchers = []
    try:
        if args.agent_config:
            configured_launcher = _preset_launcher(
                args.agent_config, args, args.agent_set
            )
        if args.agent_matrix:
            from roborsi.evaluation.matrix import load_matrix

            matrix_launchers = [
                (
                    entry,
                    _preset_launcher(
                        entry.config,
                        args,
                        (*entry.overrides, *args.agent_set),
                        image=entry.image,
                        python=entry.python,
                    ),
                )
                for entry in load_matrix(args.agent_matrix)
            ]
        has_container = bool(args.agent_image) or any(
            isinstance(launcher, DockerAgentLauncher)
            for _, launcher in matrix_launchers
        )
        if (
            args.agent_matrix
            and not has_container
            and (
                args.agent_cpus is not None
                or args.agent_memory_mb is not None
                or args.agent_credential_env
            )
        ):
            raise ValueError(
                "matrix Agent resource and credential options require an Agent image"
            )
        if args.sco_config and has_container:
            raise ValueError("--sco-config cannot use Docker Agents")
    except (ValueError, OSError, ImportError) as exc:
        evaluate.error(str(exc))
    benchmarks = tuple(
        name.strip() for group in args.benchmark for name in group.split(",")
    )
    tasks: dict[str, list[str]] = defaultdict(list)
    for selection in args.task or ():
        benchmark, separator, task_id = selection.partition(":")
        if not separator or not benchmark or not task_id:
            evaluate.error("--task must be benchmark:task_id")
        tasks[benchmark].append(task_id)
    try:
        if args.benchmark_url:
            endpoints = {}
            for item in args.benchmark_url:
                name, separator, url = item.partition("=")
                if not separator or not name or name in endpoints:
                    evaluate.error("--benchmark-url must be a unique name=URL")
                endpoints[name] = url
            if args.max_concurrent != 1:
                evaluate.error(
                    "remote service instances currently require --max-concurrent 1"
                )
            token = (
                os.environ.get(args.benchmark_token_env)
                if args.benchmark_token_env
                else None
            )
            if args.benchmark_token_env and not token:
                evaluate.error("Benchmark token variable is not set")
            if has_container and not token:
                evaluate.error(
                    "Docker Agents require authenticated remote Benchmarks (--benchmark-token-env)"
                )
            registry = HttpBenchmarkRegistry(
                endpoints, token=token, timeout_s=args.timeout_s
            )
            executor = HttpBenchmarkExecutor()
        else:
            if args.benchmark_token_env:
                evaluate.error("--benchmark-token-env requires --benchmark-url")
            if args.sco_config:
                from roborsi.evaluation.sco_jobs import ScoConfig
                from roborsi.evaluation.sco_runtime import (
                    ScoBenchmarkRegistry,
                    ScoExecutor,
                )

                registry = ScoBenchmarkRegistry([args.registry_root])
                executor = ScoExecutor(ScoConfig.from_file(args.sco_config))
            else:
                registry = BenchmarkRegistry([args.registry_root])
                executor = DockerExecutor()
        cells = expand_cases(
            registry, benchmarks, task_ids=tasks or None, seeds=args.seed
        )
        store = ResultStore(args.results_root, status_root=args.status_root)
        if matrix_launchers:
            from roborsi.evaluation.matrix import MatrixAgent, run_matrix

            matrix_agents = []
            for entry, selected_launcher in matrix_launchers:
                if entry.benchmarks is not None and set(entry.benchmarks) - set(
                    benchmarks
                ):
                    raise ValueError(
                        f"Agent {entry.name} selects an unrequested Benchmark"
                    )
                selected_cases = tuple(
                    cell
                    for cell in cells
                    if entry.benchmarks is None or cell.benchmark in entry.benchmarks
                )
                evaluator = BatchEvaluator(
                    registry,
                    store,
                    CaseOrchestrator(registry, store, selected_launcher, executor),
                )
                matrix_agents.append(
                    MatrixAgent(
                        entry.name,
                        selected_launcher.agent_version,
                        evaluator,
                        selected_cases,
                    )
                )
            document = run_matrix(
                store,
                args.run_id,
                matrix_agents,
                max_concurrent=args.max_concurrent,
                timeout_s=args.timeout_s,
                case_timeout_s=args.case_timeout_s,
                resume_from=args.resume_from,
                dry_run=args.dry_run,
            )
        else:
            if configured_launcher is not None:
                launcher = configured_launcher
            elif args.agent_image:
                launcher = DockerAgentLauncher(
                    args.agent_image,
                    args.agent_factory,
                    cpus=args.agent_cpus if args.agent_cpus is not None else 1.0,
                    memory_mb=args.agent_memory_mb
                    if args.agent_memory_mb is not None
                    else 2048,
                    credential_env=args.agent_credential_env,
                )
            elif args.agent_artifact_store:
                launcher = (
                    AgentArtifactStore(args.agent_artifact_store)
                    .load(args.agent_version)
                    .launcher(
                        python_executable=args.agent_python or sys.executable,
                        credential_env=args.agent_credential_env,
                    )
                )
            elif args.agent_url:
                launcher = HttpAgentLauncher(args.agent_url)
            else:
                launcher = PythonAgentLauncher(
                    args.agent_factory,
                    python_executable=args.agent_python or sys.executable,
                )
            if (args.agent_image or args.agent_config) and args.agent_version is None:
                args.agent_version = launcher.agent_version
            orchestrator = CaseOrchestrator(registry, store, launcher, executor)
            evaluator = BatchEvaluator(registry, store, orchestrator)
            report = (evaluator.preflight if args.dry_run else evaluator.run)(
                args.run_id,
                cells,
                args.agent_version,
                max_concurrent=args.max_concurrent,
                timeout_s=args.timeout_s,
                case_timeout_s=args.case_timeout_s,
                resume_from=args.resume_from,
            )
            document = report if args.dry_run else report.as_dict()
    except (ValueError, FileNotFoundError, FileExistsError, RpcError) as exc:
        evaluate.error(str(exc))
    print(json.dumps(document, allow_nan=False, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

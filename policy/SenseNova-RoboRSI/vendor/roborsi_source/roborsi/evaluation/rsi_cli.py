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

"""CLI wiring for saved-Agent RSI rounds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.agent_artifact import AgentArtifactStore
from roborsi.evaluation.batch import BatchEvaluator, expand_cases
from roborsi.evaluation.docker_runtime import DockerExecutor
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.rsi import (
    EvaluationSettings,
    NoRegressionRule,
    RSIController,
    RSIStore,
)
from roborsi.evaluation.service_host import load_factory
from roborsi.evaluation.store import ComparisonPolicy, ResultStore


def add_parser(commands: argparse._SubParsersAction) -> None:
    """Register rsi run/status without changing existing eval options."""
    parser = commands.add_parser(
        "rsi", help="evaluate, improve and retest a saved Agent"
    )
    operations = parser.add_subparsers(dest="rsi_command", required=True)
    status = operations.add_parser("status", help="read a saved improvement round")
    status.add_argument("--rsi-root", type=Path, required=True)
    status.add_argument("--run-id", required=True)
    status.set_defaults(rsi_parser=status)
    run = operations.add_parser("run", help="start one fresh improvement round")
    run.set_defaults(rsi_parser=run)
    run.add_argument("--rsi-root", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--registry-root", type=Path, required=True)
    run.add_argument(
        "--sco-config", type=Path, help="SCO JSON deployment configuration"
    )
    run.add_argument("--results-root", type=Path, required=True)
    run.add_argument("--status-root", type=Path)
    run.add_argument("--agent-artifact-store", type=Path, required=True)
    run.add_argument("--baseline-version", required=True)
    run.add_argument("--agent-python", default=sys.executable)
    run.add_argument("--agent-credential-env", action="append", default=[])
    run.add_argument(
        "--improver",
        required=True,
        help="trusted module:function returning an improver",
    )
    run.add_argument(
        "--improver-config",
        type=Path,
        help="nonsecret JSON object passed to the improver factory",
    )
    run.add_argument(
        "--comparison-policy",
        type=Path,
        help="JSON ComparisonPolicy; defaults to evaluation conditions",
    )
    run.add_argument("--min-gain", type=float, default=0.0)
    run.add_argument("--benchmark", action="append", required=True)
    run.add_argument("--task", action="append")
    run.add_argument("--seed", type=int, action="append")
    run.add_argument("--max-concurrent", type=int, default=1)
    run.add_argument("--timeout-s", type=float, default=30.0)
    run.add_argument("--case-timeout-s", type=float, default=3600.0)


def execute(args: argparse.Namespace) -> int:
    """Run or query RSI and write one finite JSON record to stdout."""
    try:
        runs = RSIStore(args.rsi_root)
        if args.rsi_command == "status":
            record = runs.status(args.run_id)
        else:
            policy = (
                ComparisonPolicy.from_dict(
                    json.loads(args.comparison_policy.read_text())
                )
                if args.comparison_policy
                else ComparisonPolicy()
            )
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
            results = ResultStore(args.results_root, status_root=args.status_root)
            artifacts = AgentArtifactStore(args.agent_artifact_store)
            settings = EvaluationSettings(
                max_concurrent=args.max_concurrent,
                timeout_s=args.timeout_s,
                case_timeout_s=args.case_timeout_s,
            )
            rule = NoRegressionRule(args.min_gain)
            tasks = defaultdict(list)
            for selection in args.task or ():
                benchmark, sep, task = selection.partition(":")
                if not sep or not benchmark or not task:
                    raise ValueError("--task must be benchmark:task_id")
                tasks[benchmark].append(task)
            benchmarks = [
                name.strip() for group in args.benchmark for name in group.split(",")
            ]
            cases = expand_cases(
                registry, benchmarks, task_ids=tasks or None, seeds=args.seed
            )
            config = (
                json.loads(args.improver_config.read_text())
                if args.improver_config
                else {}
            )
            if not isinstance(config, dict):
                raise ValueError("improver config must be a JSON object")
            improver = load_factory(args.improver)(config)

            def runner(artifact):
                launcher = artifact.launcher(
                    python_executable=args.agent_python,
                    credential_env=args.agent_credential_env,
                )
                return BatchEvaluator(
                    registry,
                    results,
                    CaseOrchestrator(registry, results, launcher, executor),
                )

            record = RSIController(artifacts, results, runs, runner).run(
                args.run_id,
                args.baseline_version,
                cases,
                improver,
                settings=settings,
                rule=rule,
                comparison_policy=policy,
            )
    except (ValueError, OSError, TypeError) as exc:
        args.rsi_parser.error(str(exc))
    print(json.dumps(record.as_dict(), allow_nan=False, ensure_ascii=False))
    return 0

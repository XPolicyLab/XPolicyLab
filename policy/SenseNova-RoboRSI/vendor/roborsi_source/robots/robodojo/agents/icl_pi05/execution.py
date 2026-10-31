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

"""Run one fresh original ICL episode, or continue this deployment's own run.

The controller is a separate process. Its public Agent boundary and the native
Benchmark boundary both use the framework HTTP contracts. This entry owns
shutdown, so cancellation can stop a blocked controller before RPC cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import threading
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.benchmark_http import BenchmarkApplication, BenchmarkHttpServer
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import EvaluationResult, Provenance, ServiceError, Task
from roborsi.evaluation.result import write_json_atomic
from roborsi.evaluation.runner import CaseRunner
from roborsi.evaluation.services import AgentClient, AgentFacade
from roborsi.evaluation.store import ResultStore
from roborsi.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer

from .bridge import IclRoboRSIAgent
from .factory import build_resume_submission, make_resumed_agent, prepare_case
from .service import IclBenchmark, _document


class _OperatorCancellation:
    """Cancel owned execution without adding another elapsed wall-time limit."""

    def __init__(self, event: threading.Event):
        self.event = event

    def check(self) -> None:
        if self.event.is_set():
            raise CaseCancelled("Owned ICL case cancelled by operator")

    def remaining(self, limit_s: float | None = None) -> float:
        self.check()
        return 0 if limit_s is None else limit_s


@contextmanager
def _operator_cancellation():
    event = threading.Event()
    previous = {}
    try:
        for name in (signal.SIGINT, signal.SIGTERM):
            previous[name] = signal.signal(name, lambda *_: event.set())
        yield _OperatorCancellation(event)
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _catalog_task(template: dict, instance: str) -> tuple[dict, Task]:
    """Select an authenticated export, keeping its actual instruction/identity."""
    _, catalog = _document(template["catalog"])
    task_id, seed = instance.rsplit("-s", 1)
    matches = [
        Task.from_dict(row)
        for row in catalog["tasks"]
        if row["task_id"] == task_id
        and row["native"].get("eval_seed") == int(seed)
        and row["seed"] == 0
    ]
    if len(matches) != 1:
        raise ValueError("Deployment catalog does not contain this original task/seed")
    task = matches[0]
    if task.capabilities.get("recoverable_action_rejections") is not True:
        raise ValueError("Native export must enable original non-moving feedback")
    # Framework transport capability only; no model/native input is rewritten.
    task = task.model_copy(
        update={
            "capabilities": {
                **task.capabilities,
                "controller_owned_wait": True,
            }
        }
    )
    return catalog, task


def _bind_service(
    template: dict, catalog: dict, task: Task, native_config: Path, directory: Path
) -> Path:
    """Bind a single own-episode config; never import author execution state."""
    output = directory / "icl/deployment"
    output.mkdir(parents=True, exist_ok=False)
    export = {**catalog, "tasks": [task.as_dict()]}
    catalog_path = output / "catalog.json"
    write_json_atomic(catalog_path, export)
    document = {
        **template,
        "catalog": {"path": str(catalog_path), "sha256": _sha(catalog_path)},
        "configs": [
            {
                "task_id": task.task_id,
                "layout_id": task.seed,
                "eval_seed": task.native["eval_seed"],
                "path": str(native_config),
                "sha256": _sha(native_config),
            }
        ],
    }
    config_path = output / "benchmark.json"
    write_json_atomic(config_path, document)
    write_json_atomic(
        output / "binding.json",
        {
            "source_catalog": template["catalog"],
            "instance_task": task.as_dict(),
            "changes": [
                "select_one_exported_task",
                "declare_controller_owned_wait",
                "bind_this_experiments_native_config",
            ],
            "author_evaluation_state_loaded": False,
        },
    )
    return config_path


def run_owned_case(
    *,
    store: ResultStore,
    run_id: str,
    case_id: str,
    episode_id: str,
    task: Task,
    provenance: Provenance,
    directory: Path,
    agent: IclRoboRSIAgent,
    benchmark: IclBenchmark,
    token: str,
    deadline: CaseDeadline | _OperatorCancellation | None = None,
) -> EvaluationResult:
    """Use original model waiting; native calls retain their 3600-second budget.

    ``deadline`` is an explicit operator cancellation/limit, not a default
    replacement for the original controller's episode budget. Both services
    are owned here; this is not a connection to somebody else's running Agent.
    """
    if not token:
        raise ValueError("ICL requires an independent private Benchmark token")
    if task.capabilities.get("controller_owned_wait") is not True:
        raise ValueError("Original controller waiting must be explicitly declared")
    if agent.directory != (directory / "icl/controller").absolute():
        raise ValueError("Owned Agent output differs from this Runner attempt")
    media = CaseMediaChannel(directory / "media")
    if agent.media.media_dir != media.media_dir:
        raise ValueError("Owned Agent and Runner need the same media directory")
    with ExitStack() as lifecycle:
        # Register each resource immediately, including partial startup paths.
        lifecycle.callback(agent.close)
        application = BenchmarkApplication(
            benchmark, directory / "icl/private-benchmark"
        )
        lifecycle.callback(application.close)
        native_server = BenchmarkHttpServer(("127.0.0.1", 0), application, token=token)
        lifecycle.callback(native_server.server_close)
        agent_server = HttpRpcServer(
            ("127.0.0.1", 0), AgentFacade(agent)._dispatch, token=token
        )
        lifecycle.callback(agent_server.server_close)
        for server in (native_server, agent_server):
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
            )
            thread.start()
            lifecycle.callback(thread.join, 2)
            lifecycle.callback(server.shutdown)
        agent_client = AgentClient(
            HttpRpcClient(
                f"http://127.0.0.1:{agent_server.server_address[1]}", token=token
            ),
            close_timeout_s=0,
        )
        native_client = BenchmarkClient(
            f"http://127.0.0.1:{native_server.server_address[1]}",
            token=token,
            close_timeout_s=0,
        )
        # Stop a live controller before any owned server/application cleanup.
        lifecycle.callback(agent.close)
        return CaseRunner(store).run(
            run_id,
            case_id,
            episode_id,
            task,
            provenance,
            agent_client,
            native_client,
            timeout_s=3600,
            agent_timeout_s=0,
            preclaimed=True,
            deadline=deadline,
            media_channel=media,
            failure_cleanup=agent.close,
            runtime_cleanup=lifecycle.close,
        )


def main(argv: list[str] | None = None) -> int:
    """Prepare and run one case. This entry does not submit cluster jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        required=True,
        help="Explicit native catalog/runtime/assets deployment template",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--agent-config", help="Fresh Agent preset")
    choice.add_argument(
        "--resume-case", type=Path, help="Only this experiment's original case"
    )
    parser.add_argument(
        "--source-run", type=Path, help="Stopped own run, required for resume"
    )
    parser.add_argument("--infrastructure-reason")
    args = parser.parse_args(argv)
    token = os.environ.get("ROBORSI_BENCHMARK_TOKEN")
    if not token:
        parser.error("Set ROBORSI_BENCHMARK_TOKEN; never put it in command arguments")
    if bool(args.source_run) != bool(args.resume_case):
        parser.error("--source-run is required only with --resume-case")
    if args.infrastructure_reason and not args.resume_case:
        parser.error("--infrastructure-reason requires --resume-case")
    template = json.loads(args.benchmark_config.absolute().read_text())
    _, runtime = _document(template["runtime"])
    store = ResultStore(args.output.absolute())
    destination = store._case_dir(args.run_id, args.case_id)
    if destination.exists() or any(
        path.is_symlink() for path in (destination, *destination.parents)
    ):
        raise ValueError(
            "Evaluation output must be a new case without linked ancestors"
        )
    episode_id = uuid.uuid4().hex
    if args.agent_config:
        preset = AgentConfig.load(args.agent_config)
        instance = preset.parameters["instance"]
        catalog, task = _catalog_task(template, instance)
        agent_version = preset.fingerprint
        submission = None
    else:
        reference = json.loads(
            (args.resume_case / "icl/runtime-reference.json").read_text()
        )
        instance = json.loads(
            (
                Path(reference["runtime"]) / "preparation/PREPARATION_RECEIPT.json"
            ).read_text()
        )["instance"]
        catalog, exported_task = _catalog_task(template, instance)
        if not args.output.absolute().is_relative_to(
            args.resume_case.absolute() / "attempts"
        ):
            raise ValueError(
                "Resume output must be below its own case's attempts directory"
            )
        if not args.output.resolve().is_relative_to(
            args.resume_case.resolve() / "attempts"
        ):
            raise ValueError(
                "Resume output resolves outside its own case's attempts directory"
            )
        old_plan = json.loads((args.resume_case / "plan.json").read_text())
        original_task = Task.from_dict(old_plan["task"])
        if original_task != exported_task:
            raise ValueError(
                "Own original case differs from the authenticated deployment export"
            )
        submission = build_resume_submission(
            args.resume_case.absolute(),
            args.source_run.absolute(),
            infrastructure_reason=args.infrastructure_reason,
        )
        if not submission["accepted"]:
            print(
                json.dumps({"resume_accepted": False, "receipt": submission["receipt"]})
            )
            return 2
        task = Task.from_dict(submission["task"])
        if (task.task_id, task.seed, task.native, task.instruction) != (
            exported_task.task_id,
            exported_task.seed,
            exported_task.native,
            exported_task.instruction,
        ):
            raise ValueError(
                "Own resume identity differs from the original deployment export"
            )
        agent_version = old_plan["provenance"]["agent_version"]
    provenance = Provenance(
        benchmark_version=runtime["benchmark_version"],
        benchmark_commit=runtime["benchmark_commit"],
        adapter_version=runtime["adapter_version"],
        assets_version=_digest(
            [
                {key: row[key] for key in ("name", "kind", "sha256")}
                for row in template["assets"]
            ]
        ),
        agent_version=agent_version,
    )
    directory = store.claim(
        args.run_id, args.case_id, task, provenance, episode_id=episode_id
    )
    agent = None
    stage = "icl.prepare"
    try:
        media_dir = directory / "media"
        if submission is None:
            request = prepare_case(directory, config=preset)
            agent = IclRoboRSIAgent(
                request=request,
                directory=directory / "icl/controller",
                media_dir=media_dir,
            )
            native_config = Path(request["config"])
        else:
            agent = make_resumed_agent(directory, submission, media_dir=media_dir)
            native_config = Path(submission["native_config"])
            write_json_atomic(directory / "icl/resume-submission.json", submission)
        stage = "icl.service.prepare"
        bound_service = _bind_service(template, catalog, task, native_config, directory)
        stage = "icl.service.verify"
        benchmark = IclBenchmark(bound_service)
        stage = "icl.run"
        with _operator_cancellation() as cancellation:
            result = run_owned_case(
                store=store,
                run_id=args.run_id,
                case_id=args.case_id,
                episode_id=episode_id,
                task=task,
                provenance=provenance,
                directory=directory,
                agent=agent,
                benchmark=benchmark,
                token=token,
                deadline=cancellation,
            )
    except Exception as error:
        cleanup_errors = []
        if agent is not None:
            try:
                agent.close()
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        result = EvaluationResult(
            task=task,
            episode_id=episode_id,
            status="infrastructure_error",
            success=None,
            score=None,
            provenance=provenance,
            error=ServiceError(
                code="infrastructure_error",
                message=str(error),
                details={"stage": stage, "cleanup_errors": cleanup_errors},
            ),
        )
        # Runner owns publication after execution begins. Startup has no native
        # outcome and is persisted as a technical interruption, never a score.
        if store.status(args.run_id, args.case_id).result_published:
            raise
        store.publish(args.run_id, args.case_id, result)
    print(
        json.dumps(
            {
                "status": result.status,
                "success": result.success,
                "score": result.score,
                "directory": str(directory),
            }
        )
    )
    return 0 if result.status in {"success", "task_failure"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

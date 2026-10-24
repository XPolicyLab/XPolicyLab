"""Use the original adaptive RoboDojo controller through an EmbodiedAgent."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import queue
import shutil
import socketserver
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import EmbodiedAgent
from roborsi.planner.base import PlannerResult
from roborsi.planner.codex_cli_config import CodexCliConfig

try:
    from .hybrid_backend import BackendClosed
    from .hybrid_launcher import HybridCancelled, HybridCodexCliPlanner
except ImportError:
    from hybrid_backend import BackendClosed
    from hybrid_launcher import HybridCancelled, HybridCodexCliPlanner


@dataclass(frozen=True)
class HybridRunContext:
    """Explicit per-episode resources; the backend owns no simulator or scorer."""

    reference: Any
    backend: Any
    cfg: dict
    episode_dir: Path
    prepared_workspace: Path
    training_set: Path
    public_instructions: Path
    agent_python: Path
    bubblewrap: Path
    bubblewrap_sha256: str
    resume: dict | None = None
    trajectory_dir: Path | None = None


def build_evaluation_prompt(reference, cfg: dict, *, own_resume: bool = False) -> str:
    """Extract the original prompt expressions, preserving their exact bytes."""
    options = cfg.get("options", {})
    if not options.get("evaluation_independent") or options.get("review_allow_reset"):
        raise ValueError(
            "Hybrid requires the reference independent no-reset evaluation"
        )
    if (cfg.get("continue_from_run") or options.get("replay")) and not own_resume:
        raise ValueError("A new official episode cannot adopt author evaluation state")
    source = inspect.getsource(reference.runner.run_experiment)
    tree = ast.parse(source)
    guidance, stop, prompt, prefix, recovery = [], [], [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            condition = ast.get_source_segment(source, node.test) or ""
            if "replay_fresh_session" in condition and isinstance(
                node.body[0], ast.AugAssign
            ):
                recovery.append(node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name) and name.id == "review_guidance":
                guidance.append(node)
            if (
                isinstance(name, ast.Name)
                and name.id == "prompt"
                and isinstance(node.value, ast.JoinedStr)
                and "Evaluation instance" in ast.get_source_segment(source, node)
            ):
                prompt.append(node)
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and isinstance(node.test.func.value, ast.Name)
            and node.test.func.value.id == "options"
            and node.test.func.attr == "get"
            and node.test.args
            and isinstance(node.test.args[0], ast.Constant)
            and node.test.args[0].value == "stop_min_steps"
        ):
            stop.append(node)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(
                "\nThis is an independent evaluation with a copy of your task's preparation skills;"
            )
        ):
            prefix.append(node.value)
    if not all(len(nodes) == 1 for nodes in (guidance, stop, prompt, prefix)):
        raise ValueError("Original evaluation prompt structure changed")
    if len(recovery) != 1:
        raise ValueError("Original recovery prompt structure changed")
    expressions = ast.Module(
        body=[guidance[0], stop[0], recovery[0], prompt[0]], type_ignores=[]
    )
    namespace = {
        "options": options,
        "experiment": SimpleNamespace(
            review_allow_reset=False,
            review_window_steps=options.get("review_window_steps"),
        ),
        "index": 0,
        "repeat_limit": options["max_repeat"],
        "resume_seconds": cfg["episode_seconds"],
        "continuation": cfg.get("continue_from_run") if own_resume else None,
        "replay": options.get("replay") if own_resume else None,
        "episode": cfg["test_episodes"][0],
    }
    exec(
        compile(
            ast.fix_missing_locations(expressions),
            "<original-evaluation-prompt>",
            "exec",
        ),
        namespace,
    )
    return reference.runner.INSTRUCTIONS + prefix[0] + namespace["prompt"]


class HybridEmbodiedAgent(EmbodiedAgent):
    """Bind an official episode, then delegate its loop to original run_segment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reference_context = None
        self._hybrid_planner = None
        self._cancelled = threading.Event()

    def bind_reference(self, context: HybridRunContext):
        if self._reference_context is not None or self._cancelled.is_set():
            raise ValueError("Hybrid Agent owns one freshly initialized episode")
        if Path(context.episode_dir).resolve() != Path(self.output_dir).resolve():
            raise ValueError("Hybrid context differs from the Agent case directory")
        self._reference_context = context
        return self

    def run_reference(self) -> PlannerResult:
        # EmbodiedAgent.run calls this subclass's _run. These labels never enter
        # model context; the original controller constructs its own exact prompt.
        return self.run(
            "original RoboDojo evaluation", system_prompt="original controller"
        )

    def _run(
        self,
        task,
        *,
        system_prompt,
        skills,
        skill_paths,
        initial_context,
        output_dir,
        memory=(),
        bridge=None,
        deferred_tools=(),
        include_finish=True,
        episode_ctx=None,
    ):
        if any(
            (
                skills,
                skill_paths,
                initial_context,
                memory,
                bridge,
                deferred_tools,
                episode_ctx,
            )
        ):
            raise ValueError(
                "Original hybrid owns its prompt, tools, skills and context"
            )
        if self._reference_context is None:
            raise ValueError("Bind an official hybrid context before running the Agent")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Hybrid Agent already has an active episode")
        try:
            return self._run_reference(self._reference_context)
        finally:
            self._run_lock.release()

    def _run_reference(self, context):
        if self._cancelled.is_set():
            raise HybridCancelled("Hybrid episode was cancelled")
        root = Path(context.episode_dir).resolve()
        if not root.is_dir():
            raise ValueError("Create the owned hybrid episode directory first")
        workspace = root / "workspace"
        if workspace.exists():
            if (
                workspace.is_symlink()
                or not workspace.is_dir()
                or any(workspace.iterdir())
            ):
                raise ValueError("Hybrid requires a fresh writable workspace")
            workspace.rmdir()
        source = (
            Path(context.resume.get("workspace_source", context.resume["source"])) / "agent/workspace"
            if context.resume
            else Path(context.prepared_workspace)
        )
        if (
            source.is_symlink()
            or not source.is_dir()
            or (
                context.resume is None
                and any(p.is_symlink() for p in source.rglob("*"))
            )
        ):
            raise ValueError("Hybrid requires a complete materialized ICL workspace")
        if (
            not Path(context.training_set).is_dir()
            or not Path(context.public_instructions).is_file()
        ):
            raise ValueError("Hybrid public training material and interface must exist")
        shutil.copytree(source, workspace, symlinks=True)
        run = root / "controller"
        run.mkdir(mode=0o700)
        trajectory = None
        if context.trajectory_dir is not None:
            trajectory = Path(context.trajectory_dir).resolve() / "controller"
            trajectory.mkdir(mode=0o700)
        for name in ("tools", "observations", "checkpoints", "episodes", "segments"):
            if trajectory is not None and name in {"observations", "episodes"}:
                (trajectory / name).mkdir()
                (run / name).symlink_to(trajectory / name, target_is_directory=True)
            else:
                (run / name).mkdir()
        if trajectory is not None:
            # The original append-only controller records remain unmodified.
            # Config, CLI home, sessions, tools and model logs stay private.
            for name in ("requests.jsonl", "responses.jsonl", "episodes.jsonl"):
                (trajectory / name).touch(exist_ok=False)
                (run / name).symlink_to(trajectory / name)
        (run / "workspace").symlink_to(workspace, target_is_directory=True)
        shutil.copy2(
            context.reference.source_dir / "code/policylab/robot.py",
            run / "tools/robot.py",
        )
        shutil.copy2(context.public_instructions, run / "tools/BENCHMARK.md")
        cfg = copy.deepcopy(context.cfg)
        cfg["model"] = self.model
        if len(cfg.get("test_episodes", [])) != 1:
            raise ValueError("Hybrid Controller must own exactly one official episode")
        cfg.update(
            training_set=str(Path(context.training_set).resolve()),
            agent_workspace=str(workspace),
            public_instructions=str(run / "tools/BENCHMARK.md"),
            agent_python=str(Path(context.agent_python).absolute()),
            codex_binary=self.codex_cli.executable,
            environment_dir=str(context.reference.source_dir),
            adapter=str(context.reference.source_dir / "hybrid_adapter.py")
            + ":Adapter",
        )
        (run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        if trajectory is not None:
            config_path = trajectory / "config.json"
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            config_path.chmod(0o600)
        prompt = build_evaluation_prompt(
            context.reference, cfg, own_resume=context.resume is not None
        )
        (run / "provenance.json").write_text(
            json.dumps(
                {
                    "reference_manifest_sha256": context.reference.manifest_sha256,
                    "evaluation_prompt_bytes": len(prompt.encode()),
                    "evaluation_prompt_sha256": hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                    "public_instructions_sha256": hashlib.sha256(
                        (run / "tools/BENCHMARK.md").read_bytes()
                    ).hexdigest(),
                    "robot_tool_sha256": hashlib.sha256(
                        (run / "tools/robot.py").read_bytes()
                    ).hexdigest(),
                    "model_scoring": "official evaluator only; no controller-generated success",
                },
                indent=2,
            )
            + "\n"
        )
        if trajectory is not None:
            shutil.copyfile(run / "provenance.json", trajectory / "provenance.json")
        planner = HybridCodexCliPlanner(
            reference=context.reference,
            bubblewrap=context.bubblewrap,
            bubblewrap_sha256=context.bubblewrap_sha256,
            output_dir=run,
            repo_root=workspace,
            model=self.model,
            timeout_s=None,
            dashboard_events=None,
            reasoning_effort=self.reasoning_effort,
            config=self.codex_cli,
        )
        self._hybrid_planner = planner
        if self._cancelled.is_set():
            planner.cancel()
            raise HybridCancelled(
                "Hybrid episode was cancelled before controller startup"
            )
        reference = context.reference

        class Experiment(reference.runner.Experiment):
            request = reference.hybrid_host.request

        experiment = Experiment(cfg, run, context.backend)
        experiment.phase = "evaluation"
        socket_directory = tempfile.TemporaryDirectory(prefix="srs-bridge-", dir="/tmp")
        bridge_dir = run / "bridge"
        bridge_dir.symlink_to(socket_directory.name, target_is_directory=True)
        try:
            server = socketserver.UnixStreamServer(
                str(Path(socket_directory.name) / "robot.sock"),
                reference.runner.Handler,
            )
        except BaseException:
            bridge_dir.unlink(missing_ok=True)
            socket_directory.cleanup()
            raise
        server.experiment = experiment
        server_thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        server_thread.start()
        attempt_started = time.monotonic()
        summary = None
        failure = None
        budget_marker = os.environ.get("ROBORSI_EXECUTION_BUDGET_FILE")
        try:
            if context.resume and budget_marker:
                from .case_budget import write_execution_budget

                write_execution_budget(Path(budget_marker), cfg["episode_seconds"], phase="replay")
            with experiment.lock:
                experiment.reset_episode(cfg["test_episodes"][0])
            if context.resume:
                # Physical replay is part of this episode's cumulative wall budget.
                replay_seconds = time.monotonic() - attempt_started
                cfg["episode_seconds"] -= replay_seconds
                if cfg["episode_seconds"] <= 0:
                    raise TimeoutError(
                        "Original cumulative episode wall budget exhausted during replay"
                    )
                (run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
                prompt = build_evaluation_prompt(
                    context.reference, cfg, own_resume=True
                )
                if trajectory is not None:
                    shutil.copyfile(run / "config.json", trajectory / "config.json")
                provenance_path = run / "provenance.json"
                provenance = json.loads(provenance_path.read_text())
                provenance.update(
                    evaluation_prompt_bytes=len(prompt.encode()),
                    evaluation_prompt_sha256=hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                )
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                if trajectory is not None:
                    shutil.copyfile(provenance_path, trajectory / "provenance.json")
                # Original replay must have validated before any CLI launch.
                from .hybrid_resume import read_json

                validation = (
                    read_json(
                        run / "episodes/evaluation-0001/native/replay_validation.json"
                    )
                    if context.resume["step_offset"]
                    else {"matched": True, "replayed_steps": 0}
                )
                if (
                    validation.get("matched") is not True
                    or validation["replayed_steps"] != context.resume["step_offset"]
                ):
                    raise ValueError("Native replay validation did not pass")
                (run / "resume-validation.json").write_text(
                    json.dumps(validation, indent=2) + "\n"
                )
            if budget_marker:
                from .case_budget import write_execution_budget

                write_execution_budget(Path(budget_marker), cfg["episode_seconds"], phase="agent")
            summary = reference.runner.run_segment(
                experiment,
                {**cfg, "_observation_episode": experiment.active["name"]},
                run,
                root / "codex-home",
                "evaluation-0001",
                prompt,
                cfg["episode_seconds"],
                None,
                launch=planner.launch_segment,
            )
            failed = (
                bool(experiment.fault)
                or (
                    experiment.review_window_steps is not None
                    and summary["status"] != "finished"
                )
                or (
                    summary["returncode"] != 0
                    and summary["status"] not in {"finished", "budget_exhausted"}
                )
            )
            failure = (
                "Original hybrid controller interrupted before native termination"
                if failed
                else None
            )
            return PlannerResult(
                stats={"hybrid_segment": summary, "native_success": None},
                error="Original hybrid controller interrupted before native termination"
                if failed
                else None,
                error_kind="hybrid_controller" if failed else None,
                stop_reason=summary["status"],
            )
        except BaseException as exc:
            failure = type(exc).__name__ + ": " + str(exc)
            if (
                context.resume
                and isinstance(exc, BackendClosed)
                and not (run / "resume-validation.json").exists()
            ):
                # Deferred public actions report a broken native ACK channel
                # as BackendClosed. Bind it to the original infrastructure
                # recovery category; state/scene validation errors stay fatal.
                failure = "Simulator exited during native replay: " + failure
            raise
        finally:
            try:
                from .hybrid_resume import write_json

                if summary is None:
                    # Record measured interruption evidence, not a fabricated native outcome.
                    summary = {
                        "status": "interrupted",
                        "returncode": None,
                        "wall_seconds": time.monotonic() - attempt_started,
                        "measurement_source": "v4_controller_finally",
                    }
                    write_json(run / "segments/evaluation-0001/result.json", summary)
                write_json(
                    run / "summary.json",
                    {
                        "status": "error" if failure else "completed",
                        "error": failure,
                        "segments": [summary],
                        "native_success": None,
                    },
                )
            finally:
                # A handler may be waiting for an official action acknowledgement
                # while holding Experiment.lock. Release backend waits before join.
                context.backend.cancel()
                planner.cancel()
                server.shutdown()
                server.server_close()
                server_thread.join(2)
                bridge_dir.unlink(missing_ok=True)
                socket_directory.cleanup()

    def close(self):
        self._cancelled.set()
        if self._reference_context is not None:
            self._reference_context.backend.cancel()
        planner = self._hybrid_planner
        if planner is not None:
            planner.cancel()


def make_agent(case_dir: Path, *, config: AgentConfig) -> HybridEmbodiedAgent:
    """AgentConfig custom factory; no SDK/MCP planner or task owner is created."""
    options = config.document.get("agent", {})
    if (
        options.get("planner") != "codex_cli"
        or not isinstance(options.get("model"), str)
        or not options["model"].strip()
        or options.get("reasoning_effort") != "max"
        or options.get("max_turns") is not None
        or options.get("max_tokens") is not None
    ):
        raise ValueError(
            "Hybrid Agent requires the reference CLI, model, effort and budgets"
        )
    if (
        config.document.get("prompt_file")
        or config.document.get("skills")
        or config.document.get("runtime")
    ):
        raise ValueError("The original hybrid controller owns all model context")
    options["codex_cli"] = CodexCliConfig.from_dict(options.get("codex_cli", {}))
    if options["codex_cli"].tools != "external":
        raise ValueError("Hybrid Agent requires the original external robot tools")
    return HybridEmbodiedAgent(
        mcp_servers=(), output_dir=case_dir, local_tools=(), **options
    )


CONTROLLER_CLEANUP_TIMEOUT_SECONDS = 3600.0


class HybridController:
    """Thread owner exposing only the backend's queued official joint requests."""

    def __init__(self, agent: HybridEmbodiedAgent, context: HybridRunContext):
        self.agent = agent.bind_reference(context)
        self.context = context
        self.result = None
        self.error = None
        self._done = threading.Event()
        self._cancelled = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="roborsi-hybrid", daemon=True
        )

    def _worker(self):
        try:
            self.result = self.agent.run_reference()
            if self.result.error:
                self.error = RuntimeError(self.result.error)
        except BaseException as error:
            if not self._cancelled.is_set():
                self.error = error
        finally:
            self._done.set()
            self.context.backend.cancel()

    def start(self):
        self._thread.start()
        return self

    @property
    def done(self):
        return self._done.is_set()

    def next_action(self, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._done.is_set() or self._cancelled.is_set():
                if self.error is not None:
                    raise RuntimeError("Original hybrid Agent failed") from self.error
                return None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise queue.Empty
            try:
                return self.context.backend.next_action(
                    timeout=min(0.1, remaining) if remaining is not None else 0.1
                )
            except queue.Empty:
                continue
            except BackendClosed:
                if self._done.wait(0.1):
                    continue
                if self._cancelled.is_set():
                    return None
                raise

    def acknowledge(self, *args, **kwargs):
        return self.context.backend.acknowledge(*args, **kwargs)

    def cancel(self, timeout: float = CONTROLLER_CLEANUP_TIMEOUT_SECONDS):
        self._cancelled.set()
        self.context.backend.cancel()
        self.agent.close()
        if (
            self._thread.ident is not None
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("Original hybrid controller survived cancellation")

    close = cancel

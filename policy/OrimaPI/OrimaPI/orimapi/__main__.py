# coding=utf-8
"""Command-line entrypoint and construction of the core runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .policy import CodeSkillConfig, PolicyConfig, SKILL_NAMES, create_policy
from .policy_library import PolicyCandidate, policy_library
from .runtime import Runtime, RuntimeConfig
from .state import Mode
from .supervisor import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORTS,
    validate_api_settings,
)
from .api_client import ConfiguredAPISupervisor as APISupervisor
from .llm_auth import CommandCredential, LocalBridgeCredential, validate_auth
from .copilot_auth import CopilotCredential
from .trace import TraceWriter
from .invocation import capture_invocation, record_invocation


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    policies: dict[str, PolicyCandidate]
    supervisor: dict
    task: str | None
    episodes: int
    output_root: Path
    fixed_policy: str | None = None
    xpolicylab: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fixed_policy is not None and (
            not isinstance(self.fixed_policy, str)
            or self.fixed_policy not in self.policies
        ):
            raise ValueError("fixed_policy must name a configured policy candidate")

    def public(self) -> dict:
        return {
            "runtime": asdict(self.runtime),
            "policies": {key: value.public() for key, value in self.policies.items()},
            "fixed_policy": self.fixed_policy,
            "xpolicylab": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in self.xpolicylab.items()
            },
            "supervisor": self.supervisor,
            "evaluation": {
                "task": self.task,
                "episodes": self.episodes,
                "output_root": str(self.output_root),
            },
        }


def _section(document: dict, name: str, allowed: set[str]) -> dict:
    value = document.get(name, {})
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError(f"Invalid or unknown settings in {name}")
    return dict(value)


def _path(value, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured paths must be nonempty strings")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path, *, mode: str | None = None) -> AppConfig:
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "Install project dependencies: pip install -e '.[dev]'"
        ) from None
    path = Path(path).resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {error}") from None
    if isinstance(document, dict) and "policy" in document:
        raise ValueError(
            "Replace the single policy section with a named policies catalogue; "
            "use fixed_policy only for an explicit fixed-policy experiment"
        )
    if not isinstance(document, dict) or document.keys() - {
        "runtime",
        "policies",
        "fixed_policy",
        "supervisor",
        "evaluation",
        "xpolicylab",
    }:
        raise ValueError("Configuration must contain the declared application sections")
    runtime = _section(document, "runtime", set(RuntimeConfig.__dataclass_fields__))
    if mode is not None:
        runtime["mode"] = mode
    raw_policies = document.get("policies")
    if not isinstance(raw_policies, dict) or not raw_policies:
        raise ValueError("policies must be a nonempty catalogue of named candidates")
    policies = {}
    for policy_id in raw_policies:
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("Policy candidate IDs must be nonempty strings")
        settings = _section(
            raw_policies,
            policy_id,
            set(PolicyConfig.__dataclass_fields__)
            | {"description", "supports_subgoals"},
        )
        description = settings.pop("description", None)
        supports_subgoals = settings.pop("supports_subgoals", None)
        for field in (
            "checkpoint",
            "code_path",
            "code_bundle",
            "code_selection_prompt",
            "tokenizer_path",
        ):
            settings[field] = _path(settings.get(field), path.parent)
        raw_skills = settings.get("code_skills", {})
        if not isinstance(raw_skills, dict):
            raise ValueError("code_skills must be a named sub-skill mapping")
        if raw_skills and policy_id != "code-policy":
            raise ValueError(
                "Register code sub-skills under policies.code-policy.code_skills"
            )
        skills = {}
        for skill_id in raw_skills:
            child = _section(
                raw_skills, skill_id, set(CodeSkillConfig.__dataclass_fields__)
            )
            for field in ("code_path", "code_bundle"):
                child[field] = _path(child.get(field), path.parent)
            child["code_parameters"] = _code_parameters(
                child.get("code_parameters", {}), path.parent
            )
            skills[skill_id] = CodeSkillConfig(**child)
        settings["code_skills"] = skills
        if settings["code_bundle"] is not None or skills:
            settings["code_parameters"] = _code_parameters(
                settings.get("code_parameters", {}), path.parent
            )
        policies[policy_id] = PolicyCandidate(
            PolicyConfig(**settings),
            description,
            supports_subgoals,
        )
    supervisor = _section(
        document,
        "supervisor",
        {
            "endpoint",
            "model",
            "api_key_env",
            "prompt_path",
            "initial_prompt_path",
            "review_prompt_path",
            "selection_prompt_path",
            "wire_api",
            "headers",
            "auth",
            "stream",
            "reasoning_effort",
            "require_model_match",
        },
    )
    supervisor["endpoint"] = os.environ.get("ORIMAPI_LLM_ENDPOINT") or supervisor.get(
        "endpoint"
    )
    supervisor["model"] = os.environ.get("ORIMAPI_LLM_MODEL") or supervisor.get("model")
    supervisor["api_key_env"] = supervisor.get("api_key_env", "OPENAI_API_KEY")
    for name in ("endpoint", "model", "api_key_env"):
        value = supervisor.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"supervisor.{name} must be a nonempty string or null")
    if supervisor["api_key_env"] is None:
        raise ValueError("supervisor.api_key_env cannot be null")
    supervisor.setdefault("wire_api", "chat_completions")
    supervisor.setdefault("headers", {})
    supervisor.setdefault("stream", False)
    supervisor.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    supervisor.setdefault("require_model_match", False)
    if type(supervisor["require_model_match"]) is not bool:
        raise ValueError("supervisor.require_model_match must be a boolean")
    validate_api_settings(
        supervisor["wire_api"],
        supervisor["headers"],
        supervisor["stream"],
        supervisor["reasoning_effort"],
    )
    supervisor["auth"] = validate_auth(supervisor.get("auth", {}))
    if supervisor["auth"]["type"] == "copilot":
        supervisor["auth"]["token_file"] = str(
            _path(supervisor["auth"]["token_file"], path.parent)
        )
    prompt_path = _path(
        supervisor.get("prompt_path", "../prompts/robodojo_supervisor.md"), path.parent
    )
    if prompt_path is None:
        raise ValueError("supervisor.prompt_path is required")
    supervisor["prompt_path"] = str(prompt_path)
    for name in ("initial_prompt_path", "review_prompt_path"):
        if name in supervisor:
            stage_path = _path(supervisor[name], path.parent)
            supervisor[name] = str(stage_path) if stage_path is not None else None
    selection_path = _path(
        supervisor.get(
            "selection_prompt_path", "../prompts/robodojo_policy_selection.md"
        ),
        path.parent,
    )
    if selection_path is None:
        raise ValueError("supervisor.selection_prompt_path is required")
    supervisor["selection_prompt_path"] = str(selection_path)
    evaluation = _section(document, "evaluation", {"task", "episodes", "output_root"})
    task = evaluation.get("task")
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ValueError("evaluation.task must be a nonempty string or null")
    episodes = evaluation.get("episodes", 1)
    if type(episodes) is not int or episodes < 1:
        raise ValueError("evaluation.episodes must be a positive integer")
    output_root = _path(evaluation.get("output_root", "../runs"), path.parent)
    if output_root is None:
        raise ValueError("evaluation.output_root is required")
    from .xpolicylab.deploy import load_settings

    xpolicylab = load_settings(document.get("xpolicylab", {}), path.parent)
    return AppConfig(
        RuntimeConfig(**runtime),
        policies,
        supervisor,
        task,
        episodes,
        output_root,
        document.get("fixed_policy"),
        xpolicylab,
    )


def _code_parameters(value, base):
    if not isinstance(value, dict):
        raise ValueError("code_parameters must be a JSON object")
    parameters = dict(value)
    if "vision_resources_path" in parameters or "vision_model_root" in parameters:
        if "vision_models" in parameters:
            raise ValueError(
                "Use vision_resources_path or inline vision_models, not both"
            )
        resources_path = _path(parameters.pop("vision_resources_path", None), base)
        model_root = _path(parameters.pop("vision_model_root", None), base)
        if resources_path is None or model_root is None:
            raise ValueError(
                "vision_resources_path and vision_model_root are both required"
            )
        resources = json.loads(resources_path.read_text(encoding="utf-8"))
        if not isinstance(resources, dict) or not resources:
            raise ValueError("Vision resources must be a nonempty model mapping")
        models = {}
        for name, resource in resources.items():
            if (
                not isinstance(name, str)
                or not isinstance(resource, dict)
                or not isinstance(resource.get("files"), dict)
            ):
                raise ValueError("Each vision resource requires a path and file hashes")
            model_path = _path(resource.get("path"), model_root)
            if model_path is None:
                raise ValueError("Vision model paths cannot be null")
            models[name] = {**resource, "path": str(model_path)}
        parameters["vision_models"] = models
    if "geometry" in parameters:
        geometry = parameters["geometry"]
        if not isinstance(geometry, dict):
            raise ValueError("code_parameters.geometry must be a path mapping")
        resolved = {key: _path(path, base) for key, path in geometry.items()}
        if any(path is None for path in resolved.values()):
            raise ValueError("Geometry paths cannot be null")
        parameters["geometry"] = {key: str(path) for key, path in resolved.items()}
    if "tool_limits" in parameters or "depth_estimation" in parameters:
        from .skills.code_tools import settings

        tools = settings(parameters)
        if tools.get("depth"):
            tools["depth"]["model_path"] = str(
                _path(tools["depth"]["model_path"], base)
            )
            parameters["depth_estimation"] = tools["depth"]
    return parameters


def build_runtime(config: AppConfig, trace: TraceWriter) -> Runtime:
    def factory(settings):
        policy = create_policy(settings)
        if settings.code_bundle is not None or settings.code_skills:
            policy.configure_generation(
                config.supervisor, timeout_s=config.runtime.supervisor_timeout_s
            )
        return policy

    library = policy_library(config.policies, factory=factory)
    supervisor = None
    if config.fixed_policy is None or config.runtime.mode != Mode.DIRECT:
        settings = config.supervisor
        if not settings["endpoint"] or not settings["model"]:
            raise ValueError("Configure a vision model and supervisor endpoint")
        auth = validate_auth(settings.get("auth", {}))
        credentials = {}
        if auth["type"] == "command":
            credentials["credential"] = CommandCredential(auth)
        elif auth["type"] == "copilot":
            credentials["credential"] = CopilotCredential(auth)
        elif auth["type"] == "none":
            credentials["credential"] = LocalBridgeCredential()
        else:
            api_key = os.environ.get(settings["api_key_env"])
            if not api_key:
                raise ValueError(
                    f"Missing API credential environment variable: {settings['api_key_env']}"
                )
            credentials["api_key"] = api_key
        prompt = Path(settings["prompt_path"]).read_text(encoding="utf-8")
        initial_prompt = (
            Path(settings["initial_prompt_path"]).read_text(encoding="utf-8")
            if settings.get("initial_prompt_path")
            else prompt
        )
        review_prompt = (
            Path(settings["review_prompt_path"]).read_text(encoding="utf-8")
            if settings.get("review_prompt_path")
            else prompt
        )
        selection_prompt = Path(settings["selection_prompt_path"]).read_text(
            encoding="utf-8"
        )
        trace.write(
            "supervisor_configured",
            model=settings["model"],
            wire_api=settings.get("wire_api", "chat_completions"),
            reasoning_effort=settings.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            system_prompt=prompt,
            initial_prompt=initial_prompt,
            initial_prompt_sha256=hashlib.sha256(initial_prompt.encode()).hexdigest(),
            review_prompt=review_prompt,
            review_prompt_sha256=hashlib.sha256(review_prompt.encode()).hexdigest(),
            selection_prompt_sha256=hashlib.sha256(
                selection_prompt.encode()
            ).hexdigest(),
            selection_prompt=selection_prompt,
        )
        supervisor = APISupervisor(
            endpoint=settings["endpoint"],
            model=settings["model"],
            **credentials,
            wire_api=settings.get("wire_api", "chat_completions"),
            headers=settings.get("headers", {}),
            stream=settings.get("stream", False),
            reasoning_effort=settings.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
            system_prompt=prompt,
            require_model_match=settings.get("require_model_match", False),
            initial_prompt=initial_prompt,
            review_prompt=review_prompt,
            selection_prompt=selection_prompt,
            timeout_s=config.runtime.supervisor_timeout_s,
            trace=trace,
        )
    return Runtime(
        policy_library=library,
        fixed_policy=config.fixed_policy,
        supervisor=supervisor,
        trace=trace,
        config=config.runtime,
    )


def evaluate(config: AppConfig, *, invocation=None) -> int:
    from .xpolicylab.deploy import run_evaluation

    if not config.task:
        raise ValueError("Provide --task or evaluation.task")
    name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    runtime = None
    with TraceWriter(config.output_root / name) as trace:
        trace.write_json("config.json", config.public())
        record_invocation(trace, invocation)

        def runtime_factory() -> Runtime:
            nonlocal runtime
            if runtime is not None:
                raise RuntimeError("This evaluation already owns a runtime")
            runtime = build_runtime(config, trace)
            return runtime

        try:
            result = run_evaluation(
                runtime_factory=runtime_factory,
                task=config.task,
                episodes=config.episodes,
                settings=config.xpolicylab,
                output_root=trace.root / "evaluator",
                raw_images=any(
                    candidate.config.name == "code-policy"
                    and (
                        config.runtime.mode == Mode.DIRECT
                        or candidate.supports_subgoals
                    )
                    for policy_id, candidate in config.policies.items()
                    if config.fixed_policy is None or policy_id == config.fixed_policy
                ),
            )
            if not isinstance(result, dict):
                raise RuntimeError(
                    "The evaluator did not return an official result record"
                )
            trace.write_json(
                "summary.json",
                {"status": "evaluation_returned", "official_result": result},
            )
            return 0
        except KeyboardInterrupt:
            trace.write("run_interrupted")
            trace.write_json("summary.json", {"status": "interrupted"})
            print(f"Evaluation interrupted. Run records: {trace.root}", file=sys.stderr)
            return 130
        except Exception as error:
            status = "failed"
            trace.write(
                "run_failed", status=status, error=f"{type(error).__name__}: {error}"
            )
            trace.write_json("summary.json", {"status": status, "error": str(error)})
            print(f"{error}\nRun records: {trace.root}", file=sys.stderr)
            return 1
        finally:
            if runtime is not None:
                runtime.close()


def check_skill(config: PolicyConfig, *, load: bool = False) -> int:
    from .skills.openpi_backend import inspect_skill

    report = inspect_skill(config)
    policy = None
    if load and report["ready_for_load"]:
        try:
            policy = create_policy(config)
            policy.load()
            report.update(
                loaded=True,
                capability=policy.describe(),
                scope="Skill loaded; no observations, actions or physical execution tested",
            )
        except Exception as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
            report["ready_for_load"] = False
        finally:
            if policy is not None:
                policy.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_load"] else 1


def _add_policy_arguments(parser):
    parser.add_argument(
        "--policy-id",
        help="Candidate to inspect or configure; does not fix the episode selection",
    )
    parser.add_argument(
        "--skill",
        choices=SKILL_NAMES,
        help="Target the unique candidate using this backend",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Exact checkpoint directory containing params/ and assets/",
    )
    parser.add_argument(
        "--train-config",
        dest="train_config_name",
        help="OrimaPI inference preset or official OpenPI configuration name",
    )
    parser.add_argument("--normalization-id")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Local PaliGemma tokenizer model file for the RoboDojo inference preset",
    )
    parser.add_argument(
        "--code-path", type=Path, help="Trusted local Python policy program"
    )
    parser.add_argument(
        "--code-entrypoint", help="Program factory name; default create_policy"
    )
    parser.add_argument("--code-sha256", help="Optional expected source SHA-256")
    parser.add_argument(
        "--code-depth-model-path",
        type=Path,
        help="Local metric Depth Anything checkpoint; GPT decides whether to use it",
    )
    parser.add_argument(
        "--code-depth-device", help="Depth inference device: cpu or cuda:N"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OrimaPI RoboDojo feedback agent")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("skills", help="List the available execution skills")
    descriptions = (
        ("check-config", "Validate configuration without loading any skill"),
        (
            "check-skill",
            "Check skill assets/dependencies; --load loads a skill locally",
        ),
        ("eval", "Run evaluation with installed RoboDojo and XPolicyLab"),
        (
            "eval-suite",
            "Run RoboDojo tasks with resident policies and official scoring",
        ),
    )
    for name, help_text in descriptions:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path)
        command.add_argument("--mode", choices=[mode.value for mode in Mode])
        command.add_argument(
            "--record-frames",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Save supervisor input frames; --no-record-frames keeps text logs only",
        )
        command.add_argument(
            "--policy-backend",
            choices=SKILL_NAMES,
            help="Select a top-level backend; code-policy still uses GPT to select its sub-skill",
        )
        command.add_argument(
            "--exec-horizon",
            metavar="N",
            type=int,
            help=(
                "Maximum actions executed per policy prediction; positive integer, "
                "independent of the GPT review interval (default: configuration value)"
            ),
        )
        command.add_argument(
            "--fast-trajectory-chunking",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Dev/test-only acceleration: batch interior trajectory steps into "
                "multi-step chunks and skip intermediate camera renders (default: off)"
            ),
        )
        command.add_argument(
            "--fast-trajectory-chunk-size",
            metavar="N",
            type=int,
            default=None,
            help="Maximum interior trajectory steps per fast chunk (default: 8 when enabled)",
        )
        command.add_argument(
            "--replan-interval-chunks",
            "--replan-interval-steps",
            dest="replan_interval_steps",
            metavar="N",
            type=int,
            help=(
                "Completed action chunks between blocking GPT reviews, independent "
                "of exec_horizon; legacy async mode counts environment actions"
            ),
        )
        command.add_argument("--llm-model", help="Vision model ID for GPT supervision")
        command.add_argument(
            "--llm-reasoning-effort",
            choices=REASONING_EFFORTS,
            help="Optional reasoning effort for GPT calls (default: provider setting)",
        )
        command.add_argument("--llm-endpoint", help="Full supervisor HTTP endpoint")
        command.add_argument(
            "--llm-initial-prompt",
            type=Path,
            help="Initial subgoal prompt file; relative to the working directory",
        )
        command.add_argument(
            "--llm-review-prompt",
            type=Path,
            help="Mid-episode review prompt file; relative to the working directory",
        )
        command.add_argument(
            "--llm-wire-api", choices=("chat_completions", "responses")
        )
        command.add_argument(
            "--llm-auth-command",
            nargs="+",
            metavar="ARG",
            help="Trusted credential command argv; stdout must be a token",
        )
        command.add_argument(
            "--llm-header",
            action="append",
            metavar="NAME=VALUE",
            help="Non-secret integration header; repeat for multiple headers",
        )
        command.add_argument(
            "--llm-stream",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Use Responses SSE streaming",
        )
        command.add_argument(
            "--llm-timeout-s",
            type=float,
            help="Total supervision request budget including credential refresh",
        )
        command.add_argument(
            "--fixed-policy",
            help="Fix the top-level policy; code-policy still selects its sub-skill with GPT",
        )
        _add_policy_arguments(command)
        if name == "check-skill":
            command.add_argument(
                "--load",
                action="store_true",
                help="Load the supplied weights or execute the configured program module",
            )
        if name == "eval":
            command.add_argument("--task")
            command.add_argument("--episodes", type=int)
            command.add_argument("--output-root", type=Path)
        if name == "eval-suite":
            tasks = command.add_mutually_exclusive_group()
            tasks.add_argument("--tasks", nargs="+")
            tasks.add_argument("--tasks-file", type=Path)
            tasks.add_argument("--all-tasks", action="store_true", default=None)
            command.add_argument("--profile", choices=("official", "smoke"))
            command.add_argument("--episodes", type=int)
            command.add_argument("--seeds", nargs="+", type=int)
            command.add_argument("--policy-seed", type=int)
            command.add_argument("--policy-gpu", type=int)
            command.add_argument("--simulator-gpu", type=int)
            command.add_argument("--fail-fast", action="store_true", default=None)
            command.add_argument("--min-free-gib", type=float)
            command.add_argument("--job-timeout-s", type=float)
            command.add_argument(
                "--resume",
                type=Path,
                help="Existing suite directory; completed jobs are verified and skipped",
            )
            command.add_argument("--output-root", type=Path)
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        invocation = (
            capture_invocation(argv) if args.command in {"eval", "eval-suite"} else None
        )
        if args.command == "skills":
            from .skills import catalogue

            print(json.dumps(catalogue(), indent=2))
            return 0
        resume = getattr(args, "resume", None)
        config_path = args.config or (
            resume / "config.json" if resume else Path("configs/robodojo.yaml")
        )
        config = load_config(
            config_path,
            mode=args.mode
            or ("direct" if args.command == "eval-suite" and not resume else None),
        )
        if args.record_frames is not None:
            config = replace(
                config,
                runtime=replace(config.runtime, record_frames=args.record_frames),
            )
        if args.exec_horizon is not None:
            config = replace(
                config,
                runtime=replace(config.runtime, exec_horizon=args.exec_horizon),
            )
        if args.fast_trajectory_chunk_size is not None:
            if args.fast_trajectory_chunk_size < 0 or args.fast_trajectory_chunk_size > 64:
                raise ValueError("--fast-trajectory-chunk-size must be in 0..64")
            os.environ["ORIMAPI_FAST_TRAJECTORY_CHUNKING"] = str(
                args.fast_trajectory_chunk_size
            )
        elif args.fast_trajectory_chunking is not None:
            os.environ["ORIMAPI_FAST_TRAJECTORY_CHUNKING"] = (
                "8" if args.fast_trajectory_chunking else "0"
            )
        if args.replan_interval_steps is not None:
            config = replace(
                config,
                runtime=replace(
                    config.runtime, replan_interval_steps=args.replan_interval_steps
                ),
            )
        supervisor_overrides = {}
        for stage in ("initial", "review"):
            path = getattr(args, f"llm_{stage}_prompt")
            if path is not None:
                supervisor_overrides[f"{stage}_prompt_path"] = str(
                    path.expanduser().resolve()
                )
        for field in ("model", "endpoint", "wire_api", "reasoning_effort"):
            value = getattr(args, "llm_" + field)
            if value is not None:
                if not value.strip():
                    raise ValueError(f"--llm-{field} must not be empty")
                supervisor_overrides[field] = value
        if args.llm_auth_command is not None:
            supervisor_overrides["auth"] = validate_auth(
                {
                    "type": "command",
                    "command": args.llm_auth_command,
                }
            )
        if args.llm_header is not None:
            headers = dict(config.supervisor.get("headers", {}))
            for item in args.llm_header:
                key, separator, value = item.partition("=")
                if not separator:
                    raise ValueError("--llm-header requires NAME=VALUE")
                headers[key] = value
            supervisor_overrides["headers"] = headers
        if args.llm_stream is not None:
            supervisor_overrides["stream"] = args.llm_stream
        if supervisor_overrides:
            config = replace(
                config, supervisor={**config.supervisor, **supervisor_overrides}
            )
        validate_api_settings(
            config.supervisor["wire_api"],
            config.supervisor["headers"],
            config.supervisor["stream"],
            config.supervisor["reasoning_effort"],
        )
        if args.llm_timeout_s is not None:
            config = replace(
                config,
                runtime=replace(
                    config.runtime, supervisor_timeout_s=args.llm_timeout_s
                ),
            )
        overrides = {}
        for field in (
            "checkpoint",
            "train_config_name",
            "normalization_id",
            "tokenizer_path",
            "code_path",
            "code_entrypoint",
            "code_sha256",
        ):
            value = getattr(args, field)
            if value is not None:
                overrides[field] = value.resolve() if isinstance(value, Path) else value
        policy_id = args.policy_id
        if args.skill:
            matches = [
                key
                for key, value in config.policies.items()
                if value.config.name == args.skill
            ]
            if policy_id is None:
                if len(matches) != 1:
                    raise ValueError(
                        "--skill must match one candidate; use --policy-id to disambiguate"
                    )
                policy_id = matches[0]
            elif policy_id not in matches:
                raise ValueError("--skill does not match the selected --policy-id")
        if policy_id is None and (overrides or args.command == "check-skill"):
            if len(config.policies) != 1:
                raise ValueError("Choose a candidate with --policy-id or --skill")
            policy_id = next(iter(config.policies))
        if policy_id is not None:
            if policy_id not in config.policies:
                raise ValueError(f"Unknown policy candidate: {policy_id}")
            policies = dict(config.policies)
            candidate = policies[policy_id]
            if "code_path" in overrides and candidate.config.code_skills:
                overrides.update(code_skills={}, code_selection_prompt=None)
            policies[policy_id] = replace(
                candidate, config=replace(candidate.config, **overrides)
            )
            config = replace(config, policies=policies)
        if args.fixed_policy is not None:
            config = replace(config, fixed_policy=args.fixed_policy)
        if args.policy_backend is not None:
            candidates = {
                key: candidate
                for key, candidate in config.policies.items()
                if candidate.config.name == args.policy_backend
            }
            if not candidates:
                raise ValueError("No candidates match --policy-backend")
            if (
                config.fixed_policy is not None
                and config.fixed_policy not in candidates
            ):
                raise ValueError("The fixed policy conflicts with --policy-backend")
            if len(candidates) != 1:
                raise ValueError("A backend must identify one top-level policy")
            config = replace(
                config, policies=candidates, fixed_policy=next(iter(candidates))
            )
        if args.code_depth_model_path is not None or args.code_depth_device is not None:
            from .skills.code_tools import settings as tool_settings

            if "code-policy" not in config.policies:
                raise ValueError("Code tools require the code-policy backend")
            policies = dict(config.policies)
            candidate = policies["code-policy"]
            parameters = dict(candidate.config.code_parameters)
            depth = dict(parameters.get("depth_estimation", {}))
            if args.code_depth_model_path is not None:
                depth["model_path"] = str(args.code_depth_model_path.resolve())
            if args.code_depth_device is not None:
                depth["device"] = args.code_depth_device
            parameters["depth_estimation"] = depth
            tool_settings(parameters)
            policies["code-policy"] = replace(
                candidate, config=replace(candidate.config, code_parameters=parameters)
            )
            config = replace(config, policies=policies)
        if args.command == "eval-suite":
            from .evaluation.plan import SuiteOptions, tasks_from_file
            from .evaluation.suite import run_suite

            if config.fixed_policy is None:
                matches = [
                    key
                    for key, candidate in config.policies.items()
                    if candidate.config.name == "pi05"
                ]
                if not matches:
                    matches = [
                        key
                        for key, candidate in config.policies.items()
                        if candidate.config.name == "code-policy"
                    ]
                if len(matches) != 1:
                    raise ValueError("Select a top-level backend with --fixed-policy")
                config = replace(config, fixed_policy=matches[0])
            values = {}
            if args.resume:
                values = json.loads((args.resume / "plan.json").read_text())["options"]
            if args.tasks is not None or args.tasks_file is not None or args.all_tasks:
                values["tasks"] = (
                    tasks_from_file(args.tasks_file)
                    if args.tasks_file
                    else tuple(args.tasks or ())
                )
                values["all_tasks"] = bool(args.all_tasks)
            for key in SuiteOptions.__dataclass_fields__:
                value = getattr(args, key, None)
                if value is not None and key not in {"tasks", "all_tasks"}:
                    values[key] = value
            values.setdefault("simulator_gpu", config.xpolicylab["simulator_gpu"])
            config = replace(
                config,
                output_root=(
                    args.output_root.resolve()
                    if args.output_root
                    else config.output_root
                ),
            )
            return run_suite(
                config,
                SuiteOptions(**values),
                resume=args.resume,
                dry_run=args.dry_run,
                invocation=invocation,
            )
        if args.command == "check-config":
            print(
                json.dumps(
                    {
                        "configuration_valid": True,
                        "scope": "structure only; no API, checkpoint or simulator qualification",
                        "mode": config.runtime.mode.value,
                        "policies": list(config.policies),
                        "policy_selection": (
                            "fixed" if config.fixed_policy is not None else "agent"
                        ),
                        "fixed_policy": config.fixed_policy,
                        "skill_adapter": "implemented",
                        "assets_checked": False,
                        "robodojo_evaluation": "implemented; external installation required",
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "check-skill":
            return check_skill(config.policies[policy_id].config, load=args.load)
        if args.episodes is not None and args.episodes < 1:
            raise ValueError("--episodes must be positive")
        config = replace(
            config,
            task=args.task or config.task,
            episodes=args.episodes or config.episodes,
            output_root=(
                args.output_root.resolve() if args.output_root else config.output_root
            ),
        )
        return evaluate(config, invocation=invocation)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

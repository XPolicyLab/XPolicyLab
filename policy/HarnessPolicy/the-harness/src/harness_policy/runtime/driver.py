"""High level of the runtime: the model writes programs and a kernel runs them.

One episode:
  reset the worker, claim the evaluator token, start a kernel bound to the
  episode, then loop: model reply -> <python> runs in the kernel, <verify>
  asks a separate verifier call, <finish> ends the episode. Program output,
  escalated events and fresh camera views go back to the model. The evaluator
  outcome is read with the token after the loop and is never shown to the model.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import uuid

from .. import AGENT_VERSION
from ..core.robot import RemoteEnvironment
from ..core.media import write_episode_videos
from ..core.trace import Trace, context_summary, fingerprint, provenance, write_json
from ..embodiments import embodiment_kind
from .direct import ActionError, action_program
from .prompts import assemble_prompt  # noqa: F401  (re-exported)
from .skills import LAYERS, Resolution, available_services, resolve, skills_for_layers
from .kernel import Kernel, KernelDied
from .protocol import DIRECT_KINDS, KINDS, ProtocolError, parse_reply


def _kernel_timeout(code: str, runtime: dict, config: "TaskConfig", started: float) -> float:
    """How long this cell may run before the kernel is stopped.

    A cell that hands the task to the learned policy runs until the episode's wall
    time. The short cell limit killed those runs while the policy was still moving:
    build_tower episodes 41-50 and organize_table 31-38 were stopped at 1800 s and
    scored 0. Any other cell keeps cell_timeout_seconds.
    """
    if code.strip() == "__policy_to_end__()" or "start(" in code or "run_policy(" in code:
        left = float(config.max_wall_seconds) - (time.perf_counter() - started)
        return max(1.0, left)
    return float(runtime.get("cell_timeout_seconds", 600))


VERIFIER_SYSTEM = (
    "You check a claim about a robot workspace from camera images. Answer on the first "
    "line with exactly YES or NO, then one sentence of evidence from the images. "
    "Answer NO when the images do not show enough to confirm the claim."
)
# What the driver tells the model about each embodiment kind (embodiments.KINDS): the
# verifier's instructions, the label and keys of the state in every message, and the
# image streams the model sees unless runtime.cameras names others.
EMBODIMENT_TEXT = {
    "robot": {"verifier": VERIFIER_SYSTEM, "state_label": "Robot state",
              "state_keys": ("step", "step_limit", "joint_position", "left_ee_pose", "right_ee_pose"),
              "cameras": ("cam_head", "cam_left_wrist", "cam_right_wrist")},
    "desktop": {"verifier": ("You check a claim about a computer from screenshots of its screen. Answer on the "
                             "first line with exactly YES or NO, then one sentence of evidence from the "
                             "screenshots. Answer NO when the screenshots do not show enough to confirm the claim."),
                "state_label": "Computer state", "state_keys": ("step", "step_limit", "active_window", "screen"),
                "cameras": ("screen",)},
}


@dataclass
class TaskConfig:
    """The driver's configuration for one task: the worker endpoint, the main model and
    the verifier, runtime settings, the prompt, the seeds and the per-episode budgets.
    Experiment.task_config builds one from an experiment."""

    path: Path
    environment: dict
    main: dict
    runtime: dict
    system_prompt: str
    task: str
    seeds: list[int]
    max_model_calls: int
    max_wall_seconds: float
    verifier: dict = field(default_factory=dict)
    # The harness prompt without library text; attach_library builds on it.
    base_prompt: str = ""
    # Task description appended to the first message ("with task description").
    task_description: str = ""
    # Stop the episode once model and verifier calls have cost this much (USD).
    max_cost_usd: float | None = None
    # The resolved agent's lock and identity (config/agent.py), and where its services
    # ran; None for configurations built in code.
    lock: dict | None = None
    agent_id: str | None = None
    placement: dict | None = None
    # The embodiment kind the worker drives (embodiments.KINDS).
    embodiment: str = "robot"

    def __post_init__(self):
        if not self.base_prompt:
            self.base_prompt = self.system_prompt

    def snapshot(self) -> dict:
        """The configuration as episode manifests and experiment.json record it (all but
        `path` and `base_prompt`; the lock as `agent_lock`; the embodiment kind when it is
        not the default robot, so robot records are as before)."""
        return {"environment": self.environment, "agent_version": AGENT_VERSION, "main": self.main,
                "runtime": self.runtime,
                "verifier": self.verifier, "system_prompt": self.system_prompt, "task": self.task,
                "seeds": self.seeds, "max_model_calls": self.max_model_calls,
                "max_wall_seconds": self.max_wall_seconds, "task_description": self.task_description,
                "max_cost_usd": self.max_cost_usd, "agent_id": self.agent_id, "agent_lock": self.lock,
                "placement": self.placement,
                **({"embodiment": self.embodiment} if self.embodiment != "robot" else {})}


# The kernel bootstrap: the embodiment's session with its skills (runtime/episode.py).
DEFAULT_BOOTSTRAP = "harness_policy.runtime.session:create_session"


def layer_settings(runtime: dict) -> dict:
    """The interface layers (R0-R4) a harness switches: learned, reactive, handlers."""
    layers = {"learned": False, "reactive": True, "handlers": True, **runtime.get("layers", {})}
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"Unknown runtime layers: {sorted(unknown)}")
    return layers


def model_services(runtime: dict, main: dict, verifier: dict) -> dict:
    """The models and servers the kernel's skills call: VLM, pointer, segmenter,
    policies, world models, and the success verifier (Tab. 5: it spans layers; direct
    control has none)."""
    return {
        "vlm": runtime.get("vlm"),
        "segmenter": runtime.get("segmenter"),
        "pointer": runtime.get("pointer"),
        "policies": runtime.get("policies", {}),
        "world_models": runtime.get("world_models", {}),
        "mcp": runtime.get("mcp", {}),
        "verifier": ({**main, **verifier}
                     if runtime.get("verifier", True) and not runtime.get("mode") == "direct" else None),
    }


def resolve_skills(runtime: dict, main: dict, verifier: dict, embodiment: str = "robot") -> Resolution:
    """The skills of this configuration that can run on the embodiment kind with its
    services: `runtime.skills` when it names them, else those of its interface layers."""
    requested = runtime.get("skills") or skills_for_layers(layer_settings(runtime), embodiment=embodiment)
    return resolve(requested, available_services(model_services(runtime, main, verifier)),
                   strict=bool(runtime.get("strict_skills", False)), embodiment=embodiment)


def attach_library(config: "TaskConfig", directory, layers=("notes", "skills", "reactive")):
    """Return a copy of `config` that uses the library at `directory` (frozen at its
    current version) on top of the harness prompt without any library."""
    from copy import deepcopy

    from .library import Library

    library = Library(directory, layers)
    updated = deepcopy(config)
    updated.runtime["library"] = {"path": str(library.directory), "layers": list(layers),
                                  "manifest": library.manifest()}
    skills = updated.runtime.get("skills") or resolve_skills(updated.runtime, config.main, config.verifier,
                                                             config.embodiment).enabled
    reactive = "reactive" in skills
    updated.system_prompt = config.base_prompt + library.api_summary(reactive) + library.notes()
    return updated


def _library(runtime: dict, base: Path):
    from .library import Library

    config = runtime["library"]
    return Library((base / config["path"]).resolve() if not Path(config["path"]).is_absolute()
                   else config["path"], config.get("layers", ("notes", "skills", "reactive")))


def load_task_config(path, *, task=None, seeds=None, model=None, deploy=None, sets=()) -> TaskConfig:
    """The driver's configuration of an experiment file (configs/experiments/*.toml) for
    one task: `task` (default: the experiment's first), `seeds`, `model` (a model file's
    name or path, configs/models/), `deploy` and --set items override the file."""
    from ..config.experiments import load_experiment

    experiment = load_experiment(path, deploy=deploy, sets=sets, tasks=[task] if task else None,
                                 seeds=list(seeds) if seeds is not None else None, model=model)
    return experiment.task_config(task, seeds)


def _image_parts(images: dict, cameras) -> list[dict]:
    parts = []
    for camera in cameras:
        if camera in images:
            parts.append({"type": "input_text", "text": f"Camera: {camera}"})
            parts.append({"type": "input_image", "image_url": images[camera], "detail": "auto"})
    return parts


def _state_text(state: dict, embodiment: str = "robot") -> str:
    keys = EMBODIMENT_TEXT[embodiment]["state_keys"]
    return json.dumps({key: state[key] for key in keys if key in state},
                      default=lambda v: round(v, 4) if isinstance(v, float) else v)


def _round(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round(item) for item in value]
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    return value


def _cell_feedback(result: dict, hz: float | None = None, embodiment: str = "robot") -> str:
    lines = [f"Program status: {result['status']} ({result.get('seconds', 0):.1f} s)"]
    if result.get("stdout"):
        lines.append("Output:\n" + result["stdout"].rstrip())
    if result.get("error"):
        lines.append("Error:\n" + result["error"].rstrip())
    trace = result.get("escalation") or result.get("episode_ended")
    if trace:
        event = trace["event"]
        lines.append("Unhandled event: " + json.dumps(_round({
            "kind": event["kind"], "source": event["source"], "detail": event["detail"],
            "skill": trace["skill"], "parameters": trace["parameters"],
            "ticks": trace["end_tick"] - trace["start_tick"]})))
    if result.get("events"):
        executed = [f"{item['skill']}: {item['event']['kind']}" + (" (handled)" if item.get("handled") else "")
                    for item in result["events"]]
        lines.append("Executions: " + "; ".join(executed[-20:]))
    if (result.get("report") or {}).get("text"):
        lines.append(result["report"]["text"])
    if result.get("state"):
        lines.append(f"{EMBODIMENT_TEXT[embodiment]['state_label']}: " + _state_text(result["state"], embodiment))
        if result["state"].get("step") is not None:
            lines.append(_steps_text(result["state"], hz))
    return "\n".join(lines)


def _steps_text(state: dict, hz: float | None) -> str:
    """The env steps used, and left where the episode's limit is reported (Z2 showed both
    every turn): every motion tick counts against the limit."""
    step, limit = int(state["step"]), state.get("step_limit")
    used = f"Env steps used: {step}" + (f" ({hz:g} per second)" if hz else "")
    return used + (f"; left: {int(limit) - step}" if limit else "; the limit is not reported")


def _prune_images(messages: list[dict], keep: int) -> None:
    """Keep images only in the last `keep` user messages."""
    seen = 0
    for message in reversed(messages):
        if message["role"] != "user" or isinstance(message["content"], str):
            continue
        if any(part["type"] == "input_image" for part in message["content"]):
            seen += 1
            if seen > keep:
                message["content"] = [part if part["type"] != "input_image"
                                      else {"type": "input_text", "text": "[earlier image omitted]"}
                                      for part in message["content"]]


def _window(messages: list[dict], turns: int | None) -> list[dict]:
    """The first message (instruction and initial views) plus the last `turns`
    assistant/user exchanges. Program state lives in the kernel, so older turns
    can drop out of the prompt without losing variables."""
    if not turns or len(messages) <= 1 + 2 * turns:
        return messages
    # messages alternate user, assistant, user, ...; keep the alternation.
    recent = messages[-2 * turns:]
    if recent[0]["role"] != "assistant":
        recent = recent[1:]
    omitted = len(messages) - 1 - len(recent)
    first = dict(messages[0])
    note = {"type": "input_text", "text": f"[{omitted} later messages are omitted here; kernel variables persist]"}
    first["content"] = (list(first["content"]) + [note] if isinstance(first["content"], list)
                        else f"{first['content']}\n{note['text']}")
    return [first] + recent


def _cell_usage(summary: dict, result: dict) -> None:
    """Fold what a program did below the high level into the episode summary: events
    its handlers resolved, the one that escalated, and model calls made from inside
    the program (pointing and questions, the success verifier)."""
    summary["events_handled"] += sum(1 for item in result.get("events", []) if item.get("handled"))
    summary["events_escalated"] += 1 if result.get("status") == "escalated" else 0
    for name, usage in (result.get("usage") or {}).items():
        summary["input_tokens"] += int(usage.get("input_tokens") or 0)
        summary["output_tokens"] += int(usage.get("output_tokens") or 0)
        summary["cost_usd"] += float(usage.get("cost_usd") or 0.0)
        if name == "verifier":
            summary["verifier_calls"] += int(usage.get("calls") or 0)
        else:
            summary["vision_calls"] += int(usage.get("calls") or 0)
            summary["vision_cost_usd"] += float(usage.get("cost_usd") or 0.0)


def _usage(total: dict, usage: dict) -> None:
    for key in ("input_tokens", "output_tokens"):
        total[key] += int(usage.get(key) or 0)
    if usage.get("cost") is not None:
        total["cost_usd"] += float(usage["cost"])


def _wrap_up(kernel, trace):
    """The benchmark's end-of-episode convention, the same for every layer: the kernel
    returns the arms to their start pose when the embodiment profile asks for it."""
    try:
        result = kernel.run("__wrap_up__()", timeout=120)
        trace.emit("wrap_up", status=result["status"], error=result.get("error"),
                   events=[event.get("event", {}).get("kind") for event in result.get("events") or []])
    except Exception as error:
        trace.emit("cleanup_error", message=f"wrap_up: {type(error).__name__}: {error}")


def run_program_episode(config: TaskConfig, seed: int, directory: Path, *,
                        chat=None, verifier=None, environment=None) -> dict:
    """Run one episode of `config.task` with `seed` and return its summary (status,
    evaluator outcome, model calls, tokens, cost, times), also written with the trace and
    manifest under `directory`. `chat`, `verifier` and `environment` replace the configured
    models and worker. An error ends the episode with status "error" instead of raising."""
    from .chat import ChatModel

    trace = Trace(directory)
    snapshot = config.snapshot()
    write_json(directory / "manifest.json", {
        "schema_version": 1, "kind": "program", "config": snapshot,
        "config_sha256": fingerprint(snapshot), "seed": seed, "provenance": provenance(),
    })
    runtime = config.runtime
    embodiment = config.embodiment
    told = EMBODIMENT_TEXT[embodiment]
    # Resolved when the configuration was loaded; a config built in code may leave it out.
    skills = runtime.get("skills") or resolve_skills(runtime, config.main, config.verifier, embodiment).enabled
    summary = {"run": str(directory), "task": config.task, "seed": seed, "status": "error",
               "success": False, "reward": None, "process_score": None, "steps": 0,
               "model_calls": 0, "verifier_calls": 0, "cells": 0, "input_tokens": 0,
               "output_tokens": 0, "cost_usd": 0.0, "agent_seconds": 0.0, "kernel_seconds": 0.0,
               # Why the high level was called (Sec. 3.2: at the start and when an event
               # reaches it), and what the lower layers resolved without it.
               "decisions": {}, "events_handled": 0, "events_escalated": 0,
               "vision_calls": 0, "vision_cost_usd": 0.0}
    started = time.perf_counter()
    kernel = None
    token = None
    env = environment
    try:
        chat = chat or ChatModel(config.main)
        if runtime.get("verifier", True):
            verifier = verifier or ChatModel({**config.main, **config.verifier})
        env = env or RemoteEnvironment(config.environment["endpoint"],
                                       timeout=float(config.environment.get("timeout_seconds", 600)))
        env.tools()
        observation = env.reset(config.task, seed)
        episode = observation.episode_id
        token = env.call("claim_outcome", episode_id=episode)["token"]
        # The embodiment profile is part of the frozen configuration (App. A).
        profile = {key: value for key, value in env.call("profile").items()
                   if key not in ("task", "step_limit", "fk_check_error_m")}
        # The skills, prompt and session were chosen for one embodiment kind; the worker
        # says which it drives.
        if embodiment_kind(profile) != embodiment:
            raise RuntimeError(f"The worker drives a {embodiment_kind(profile)} embodiment; this configuration "
                               f"was resolved for a {embodiment} one")
        write_json(directory / "manifest.json", {
            "schema_version": 1, "kind": "program", "config": snapshot,
            "config_sha256": fingerprint(snapshot), "embodiment_profile": profile,
            "embodiment_sha256": fingerprint(profile), "seed": seed, "provenance": provenance(),
        })
        trace.emit("reset", observation=observation.to_dict())
        library = _library(runtime, Path("/")) if runtime.get("library") else None
        if library is not None:
            frozen = runtime["library"]["manifest"]["version"]
            if library.version() != frozen:
                raise RuntimeError(f"Library {library.directory} changed after the configuration was loaded "
                                   f"(expected version {frozen[:12]}); evaluation needs a frozen library")
        kernel = Kernel(runtime.get("bootstrap", DEFAULT_BOOTSTRAP), {
            "endpoint": config.environment["endpoint"], "episode_id": episode, "embodiment": embodiment,
            "reactive_budget_ms": runtime.get("reactive_budget_ms", 5.0),
            "default_max_seconds": runtime.get("default_max_seconds", 30.0),
            "skills": skills,
            **model_services(runtime, config.main, config.verifier),
            "memory": library.memory_entries() if library is not None else [],
            # App. A: programs see only the Python installation and the harness source,
            # and reach only the services named here (runtime/isolation.py).
            "isolation": runtime.get("isolation", True),
            "strategy_first": bool(runtime.get("strategy_first", False)),
            "policy_progress_seconds": float(runtime.get("policy_progress_seconds") or 0.0),
        }, python=runtime.get("python", __import__("sys").executable), log_path=directory / "kernel.log")
        if library is not None:
            code = library.kernel_code("reactive" in skills)
            if code:
                result = kernel.run(code, timeout=120)
                trace.emit("library_loaded", version=frozen, status=result["status"], error=result.get("error"))
                if result["status"] != "ok":
                    raise RuntimeError(f"Library code failed in the kernel: {result.get('error')}")
        cameras = runtime.get("cameras", list(told["cameras"]))
        opening = f"Instruction: {observation.instruction}\n"
        if config.task_description:
            opening += f"\nTask description:\n{config.task_description}\n\n"
        messages = [{"role": "user", "content": [
            {"type": "input_text",
             "text": opening + f"{told['state_label']}: {_state_text(observation.state, embodiment)}"},
            *_image_parts(observation.images, cameras)]}]
        summary["status"] = "budget_exhausted"
        logged = 0
        direct = runtime.get("mode", "program") == "direct"
        # History turns sent to the model; lowered for the rest of the episode after
        # the model server runs out of memory.
        turns = runtime.get("max_history_turns")
        # Index of the first reply without a valid block since the last valid one. Such
        # exchanges leave the history once a valid reply arrives: a model shown its own
        # malformed replies tends to repeat them.
        malformed_from = None
        reason = "start"
        # The model's calls after the first run of the task handed to the learned policy are
        # bounded: finishing with its own tools there took about half of the development
        # run's spend for no score over the policy alone (agent 1.4.0's run, 7 episodes:
        # 12.1 against Simate alone's 10.7 on the same layouts, $0.46 of $0.97). Once they
        # are used up the policy runs on to the end of the episode.
        post_calls = int(runtime.get("post_policy_calls") or 0)
        post_from = None
        for _ in range(config.max_model_calls):
            # 0 means the policy runs on at once: the model's next calls were where a
            # finished handover got grasped again. A positive count is the old bound.
            if post_from is not None and summary["model_calls"] - post_from >= post_calls:
                result = kernel.run("__policy_to_end__()",
                                    timeout=_kernel_timeout("__policy_to_end__()", runtime, config, started))
                trace.emit("policy_to_end", status=result["status"], error=result.get("error"))
                _cell_usage(summary, result)
                if result.get("strategy"):
                    summary["strategy"] = result["strategy"]
                if result["status"] == "episode_ended":
                    summary["status"] = "episode_ended"
                else:
                    trace.emit("finish", text="the model's calls after the policy's run were used up",
                               source="driver")
                    summary["status"] = "finished"
                    _wrap_up(kernel, trace)
                break
            if time.perf_counter() - started > config.max_wall_seconds:
                summary["status"] = "wall_time_exhausted"
                break
            if config.max_cost_usd is not None and summary["cost_usd"] >= config.max_cost_usd:
                summary["status"] = "cost_exhausted"
                break
            _prune_images(messages, int(runtime.get("keep_images", 2)))
            request = _window(messages, turns)
            # Log only messages added since the previous call; the request is the first
            # message plus the last max_history_turns exchanges (older images pruned).
            trace.emit("model_request", call=summary["model_calls"], reason=reason, total_messages=len(messages),
                       sent_messages=len(request), new_messages=messages[logged:],
                       context=context_summary(request, config.system_prompt))
            summary["decisions"][reason] = summary["decisions"].get(reason, 0) + 1
            logged = len(messages)
            summary["model_calls"] += 1
            before = time.perf_counter()
            while True:
                try:
                    reply = chat.complete(config.system_prompt, request)
                    break
                except RuntimeError as error:
                    # A self-hosted model can run out of memory on a long context; retry
                    # with fewer history turns instead of ending the episode.
                    if "out of GPU memory" not in str(error) or len(request) <= 2:
                        raise
                    turns = max(1, (turns or (len(messages) - 1) // 2) - 1)
                    shorter = _window(messages, turns)
                    if len(shorter) >= len(request):
                        raise
                    trace.emit("context_shortened", history_turns=turns, sent_messages=len(shorter))
                    request = shorter
            summary["agent_seconds"] += time.perf_counter() - before
            _usage(summary, reply.usage)
            trace.emit("model_reply", call=summary["model_calls"], text=reply.text, usage=reply.usage,
                       response_id=reply.response_id, seconds=reply.seconds,
                       **({"reasoning": reply.reasoning} if reply.reasoning else {}))
            messages.append({"role": "assistant", "content": reply.text})
            try:
                parsed = parse_reply(reply.text, DIRECT_KINDS if direct else KINDS)
                program = (action_program(parsed.body, allow_policy="vla" in skills,
                                          allow_skills=bool(runtime.get("fixed_skills", False)))
                           if parsed.kind == "action" else parsed.body)
            except (ProtocolError, ActionError) as error:
                blocks = "<action> or <finish>" if direct else "<python>, <verify>, or <finish>"
                if reply.usage.get("finish_reason") == "length":
                    # The reply hit the output limit before its closing tag.
                    text = (f"Your reply was cut off after {reply.usage.get('output_tokens')} tokens, before the "
                            f"closing tag, so nothing ran. Write the program again more compactly: short "
                            f"comments, helper functions for repeated steps.")
                else:
                    text = (f"Format error: {error}. Nothing ran. Reply with the {blocks} block itself, "
                            f"starting with its opening tag.")
                if malformed_from is None:
                    malformed_from = len(messages) - 1
                messages.append({"role": "user", "content": text})
                trace.emit("protocol_error", message=str(error), truncated=reply.usage.get("finish_reason") == "length")
                reason = "format"
                continue
            if malformed_from is not None:
                dropped = len(messages) - 1 - malformed_from
                del messages[malformed_from:-1]
                logged -= dropped
                trace.emit("malformed_dropped", messages=dropped)
                malformed_from = None
            if parsed.kind == "finish":
                trace.emit("finish", text=parsed.body)
                summary["status"] = "finished"
                _wrap_up(kernel, trace)
                break
            if parsed.kind == "verify":
                view = env.call("observe", episode_id=episode, cameras=cameras)
                summary["verifier_calls"] += 1
                if verifier is None:
                    answer_text = "Verifier disabled in this configuration."
                else:
                    before = time.perf_counter()
                    answer = verifier.complete(told["verifier"], [{"role": "user", "content": [
                        {"type": "input_text", "text": f"Claim: {parsed.body}"},
                        *_image_parts(view["images"], cameras)]}])
                    summary["agent_seconds"] += time.perf_counter() - before
                    _usage(summary, answer.usage)
                    answer_text = answer.text.strip()
                trace.emit("verify", claim=parsed.body, answer=answer_text)
                messages.append({"role": "user", "content": [
                    {"type": "input_text", "text": f"Verifier: {answer_text}\n"
                                                   f"{told['state_label']}: {_state_text(view['state'], embodiment)}"},
                    *_image_parts(view["images"], cameras)]})
                reason = "verify"
                continue
            summary["cells"] += 1
            before = time.perf_counter()
            try:
                result = kernel.run(program, timeout=_kernel_timeout(program, runtime, config, started))
            except KernelDied as error:
                summary["kernel_seconds"] += time.perf_counter() - before
                trace.emit("kernel_died", message=str(error))
                raise
            summary["kernel_seconds"] += time.perf_counter() - before
            recorded = {k: v for k, v in result.items() if k not in ("attachments", "report")}
            if result.get("report"):
                recorded["report"] = {"text": result["report"].get("text", ""),
                                      "images": sorted(result["report"].get("images") or {})}
            trace.emit("cell", code=program, action=parsed.body if direct else None, result=recorded,
                       attachments=len(result.get("attachments", [])))
            _cell_usage(summary, result)
            if result.get("strategy"):
                summary["strategy"] = result["strategy"]
                if post_from is None and result["strategy"].get("policy_ran"):
                    post_from = summary["model_calls"]
            if result["status"] == "episode_ended":
                summary["status"] = "episode_ended"
                break
            if result["status"] == "finished":
                trace.emit("finish", text=result.get("finish", ""), source="program")
                summary["status"] = "finished"
                _wrap_up(kernel, trace)
                break
            reason = {"escalated": "escalation", "error": "error"}.get(result["status"], "program_end")
            if direct:
                reason = "action"
            calls_left = config.max_model_calls - summary["model_calls"]
            note = ("\nThe program ended without calling finish(). If the task is done, finish; otherwise "
                    "write a program for the rest of the task." if reason == "program_end" else "")
            if post_from is not None and post_calls:
                calls_left = min(calls_left, post_calls - (summary["model_calls"] - post_from))
                note += ("\nAfter these calls the learned policy runs on by itself to the end of the episode."
                         if calls_left > 0 else "")
            feedback = _cell_feedback(result, profile.get("control_hz"), embodiment)
            content = [{"type": "input_text", "text": feedback + note + f"\nModel calls left: {calls_left}"}]
            for item in result.get("attachments", [])[:6]:
                content.append({"type": "input_text", "text": f"show(): {item['caption']}"})
                content.append({"type": "input_image", "image_url": item["image_url"], "detail": "auto"})
            view = env.call("observe", episode_id=episode, cameras=cameras)
            # A skill's report (the tracked scene) may outline objects in the same views.
            images = {**view["images"], **((result.get("report") or {}).get("images") or {})}
            content.extend(_image_parts(images, cameras))
            messages.append({"role": "user", "content": content})
            if view["state"].get("ended"):
                summary["status"] = "episode_ended"
                break
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        trace.emit("interrupted")
        raise
    except Exception as error:
        summary["status"] = "error"
        summary["error"] = f"{type(error).__name__}: {error}"
        trace.emit("error", message=summary["error"])
    finally:
        if kernel is not None:
            try:
                kernel.close()
            except Exception as error:
                trace.emit("cleanup_error", message=f"kernel: {type(error).__name__}: {error}")
        if token is not None:
            try:
                outcome = env.call("outcome", token=token)
                videos = outcome.pop("videos", None)
                video_error = outcome.pop("video_error", None)
                trace.emit("outcome", outcome=outcome)
                if video_error:
                    trace.emit("cleanup_error", message=f"video: {video_error}")
                if videos:
                    try:
                        trace.emit("episode_video", files=write_episode_videos(directory, videos))
                    except Exception as error:
                        trace.emit("cleanup_error", message=f"video: {type(error).__name__}: {error}")
                summary.update(success=outcome["success"], reward=outcome["reward"],
                               process_score=outcome["process_score"], steps=outcome["steps"])
                if outcome.get("unscorable") and summary["status"] != "interrupted":
                    # The environment dropped this episode (e.g. RoboDojo marked it unstable
                    # mid-way): neither a success nor a failure of the policy.
                    summary["status"] = "error"
                    summary["error"] = f"Unscorable: {outcome['unscorable']}"
                    trace.emit("error", message=summary["error"])
            except Exception as error:
                trace.emit("cleanup_error", message=f"outcome: {type(error).__name__}: {error}")
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(directory / "summary.json", summary)
    return summary


def run_task(config: TaskConfig, output: Path, *, script: list | dict | None = None) -> dict:
    """Run every seed. With `script` (a list of replies, or {"replies", "verifier"}),
    replies are replayed and no model is called."""
    from .chat import scripted_models

    directory = output.resolve() / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    episodes = []
    result = {"experiment": str(directory), "kind": "program", "config": config.snapshot(), "episodes": episodes}
    try:
        for seed in config.seeds:
            scripted = {} if script is None else scripted_models(script)
            episodes.append(run_program_episode(config, seed, directory / f"seed-{seed}", **scripted))
            write_json(directory / "experiment.json", result)
    finally:
        result["success_rate"] = sum(item["success"] for item in episodes) / len(episodes) if episodes else None
        result["completed_cases"] = len(episodes)
        result["planned_cases"] = len(config.seeds)
        write_json(directory / "experiment.json", result)
    return result

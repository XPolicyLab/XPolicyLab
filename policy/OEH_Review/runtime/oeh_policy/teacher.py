"""The review-mode teacher: one persistent GPT conversation per episode, attached to the kernel as an SDK brain.

The conversation follows the paper's hybrid controller (GPT-as-Policy 79f8be5 ``skill/run.py``): developer
instructions = skill + unchanged gate prompt + task/contract context; one tool; every observation packet answers the
previous tool call; the whole history (with the model's encrypted reasoning) is sent back each call. It talks to the
OpenAI Responses API directly with the user's key: the key is read from a file named in the frozen config and is sent
only in the Authorization header. Nothing in this module logs, records or returns it, and provider error bodies are
reduced to their status, type and code (with key-like strings masked) before anything is recorded.

The brain never acts on its own: a valid decision becomes one ``oeh_manipulate`` proposal through ``TurnCtx.propose``,
the admission path every other brain uses. The runtime owns when a proposal may be made (``ReviewState.window``).

Threads: the HTTP call runs in a worker thread; everything that records (events, artifacts) happens on the runtime's
event loop after the call returns, and a closed client neither retries nor sleeps on.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from oeh_sdk import Brain

from .contracts import CAPABILITY, digest, encoded
from .profiles import normalize_benchmark
from .review import ARMS, intent_from_decision, rgb_views, tool_spec, validate_teacher_response
from .takeover import SCHEDULE_FILES
from .task_tree import policy_name

# The teacher's prompt per benchmark: RoboDojo's, the one the formal rounds froze, in teacher/; every other benchmark's
# in teacher/<benchmark>/ (its SKILL.md, teacher_context.md and task_context.json). The paper's gate prompt, and the
# schedule addenda, are shared by all.
PROMPT_DIR = Path(__file__).with_name("teacher")
PROMPT_FILES = ("SKILL.md", "gate_prompt.md", "teacher_context.md")
SHARED_PROMPT_FILES = ("gate_prompt.md",)
# Status codes worth another attempt; everything else is the request's own fault.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
# Errors no retry or fallback can fix: the attempt must stop visibly, not turn into a student-only run.
FATAL_STATUS = {401, 403, 404}
# A 429 carries both rate limits (retry) and exhausted budgets (no retry can help): the error type and code tell them
# apart. Seen 2026-09-28: {"type": "insufficient_quota", "code": "project_spend_limit_exceeded"}.
FATAL_CODES = {"insufficient_quota", "invalid_api_key", "model_not_found", "account_deactivated",
               "project_spend_limit_exceeded", "billing_hard_limit_reached", "billing_not_active"}
FATAL_TYPES = {"insufficient_quota", "authentication_error", "permission_error"}
# The code the runtime reports when the teacher cannot be reached at preflight, before any episode step: the
# evaluator counts that attempt as infrastructure (oeh_eval.runtime_config.PREFLIGHT_UNREACHABLE).
PREFLIGHT_UNREACHABLE = "unreachable_before_episode"
TEACHER_WAKE = "obs/teacher_packet"
WINDOW_CLOSED = "window closed"
# Outcomes of a proposal that lost the call to a harness hold: the same decision is proposed at the next call.
RETRY_OUTCOMES = {"preempted", "cancelled"}


def first_intro(views: int = 3) -> str:
    """The text before the first packet (the Responses, Messages and Chat transports), for ``views`` camera views."""
    return ("Act as the autonomous policy agent for this single simulation rollout. This message is the first "
            "observation packet; answer it with one robodojo_execute call. The user authorizes sending this "
            f"episode's {rgb_views(views)}, end-effector state, the OpenWAM proposal, the task text and same-episode "
            "history to the model for online decisions.")


FIRST_INTRO = first_intro()
# Credentials a provider's error text may echo, masked before anything is recorded: OpenAI/Anthropic/DeepSeek `sk-`
# keys, Google `AIza` keys, Hugging Face, GitHub and xAI tokens, bearer tokens, and `key=`/`token:` style pairs.
KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_*\-]{4,}|AIza[0-9A-Za-z_\-]{20,}|hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}"
                      r"|xai-[A-Za-z0-9]{16,}|(?i:bearer)\s+[A-Za-z0-9._~+/=\-]{8,}"
                      r"|(?i:api[_-]?key|x-api-key|access[_-]?token|token)[\"'\s:=]+[A-Za-z0-9._\-]{16,}")


def mask(text) -> str:
    """``text`` with every key-like string replaced by ``[masked]``."""
    return KEY_LIKE.sub("[masked]", str(text))
PRUNED_VIEW = {"type": "input_text", "text": "[earlier camera view omitted]"}
TEACHER_METHODS = {"codex_app_server": "teacher_codex", "anthropic_messages": "teacher_anthropic",
                   "claude_code": "teacher_claude_code", "openai_chat": "teacher_chat"}


class TeacherUnavailable(RuntimeError):
    """The model could not be reached within the configured attempts."""

    def __init__(self, message, errors=()):
        super().__init__(message)
        self.errors = list(errors)


class TeacherFatal(TeacherUnavailable):
    """An unrecoverable provider, budget or context failure: no fallback may bypass it."""




def prompt_dir(benchmark: str = "robodojo") -> Path:
    """The directory of one benchmark's own prompt files."""
    name = normalize_benchmark(benchmark)
    return PROMPT_DIR if name == "robodojo" else PROMPT_DIR / name


def prompt_files(benchmark: str = "robodojo") -> dict[str, Path]:
    """The developer prompt's files in order: the benchmark's skill, the shared gate prompt, its context."""
    own = prompt_dir(benchmark)
    return {name: (PROMPT_DIR if name in SHARED_PROMPT_FILES else own) / name for name in PROMPT_FILES}


def prompt_text(schedule: str | None = None, benchmark: str = "robodojo", context: dict | None = None) -> str:
    """The developer prompt. ``context`` is the declared manifest (review.context); None, or every switch the
    developer prompt depends on at its off value, gives today's bytes (oeh_policy.review_context)."""
    if context is not None:
        from .review_context import prompt_moved, prompt_text_for
        if prompt_moved(context):
            return prompt_text_for(context, schedule, benchmark)
    skill, gate, context = prompt_files(benchmark).values()
    text = (skill.read_text(encoding="utf-8") + "\n\n# Unchanged baseline gate prompt\n\n"
            + gate.read_text(encoding="utf-8") + "\n\n" + context.read_text(encoding="utf-8"))
    if schedule in SCHEDULE_FILES:
        text += "\n\n" + (PROMPT_DIR / SCHEDULE_FILES[schedule]).read_text(encoding="utf-8")
    return text


def prompt_identity(schedule: str | None = None, benchmark: str = "robodojo", arms: tuple[str, ...] = ARMS,
                    gripper: bool = True, context: dict | None = None) -> dict:
    """The prompt files' hashes (plus every file a context switch adds), the prompt's and the tool's; ``context``
    None, or its file and prompt switches off, gives today's identity."""
    if context is not None:
        from .review_context import identity_moved, prompt_identity_for
        if identity_moved(context):
            return prompt_identity_for(context, schedule, benchmark, arms, gripper)
    files = {name: digest(path.read_text(encoding="utf-8")) for name, path in prompt_files(benchmark).items()}
    files["task_context.json"] = digest(json.loads((prompt_dir(benchmark) / "task_context.json")
                                                   .read_text(encoding="utf-8")))
    if schedule in SCHEDULE_FILES:
        name = SCHEDULE_FILES[schedule]
        files[name] = digest((PROMPT_DIR / name).read_text(encoding="utf-8"))
    return {"files": files, "prompt_sha256": digest(prompt_text(schedule, benchmark)),
            "tool_sha256": digest(tool_spec(arms, gripper, schedule))}


def task_context(task: str | None, benchmark: str = "robodojo", context: dict | None = None) -> dict | None:
    """The first packet's ``task_context``: the benchmark's note on this task (task_context.json ``notes``, keyed by
    the task with any ``_random`` suffix removed) and, where the file lists the tasks exempt from it
    (``no_arm_return``), whether success also needs the arms back at their start poses. ``context`` is the declared
    manifest: its ``task.note`` level (oeh_policy.review_context.task_context_for); None or brief, today's dict."""
    if context is not None:
        from .review_context import note_level, task_context_for
        if note_level(context) != "brief":
            return task_context_for(context, task, benchmark)
    if not task:
        return None
    data = json.loads((prompt_dir(benchmark) / "task_context.json").read_text(encoding="utf-8"))
    base = task[:-len("_random")] if task.endswith("_random") else task
    note = data["notes"].get(base)
    if note is None:
        return None
    context = {"task": base, "process": note}
    if "no_arm_return" in data:
        context["requires_arm_return"] = base not in data["no_arm_return"]
    return context


def image_items(arrays: dict, max_edge: int, directory: Path, prefix: str, *, downsample_k: int = 1) -> tuple[list[dict], list[dict]]:
    """Each camera view as a Responses ``input_image`` (PNG data URL, downscaled with LANCZOS to ``max_edge`` like the
    paper's previews) and its metadata; the PNG is also saved under ``directory``."""
    if type(downsample_k) is not int or downsample_k < 1:
        raise ValueError("image_downsample_k must be a positive integer")
    items, meta = [], []
    for name, array in arrays.items():
        image = Image.fromarray(np.asarray(array, dtype=np.uint8))
        original = image.size
        if downsample_k > 1:
            image = image.resize(tuple(max(1, n // downsample_k) for n in original), Image.Resampling.LANCZOS)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        data = stream.getvalue()
        path = directory / f"{prefix}-{name}.png"
        path.write_bytes(data)
        items.append({"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(data).decode()})
        meta.append({"camera": name, "size": list(image.size), "original_size": list(original),
                     "sha256": hashlib.sha256(data).hexdigest(), "artifact": path.name,
                     "image_downsample_k": downsample_k, "png_bytes": len(data)})
    return items, meta


def read_api_key(path: str) -> str:
    """``NAME=value`` or a bare key in a private file. Errors never include the file's contents."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = (line.split("=", 1)[1] if "=" in line else line).strip().strip("'\"")
        if value:
            return value
    raise ValueError(f"no API key in {path}")


def provider_error(status: int | None, body: bytes | str | None) -> dict:
    """What may be recorded of a provider error: status, type and code; the message only for request errors, with
    key-like strings masked (an authentication error echoes a masked key)."""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else (body or "")
    try:
        error = json.loads(text).get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    record = {"status": status, "type": error.get("type"), "code": error.get("code")}
    if status not in FATAL_STATUS and status != 429:
        record["message"] = mask(str(error.get("message") or text))[:400]
    return record


class ResponsesClient:
    """POST /v1/responses with bounded retries. Blocking: the brain runs it in a worker thread. The thread never
    records anything; attempt errors travel back with the result or the exception."""

    def __init__(self, cfg: dict, key: str | None = None):
        self.cfg = cfg
        self.base = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self._key = key if key is not None else read_api_key(cfg["api_key_file"])
        self.closed = threading.Event()

    def _request(self, path: str, body: dict | None, timeout: float):
        request = urllib.request.Request(self.base + path, method="POST" if body is not None else "GET",
                                         data=None if body is None else json.dumps(body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self._key,
                                                  "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def preflight(self) -> dict:
        """The key can read the frozen model (a model lookup costs no tokens). Raises TeacherFatal otherwise."""
        try:
            data = self._request(f"/models/{self.cfg['model']}", None, 30)
        except urllib.error.HTTPError as exc:
            error = provider_error(exc.code, exc.read()[:2000])
            raise TeacherFatal(f"teacher preflight refused: {error}", [error]) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TeacherUnavailable(f"teacher preflight unreachable: {type(exc).__name__}: {exc}") from None
        if data.get("id") != self.cfg["model"]:
            raise TeacherFatal(f"teacher preflight: the API reports model {data.get('id')!r}")
        return {"model": data.get("id"), "owned_by": data.get("owned_by")}

    def create(self, body: dict) -> tuple[dict, list[dict]]:
        attempts = self.cfg.get("max_attempts", 4)
        delays = self.cfg.get("retry_delays_s") or [5]
        errors = []
        for attempt in range(attempts):
            if self.closed.is_set():
                raise TeacherUnavailable("teacher client closed", errors)
            started = time.monotonic()
            try:
                data = self._request("/responses", body, self.cfg["request_timeout_s"])
                if data.get("status") not in (None, "completed"):
                    raise RuntimeError(f"response status {data.get('status')}: {data.get('incomplete_details')}")
                return data, errors
            except urllib.error.HTTPError as exc:
                error = provider_error(exc.code, exc.read()[:2000])
                fatal = (exc.code in FATAL_STATUS or error.get("code") in FATAL_CODES
                         or error.get("type") in FATAL_TYPES)
                retry = not fatal and exc.code in RETRYABLE_STATUS
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                error = {"status": None, "type": type(exc).__name__,
                         "message": mask(str(exc))[:400]}
                fatal, retry = False, True
            errors.append({**error, "attempt": attempt + 1, "seconds": round(time.monotonic() - started, 3),
                           "retry": retry and attempt + 1 < attempts, "fatal": fatal})
            if fatal:
                raise TeacherFatal(f"teacher refused: {error}", errors)
            if not retry or attempt + 1 == attempts:
                break
            if self.closed.wait(delays[min(attempt, len(delays) - 1)]):
                break
        raise TeacherUnavailable(f"teacher unavailable after {len(errors)} attempt(s): {errors[-1] if errors else ''}",
                                 errors)


def _compact_packet_text(text: str) -> str:
    """An older packet without its per-row proposal table (its diagnostics, state and assessment context stay)."""
    try:
        data = json.loads(text)
    except ValueError:
        return text
    proposal = data.get("student_proposal")
    if isinstance(proposal, dict) and isinstance(proposal.get("table"), list):
        data["student_proposal"] = {**proposal, "table": "[per-row table omitted from an earlier packet]"}
        return json.dumps(data, separators=(",", ":"))
    return text


class Conversation:
    """The persistent input list. When the last call's input passed ``prune_above_tokens``, every packet older than
    the latest ``context_images`` loses its camera views and per-row proposal table at once (one prefix change, then
    the cache resumes until the threshold is passed again), so a long episode stays below the long-context price."""

    def __init__(self, cfg: dict, episode_id: str, prompt: str, tool: dict | None = None):
        self.cfg = cfg
        self.tool = tool if tool is not None else tool_spec()
        self.items: list = [{"role": "developer", "content": prompt}]
        self.cache_key = f"oeh-review-{episode_id}"
        self.pending_call: str | None = None
        self.last_input_tokens = 0
        self.prunes = 0
        self.recorded = 0

    def body(self) -> dict:
        body = {"model": self.cfg["model"], "input": self.items, "tools": [self.tool], "tool_choice": "required",
                "parallel_tool_calls": False, "store": False, "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": self.cache_key, "reasoning": {"effort": self.cfg["reasoning_effort"]}}
        if self.cfg.get("max_output_tokens"):
            body["max_output_tokens"] = self.cfg["max_output_tokens"]
        return body

    def _answer(self, content: list):
        if self.pending_call is None:
            self.items.append({"role": "user", "content": content})
        else:
            self.items.append({"type": "function_call_output", "call_id": self.pending_call, "output": content})
            self.pending_call = None

    def add_packet(self, text: str, images: list[dict], intro: str | None = None):
        head = [{"type": "input_text", "text": intro}] if intro else []
        self._answer(head + [{"type": "input_text", "text": text}, *copy.deepcopy(images)])

    def add_error(self, text: str):
        self._answer([{"type": "input_text", "text": text}])

    def add_output(self, output: list) -> dict | None:
        """Append the model's output items; return its tool call (the last one), if any."""
        call = None
        for item in output:
            if item.get("type") == "reasoning":
                kept = {k: item[k] for k in ("type", "id", "encrypted_content", "summary") if k in item}
            else:
                kept = {k: v for k, v in item.items() if k != "status"}
            self.items.append(kept)
            if item.get("type") == "function_call":
                call = item
        if call is not None:
            self.pending_call = call["call_id"]
        return call

    def unrecorded(self) -> list:
        items, self.recorded = self.items[self.recorded:], len(self.items)
        return items

    def prune(self) -> dict | None:
        if self.last_input_tokens < self.cfg.get("prune_above_tokens", 180_000):
            return None
        keep, seen, views, tables = self.cfg.get("context_images", 6), 0, 0, 0
        for item in reversed(self.items):
            key = "output" if item.get("type") == "function_call_output" else "content"
            content = item.get(key)
            if not isinstance(content, list) or not any(c.get("type") == "input_image" or
                                                        (c.get("type") == "input_text" and '"student_proposal"' in
                                                         c.get("text", "")) for c in content):
                continue
            seen += 1
            if seen <= keep:
                continue
            replaced = []
            for part in content:
                if part.get("type") == "input_image":
                    replaced.append(dict(PRUNED_VIEW))
                    views += 1
                elif part.get("type") == "input_text" and '"student_proposal"' in part.get("text", ""):
                    compact = _compact_packet_text(part["text"])
                    tables += compact != part["text"]
                    replaced.append({**part, "text": compact})
                else:
                    replaced.append(part)
            item[key] = replaced
        if not views and not tables:
            return None
        self.prunes += 1
        self.recorded = min(self.recorded, 1)  # the next call artifact records the whole pruned history once
        return {"views_removed": views, "tables_removed": tables, "kept_packets": keep,
                "input_tokens_before": self.last_input_tokens}


def make_client(cfg: dict, runtime):
    """The frozen transport: the OpenAI Responses API with the key file, the Codex app-server with the listed
    logins (``codex_teacher``), the Anthropic Messages API with the key file (``anthropic_teacher``), Claude Code with
    the listed logins (``claude_code_teacher``), or any OpenAI-compatible Chat Completions endpoint with the key file
    (``chat_teacher``: Gemini, Qwen, DeepSeek, vLLM and the like)."""
    if cfg.get("api") == "anthropic_messages":
        from .anthropic_teacher import AnthropicClient
        return AnthropicClient(cfg)
    if cfg.get("api") == "openai_chat":
        from .chat_teacher import ChatClient
        return ChatClient(cfg)
    if cfg.get("api") == "claude_code":
        from .claude_code_teacher import ClaudeCodeClient
        directory = runtime.output / "teacher"
        directory.mkdir(parents=True, exist_ok=True)
        return ClaudeCodeClient(cfg, workdir=directory, prompt=runtime_prompt(runtime),
                                schema=runtime_tool(runtime)["parameters"])
    if cfg.get("api") == "codex_app_server":
        from .codex_teacher import CodexClient
        directory = runtime.output / "teacher"
        directory.mkdir(parents=True, exist_ok=True)
        return CodexClient(cfg, workdir=directory, prompt=runtime_prompt(runtime))
    return ResponsesClient(cfg)


def runtime_limits(runtime):
    from .review import DEFAULT_LIMITS
    return {**DEFAULT_LIMITS, **runtime.cfg.get("review",{}).get("limits",{})}


def execution_contract_prompt(prompt, runtime):
    limits=runtime_limits(runtime)
    # Replace illustrative RoboDojo limits with the frozen runtime limits, once per thread.
    for mode,key in [("student","student_max_steps"),("edit","correction_max_steps"),("eef","correction_max_steps")]:
        prompt=re.sub(r"(`"+mode+r"`: (?:execute the first )?)1[–-]\d+",lambda m:m[1]+"1–"+str(limits[key]),prompt)
    return prompt + ("\n# Active execution contract\n"
        f"steps per mode: student=1..{limits['student_max_steps']} (also limited by candidate rows), "
        f"edit=1..{limits['correction_max_steps']} (also limited by candidate rows), "
        f"eef=1..{limits['correction_max_steps']}, home=1..{limits['home_max_steps']}. "
        "These configured bounds override illustrative defaults. A request outside its mode/candidate bound is rejected and executes zero rows; submit a corrected request. Valid requests execute exactly their requested steps with no padding. Other invalid fields execute zero rows. "
        "task_started=false means no non-hold task action has executed: report not_started even after initialization holds or query retries. "
        "A hold may advance simulation time and native_step without starting the task. task_started=true means use another execution status.\n")


def runtime_prompt(runtime) -> str:
    """Compatibility entry point; context assembly belongs to ContextController."""
    from .context_controller import initial_prompt
    return initial_prompt(runtime)


def runtime_tool(runtime) -> dict:
    """The tool of a runtime's teacher, for its review method's arms (and whether they have a gripper), its
    schedule, and under review-stmem the memory tool."""
    original = tool_spec(runtime.method.arms, getattr(runtime.method, "gripper", True),
                         runtime.cfg["review"].get("schedule"))  # the segment schedule's tool takes next_look (G13)
    if runtime.cfg.get("mode") != "review-stmem":
        return original
    limits=runtime_limits(runtime)
    steps=original["parameters"]["properties"]["steps"]
    steps.update(minimum=1,
        description=f"student 1..{limits['student_max_steps']}; edit/eef 1..{limits['correction_max_steps']}; home 1..{limits['home_max_steps']}. Student/edit are also capped by candidate rows. Out-of-range steps are rejected without execution; valid steps execute as requested, never padded.")
    if runtime.cfg.get("mode") == "review-stmem":
        from .stmem_runtime import tool
        is_tree = policy_name(runtime.cfg.get("semmem")) == "hierarchical-policy"
        return tool(original, task_tree=is_tree)
    return original


def make_conversation(cfg: dict, episode_id: str, prompt: str, tool: dict | None = None, views: int = 3):
    """The transport's conversation. ``tool`` is the teacher's tool (default RoboDojo's, ``tool_spec()``); ``views``
    the camera views every packet carries (the Codex and Claude Code transports name them in their first text)."""
    if cfg.get("api") == "anthropic_messages":
        from .anthropic_teacher import AnthropicConversation
        return AnthropicConversation(cfg, episode_id, prompt, tool)
    if cfg.get("api") == "openai_chat":
        from .chat_teacher import ChatConversation
        return ChatConversation(cfg, episode_id, prompt, tool)
    if cfg.get("api") == "claude_code":
        from .claude_code_teacher import ClaudeCodeConversation
        return ClaudeCodeConversation(cfg, episode_id, prompt, tool, views)
    if cfg.get("api") == "codex_app_server":
        from .codex_teacher import CodexConversation
        return CodexConversation(cfg, episode_id, prompt, tool, views)
    return Conversation(cfg, episode_id, prompt, tool)


class TeacherBrain(Brain):
    """Wakes on ``obs/teacher_packet``; answers each packet with one admitted decision."""

    def __init__(self, runtime, client: ResponsesClient | None = None):
        super().__init__("teacher", {"default": "ignore", "rules": {TEACHER_WAKE: "wake", "safety/estop": "interrupt"},
                                     "budget": {"max_consecutive_wakes": 100000, "min_interval_ms": 0}})
        self.runtime = runtime
        self.cfg = runtime.cfg["review"]["teacher"]
        self.client = client if client is not None else make_client(self.cfg, runtime)
        self.prompt = runtime_prompt(runtime)
        held = getattr(self.client, "prompt", None)
        if isinstance(held, str) and held != self.prompt:
            # A client started before the runtime (codex_teacher.prestart, from model.Model) holds the prompt it was
            # built with; under a declared context manifest that is not this prompt. Its thread opens at the first
            # call, so it takes this one; a client whose thread is already open cannot, and the start fails.
            if getattr(self.client, "thread_id", None) is not None:
                raise ValueError("the teacher client's conversation was opened with different instructions (a "
                                 "different developer prompt)")
            self.client.prompt = self.prompt
        self.arms = runtime.method.arms
        self.gripper = getattr(runtime.method, "gripper", True)
        self.tool = runtime_tool(runtime)
        self.views = len(runtime.contract.review_cameras)  # every packet carries these views
        self.conversation = make_conversation(self.cfg, runtime.episode_id, self.prompt, self.tool, self.views)
        self.calls = self.failed_calls = self.invalid = 0
        self.tokens = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
        self.working_set={"phase":"unknown","entity_ids":[],"reason":"Awaiting LLM selection"}
        self.cost = 0.0
        self._input_history_tokens = 0
        self.input_ledger = None
        if self.cfg.get("input_context") and self.cfg["input_context"].get("enforce_budget", True) and not runtime.cfg["review"].get("llm_token_budget"):
            from .input_budget import InputBudget
            self.input_ledger = InputBudget(runtime.output/"input-budget.json", self.cfg["input_context"])
        if runtime.cfg["review"].get("llm_token_budget"):
            from .llm_budget import BudgetedClient,EpisodeLLMBudget
            self.client=BudgetedClient(self.client,EpisodeLLMBudget(runtime.output/"llm-token-budget.json",runtime.cfg["review"]["llm_token_budget"]),"main",self.prompt,self.tool)
        self.directory = runtime.output / "teacher"
        self.directory.mkdir(exist_ok=True)
        (self.directory / "prompt.md").write_text(self.prompt, encoding="utf-8")
        (self.directory / "tool.json").write_bytes(encoded(self.tool))

    @property
    def context_controller(self):
        if not hasattr(self, "_context_controller"):
            from .context_controller import BaseContextController
            self._context_controller = BaseContextController(self.runtime)
        return self._context_controller

    @property
    def memory_replies(self):
        return self.context_controller.replies

    @memory_replies.setter
    def memory_replies(self, value):
        self.context_controller.replies = value

    @property
    def semmem_evidence(self):
        return self.context_controller.evidence

    @semmem_evidence.setter
    def semmem_evidence(self, value):
        self.context_controller.evidence = value

    @property
    def task_tree_requires_recovery(self):
        return self.context_controller.requires_recovery

    @task_tree_requires_recovery.setter
    def task_tree_requires_recovery(self, value):
        self.context_controller.requires_recovery = value

    @property
    def task_tree_discard_response(self):
        return self.context_controller.discard_response

    @task_tree_discard_response.setter
    def task_tree_discard_response(self, value):
        self.context_controller.discard_response = value

    def close(self):
        self.client.closed.set()
        stop = getattr(self.client, "close", None)  # the Codex and Claude Code transports own a process
        if stop is not None:
            stop()

    def summary(self) -> dict:
        return {"model": self.cfg["model"], "calls": self.calls, "failed_calls": self.failed_calls,
                "invalid_or_refused": self.invalid, "tokens": dict(self.tokens), "cost_usd_estimate": round(self.cost, 4),
                "context_prunes": self.conversation.prunes}

    async def on_turn(self, ctx):
        state = self.runtime.review
        packet = state.packet
        request_id = packet["request_id"] if packet else None
        try:
            # An on_stall boundary where OpenWAM runs alone holds an `auto` packet: never the teacher's to answer.
            if packet is not None and not packet.get("auto") and not state.closed(request_id):
                await self._decide(ctx, packet, state.images)
        except asyncio.CancelledError:
            raise
        except TeacherFatal as exc:
            self.runtime.emit("teacher_fatal", {"request_id": request_id, "error": str(exc)[:800]})
            state.fatal = str(exc)[:800]
            state.fatal_error = exc.errors[-1] if exc.errors else None  # the provider's own status, type and code
            state.give_up(request_id, "fatal")
        except Exception as exc:  # the teacher gives up on this packet; the runtime falls back
            self.runtime.emit("teacher_failed", {"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"[:800]})
            state.give_up(request_id, type(exc).__name__)
        finally:
            try:
                await ctx.done("decided")
            except Exception:
                pass

    async def _decide(self, ctx, packet: dict, images: list[dict]):
        state = self.runtime.review
        from .task_tree_runtime import ensure_tree
        ensure_tree(self.runtime, packet)
        projection = self.runtime.cfg.get("stmem") if self.runtime.cfg.get("stmem", {}).get("context_max_entities") else None
        reset_every = 0 if getattr(self.runtime, "stmem", None) is not None else self.cfg.get("context_reset_every", 0)
        if reset_every and packet["decision"] and packet["decision"] % reset_every == 0:
            reset = getattr(self.client, "reset_context", None)
            if self.cfg["api"] == "codex_app_server" and reset is None:
                raise TeacherUnavailable("Codex client cannot reset bounded semantic context")
            if reset:
                reset()
            self.conversation = make_conversation(self.cfg, self.runtime.episode_id, self.prompt, self.tool, self.views)
            self._input_history_tokens = 0
            self.task_tree_requires_recovery = True
            self.runtime.emit("teacher_context_reset", {"decision":packet["decision"],
                "reason":"bounded_semantic_checkpoint", "semantic_context":packet.get("semantic_context")})
        self.context_controller.add_observation(self, packet, images,
            first_intro(self.views) if packet["decision"] == 0 or
            reset_every and packet["decision"] % reset_every == 0 else None)
        refusals = 0
        queries = 0
        recovery_retries = 0
        while not state.closed(packet["request_id"]):
            call = await self._call(packet)
            if state.closed(packet["request_id"]):
                return  # answered too late: the harness has moved on; the next packet answers this call
            if getattr(self, "task_tree_discard_response", False):
                self.task_tree_discard_response = False
                recovery_retries += 1
                if recovery_retries >= self.cfg.get("max_invalid", 6):
                    raise TeacherFatal("repeated context loss prevents complete memory context recovery",
                        [{"code": "context_recovery_exhausted", "fatal": True}])
                continue  # Full tree + queues + current views must be delivered before acting.
            problem = None
            task_binding = None
            task_attempt_id = None
            if call is None:
                problem = {"error": "answer with exactly one robodojo_execute call", "no_execution": True}
            else:
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                    if getattr(self.runtime, "stmem", None) is not None:
                        from .stmem_events import validate_working_set
                        raw=getattr(self.runtime.stmem,"last_context",None) or packet["spatial_context"]
                        if arguments.get("stmem_working_set") is not None:
                            self.working_set=validate_working_set(arguments["stmem_working_set"],raw)
                            writer=getattr(self.runtime.stmem,"set_working_set",None)
                            if writer is not None:await writer(self.working_set,packet["sim_time_s"],packet.get("observation_id"))
                            self.runtime.emit("stmem_working_set",{"observation_id":packet.get("observation_id"),"request_id":packet["request_id"],**self.working_set})
                        if arguments.get("action") is None:
                            queries+=1
                            if queries>self.runtime.cfg["stmem"]["max_query_calls"]:
                                raise ValueError("auxiliary-only budget exhausted; choose an execution action")
                            if arguments.get("memory_update") is None and not arguments.get("memory_query") and arguments.get("stmem_working_set") is None and arguments.get("semmem_update") is None:
                                raise ValueError("action=null requires at least one auxiliary operation")
                        call_id = call.get("call_id", f"teacher:{getattr(self, 'calls', 0)}")
                        from .task_tree_runtime import update as task_update
                        task_update(self, arguments, packet, call_id)
                        task_binding = arguments.get("task_binding")
                        task_attempt_id = call_id
                        from .stmem_updates import adjuncts
                        await adjuncts(self,arguments,packet,call.get("call_id",f"teacher:{getattr(self,'calls',0)}"))
                        arguments = arguments["action"]
                        if arguments is None:
                            continue
                        if not isinstance(arguments, dict):
                            raise ValueError("action must be an object")
                        if arguments.get("kind") == "memory_query":
                            queries += 1
                            if queries > self.runtime.cfg["stmem"]["max_query_calls"]:
                                raise ValueError("spatial query budget exhausted; choose an execution action")
                            result = await self.runtime.stmem.query(arguments, packet["sim_time_s"])
                            query_images = result.pop("_image_items", [])
                            if "spatial_context" in result:
                                packet["spatial_context"] = result["spatial_context"]
                            self.runtime.emit("stmem_query", {"query": arguments, "result": result})
                            self.context_controller.add_legacy_query_reply(
                                self, packet, result, query_images, projection)
                            continue
                    decision = validate_teacher_response(arguments, packet, self.runtime.method.limits,
                                                         final_home_steps=self.runtime.final_home_steps,
                                                         arms=self.arms, gripper=self.gripper)
                    from .task_tree_runtime import validate_binding
                    validate_binding(self.runtime, task_binding, decision)
                except (ValueError, TypeError, KeyError) as exc:
                    self.runtime.emit("teacher_invalid", {"request_id": packet["request_id"], "error": str(exc)[:800]})
                    problem = {"error": str(exc), "error_type": "recoverable_tool_input", "no_execution": True}
                else:
                    # The review guard (oeh_policy.review_guard), when set, refuses before admission.
                    refusal = self.runtime.review_guard_refusal(packet, decision, arguments.get("assessment"))
                    if refusal is not None:
                        problem = {"error": refusal, "error_type": "review_guard", "no_execution": True}
                    else:
                        tree = getattr(self.runtime, "task_tree", None)
                        if tree is not None and decision["mode"] != "hold":
                            from .task_tree_runtime import source
                            try:
                                tree.record_attempt(task_attempt_id, task_binding,
                                    source=source(packet, task_attempt_id), action=decision)
                            except ValueError as exc:
                                self.context_controller.add_feedback(self, {"error": str(exc), "no_execution": True})
                                self.invalid += 1
                                if self.invalid >= self.cfg.get("max_invalid", 6):
                                    raise RuntimeError("repeated task action attempt")
                                continue
                        self.runtime.task_action_binding = copy.deepcopy(task_binding)
                        self.task_proposal_result = None
                        try:
                            refusal = await self._propose(ctx, packet, decision, arguments.get("assessment"))
                        finally:
                            self.runtime.task_action_binding = None
                        if tree is not None and decision["mode"] != "hold":
                            if self.task_proposal_result is not None:
                                tree.record_result(task_attempt_id, self.task_proposal_result)
                            elif refusal is not None:
                                tree.record_result(task_attempt_id, {"status": "admission_refused", "reason": refusal})
                            # A closed packet without an execution result leaves the attempt unknown.
                        if refusal is None:
                            return
                        problem = {"error": refusal, "error_type": "oeh_admission_refused", "no_execution": True}
            refusals += 1
            self.invalid += 1
            if refusals >= self.cfg.get("max_invalid", 6):
                raise RuntimeError(f"{refusals} invalid or refused decisions for one packet")
            problem.update(request_id=packet["request_id"],
                           correction="Submit a corrected robodojo_execute call for the same packet.")
            self.context_controller.add_feedback(self, problem)

    async def _call(self, packet: dict) -> dict | None:
        runtime = self.runtime
        pruned = None if getattr(runtime,"stmem",None) is not None else self.conversation.prune()
        if pruned:
            runtime.emit("teacher_context_pruned", {"request_id": packet["request_id"], **pruned})
        self.calls += 1
        request_id = f"teacher:{packet['request_id']}:{self.calls}"
        notification = self.context_controller.prepare_turn(self, packet, pruned=pruned)
        body = self.conversation.body()
        max_chars = self.cfg.get("max_input_chars")
        if max_chars:
            chars = len(json.dumps(body, ensure_ascii=True, separators=(",", ":")))
            runtime.emit("teacher_input_budget", {"request_id":request_id, "serialized_chars":chars,
                "max_chars":max_chars,
                "image_downsample_k":self.cfg.get("image_downsample_k",1)})
            if chars > max_chars:
                self.failed_calls += 1
                runtime.emit("teacher_call_error", {"request_id":request_id, "code":"local_input_budget", "retry":False})
                raise TeacherUnavailable("teacher input exceeds local semantic input budget", [{"code":"local_input_budget", "retry":False}])
        reservation = None
        if self.cfg.get("input_context"):
            from .input_context import input_estimate
            from .input_budget import InputBudgetExceeded
            try:
                estimate = input_estimate(body,self.prompt,self.tool,self.cfg["input_context"])
            except Exception as exc:
                raise TeacherFatal("input estimator unavailable",[{"code":"input_estimator_unavailable","fatal":True}]) from exc
            if body.get("transport")=="codex_app_server":
                estimate["estimated_input"] += int(getattr(self,"_input_history_tokens",0)*self.cfg["input_context"]["tokenizer_multiplier"])
            runtime.emit("teacher_input_attribution",estimate)
            try:
                reservation = self.input_ledger.reserve("main",estimate["estimated_input"]) if self.input_ledger is not None else None
            except InputBudgetExceeded as exc:
                raise TeacherFatal(str(exc),[{"code":"input_budget_exhausted","fatal":True}]) from exc
        path = self.directory / f"call-{self.calls:04d}.json"
        # The input is the previous calls' input and output plus these new items (camera views as the sha256 of the
        # PNG saved beside their packet, encrypted reasoning as its hash); after a prune, the whole pruned history.
        artifact = {"request_id": request_id, "packet_request_id": packet["request_id"],
                    "started_monotonic_ns":time.monotonic_ns(), "started_utc":time.time(),
                    "full_input_sha256": digest(body), "input_items": len(body["input"]),
                    "new_items_from": self.conversation.recorded,
                    "new_items": redact(self.conversation.unrecorded()), "model": body["model"],
                    "reasoning": body["reasoning"], "prompt_cache_key": body["prompt_cache_key"],
                    "tool_sha256": digest(body["tools"])}
        if notification is not None:
            artifact["stmem_notification"] = notification
        path.write_bytes(encoded(artifact))
        runtime.emit("model_call", {"model_id": self.cfg["model"], "request_id": request_id,
                                    "input_sha256": digest(artifact), "input_artifact": str(path.relative_to(runtime.output)),
                                    "full_input_sha256": artifact["full_input_sha256"],
                                    "method": TEACHER_METHODS.get(self.cfg.get("api"), "teacher_responses")})
        started = time.monotonic_ns()
        try:
            from .llm_budget import LLMTokenBudgetExceeded
            scene=packet.get('scene_frame')
            if scene and getattr(runtime.stmem,'scene_frames',None):
                runtime.stmem.scene_frames.delivered(scene['frame_id'],request_id)
            data, errors = await asyncio.to_thread(self.client.create, body)
        except LLMTokenBudgetExceeded as exc:
            runtime.emit("llm_token_budget_stop",{"channel":"main","reason":str(exc)})
            raise TeacherFatal(str(exc),[{"code":"llm_token_budget_exceeded","fatal":True}]) from exc
        except TeacherUnavailable as exc:
            self.failed_calls += 1
            if not runtime.closed:
                for error in exc.errors:
                    runtime.emit("teacher_call_error", {"request_id": request_id, **error})
                runtime.emit("model_failure", {"model_id": self.cfg["model"], "request_id": request_id,
                                               "reason": str(exc)[:800]})
                runtime.emit("model_timing", {"request_id": request_id,
                                              "duration_ms": (time.monotonic_ns() - started) / 1e6})
            raise
        if runtime.closed:
            raise TeacherUnavailable("episode closed during the call")
        self.context_controller.on_delivery(data)
        for error in errors:
            runtime.emit("teacher_call_error", {"request_id": request_id, **error})
        runtime.emit("model_timing", {"request_id": request_id, "duration_ms": (time.monotonic_ns() - started) / 1e6})
        usage = data.get("usage") or {}
        if reservation is not None:
            self.input_ledger.settle(reservation,usage)
        if self.cfg.get("input_context"):
            from .input_context import text_tokens
            output = data.get("output",[])
            # Codex retains the raw agent JSON, not our escaped Responses-style log wrapper.
            retained_output = "\n".join(item.get("arguments", "") for item in output
                                        if item.get("type") == "function_call") if body.get("transport") == "codex_app_server" else output
            self._input_history_tokens += estimate["parts"]["current_input"] + estimate["parts"]["images_estimate"] + text_tokens(retained_output)
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        self.conversation.last_input_tokens = usage.get("input_tokens") or 0
        prices = self.cfg.get("prices_usd_per_million")
        cost = None
        if prices and usage:
            # Anthropic bills cache writes apart (``cache_write``); without that price they count as plain input.
            written = (usage.get("input_tokens_details") or {}).get("cache_creation_tokens") or 0
            write_price = prices.get("cache_write", prices["input"])
            cost = ((usage.get("input_tokens", 0) - cached - written) * prices["input"] + written * write_price
                    + cached * prices["cached_input"] + usage.get("output_tokens", 0) * prices["output"]) / 1e6
            self.cost += cost
        self.tokens["input"] += usage.get("input_tokens") or 0
        self.tokens["cached_input"] += cached
        self.tokens["output"] += usage.get("output_tokens") or 0
        self.tokens["reasoning"] += reasoning or 0
        runtime.emit("usage", {"model_id": self.cfg["model"], "request_id": request_id,
                               "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                               "cached_input_tokens": cached, "reasoning_output_tokens": reasoning,
                               "call_complete": True, "total_tokens": usage.get("total_tokens"), "cost_usd": cost,
                               "cost_source": "list-price estimate" if cost is not None else None,
                               "source": "provider_reported"})
        output = data.get("output") or []
        call = self.conversation.add_output(output)
        self.conversation.recorded = len(self.conversation.items)
        (self.directory / f"call-{self.calls:04d}-output.json").write_bytes(encoded({
            "request_id": request_id, "response_id": data.get("id"), "usage": usage, "output": redact(output),
            "completed_monotonic_ns":time.monotonic_ns(), "completed_utc":time.time(),
            **{key: data[key] for key in ("stop", "codex", "claude_code") if data.get(key)}}))
        return call

    async def _propose(self, ctx, packet: dict, decision: dict, assessment) -> str | None:
        """Propose once the harness is waiting for a decision. None once executed or moot, else the refusal text."""
        runtime, state = self.runtime, self.runtime.review
        intent = intent_from_decision(decision, packet["observation_id"], packet["candidate_id"])
        state.assessments[packet["request_id"]] = {"assessment": assessment, "trigger": decision["trigger"]}
        state.waiting.add(packet["request_id"])
        try:
            while True:
                await state.wait_window(packet["request_id"])
                if state.closed(packet["request_id"]):
                    return None
                runtime.emit("teacher_decision", {"request_id": packet["request_id"], "intent": intent,
                                                  "trigger": decision["trigger"]})
                outcome = await ctx.propose(CAPABILITY, intent, timeout=runtime.cfg["review"]["turn_seconds"])
                self.task_proposal_result = copy.deepcopy(outcome)
                if outcome.get("outcome") == "executed":
                    return None
                reason = str(outcome.get("reason") or outcome.get("error") or outcome.get("outcome"))
                if WINDOW_CLOSED in reason or outcome.get("outcome") in RETRY_OUTCOMES:
                    continue  # a hold took this call; the same decision is proposed at the next one
                return reason
        finally:
            state.waiting.discard(packet["request_id"])


def redact(items):
    """Artifacts keep text and structure; image bytes and encrypted reasoning become their hashes."""
    def walk(value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key in ("image_url", "url") and isinstance(item, str) and item.startswith("data:image/png;base64,"):
                    data = base64.b64decode(item.split(",", 1)[1])
                    out[key] = {"png_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                elif (key == "data" and isinstance(item, str) and value.get("type") == "base64"
                      and str(value.get("media_type", "")).startswith("image/")):  # a Messages API image
                    data = base64.b64decode(item)
                    out[key] = {"png_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                elif key in ("image_url", "encrypted_content", "signature") and isinstance(item, str):
                    out[key] = {"sha256": digest({"value": item}), "chars": len(item)}
                else:
                    out[key] = walk(item)
            return out
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value
    return walk(copy.deepcopy(items))

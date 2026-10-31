"""One episode, from the first observation to the verdict.

The model works through two tools: ``python`` runs a cell in the sandboxed worker (the cell's
``robot.*`` calls come back here and are answered by the session, which drives the robot) and
``finish`` stops the episode.  After every reply its cells run in order; the pictures they asked
for with ``robot.look`` follow in one ``user`` message.  The loop ends when the benchmark ends the
episode (success or the step budget), when the model calls ``finish``, or at a limit (replies,
time, the gateway giving up); the scene is then judged as it stands.

What depends on the benchmark (observations, the ``robot`` calls, the task card, the verdict) is
the :class:`~hexashell.runtime.session.Session`'s; RoboDojo's is built by :func:`play`.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..kernel.host import Kernel
from . import card as cards
from .conversation import Conversation
from .llm import AnthropicGateway, Gateway, GatewayError
from .records import Records
from .session import Session

#: Longest tool result kept in the conversation (characters); the middle of a longer one is cut.
TOOL_TEXT_MAX = 16_000
#: Pictures one turn may add (the latest are kept).
PICTURES_PER_TURN = 8

TOOLS = [
    {"type": "function", "function": {
        "name": "python",
        "description": ("Run Python code in your persistent session (`robot`, `np` and `math` are defined; "
                        "variables stay between calls). Returns what the code printed, the value of a final "
                        "expression, and any error. Pictures from robot.look() follow the result."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": ("End the episode now; the scene is judged as it is. The episode ends by itself the moment "
                        "the task is complete, so only call this when nothing you can still do would raise the "
                        "score."),
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["no_budget", "stuck", "keep_partial"],
                       "description": ("no_budget: what is left cannot cover another attempt; stuck: no way "
                                       "forward; keep_partial: more attempts would risk the partial result.")},
            "summary": {"type": "string", "description": "What was achieved and what is missing."}},
            "required": ["status", "summary"]}}},
]

NUDGE = ("The episode is still running, so the task is not complete yet (it ends by itself the moment it is). "
         "Continue with the python tool; call finish only if nothing you can still do would raise the score.")


@dataclass
class Settings:
    #: "openai" (chat completions) or "anthropic" (Messages API, for Claude)
    api: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    api_key_file: str = "~/.config/hexashell/api-key"
    #: a private CA for the gateway's certificate (empty: the usual ones)
    ca_file: str = ""
    model: str = "gpt-6-astra"
    reasoning_effort: str = "high"
    #: Anthropic only: what the thinking blocks show ("summarized" or "omitted")
    thinking_display: str = "summarized"
    max_tokens: int = 8192
    skills: bool = False
    ladder: bool = True
    max_turns: int = 100
    budget_s: float = 5400.0
    nudges: int = 3
    #: the picture window (runtime/conversation.py): at most images_max pictures and images_bytes_max bytes of
    #: data URIs (as sent) per request; beyond either the oldest go, down to images_low and 70% of the bytes
    images_max: int = 90
    images_low: int = 60
    images_bytes_max: int = 10_000_000
    jpeg_quality: int = 90
    cell_cpu_s: float = 120.0
    cell_idle_s: float = 150.0
    cell_total_s: float = 1800.0
    sam3_url: str = ""
    request_timeout_s: float = 900.0
    #: stream the replies (the gateway allows each user only a few non-streaming requests a minute)
    stream: bool = True
    #: only for a self-hosted Qwen on vLLM (api openai): send each reply's reasoning back with it in the
    #: history (a "reasoning" key, which vLLM hands to the chat template as reasoning_content), as a model
    #: trained on whole conversations with their thinking expects.  Off by default; keep it off for
    #: gpt-6-astra and other hosted models (only short reasoning titles, unused; a gateway may refuse the key).
    history_reasoning: bool = False
    #: motion speed profile (motion/ik.py SPEEDS): "v1" every action at the end limits, as in R0
    motion: str = "v1"
    #: a file of cells to run instead of asking the model (see runtime/script.py); {task}, {seed} and
    #: {layout} in the path are filled in per episode
    script: str = ""

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Settings":
        """deploy.yml values, overridden by HEXASHELL_<NAME> environment variables."""
        out = cls()
        for f in fields(cls):
            value = cfg.get(f.name, getattr(out, f.name))
            env = os.environ.get("HEXASHELL_" + f.name.upper())
            if env is not None:
                value = env
            if value is None:
                continue
            kind = type(getattr(out, f.name))
            if kind is bool and isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            setattr(out, f.name, kind(value))
        return out

    def api_key(self) -> str:
        path = Path(self.api_key_file).expanduser() if self.api_key_file else None
        if path is not None and path.exists():
            return path.read_text().strip()
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(f"no API key: {path} does not exist and OPENAI_API_KEY is not set")
        return key


def play(bridge, info: dict[str, Any], cfg: dict[str, Any], result_dir: Path) -> None:
    """A RoboDojo episode: XPolicyLab's observations arrive through ``bridge``."""
    from ..xpl.session import RoboDojoSession

    Episode(RoboDojoSession(bridge, info), Settings.from_config(cfg), Path(result_dir)).run()


class Episode:
    def __init__(self, session: Session, settings: Settings, result_dir: Path):
        self.session = session
        self.info = session.info
        self.s = settings
        self.rec = Records(result_dir, session.info, name=session.record_name())
        self.t0 = time.monotonic()
        # the budget is the agent's own time, from when the episode is open (a simulator may take minutes
        # to start): what gateway failures cost is given back, up to one more budget, so this is the
        # latest end
        self.t_agent = self.t0
        self.open_s: float | None = None
        self.deadline = self.t0 + 2 * settings.budget_s
        self.conv: Conversation | None = None
        self.gateway: Gateway | None = None
        self.kernel: Kernel | None = None
        self.turn_pictures: list = []
        self.turns = 0
        self.cells = 0
        self.tokens = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
        self.robot_calls: dict[str, int] = {}
        self.end = "unknown"
        self.finish: dict[str, str] | None = None
        self.errors: list[str] = []
        self.used: dict[str, Any] = {}

    # -- the whole episode -----------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._setup()
            self._loop()
        except Exception as exc:  # noqa: BLE001 - recorded; the scene is judged as it stands
            self.end = "error"
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self.rec.event({"kind": "error", "traceback": traceback.format_exc()})
        finally:
            self._wrap_up()

    def _setup(self) -> None:
        session, s = self.session, self.s
        session.open(s, self.rec, self._attach)
        self.t_agent = time.monotonic()
        self.open_s = round(self.t_agent - self.t0, 1)
        self.deadline = self.t_agent + 2 * s.budget_s
        self.conv = Conversation(cards.system_prompt(session.prompt_folder(), env_icl=session.env_icl),
                                 images_max=s.images_max, images_low=s.images_low,
                                 bytes_max=s.images_bytes_max, quality=s.jpeg_quality)
        if s.script:
            from .script import ScriptGateway

            self.gateway = ScriptGateway(s.script.format(**session.labels()), log=self.rec.event)
        elif s.api == "anthropic":
            self.gateway = AnthropicGateway(s.api_base, s.api_key(), s.model, display=s.thinking_display,
                                            reasoning_effort=s.reasoning_effort, max_tokens=s.max_tokens,
                                            request_timeout_s=s.request_timeout_s, stream=s.stream, log=self.rec.event,
                                            ca_file=s.ca_file or None)
        elif s.api == "openai":
            self.gateway = Gateway(s.api_base, s.api_key(), s.model, reasoning_effort=s.reasoning_effort,
                                   max_tokens=s.max_tokens, request_timeout_s=s.request_timeout_s, stream=s.stream,
                                   log=self.rec.event, ca_file=s.ca_file or None)
        else:
            raise ValueError(f"unknown api {s.api!r}: use openai or anthropic")
        self.kernel = Kernel(self.rec.workspace, session.dispatch, cpu_s=s.cell_cpu_s, idle_s=s.cell_idle_s,
                             total_s=s.cell_total_s, proxy=session.proxy, log=self.rec.event)
        self.kernel.start()
        text, self.used = session.card(s)
        text += (f"\nYou have up to {s.max_turns} replies and {s.budget_s / 60:.0f} minutes for this episode; "
                 "the simulation stands still while you think.\n")
        pictures = [self._keep(self.conv.picture(image, f"{cam} at the start", pinned=True, kind=f"start_{cam}"))
                    for cam, image in session.start_pictures()]
        self.conv.user(text, pictures)
        self.rec.write_text("prompt.md", self.conv.messages[0]["content"] + "\n\n" + text)
        self.rec.event({"kind": "start", "settings": self._settings_record(), "card": self.used,
                        "perception": session.perception})

    def _loop(self) -> None:
        s = self.s
        nudges = s.nudges
        while True:
            if self.session.over:
                self.end = "episode_over"
                return
            if self.turns >= s.max_turns:
                self.end = "turn_limit"
                return
            if self._agent_seconds() >= s.budget_s or time.monotonic() >= self.deadline:
                self.end = "time_limit"
                return
            dropped = self.conv.trim()
            if dropped:
                self.rec.event({"kind": "pictures_dropped", "numbers": dropped, "total": self.conv.dropped,
                                "thinking_cleared": self.conv.thinking_cleared})
            try:
                reply = self.gateway.ask(self.conv.wire(), TOOLS, deadline=self.deadline,
                                         stop=lambda: self.session.over)
            except GatewayError as exc:
                self.end = {"time": "time_limit", "stopped": "episode_over"}.get(exc.kind, "gateway_error")
                if exc.kind in ("refused", "failing"):
                    self.errors.append(str(exc))
                return
            self.turns += 1
            for k in self.tokens:
                self.tokens[k] += reply.usage.get(k, 0)
            calls = [self._normalise(c, i) for i, c in enumerate(reply.calls)]
            self.conv.assistant(reply.text, calls, reply.raw_message.get("_blocks"),
                                reply.reasoning if self.s.history_reasoning and self.s.api == "openai" else None)
            self.rec.event({"kind": "reply", "turn": self.turns, "text": reply.text,
                            "reasoning": reply.reasoning, "calls": [c["function"]["name"] for c in calls],
                            "finish_reason": reply.finish_reason, "usage": reply.usage,
                            "seconds": round(reply.seconds, 2), "attempts": reply.attempts})
            if not calls:
                if nudges <= 0:
                    self.end = "no_tool_call"
                    return
                nudges -= 1
                self.conv.user(NUDGE)
                self.rec.event({"kind": "nudge", "left": nudges})
                continue
            self.turn_pictures = []
            for call in calls:
                self.conv.tool(call["id"], self._answer(call))
            if self.turn_pictures:
                pics = self.turn_pictures[-PICTURES_PER_TURN:]
                skipped = len(self.turn_pictures) - len(pics)
                intro = "Pictures from robot.look() in this turn" + (
                    f" (the first {skipped} were left out: at most {PICTURES_PER_TURN} per turn)" if skipped else "")
                self.conv.user(intro + ":", pics)
            self.session.pictures_shown()
            if self.finish is not None:
                self.end = "finish"
                return

    def _normalise(self, call: dict[str, Any], index: int) -> dict[str, Any]:
        function = call.get("function") or {}
        return {"id": call.get("id") or f"call_{self.turns}_{index}", "type": "function",
                "function": {"name": str(function.get("name") or ""),
                             "arguments": function.get("arguments") if isinstance(function.get("arguments"), str)
                             else json.dumps(function.get("arguments") or {})}}

    def _answer(self, call: dict[str, Any]) -> str:
        name = call["function"]["name"]
        if self.finish is not None:
            return "Not run: finish was called earlier in this reply."
        if self.session.over:
            return "Not run: " + self._over_text()
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("the arguments are not a JSON object")
        except ValueError as exc:
            return f"Could not read the arguments of {name}: {exc}. Send them as a JSON object."
        if name == "python":
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                return "python needs `code`: the Python source to run."
            return self._cell(code)
        if name == "finish":
            status = str(args.get("status") or "")
            self.finish = {"status": status, "summary": str(args.get("summary") or "")}
            self.rec.event({"kind": "finish", **self.finish})
            return "Finishing: the episode ends now and the scene is judged as it stands."
        return f"There is no tool named {name!r}; use python or finish."

    # -- cells -----------------------------------------------------------------------------------
    def _cell(self, code: str) -> str:
        self.cells += 1
        number = self.cells
        session = self.session
        before = session.step
        result = self.kernel.run(code)
        for c in result.calls:
            self.robot_calls[c["name"]] = self.robot_calls.get(c["name"], 0) + 1
        text = result.output
        if result.error:
            text = (text.rstrip("\n") + "\n" if text.strip() else "") + result.error
        text = _clip(text.strip("\n") or "(no output)")
        after = session.step
        moved = after is not None and before is not None and after != before
        status = [f"step {after} of {session.step_limit}" if after is not None else "",
                  f"reply {self.turns} of {self.s.max_turns}"]
        if result.restarted:
            status.append("the Python session was restarted")
        text += "\n— " + " · ".join(x for x in status if x)
        if result.stopped or session.over:
            text += "\n" + self._over_text()
        self.rec.cell(number, code, {"output": result.output, "error": result.error, "stopped": result.stopped,
                                     "restarted": result.restarted, "seconds": round(result.elapsed, 2),
                                     "calls": result.calls, "step_before": before, "step_after": after,
                                     "turn": self.turns})
        if moved:
            self.rec.frames(f"cell{number:03d}", session.frames())
        return text

    def _over_text(self) -> str:
        return self.session.over_text()

    def _attach(self, kind: str, image: np.ndarray, caption: str) -> None:
        self.turn_pictures.append(self._keep(self.conv.picture(image, caption, kind=kind)))

    def _keep(self, pic):
        pic.file = self.rec.picture(pic.number, pic.kind, pic.data)
        return pic

    # -- the end ---------------------------------------------------------------------------------
    def _wrap_up(self) -> None:
        verdict = None
        try:
            verdict = self.session.conclude(60.0)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"verdict: {exc}")
        for closer in (self.kernel, self.gateway, self.session):
            try:
                if closer is not None:
                    closer.close()
            except Exception:  # noqa: BLE001
                pass
        v = verdict or {}
        try:
            head = self.session.result_fields(v)
        except Exception as exc:  # noqa: BLE001 - the record is still written
            head = {"task": self.info.get("task_name"), "seed": self.info.get("eval_seed")}
            self.errors.append(f"result fields: {exc}")
        if self.session.bench != "robodojo":
            head = {"bench": self.session.bench, **head}
        result = {
            **head,
            "success": bool(v.get("success")), "progress": v.get("progress"),
            "step": v.get("step"), "step_limit": v.get("step_limit"), "verdict": v,
            "end": self.end, "finish": self.finish,
            "turns": self.turns, "cells": self.cells, "robot_calls": self.robot_calls,
            "tokens": self.tokens,
            "pictures": {"shown": len(self.conv.pictures) if self.conv else 0,
                         "dropped": self.conv.dropped if self.conv else 0},
            "seconds": round(time.monotonic() - self.t0, 1), "open_s": self.open_s,
            "gateway_lost_s": round(self._lost_seconds(), 1),
            "settings": self._settings_record(), "card": self.used,
            "errors": self.session.errors + self.errors, "hexashell": __version__,
        }
        try:
            if self.conv is not None:
                self.rec.write_json("transcript.json", self.conv.for_record())
            self.rec.write_json("result.json", result)
        finally:
            self.rec.close()

    def _lost_seconds(self) -> float:
        """Seconds the gateway's failed attempts and the pauses after them took so far."""
        return float(getattr(self.gateway, "lost_s", 0.0) or 0.0)

    def _agent_seconds(self) -> float:
        """Seconds since the episode opened, less what gateway failures took (given back up to one budget)."""
        return time.monotonic() - self.t_agent - min(self._lost_seconds(), self.s.budget_s)

    def _settings_record(self) -> dict[str, Any]:
        s = {f.name: getattr(self.s, f.name) for f in fields(self.s)}
        s.pop("api_key_file", None)
        if self.gateway is not None:
            s["sent"] = self.gateway.settings()  # what every request carried
        return s


def _clip(text: str) -> str:
    if len(text) <= TOOL_TEXT_MAX:
        return text
    head = text[: TOOL_TEXT_MAX * 5 // 8]
    tail = text[-TOOL_TEXT_MAX * 3 // 8:]
    return f"{head}\n[... {len(text) - len(head) - len(tail)} characters cut ...]\n{tail}"

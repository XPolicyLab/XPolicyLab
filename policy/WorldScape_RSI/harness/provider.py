"""Sticky inference-only client based on the archive's Responses transport."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

from .control_guard import policy_review_tool

from .responses_util import (ReasoningReplayStore, chat_messages_to_responses_input,
                             chat_tools_to_responses_tools, responses_to_chat_completion)


class ResponsesClient:
    def __init__(self, spec):
        self.config = {"model": spec["model"], "effort": spec["effort"],
                       "url": os.environ.get(spec["api_base_env"], "")}
        self.token = os.environ.get(spec["api_key_env"], "")
        if not self.config["url"] and not self.token and os.environ.get("DEXBOTIC_SECRETS"):
            from .provider_profiles import load_vlm_profile
            profile = load_vlm_profile(secrets_path=os.environ["DEXBOTIC_SECRETS"])
            if profile["model"] != spec["model"]:
                raise ValueError("The selected provider profile differs from the frozen Agent model")
            self.config["url"], self.token = profile["base_url"], profile["api_key"]
        if not self.config["url"].startswith(("https://", "http://")) or not self.token:
            raise ValueError("Set the Agent API base and API-key environment variables")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.lock = threading.Lock()
        self.closed = False
        self.timeout_s = float(spec.get("timeout_s", 300))
        self.release_path = os.environ.get("R2SI_SESSION_RELEASE_PATH", "")
        if self.release_path and (
                not self.release_path.startswith("/") or self.release_path.startswith("//")
                or self.release_path.count("{session_id}") != 1):
            raise ValueError("Invalid provider session release path")
        self.requested = False
        self.reset()

    def _record(self, phase, **details):
        directory = os.environ.get("R2SI_DIAGNOSTICS_DIR")
        if not directory:
            return
        directory = Path(directory).resolve()
        roots = os.environ.get("WSP_WRITABLE_ROOTS", "").split(os.pathsep)
        if not any(root and directory.is_relative_to(Path(root).resolve()) for root in roots):
            raise ValueError("Provider diagnostic directory is outside the writable workspace")
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(phase=phase, time=time.time(), session_id=self.session_id, **details)
        target = directory / (self.session_id + ".jsonl")
        with target.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row) + "\n")

    def reset(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Inference client is closed")
            if self.requested:
                raise RuntimeError("Close the active inference session before reset")
            self.session_id = uuid.uuid4().hex
            self.messages = []
            self.tools = None
            self.reasoning = ReasoningReplayStore()

    def complete(self, messages, tools):
        with self.lock:
            if self.closed:
                raise RuntimeError("Inference client is closed")
            if len(messages) < len(self.messages) or messages[:len(self.messages)] != self.messages:
                raise RuntimeError("Completed episode context changed")
            review = tools == [policy_review_tool()]
            if review and not self.tools:
                raise RuntimeError("A Policy review requires an existing Agent decision context")
            if self.tools is not None and tools != self.tools and not review:
                raise RuntimeError("Tool definitions changed within an episode")
            # The upstream pool pins tool definitions for the whole session too.
            # Declare the read-only review tool from the first request, then only
            # change tool_choice at an acknowledged execution boundary.
            baseline_tools = self.tools if self.tools is not None else tools
            declared = [*baseline_tools, policy_review_tool()] if baseline_tools else []
            flattened = chat_tools_to_responses_tools(declared)
            active = chat_tools_to_responses_tools(tools)
            if any(item.get("type") != "function" for item in flattened):
                raise ValueError("Only local Harness function proposals are exposed")
            if review:
                choice = {"type": "function", "name": "review_policy_chunk"}
            elif active:
                choice = {"type": "allowed_tools", "mode": "required",
                          "tools": [{"type": "function", "name": item["name"]} for item in active]}
            else:
                choice = "none"
            body = {"model": self.config["model"], "reasoning": {"effort": self.config["effort"]},
                    "input": chat_messages_to_responses_input(
                        [m for m in messages if m.get("role") != "system"], reasoning_store=self.reasoning),
                    "instructions": next((m["content"] for m in messages if m.get("role") == "system"), ""),
                    "tools": flattened, "tool_choice": choice, "parallel_tool_calls": False,
                    "store": False, "include": ["reasoning.encrypted_content"],
                    "prompt_cache_key": self.session_id, "stream": False}
            headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json",
                       "session_id": self.session_id, "conversation_id": self.session_id}
            request = urllib.request.Request(self.config["url"].rstrip("/") + "/responses",
                                             data=json.dumps(body).encode(), headers=headers)
            # No automatic replay: after ambiguous inference, reset the trial explicitly.
            self.requested = True
            self._record("request_started", request_bytes=len(request.data))
            try:
                with self.opener.open(request, timeout=self.timeout_s) as response:
                    raw = json.load(response)
            except urllib.error.HTTPError as error:
                # Bounded error fields only: never prompts, image data or tokens.
                data = error.read(8192)
                try:
                    item = json.loads(data)
                    item = item.get("error", item) if isinstance(item, dict) else {}
                    details = {k: str(item[k]).replace(self.token, "[redacted]")[:1000]
                               for k in ("code", "type", "message")
                               if isinstance(item, dict) and k in item
                               and "data:image" not in str(item[k])}
                except (ValueError, TypeError):
                    details = {}
                self._record("request_failed", http_status=error.code, **details)
                raise
            if raw.get("status") != "completed":
                self._record("request_incomplete", status=raw.get("status"))
                raise RuntimeError("Model inference did not complete")
            result = responses_to_chat_completion(raw, reasoning_store=self.reasoning)
            self.messages = copy.deepcopy(messages)
            if self.tools is None:
                self.tools = copy.deepcopy(tools)
            self._record("request_completed")
            return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.requested and self.release_path:
                request = urllib.request.Request(
                    self.config["url"].rstrip("/") + self.release_path.format(session_id=self.session_id),
                    data=b"{}", headers={"Authorization": "Bearer " + self.token,
                                         "Content-Type": "application/json",
                                         "session_id": self.session_id,
                                         "conversation_id": self.session_id})
                try:
                    with self.opener.open(request, timeout=15) as response:
                        receipt = json.load(response)
                except Exception as error:
                    self._record("release_unconfirmed", error_type=type(error).__name__,
                                 http_status=getattr(error, "code", None))
                    raise
                if receipt.get("released") is not True:
                    self._record("release_unconfirmed", reason="missing_confirmation")
                    raise RuntimeError("Provider session release is unconfirmed")
                self._record("session_released", release_confirmed=True)
            self.requested = False
            self.closed = True
            self.token = ""

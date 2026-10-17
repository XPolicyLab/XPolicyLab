"""Sticky inference-only client based on the archive's Responses transport."""
from __future__ import annotations

import copy
import json
import os
import threading
import urllib.request
import uuid

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
        self.reset()

    def reset(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Inference client is closed")
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
            if self.tools is not None and tools != self.tools:
                raise RuntimeError("Tool definitions changed within an episode")
            flattened = chat_tools_to_responses_tools(tools)
            if any(item.get("type") != "function" for item in flattened):
                raise ValueError("Only local Harness function proposals are exposed")
            body = {"model": self.config["model"], "reasoning": {"effort": self.config["effort"]},
                    "input": chat_messages_to_responses_input(
                        [m for m in messages if m.get("role") != "system"], reasoning_store=self.reasoning),
                    "instructions": next((m["content"] for m in messages if m.get("role") == "system"), ""),
                    "tools": flattened, "tool_choice": "required", "parallel_tool_calls": False,
                    "store": False, "include": ["reasoning.encrypted_content"],
                    "prompt_cache_key": self.session_id, "stream": False}
            headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json",
                       "session_id": self.session_id, "conversation_id": self.session_id}
            request = urllib.request.Request(self.config["url"].rstrip("/") + "/responses",
                                             data=json.dumps(body).encode(), headers=headers)
            # No automatic replay: after ambiguous inference, reset the trial explicitly.
            with self.opener.open(request, timeout=None) as response:
                raw = json.load(response)
            if raw.get("status") != "completed":
                raise RuntimeError("Model inference did not complete")
            result = responses_to_chat_completion(raw, reasoning_store=self.reasoning)
            self.messages, self.tools = copy.deepcopy(messages), copy.deepcopy(tools)
            return result

    def close(self):
        with self.lock:
            self.closed = True
            self.token = ""

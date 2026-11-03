"""One agent's ReAct loop: a_t = pi_z(h_t; c_t), o_{t+1} = R(a_t), h_{t+1} = Update(h_t, a_t, o_{t+1}).

`step()` is one model decision. Tool calls run in order; a control tool
(interact, finish, return_to_task) ends the batch because it changes who
reasons next or what the scene looks like, and later calls in the same decision
are reported as not executed.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from knowin_agent.agents.context import ContextBuilder, strip_images
from knowin_agent.agents.tools import ControlTransfer, ToolResult, ToolSpec, execute


class AgentSession:
    def __init__(self, role: str, model, tools: list[ToolSpec], context: ContextBuilder, *, consumer: str,
                 recorder=None, contract=None, delegation_id: Optional[str] = None, max_history_decisions: int = 24):
        self.role = role
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.schemas = [t.schema() for t in tools]
        self.context = context
        self.consumer = consumer
        self.recorder = recorder
        self.contract = contract
        self.delegation_id = delegation_id
        self.max_history_decisions = max_history_decisions
        self.system = context.system_prompt(role, contract)
        self.decisions = 0
        self.groups: list[list[dict]] = []
        self.pending_images: list = []
        self.note = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_seconds": 0.0}

    def messages(self) -> list[dict]:
        out = [{"role": "system", "content": self.system}]
        for index, group in enumerate(self.groups):
            latest = index == len(self.groups) - 1
            for message in group:
                message = message if latest else strip_images(message)
                out.append({k: v for k, v in message.items() if not k.startswith("_")})
        return out

    def step(self) -> list[ToolResult]:
        self.decisions += 1
        observation = self.context.observation(self.role, self.consumer, decision=self.decisions,
                                               contract=self.contract, extra_images=self.pending_images,
                                               note=self.note)
        self.pending_images, self.note = [], ""
        group = [observation]
        self.groups.append(group)
        if len(self.groups) > self.max_history_decisions:
            self.groups = self.groups[-self.max_history_decisions:]
        started = time.time()
        turn = self.model.complete(self.messages(), self.schemas)
        self._account(turn, time.time() - started, observation)
        assistant = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}} for c in turn.tool_calls]
        group.append(assistant)
        if not turn.tool_calls:
            self.note = ("[Note] Your last decision called no tool. Act through a tool"
                         + (", or call finish." if self.role == "task" else ", or call return_to_task."))
            return []
        results, transfer = [], None
        for call in turn.tool_calls:
            if transfer is not None:
                result = ToolResult(call.name, False, {"error": "not executed: control was transferred by an earlier "
                                                                "call in this decision; decide again"})
            elif call.name not in self.tools:
                result = ToolResult(call.name, False, {"error": f"unknown tool {call.name!r} for the {self.role} agent; "
                                                                f"available: {sorted(self.tools)}"})
            elif call.parse_error:
                result = ToolResult(call.name, False, {"error": call.parse_error})
            else:
                tick_before = self.context.rt.tick
                try:
                    result = execute(self.tools[call.name], call.arguments)
                except ControlTransfer as control:
                    transfer = control
                    result = ToolResult(call.name, True, control.payload)
                self._trace(call, result, turn.text, tick_before)
                if self.tools[call.name].control and transfer is None:
                    transfer = ControlTransfer("continue", {})
            group.append({"role": "tool", "tool_call_id": call.id, "content": result.text()})
            self.pending_images += result.images
            results.append(result)
        if transfer is not None and transfer.kind != "continue":
            raise transfer
        return results

    def _account(self, turn, seconds: float, observation: dict) -> None:
        usage = turn.usage or {}
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.usage["model_seconds"] += seconds
        if self.recorder is not None:
            self.recorder.record("llm_calls", scope=self.role, delegation=self.delegation_id, decision=self.decisions,
                                 model=getattr(self.model, "name", "?"), usage=usage, seconds=round(seconds, 2))
            self.recorder.record("turns", scope=self.role, delegation=self.delegation_id, decision=self.decisions,
                                 obs_id=observation["_meta"]["obs_id"], tick=observation["_meta"]["tick"],
                                 text=turn.text, tool_calls=[{"name": c.name, "arguments": c.arguments}
                                                             for c in turn.tool_calls])

    def _trace(self, call, result: ToolResult, rationale: str, tick_before: int) -> None:
        if self.recorder is None:
            return
        self.recorder.record("tool_calls", scope=self.role, delegation=self.delegation_id, decision=self.decisions,
                             tool=call.name, arguments=call.arguments, rationale=rationale, ok=result.ok,
                             status=result.data.get("status") if isinstance(result.data, dict) else None,
                             result=result.data, tick_before=tick_before, tick_after=self.context.rt.tick,
                             obs_id=self.context.rt.obs.obs_id)

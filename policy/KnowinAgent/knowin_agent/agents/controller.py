"""Episode controller: F = (pi_Task, pi_Local, R, K).

Only one agent deliberates at a time. Task runs its ReAct loop; `interact`
opens a contract and runs Local until it returns (or its round budget ends),
then Task resumes with the report. Robot work started asynchronously keeps
running across decisions and delegations; its completions arrive as events.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

from knowin_agent.agents.context import ContextBuilder
from knowin_agent.agents.session import AgentSession
from knowin_agent.agents.tools import ControlTransfer, local_toolset, task_toolset
from knowin_agent.core.events import EventKind
from knowin_agent.core.recorder import RunRecorder
from knowin_agent.runtime.api import RobotRuntime, ToolOutcome
from knowin_agent.runtime.contract import InteractionContract
from knowin_agent.skills.library import SkillLibrary


@dataclass
class EpisodeLimits:
    max_task_decisions: int = 150
    max_local_rounds: int = 16
    max_total_decisions: int = 300
    wall_seconds: float = 5400.0
    model_error_retries: int = 2
    max_executions: Optional[int] = None  # program executions (practice budget); enforced by the exec tool


class EpisodeController:
    def __init__(self, runtime: RobotRuntime, library: SkillLibrary, *, task_model, local_model=None,
                 recorder: Optional[RunRecorder] = None, limits: EpisodeLimits = EpisodeLimits(),
                 task_brief: str = "", image_side: int = 448, kit=None):
        self.rt = runtime
        self.library = library
        self.task_model = task_model
        self.local_model = local_model or task_model
        self.recorder = recorder or RunRecorder(None)
        self.limits = limits
        self.kit = kit
        sections = ()
        if kit is not None:
            from knowin_agent.assembly import task_sections

            sections = task_sections(kit)
        self.context = ContextBuilder(runtime, library, task_brief=task_brief, image_side=image_side,
                                      recorder=self.recorder, task_sections=sections)
        self.delegations: list[dict] = []
        self.local_decisions = 0
        self.local_usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_seconds": 0.0}
        self.finish_report: Optional[dict] = None
        self.started = 0.0
        self.task: Optional[AgentSession] = None

    # -- control tools ---------------------------------------------------------------------------

    def _remember(self, args: dict) -> ToolOutcome:
        note = self.rt.ctx.notes.write(args["key"], args["text"], "task", self.rt.tick)
        return ToolOutcome({"status": "stored", "key": note.key})

    def _finish(self, args: dict):
        from knowin_agent.assembly import completion_refused

        refusal = completion_refused(self.rt, args.get("outcome"))
        if refusal is not None:
            return ToolOutcome(refusal)
        self.finish_report = {"outcome": args["outcome"], "summary": args["summary"], "tick": self.rt.tick}
        raise ControlTransfer("finish", {"status": "finished", **self.finish_report})

    @staticmethod
    def _return(args: dict):
        raise ControlTransfer("return", {"status": args["status"], "report": args["report"],
                                         "evidence": args.get("evidence", "")})

    def _interact(self, args: dict) -> ToolOutcome:
        contract = InteractionContract.from_args(args)
        contract.max_rounds = min(contract.max_rounds, self.limits.max_local_rounds)
        contract.open({arm: state.ee_pose.xyz for arm, state in self.rt.obs.arms.items()})
        delegation = f"d{len(self.delegations) + 1}"
        start_tick = self.rt.tick
        self.recorder.record("delegations", delegation=delegation, phase="open", tick=start_tick,
                             contract=contract.describe())
        self.rt.bus.publish(EventKind.DELEGATED, "task", f"delegated '{contract.goal}' ({delegation})",
                            tick=start_tick, delegation=delegation)
        self.rt.bus.mark_seen(f"local:{delegation}")
        tools = local_toolset(self.rt, self.library, contract, return_to_task=self._return)
        session = AgentSession("local", self.local_model, tools, self.context, consumer=f"local:{delegation}",
                               recorder=self.recorder, contract=contract, delegation_id=delegation)
        self.rt.contract, self.rt.active_scope, self.rt.track_owner = contract, "local", "local"
        report = None
        try:
            while session.decisions < contract.max_rounds:
                if self.rt.executor.episode_over:
                    report = {"status": "episode_over", "report": "the episode ended during local interaction"}
                    break
                if self._out_of_budget():
                    report = {"status": "unresolved", "report": "episode decision/time budget exhausted"}
                    break
                try:
                    session.step()
                except ControlTransfer as control:
                    report = dict(control.payload)
                    break
            if report is None:
                report = {"status": "round_limit",
                          "report": f"Local used its {contract.max_rounds} rounds without returning"}
        finally:
            self.rt.contract, self.rt.active_scope, self.rt.track_owner = None, "task", "task"
            self.local_decisions += session.decisions
            for key in self.local_usage:
                self.local_usage[key] += session.usage[key]
        report.update(delegation=delegation, local_decisions=session.decisions, ticks_used=self.rt.tick - start_tick)
        self.delegations.append({"delegation": delegation, "goal": contract.goal, "status": report["status"],
                                 "rounds": session.decisions, "ticks": report["ticks_used"]})
        self.recorder.record("delegations", delegation=delegation, phase="return", tick=self.rt.tick, report=report)
        self.rt.bus.publish(EventKind.RETURNED, "local", f"local returned {report['status']} ({delegation})",
                            tick=self.rt.tick, delegation=delegation)
        return ToolOutcome(report)

    # -- loop ------------------------------------------------------------------------------------

    def _total_decisions(self) -> int:
        return (self.task.decisions if self.task else 0) + self.local_decisions

    def _out_of_budget(self) -> bool:
        return (self._total_decisions() >= self.limits.max_total_decisions
                or time.time() - self.started >= self.limits.wall_seconds)

    def run(self) -> dict:
        self.started = time.time()
        self.rt.wall_window = (self.started, float(self.limits.wall_seconds))
        self.library.freeze()
        self.rt.note_start_poses()
        if self.kit is not None:
            from knowin_agent.assembly import build_task_tools

            tools = build_task_tools(self.kit, remember=self._remember, interact=self._interact, finish=self._finish)
        else:
            tools = task_toolset(self.rt, self.library, remember=self._remember, interact=self._interact,
                                 finish=self._finish)
        self.task = AgentSession("task", self.task_model, tools, self.context, consumer="task",
                                 recorder=self.recorder)
        self.recorder.record("episode", phase="start", instruction=self.rt.ctx.instruction,
                             task_model=getattr(self.task_model, "name", "?"),
                             local_model=getattr(self.local_model, "name", "?"), limits=asdict(self.limits),
                             skills=self.library.ids())
        stop, errors = None, 0
        from knowin_agent.assembly import attempt_may_restart

        while stop is None:
            if self.rt.executor.episode_over and not attempt_may_restart(self.kit):
                stop = f"episode_over:{self.rt.executor.over_reason}"
            elif self.task.decisions >= self.limits.max_task_decisions:
                stop = "task_decision_limit"
            elif self._out_of_budget():
                stop = "decision_or_wall_budget"
            if stop:
                break
            try:
                self.task.step()
                errors = 0
            except ControlTransfer as control:
                if control.kind == "finish":
                    stop = "finished"
            except RuntimeError as exc:
                errors += 1
                self.recorder.record("events", kind="model_error", error=str(exc), decision=self.task.decisions)
                if errors > self.limits.model_error_retries:
                    stop = f"model_error:{exc}"
                time.sleep(min(30, 5 * errors))
        return self._result(stop)

    def _result(self, stop: str) -> dict:
        evaluation = {}
        try:
            evaluation = self.rt.backend.evaluation()
        except Exception as exc:  # noqa: BLE001 - reporting must not hide the episode outcome
            evaluation = {"error": str(exc)}
        task_usage = self.task.usage if self.task else {}
        result = {
            "stop_reason": stop,
            "finish_report": self.finish_report,
            "decisions": {"task": self.task.decisions if self.task else 0, "local": self.local_decisions,
                          "total": self._total_decisions()},
            "delegations": self.delegations,
            "budget": self.rt.budget(),
            "wall_seconds": round(time.time() - self.started, 1),
            "usage": {"task": task_usage, "local": self.local_usage},
            "evaluation_host_only": evaluation,
        }
        self.recorder.record("episode", phase="end", **result)
        self.recorder.write_json("result.json", result)
        return result

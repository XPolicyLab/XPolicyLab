"""Assemble one episode from a run config: switches -> backend, runtime, workspace, program session, toolsets.

The API-model controller (agents/controller.py) and the coding-agent host (bridge/host.py) both build their
episode here, so every switch means the same thing in both.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from knowin_agent.agents.tools import ControlTransfer, ToolSpec, local_toolset, robot_tools, task_toolset
from knowin_agent.core.memory import TaskContext
from knowin_agent.core.recorder import RunRecorder
from knowin_agent.experiments.switches import apply as apply_switches
from knowin_agent.experiments.switches import resolve as resolve_switches
from knowin_agent.runtime.api import ApiError, RobotRuntime, ToolOutcome
from knowin_agent.skills.library import SkillLibrary

MOTION_TOOLS = ("move_to", "rotate", "advance", "gripper", "wait", "orient_held", "carry_tip_to", "center_over",
                "advance_until")
# Scoring default (hybrid): one action interface (exec) plus the non-moving queries used to write the
# scene plan. Duplicate single-step motion tools stay available under action_mode=tools.
HYBRID_TASK_TOOLS = ("locate", "inspect", "check_reach", "remember", "interact", "finish")
# 'program' mode: everything the robot does goes through exec; these are the only other task tools.
PROGRAM_MODE_TOOLS = ("remember", "interact", "finish")
PROGRAM_MODES = ("code", "hybrid", "program")
PHASES = ("practice", "test")


@dataclass
class EpisodeKit:
    config: dict
    switches: dict
    phase: str
    backend: object
    runtime: RobotRuntime
    library: SkillLibrary
    recorder: RunRecorder
    brief_text: str
    brief_manifest: dict = field(default_factory=dict)
    workspace: Optional[object] = None
    program: Optional[object] = None
    stall_gate: Optional[object] = None
    executions: int = 0         # executions counted against max_executions
    measure_only: int = 0       # executions that moved nothing and were not counted (up to max_executions)
    idle_streak: int = 0        # programs in a row that moved nothing
    setback: Optional[dict] = None  # practice: the last drop/spill/score loss and what recovering it has cost
    last_score: Optional[float] = None
    submissions: int = 0        # every exec submission, the numbering of replay artifacts
    max_executions: Optional[int] = None
    artifacts: Optional[object] = None  # per-execution video/view/state for replay (program.artifacts)

    @property
    def action_mode(self) -> str:
        return self.switches["agent.action_mode"]

    @property
    def task(self) -> str:
        return self.config.get("backend", {}).get("task", "")


def resolve_config(config: dict) -> tuple[dict, dict]:
    """Expand the profile and overrides into every switch and write them into the config."""
    switches = resolve(config)
    return apply_switches(config, switches), switches


def resolve(config: dict) -> dict:
    return resolve_switches(config.get("profile"), config.get("switches") or {})


def _brief(config: dict, switches: dict, kind: str) -> tuple[str, dict]:
    brief = config.get("brief", "robodojo" if kind == "robodojo" else None)
    if brief == "robodojo":
        from knowin_agent.knowledge import task_brief

        b = task_brief(config.get("backend", {}).get("task", ""), switches)
        return str(b), b.manifest()
    if brief and Path(brief).is_file():
        return Path(brief).read_text(encoding="utf-8"), {"file": str(brief)}
    return brief or "", {}


def _shared_library(switches: dict, knowledge_from: Optional[str] = None) -> Optional[str]:
    """'latest' means the library the frozen knowledge was trained with, else the newest published version."""
    from knowin_agent.workspace.knowledge import latest_published

    choice = switches.get("evolution.shared_library")
    if choice in (None, "off", False):
        return None
    if choice == "latest" and knowledge_from:
        manifest = Path(knowledge_from) / "manifest.json"
        if manifest.exists():
            import json

            trained = json.loads(manifest.read_text(encoding="utf-8")).get("library_version")
            if trained:
                return trained
    return latest_published() if choice == "latest" else str(choice)


def assemble(config: dict, run_dir: Path, *, backend=None, frame_sink=None,
             recorder: Optional[RunRecorder] = None) -> EpisodeKit:
    from knowin_agent.episode import build_backend, build_perceiver

    config, switches = resolve_config(config)
    run_dir = Path(run_dir)
    recorder = recorder or RunRecorder(run_dir)
    phase = config.get("phase", "test")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    kind = config.get("backend", {}).get("kind", "robodojo")
    backend = backend or build_backend(config.get("backend", {}), frame_sink=frame_sink)
    perceiver = build_perceiver(config.get("perception", {}), backend)
    runtime = RobotRuntime(backend, context=TaskContext(instruction=backend.instruction()), perceiver=perceiver,
                           budget_mode=switches["budget.mode"])
    runtime.default_path = switches["motion.path"]
    runtime.limits = dataclasses.replace(runtime.limits, transit_open_m=float(switches["motion.transit_step_m"]))
    runtime.async_enabled = bool(switches["motion.bimanual_async"])
    library = SkillLibrary()
    if config.get("skills_dir"):
        library.load_dir(Path(config["skills_dir"]))
    brief_text, brief_manifest = _brief(config, switches, kind)
    kit = EpisodeKit(config, switches, phase, backend, runtime, library, recorder, brief_text, brief_manifest,
                     max_executions=(config.get("limits") or {}).get("max_executions"))
    if switches["agent.action_mode"] in PROGRAM_MODES:
        from knowin_agent.program.session import ProgramSession
        from knowin_agent.workspace import create_workspace

        knowledge_from = config.get("knowledge_from") or switches.get("evolution.seed_knowledge")
        kit.workspace = create_workspace(run_dir / "workspace", instruction=backend.instruction(), phase=phase,
                                         knowledge_from=knowledge_from,
                                         shared_library=_shared_library(switches, knowledge_from))
        scored = phase == "practice" and bool(switches.get("evolution.practice_feedback", True))
        feedback = (lambda: backend.evaluation().get("score")) if scored else None
        kit.program = ProgramSession(runtime, workspace=kit.workspace.root, phase=phase, recorder=None,
                                     library=library, score_feedback=feedback)
        threshold = int(switches.get("evolution.stall_rule") or 0)
        if scored and threshold > 0:
            from knowin_agent.evolution.gates import StallGate

            kit.stall_gate = StallGate(kit.workspace.lessons, threshold=threshold, budget=kit.max_executions,
                                       mode=switches.get("evolution.stall_mode", "advise"))
    if frame_sink is not None and hasattr(frame_sink, "frames") and kit.program is not None:
        from knowin_agent.program.artifacts import ExecutionArtifacts

        kit.artifacts = ExecutionArtifacts(runtime, frame_sink, run_dir / "observations",
                                           fps=backend.profile.control_hz / getattr(frame_sink, "stride", 1))
    recorder.record("episode", phase="assembled", run_phase=phase, switches=switches, brief=brief_manifest,
                    workspace=str(kit.workspace.root) if kit.workspace else None)
    return kit


COMPLETION_REFUSAL_MIN_ACTIONS = 10


def completion_refused(runtime, outcome) -> Optional[dict]:
    """finish(outcome='completed') while the episode goes on is refused: success ends the episode by itself
    (all 114 Tests that scored 100 ended so, none called finish), so a live episode means some success
    condition is still false. In the 10-30/31 campaigns all 5 'completed' reports were wrong, with 13-51 % of
    the native actions left to fix them. 'partial' / 'failed', an ended episode or a spent budget pass."""
    if outcome != "completed" or runtime.executor.episode_over:
        return None
    remaining = runtime.backend.budget().remaining
    if remaining is not None and remaining < COMPLETION_REFUSAL_MIN_ACTIONS:
        return None
    return {"status": "refused", "reason": "the episode is still running, so at least one success condition is "
            "false (success ends the episode by itself)",
            "native_steps_remaining": remaining,
            "next": "check each condition of the top tier again by measurement and fix the one that fails; "
                    "finish(outcome='partial') only if nothing more can be done"}


def attempt_may_restart(kit: Optional[EpisodeKit]) -> bool:
    """In practice, an attempt the evaluator ended without success does not end the episode while executions
    remain: the agent can write the lesson and reset() (the receipt's `attempt_over` says so)."""
    if kit is None or kit.phase != "practice" or kit.program is None or not kit.runtime.executor.episode_over:
        return False
    if kit.max_executions is not None and kit.executions >= kit.max_executions:
        return False
    try:
        return not bool(kit.backend.evaluation().get("success"))
    except Exception:  # noqa: BLE001 - no verdict: end the episode as before
        return False


# -- toolsets ------------------------------------------------------------------------------------

def _guarded(kit: EpisodeKit, step: Callable) -> None:
    """Replay artifacts must never fail a program."""
    try:
        step()
    except Exception as exc:  # noqa: BLE001
        kit.recorder.record("events", kind="artifact_error", error=f"{type(exc).__name__}: {exc}")


FREE_IDLE_PROGRAMS = 2
SETBACK_PROGRAMS = 2  # recovery programs after a setback before the practice receipt suggests reset()


def _setback_notice(kit: EpisodeKit, data: dict, moved: bool, reset: bool) -> Optional[str]:
    """Practice only: after a part dropped or the score fell, count the moving programs that have not won the
    score back; from SETBACK_PROGRAMS on, say what recovering has cost and that reset() restores the scene.
    Recorded episodes spent 10-20 minutes salvaging a dropped key or spilled balls and reset only near the end."""
    if kit.phase != "practice" or reset or (data.get("native_steps") or {}).get("resets"):
        kit.setback = None
        return None
    score = data.get("score_feedback")
    score = score.get("score") if isinstance(score, dict) else score
    score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None
    results = [t.get("result") or {} for t in data.get("tracks") or []]
    dropped = any(r.get("status") == "dropped" for r in results)
    regripped = any(r.get("holding") == "yes" and float(r.get("commanded", 1.0)) < 0.5 for r in results)
    setback = kit.setback
    if setback is not None and ((score is not None and score > setback["score_before"])
                                or (setback["dropped"] and regripped and not dropped)):
        kit.setback = None
        kit.last_score = score if score is not None else kit.last_score
        return None
    if setback is None:
        fell = score is not None and kit.last_score is not None and score < kit.last_score
        if dropped or fell:
            kit.setback = {"at": kit.submissions, "what": "a held part dropped" if dropped else "the score fell",
                           "dropped": dropped, "score_before": kit.last_score if fell else (score or 0.0),
                           "programs": 0, "steps": 0}
    elif moved:
        setback["programs"] += 1
        setback["steps"] += int((data.get("native_steps") or {}).get("used") or 0)
    if score is not None:
        kit.last_score = score
    setback = kit.setback
    if setback is None or setback["programs"] < SETBACK_PROGRAMS:
        return None
    return (f"{setback['what']} at execution {setback['at']}; {setback['programs']} moving programs and "
            f"{setback['steps']} native actions since have not won the score back. reset() restores the scene: "
            "redoing the attempt from the start is usually cheaper than salvaging, and practice is for getting the "
            "whole sequence to work. Write the cause into lessons/ first.")


def _exec_tool(kit: EpisodeKit) -> list[ToolSpec]:
    from knowin_agent.evolution.gates import StallRefused
    from knowin_agent.program.tools import program_tools

    tools = {t.name: t for t in program_tools(kit.program)}
    inner = tools["exec"].handler

    def run(args: dict) -> ToolOutcome:
        code = args.get("code", "")
        if kit.max_executions is not None and kit.executions >= kit.max_executions:
            raise ApiError(f"the execution budget ({kit.max_executions}) is used up; call finish")
        if kit.stall_gate is not None:
            try:
                kit.stall_gate.admit(code)
            except StallRefused as exc:
                return ToolOutcome({"ok": False, "stop_reason": "stall_refused", "error": str(exc)})
        if kit.max_executions is not None:
            kit.program.api.executions_left = kit.max_executions - kit.executions
        if kit.artifacts is not None:
            _guarded(kit, kit.artifacts.begin)
        outcome = inner(args)
        kit.submissions += 1
        steps = outcome.data.get("native_steps") or {}
        moved = steps.get("after") != steps.get("before") or bool(steps.get("used")) or bool(steps.get("resets"))
        kit.idle_streak = 0 if moved else kit.idle_streak + 1
        # 27-52% of failing practice programs only measured; they ran the execution cap out with most native
        # actions unused. Programs that move nothing are not counted, up to the cap again -- but only two in a
        # row: free probes then took 49-59% of all programs, up to 7 in a row (insert_key, pour_balls, 10-31).
        if (moved or kit.max_executions is None or kit.measure_only >= kit.max_executions
                or kit.idle_streak > FREE_IDLE_PROGRAMS):
            kit.executions += 1
        else:
            kit.measure_only += 1
        if kit.idle_streak > FREE_IDLE_PROGRAMS:
            outcome.data["idle_notice"] = (
                f"{kit.idle_streak} programs in a row moved nothing; from the third on they are counted. Act on "
                "what you have measured: put remaining checks in the same program as the motion they decide.")
        notice = _setback_notice(kit, outcome.data, moved, "reset(" in code)
        if notice:
            outcome.data["setback_notice"] = notice
        if kit.artifacts is not None:
            _guarded(kit, lambda: kit.artifacts.end(kit.submissions))
        if kit.stall_gate is not None:
            score = outcome.data.get("score_feedback")
            score = score.get("score") if isinstance(score, dict) else score
            notice = kit.stall_gate.record(kit.submissions, score, reset="reset(" in code)
            if notice:
                outcome.data["stall_notice"] = notice
        outcome.data["executions_used"] = kit.executions
        if kit.max_executions is not None:
            outcome.data["executions_limit"] = kit.max_executions
            outcome.data["measure_only_uncounted"] = f"{kit.measure_only}/{kit.max_executions}"
        return outcome

    tools["exec"] = dataclasses.replace(tools["exec"], handler=run)
    return list(tools.values())


def _file_tools(kit: EpisodeKit) -> list[ToolSpec]:
    """Workspace files for API-model agents (coding-agent CLIs use their own file tools)."""
    ws = kit.workspace

    def list_files(args):
        base = ws.path(args.get("dir", "."))
        files = sorted(str(p.relative_to(ws.root)) for p in base.rglob("*") if p.is_file())
        return ToolOutcome({"files": files[:200]})

    def read_file(args):
        path = ws.path(args["path"])
        text = path.read_text(encoding="utf-8")
        return ToolOutcome({"path": args["path"], "text": text[:20000], "truncated": len(text) > 20000})

    def write_file(args):
        path = ws.write_text(args["path"], args["text"])
        return ToolOutcome({"status": "written", "path": str(path.relative_to(ws.root)), "chars": len(args["text"])})

    tools = [ToolSpec("list_files", "List workspace files (skills/, skills/shared/, lessons/, scratch/).",
                      {"dir": {"type": "string"}}, [], list_files),
             ToolSpec("read_file", "Read a workspace file.", {"path": {"type": "string"}}, ["path"], read_file)]
    if kit.phase == "practice":
        tools.append(ToolSpec("write_file", "Write a workspace file under skills/, lessons/ or scratch/ (practice only).",
                              {"path": {"type": "string"}, "text": {"type": "string"}}, ["path", "text"], write_file))
    return tools


def build_task_tools(kit: EpisodeKit, *, remember: Callable, interact: Callable, finish: Callable,
                     file_tools: bool = True) -> list[ToolSpec]:
    mode = kit.action_mode
    tools = task_toolset(kit.runtime, kit.library, remember=remember, interact=interact, finish=finish)
    if mode == "hybrid":
        robot = robot_tools(kit.runtime)
        extras = [robot["check_reach"]] if "check_reach" in robot else []
        keep = set(HYBRID_TASK_TOOLS)
        tools = [t for t in tools + extras if t.name in keep]
    elif mode == "code":
        tools = [t for t in tools if t.name not in MOTION_TOOLS]
    elif mode == "program":
        keep = set(PROGRAM_MODE_TOOLS)
        tools = [t for t in tools if t.name in keep]
    if not kit.switches["agent.delegation"]:
        tools = [t for t in tools if t.name != "interact"]
    if not kit.switches["perception.observe"]:
        tools = [t for t in tools if t.name != "observe"]
    if kit.program is not None:
        tools += _exec_tool(kit)
        if file_tools:
            tools += _file_tools(kit)
    if kit.phase == "practice" and getattr(kit.backend, "supports_reset", False) and mode == "tools":
        tools.append(ToolSpec("reset", "Practice only: restore the same scene and a fresh action budget.", {}, [],
                              lambda a: ToolOutcome(kit.runtime.reset())))
    return tools


def build_local_tools(kit: EpisodeKit, contract, *, return_to_task: Callable) -> list[ToolSpec]:
    return local_toolset(kit.runtime, kit.library, contract, return_to_task=return_to_task)


def task_sections(kit: EpisodeKit, *, inline_plan: bool = True) -> list[tuple[str, str]]:
    """System-prompt sections for the Task agent: principles, phase policy, program API, workspace knowledge."""
    from knowin_agent.agents.context import load_prompt

    sections = []
    if kit.program is not None:
        from knowin_agent.program.tools import doc_text

        sections += [("Principles", load_prompt("principles")), ("Phase", load_prompt(kit.phase))]
        if kit.phase == "practice":
            scored = bool(kit.switches.get("evolution.practice_feedback", True))
            sections.append(("Feedback", load_prompt("practice_scored" if scored else "practice_blind")))
        sections += [("Program API", doc_text("api")), ("Knowledge", knowledge_index(kit, inline_plan=inline_plan))]
    if kit.action_mode == "hybrid":
        sections.append(("Action mode", load_prompt("mode_hybrid")))
    elif kit.action_mode == "program":
        sections.append(("Action mode", load_prompt("mode_program")))
    elif kit.action_mode == "code":
        sections.append(("Action mode", load_prompt("mode_hybrid")))
        sections.append(("Action mode", "Perception and skill tools are also available as separate calls; "
                                        "robot motion tools are not."))
    elif kit.action_mode == "tools":
        sections.append(("Action mode", load_prompt("mode_tools")))
    return sections


def agent_workspace(kit: EpisodeKit) -> str:
    """The workspace path as the agent sees it: /workspace inside a container, never the host path."""
    contained = ((kit.config.get("agent") or {}).get("isolation") or {}).get("kind") == "docker"
    return "/workspace" if contained else str(kit.workspace.root)


def knowledge_index(kit: EpisodeKit, *, inline_plan: bool = True, limit: int = 12000) -> str:
    """What the workspace already knows: skill files with their one-line descriptions and lessons/PLAN.md."""
    if kit.workspace is None:
        return ""
    ws = kit.workspace
    lines = [f"Workspace: {agent_workspace(kit)} ({kit.phase}; skills/ and lessons/ are "
             f"{'writable' if kit.phase == 'practice' else 'read-only'}; skills/shared/ is read-only"
             + (f", library {ws.shared_version}" if ws.shared_version else "") + ")"]
    for folder in (ws.skills, ws.shared, ws.lessons):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.suffix in (".py", ".md"):
                first = next((l.strip("# ").strip() for l in path.read_text(encoding="utf-8").splitlines()
                              if l.strip()), "")
                lines.append(f"- {path.relative_to(ws.root)}: {first[:120]}")
    plan = ws.lessons / "PLAN.md"
    if inline_plan and plan.exists():
        lines += ["", "### lessons/PLAN.md", plan.read_text(encoding="utf-8")]
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n...(truncated; read the files)"

"""Agent tools of the code-action mode: `exec` (run a program) and `api_reference` (the program API docs)."""

from __future__ import annotations

from importlib import resources

from knowin_agent.agents.tools import ToolSpec
from knowin_agent.program.session import ProgramSession
from knowin_agent.runtime.api import ToolOutcome

DOCS = {"api": "api.md", "manipulation": "manipulation.md"}

EXEC_DESCRIPTION = (
    "Run a Python program on the robot. The namespace persists across calls and already holds the program API "
    "(move_to, set_gripper, check_reach, checkpoint, subgoal_done, get_observation, step, locate, inspect, observe, "
    "grasp_candidates, run_skill, both, wait, budget_panel, ...) and the manipulation library (pick, place, "
    "look_down, press_at, screw_down, rim_transfer, ...). Only `import numpy as np` and `import math`; no file "
    "access, no names starting with '_'. include lists skills/<name>.py or skills/shared/<name>.py files run "
    "first. A stall, failed checkpoint or budget refusal stops the program; the receipt gives stop_reason, the "
    "error line, stdout (print), native actions used, tracks, checkpoints and the budget panel. "
    "Call api_reference for the full API.")


def doc_text(topic: str = "api") -> str:
    if topic not in DOCS:
        raise ValueError(f"topic must be one of {sorted(DOCS)}")
    return resources.files("knowin_agent.program").joinpath("docs", DOCS[topic]).read_text(encoding="utf-8")


def program_tools(session: ProgramSession) -> list[ToolSpec]:
    def run(args: dict) -> ToolOutcome:
        receipt = session.execute(args["code"], args.get("include") or [])
        data = {k: v for k, v in receipt.items() if not (k == "error" and v is None)}
        return ToolOutcome(data, session.images())

    def reference(args: dict) -> ToolOutcome:
        topic = args.get("topic", "api")
        return ToolOutcome({"topic": topic, "text": doc_text(topic)})

    return [
        ToolSpec("exec", EXEC_DESCRIPTION,
                 {"code": {"type": "string", "description": "Python program (numpy as np, math)"},
                  "include": {"type": "array", "items": {"type": "string"},
                              "description": "files run before the program: skills/<name>.py, "
                                             "skills/shared/<name>.py"}},
                 ["code"], run),
        ToolSpec("api_reference", "The program API reference ('api') or the grasp/place rules ('manipulation').",
                 {"topic": {"type": "string", "enum": sorted(DOCS)}}, [], reference),
    ]

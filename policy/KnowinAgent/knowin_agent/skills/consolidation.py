"""Offline skill consolidation from recorded interaction traces.

Between episodes, recorded traces from both decision scales are mined for
recurring call sequences. Delegation boundaries (interact / return_to_task) and
calls that need fresh model interpretation (perception whose output the model
read before choosing the next call) end a segment. A candidate keeps links to
its source calls; a coding agent turns it into a SkillSpec module, which is
validated by loading it and running it once on the mock world.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from knowin_agent.core.recorder import read_table

BOUNDARY_TOOLS = {"interact", "return_to_task", "finish", "remember", "skill_info", "cancel_track"}
INTERPRETATION_TOOLS = {"locate", "inspect", "check_relation", "grasp_candidates", "reach_query"}
DISCRETE_ARGS = ("command", "direction", "frame", "mode", "view", "axis", "pivot", "skill_id", "precision",
                 "stop_on_contact")


@dataclass
class Step:
    run: str
    scope: str
    delegation: Optional[str]
    decision: int
    tool: str
    arguments: dict
    rationale: str
    status: Optional[str]
    ok: bool

    def signature(self, arm_roles: dict) -> str:
        parts = [self.tool]
        arm = self.arguments.get("arm")
        if arm:
            parts.append(arm_roles.setdefault(arm, f"arm{len(arm_roles)}"))
        for key in DISCRETE_ARGS:
            if key in self.arguments and not isinstance(self.arguments[key], (list, dict)):
                parts.append(f"{key}={self.arguments[key]}")
        return ":".join(str(p) for p in parts)


@dataclass
class Candidate:
    candidate_id: str
    scope: str
    signature: list
    occurrences: list = field(default_factory=list)
    varying_inputs: list = field(default_factory=list)
    constants: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def load_steps(run_dir: Path) -> list[Step]:
    rows = read_table(run_dir, "tool_calls")
    return [Step(run=Path(run_dir).name, scope=r.get("scope", "task"), delegation=r.get("delegation"),
                 decision=int(r.get("decision", 0)), tool=r["tool"], arguments=r.get("arguments") or {},
                 rationale=r.get("rationale") or "", status=r.get("status"), ok=bool(r.get("ok")))
            for r in rows]


def segments(steps: list[Step]) -> list[list[Step]]:
    """Split a run into reusable stretches: same scope/delegation, successful motion calls only."""
    out, current, key = [], [], None
    for step in steps:
        step_key = (step.run, step.scope, step.delegation)
        boundary = (step.tool in BOUNDARY_TOOLS or step.tool in INTERPRETATION_TOOLS or not step.ok
                    or step_key != key)
        if boundary and current:
            out.append(current)
            current = []
        key = step_key
        if step.tool not in BOUNDARY_TOOLS and step.tool not in INTERPRETATION_TOOLS and step.ok:
            current.append(step)
    if current:
        out.append(current)
    return [s for s in out if len(s) >= 2]


def mine(run_dirs: list[Path], *, min_len: int = 2, max_len: int = 6, min_count: int = 2) -> list[Candidate]:
    counts: dict[tuple, list] = defaultdict(list)
    for run_dir in run_dirs:
        for segment in segments(load_steps(run_dir)):
            roles: dict = {}
            sigs = [s.signature(roles) for s in segment]
            for n in range(min_len, min(max_len, len(segment)) + 1):
                for i in range(len(segment) - n + 1):
                    counts[(segment[0].scope, tuple(sigs[i:i + n]))].append(segment[i:i + n])
    ranked = sorted(((k, v) for k, v in counts.items() if len(v) >= min_count),
                    key=lambda kv: (-len(kv[1][0]) * len(kv[1]), kv[0]))
    chosen: list[Candidate] = []
    covered: set = set()
    for (scope, signature), occurrences in ranked:
        ids = {(s.run, s.decision, s.tool) for occ in occurrences for s in occ}
        if ids <= covered:
            continue
        covered |= ids
        candidate = Candidate(f"c{len(chosen) + 1}", scope, list(signature))
        for occ in occurrences:
            candidate.occurrences.append([{"run": s.run, "delegation": s.delegation, "decision": s.decision,
                                           "tool": s.tool, "arguments": s.arguments, "rationale": s.rationale[:400],
                                           "status": s.status} for s in occ])
        _parameters(candidate)
        chosen.append(candidate)
    return chosen


def _parameters(candidate: Candidate) -> None:
    """Arguments that differ across occurrences become inputs; identical ones stay constants."""
    for index, sig in enumerate(candidate.signature):
        values = defaultdict(set)
        for occ in candidate.occurrences:
            for key, value in occ[index]["arguments"].items():
                values[key].add(json.dumps(value, sort_keys=True))
        for key, seen in values.items():
            name = f"step{index + 1}.{key}"
            if len(seen) > 1 or key in ("target", "arm"):
                candidate.varying_inputs.append(name)
            else:
                candidate.constants[name] = json.loads(next(iter(seen)))


SKILL_TEMPLATE = '''"""<one-line summary>. Consolidated from <candidate id and source links>."""
import numpy as np
from knowin_agent.runtime.motion import move_flange, set_gripper
from knowin_agent.skills.base import SkillSpec, arm_binding
from knowin_agent.skills.builtin import _goto, _working_point, _spin_about, _tool_axis, _status, arm_is_free, gripper_closed


def body(ctx, rt, arm, target, ...):
    # compose motion programs with `yield from`; never call rt.move_to / run_skill here
    ...
    return {"status": "...", ...}


SKILL = SkillSpec("<skill_id>", "<summary>", {<inputs json schema>}, [<required>], body, arm_binding,
                  [arm_is_free], outputs="...", termination="...", source={"origin": "consolidated", "candidate": "<id>", "links": [...]})
'''


def build_prompt(candidate: Candidate) -> str:
    occurrences = json.dumps(candidate.occurrences[:6], ensure_ascii=False, indent=1)
    return f"""You are consolidating a recurring robot interaction into a reusable KnowinAgent skill.

Candidate {candidate.candidate_id} ({candidate.scope} scope) repeats this call sequence:
{json.dumps(candidate.signature, ensure_ascii=False)}

Source occurrences (calls, arguments, the agent's rationale, outcome):
{occurrences}

Inputs that varied across occurrences: {candidate.varying_inputs}
Arguments that stayed constant: {json.dumps(candidate.constants, ensure_ascii=False)}

Write one Python module that defines SKILL = SkillSpec(...). Rules:
- Expose object identities, arm assignment and spatial arguments as inputs (or recompute them through perception
  inside the body); keep the reusable operation structure as explicit program logic.
- Stop the procedure where fresh model interpretation is needed; return a receipt with a `status` and the evidence
  the agent needs to decide what comes next. Never claim task success.
- Declare applicability guards (arm free, gripper state, reference freshness) and termination behaviour.
- Keep links to the source calls in `source`.
Template:
{SKILL_TEMPLATE}
Reply with the module in a single ```python block."""


def extract_code(reply: str) -> str:
    match = re.search(r"```python\n(.*?)```", reply, re.S)
    if not match:
        raise ValueError("reply has no ```python block")
    return match.group(1)


def write_candidates(candidates: list[Candidate], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "candidates.json"
    path.write_text(json.dumps([c.as_dict() for c in candidates], ensure_ascii=False, indent=1), encoding="utf-8")
    for candidate in candidates:
        (out_dir / f"{candidate.candidate_id}.prompt.md").write_text(build_prompt(candidate), encoding="utf-8")
    return path


def generate(candidate: Candidate, model, out_dir: Path, skill_name: Optional[str] = None) -> Path:
    """Ask a coding model (ChatModel) to write the skill module for one candidate."""
    turn = model.complete([{"role": "system", "content": "You write small, correct Python robot skills."},
                           {"role": "user", "content": build_prompt(candidate)}], [])
    code = extract_code(turn.text)
    path = Path(out_dir) / f"{skill_name or candidate.candidate_id}.py"
    path.write_text(code, encoding="utf-8")
    return path


def validate(skill_dir: Path, *, smoke: Optional[dict] = None) -> dict:
    """Load every consolidated skill; optionally run named skills once on the mock world.

    smoke maps skill_id -> bindings for a mock run (objects 'red block' and 'blue block' exist).
    """
    from knowin_agent.backends.mock import TABLE_Z, Box, MockBackend, MockPerceiver
    from knowin_agent.core.memory import TaskContext
    from knowin_agent.runtime.api import RobotRuntime
    from knowin_agent.skills.library import SkillLibrary
    import numpy as np

    library = SkillLibrary()
    report = {"loaded": library.load_dir(Path(skill_dir)), "smoke": {}}
    for skill_id, bindings in (smoke or {}).items():
        world = MockBackend([Box("red block", np.array([-0.15, 0.10, TABLE_Z + 0.02]), (0.06, 0.03, 0.04)),
                             Box("blue block", np.array([0.15, 0.12, TABLE_Z + 0.02]), (0.04, 0.04, 0.04))])
        rt = RobotRuntime(world, context=TaskContext("validation"), perceiver=MockPerceiver(world))
        try:
            receipt = library.run(rt, skill_id, dict(bindings))
            report["smoke"][skill_id] = {"ok": True, "status": receipt.get("status")}
        except Exception as exc:  # noqa: BLE001 - report every validation failure
            report["smoke"][skill_id] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return report

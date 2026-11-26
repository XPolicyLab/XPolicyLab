"""Skill library: fixed within an episode, extended offline between episodes.

`run` binds parameters, checks guards (applicability, arm availability, the
contract when Local calls knob_cycle), then executes the body as one track and
returns an execution receipt. No agent model is invoked during a skill.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from knowin_agent.core.events import EventKind
from knowin_agent.skills.base import SkillSpec
from knowin_agent.skills.builtin import PREPARE, builtin_skills


class SkillError(RuntimeError):
    pass


class SkillLibrary:
    def __init__(self, skills: Optional[list[SkillSpec]] = None, *, include_builtin: bool = True):
        self._skills: dict[str, SkillSpec] = {}
        self.frozen = False
        for spec in (builtin_skills() if include_builtin else []) + list(skills or []):
            self.register(spec)

    def register(self, spec: SkillSpec) -> None:
        if self.frozen:
            raise SkillError("the skill library is fixed during an episode")
        self._skills[spec.skill_id] = spec

    def load_dir(self, directory: Path) -> list[str]:
        """Load consolidated skills: every *.py file that defines SKILL = SkillSpec(...)."""
        loaded = []
        for path in sorted(Path(directory).glob("*.py")):
            module_spec = importlib.util.spec_from_file_location(f"knowin_agent_skill_{path.stem}", path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            spec = getattr(module, "SKILL", None)
            if not isinstance(spec, SkillSpec):
                raise SkillError(f"{path.name} does not define SKILL = SkillSpec(...)")
            spec.source = {**spec.source, "file": path.name}
            self.register(spec)
            loaded.append(spec.skill_id)
        return loaded

    def freeze(self) -> None:
        self.frozen = True

    def ids(self) -> list[str]:
        return list(self._skills)

    def get(self, skill_id: str) -> SkillSpec:
        if skill_id not in self._skills:
            raise SkillError(f"unknown skill {skill_id!r}; available: {self.ids()}")
        return self._skills[skill_id]

    def catalog(self) -> list[dict]:
        return [{"skill_id": s.skill_id, "summary": s.summary, "required": s.required} for s in self._skills.values()]

    def info(self, skill_id: str) -> dict:
        return self.get(skill_id).info()

    def run(self, runtime, skill_id: str, bindings: dict, *, owner: str = "task", wait: bool = True) -> dict:
        spec = self.get(skill_id)
        bindings = dict(bindings or {})
        unknown = sorted(set(bindings) - set(spec.inputs))
        if unknown:
            raise SkillError(f"{skill_id} has no inputs {unknown}; inputs: {sorted(spec.inputs)}")
        if skill_id in PREPARE:
            try:
                bindings = PREPARE[skill_id](runtime, bindings)
            except (KeyError, ValueError) as exc:
                raise SkillError(f"{skill_id}: {exc}") from None
        missing = [k for k in spec.required if k not in bindings]
        if missing:
            raise SkillError(f"{skill_id} needs {missing}")
        for guard in spec.guards:
            reason = guard(runtime, bindings)
            if reason:
                raise SkillError(f"{skill_id} guard failed: {reason}")
        arms = spec.arms(bindings)
        runtime.bus.publish(EventKind.SKILL_STARTED, owner, f"skill {skill_id} on {'+'.join(arms)}",
                            tick=runtime.tick, skill=skill_id, bindings=_brief(bindings))
        track = runtime.executor.submit(arms, lambda ctx: spec.body(ctx, runtime, **bindings),
                                        label=f"skill {skill_id}", owner=owner)
        if not wait:
            return {"track": track.id, "status": "running"}
        runtime.executor.run([track])
        result = dict(track.result or {})
        result.setdefault("status", track.status.value)
        receipt = {"skill": skill_id, "track": track.id, "ticks": (track.finished_tick or 0) - (track.started_tick or 0),
                   **result}
        runtime.bus.publish(EventKind.SKILL_FINISHED, owner, f"skill {skill_id}: {receipt.get('status')}",
                            tick=runtime.tick, skill=skill_id, status=receipt.get("status"))
        return receipt


def _brief(bindings: dict) -> dict:
    return {k: (v if not isinstance(v, (list, tuple)) else [round(x, 4) if isinstance(x, float) else x for x in v])
            for k, v in bindings.items()}

"""One inner Agent for every task, with checkpoint-backed WAM Skills."""
from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path
from .release import ROOT

class Backend:
    def before(self, action): pass
    def completed(self, executed, planned, interrupted): pass

def register_vendor():
    """Expose retained private policies without replacing XPolicyLab shared helpers."""
    import XPolicyLab.policy
    curobo_root = ROOT / "vendor/runtime_support/curobo"
    loaded = sys.modules.get("curobo")
    if loaded is not None and (not getattr(loaded, "__file__", None)
                               or not Path(loaded.__file__).resolve().is_relative_to(curobo_root.resolve())):
        raise RuntimeError("An incompatible CuRobo was imported before the frozen runtime")
    if str(curobo_root) not in sys.path:
        sys.path.insert(0, str(curobo_root))
    directory = str(ROOT / "vendor/policy")
    if directory not in XPolicyLab.policy.__path__:
        XPolicyLab.policy.__path__ = [directory, *list(XPolicyLab.policy.__path__)]
    names = ("RoboDojo_Agent_L3_Inspect", "RoboDojo_Agent_L3_Inspect_EEF", "RoboRSI_RoboProbe")
    for name in names:
        module = importlib.import_module("XPolicyLab.policy." + name)
        if Path(module.__file__).resolve().parent != Path(directory) / name:
            raise RuntimeError("An incompatible Agent policy was imported before the frozen vendor")


class AgentBackend(Backend):
    def __init__(self, spec, task, deployment):
        register_vendor()
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.deploy import _debug_action_spec
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, Observation
        from XPolicyLab.policy.RoboRSI_RoboProbe.contracts import strict_settings
        from XPolicyLab.policy.RoboRSI_RoboProbe.execution_monitor import ExecutionMonitor
        from XPolicyLab.policy.RoboRSI_RoboProbe.planner import planner_for
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        from .provider import ResponsesClient
        self.action_type, self.observation_type = Action, Observation
        self.task = task
        self.steps = 0
        self.last = None
        self.current = None
        self.reasons = []
        self.spec = _debug_action_spec()  # Reads real robot URDF/config, not fixture limits.
        self.planner = planner_for()
        self.client = ResponsesClient(spec)
        self.policy = SkillEefPolicy(action_spec=self.spec, env=strict_settings(os.environ),
                                     planner=self.planner, task_id=task, client=self.client)
        self.policy._keep_all_images = True
        self.monitor = ExecutionMonitor(self.spec)
        from .wam import WamSkills
        self.wam = WamSkills(spec["wam_skills"], task, deployment)
        self.policy.wam_skills = self.wam
        self.policy.wam_call = self._wam_call
        self.policy.wam_catalog = sorted(spec["wam_skills"])

    def reset(self):
        self.policy.reset()
        self.client.reset()
        self.steps = 0
        self.current = self.last = None
        self.reasons = []
        self.wam.close()

    def observe(self, observation, executed_action):
        if executed_action is not None:
            self.steps += 1
        current = self.observation_type(
            images={key: value["color"] for key, value in observation["vision"].items()},
            state=observation["state"], instruction=observation["instruction"],
            step=self.steps, remaining_steps=None, extra={"task": self.task})
        self.reasons = []
        if executed_action is not None:
            event = self.monitor.after(self.action_type(data=executed_action, meta={}), self.current, current)
            self.reasons = event["reasons"]
        self.wam.observe(observation, executed_action, bool(self.reasons))
        self.current = current
        return bool(self.reasons)

    def plan(self):
        self.planner.set_observation(self.current)
        chunk = self.policy.act(self.current)
        if not chunk.meta.get("wam_skill"):
            self.wam.native_selected()
        self.monitor.begin(self.current)
        from XPolicyLab.policy.RoboRSI_RoboProbe.tools import HeldActions
        if isinstance(chunk.actions, HeldActions):
            from .plans import RepeatedAction
            return RepeatedAction(dict(chunk.actions.data), len(chunk.actions))
        return [dict(item.data) for item in chunk.actions]

    def before(self, action):
        self.wam.before(action)
        self.monitor.before(self.action_type(data=action, meta={}), self.current)

    def completed(self, executed, planned, interrupted):
        if self.wam.continuous is not None:
            # WAM may change grippers without the native EEF planner's latches.
            self.policy._last_targets = {}
            self.policy._gripper_goals = {
                arm: float(self.current.state[arm + "_ee_joint_state"][0])
                for arm in ("left", "right")}
        self.policy.confirm_executed(executed)
        self.policy.execution_feedback(self.current, executed, planned, self.reasons)

    def _wam_call(self, skill_id, observation):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure, MotionOutcome
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, ActionChunk
        try:
            actions = self.wam.plan(skill_id)
        except Exception as error:
            # Never convert ambiguous inference into a repair/replay request.
            raise InfrastructureFailure("WAM Skill failed; end this trial without inference replay") from error
        chunk = ActionChunk(actions=[Action(data=a, meta={}) for a in actions],
                            control_hz=self.spec.control_hz, meta={"wam_skill": skill_id})
        return MotionOutcome(chunk=chunk, tool_result="WAM proposed " + str(len(actions)) +
                             " native actions; execution is confirmed by subsequent observations.")

    def close(self):
        try:
            self.wam.close()
        finally:
            self.client.close()


def build_backend(name, spec, task, deployment):
    if name != "inner_agent" or spec.get("kind") != "agent":
        raise ValueError("Every task must use the common inner Agent")
    return AgentBackend(spec, task, deployment)

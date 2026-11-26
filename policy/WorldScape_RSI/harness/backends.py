"""One inner Agent for every task, with checkpoint-backed WAM Skills."""
from __future__ import annotations
import importlib
import json
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
    def __init__(self, spec, deployment):
        register_vendor()
        from .robot_spec import load_robot_spec, actuator_command_limits
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, Observation
        from XPolicyLab.policy.RoboRSI_RoboProbe.contracts import strict_settings
        from XPolicyLab.policy.RoboRSI_RoboProbe.execution_monitor import ExecutionMonitor
        from XPolicyLab.policy.RoboRSI_RoboProbe.planner import planner_for
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        from .provider import ResponsesClient
        self.action_type, self.observation_type = Action, Observation
        self.steps = 0
        self.last = None
        self.current = None
        self.reasons = []
        self.spec = load_robot_spec()  # Reads real robot URDF/config, not fixture limits.
        self.planner = planner_for()
        self.client = ResponsesClient(spec)
        self.policy = SkillEefPolicy(action_spec=self.spec, env=strict_settings(os.environ),
                                     planner=self.planner, client=self.client)
        self.policy._keep_all_images = True
        self.monitor = ExecutionMonitor(self.spec)
        profile = spec.get('wam_command_profile', 'native_planner_budget')
        if profile not in {'native_planner_budget', 'actuator_velocity_budget'}:
            raise ValueError('Unknown frozen WAM command profile')
        self.wam_command_limits = (actuator_command_limits(self.spec)
                                   if profile == 'actuator_velocity_budget' else self.spec.max_step)
        from .wam import WamSkills
        self.wam = WamSkills(spec["wam_skills"], deployment)
        self.policy.wam_skills = self.wam
        self.policy.wam_call = self._wam_call
        self.policy.wam_catalog = sorted(spec["wam_skills"])
        self.policy.trace_checkpoint = self._audit
        self._ensure_control()
        self._trace_cursor = 0
        self._trace_last = None

    def drain_trace(self):
        """New and completed tool records, including actual arguments and feedback."""
        records = getattr(self.policy, "_transcript", [])
        start = max(0, self._trace_cursor - 1)
        events = []
        for index in range(start, len(records)):
            row = dict(records[index])
            row["event_index"] = index
            if index >= self._trace_cursor or row != self._trace_last:
                events.append(row)
        self._trace_cursor = len(records)
        self._trace_last = dict(records[-1], event_index=len(records)-1) if records else None
        return events

    def _ensure_control(self):
        from .control_guard import ControlGuard
        if not hasattr(self,'control'): self.control=ControlGuard()
        self.policy.control_guard=self.control
        return self.control

    def review_pending(self):
        if self.wam.pending is None: return False
        guard=self._ensure_control()
        # The model boundary calls here only after update_obs acknowledged execution.
        decision=self.policy.review_wam_execution(self.current,guard.status())
        takeover=guard.review(decision)
        self._audit()
        if takeover:
            self.wam.interrupt_pending()
            self.reasons=['Agent requested takeover from fresh public observation']
        return takeover

    def _audit(self):
        """Bounded private tool/execution evidence, without prompts or images."""
        destination = os.environ.get("R2SI_DIAGNOSTICS_DIR")
        if not destination:
            return
        root = Path(destination).resolve()
        allowed = [Path(p).resolve() for p in os.environ.get("WSP_WRITABLE_ROOTS", "").split(os.pathsep) if p]
        if not any(root.is_relative_to(p) for p in allowed):
            raise RuntimeError("Agent audit directory must be inside the writable workspace")
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in getattr(self.policy, "_transcript", [])[-128:]:
            args = item.get("arguments", {})
            nested = args.get("arguments", {}) if isinstance(args, dict) else {}
            rows.append({k: item[k] for k in ("tool", "policy_step", "accepted", "execution",
                         "validation_error", "latency_s") if k in item} | {
                "skill": args.get("id") if isinstance(args, dict) else None,
                "op": nested.get("op") if isinstance(nested, dict) else None})
        if hasattr(self,'control'):
            control_path = root / ('control-'+self.client.session_id+'.json')
            control_temp = control_path.with_suffix('.tmp')
            control_temp.write_text(json.dumps({'status':self.control.status(),'reviews':self.control.history},allow_nan=False),encoding='utf8')
            control_temp.replace(control_path)
        path = root / ("actions-" + self.client.session_id + ".json")
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(step=self.steps, records=rows)), encoding="utf8")
        temp.replace(path)

    def reset(self):
        self.policy.reset()
        self._trace_cursor = 0
        self._trace_last = None
        self.client.reset()
        self.steps = 0
        self.current = self.last = None
        self.reasons = []
        self.wam.close()
        self._ensure_control().reset()

    def observe(self, observation, executed_action):
        if executed_action is not None:
            self.steps += 1
        current = self.observation_type(
            images={key: value["color"] for key, value in observation["vision"].items()},
            state=observation["state"], instruction=observation["instruction"],
            step=self.steps, remaining_steps=None, extra={})
        self.reasons = []
        if executed_action is not None:
            event = self.monitor.after(self.action_type(data=executed_action, meta={}), self.current, current)
            self.reasons = event["reasons"]
        self._ensure_control().observe(observation,self.steps,executed_action,self.wam.pending is not None)
        self.wam.observe(observation, executed_action, bool(self.reasons))
        self.current = current
        return bool(self.reasons)

    def plan(self):
        self.planner.set_observation(self.current)
        chunk = self.policy.act(self.current)
        self._audit()
        if not chunk.meta.get("wam_skill"):
            self.wam.native_selected()
        self.monitor.set_command_limits(self.wam_command_limits if chunk.meta.get('wam_skill')
                                        else self.spec.max_step)
        self.monitor.begin(self.current)
        from XPolicyLab.policy.RoboRSI_RoboProbe.tools import HeldActions
        if isinstance(chunk.actions, HeldActions):
            from .plans import RepeatedAction
            return RepeatedAction(dict(chunk.actions.data), len(chunk.actions))
        return [dict(item.data) for item in chunk.actions]

    def before(self, action):
        self.monitor.before(self.action_type(data=action, meta={}), self.current)
        self.wam.before(action)

    def completed(self, executed, planned, interrupted):
        if self.wam.continuous is not None:
            # WAM may change grippers without the native EEF planner's latches.
            self.policy._last_targets = {}
            self.policy._gripper_goals = {
                arm: float(self.current.state[arm + "_ee_joint_state"][0])
                for arm in ("left", "right")}
        self.policy.confirm_executed(executed)
        self.policy.execution_feedback(self.current, executed, planned, self.reasons)
        self._audit()

    def _wam_call(self, skill_id, observation):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure, MotionOutcome
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, ActionChunk
        try:
            self._ensure_control().consume(skill_id,self.wam.continuous)
        except ValueError as error:
            return MotionOutcome(chunk=None,tool_result=json.dumps({'error':str(error),'executed_steps':0}),repairable=True)
        try:
            actions = self.wam.plan(skill_id)
        except Exception as error:
            # Never convert ambiguous inference into a repair/replay request.
            raise InfrastructureFailure("WAM Skill failed; end this trial without inference replay") from error
        chunk = ActionChunk(actions=[Action(data=a, meta={}) for a in actions],
                            control_hz=self.spec.control_hz, meta={"wam_skill": skill_id})
        try:
            self.monitor.validate_plan(chunk.actions, observation, self.wam_command_limits)
        except InfrastructureFailure as error:
            self.wam.reject_unissued()
            return MotionOutcome(chunk=None, tool_result=json.dumps({
                "error": str(error), "executed_steps": 0,
                "proposal_discarded": True,
                "guidance": "No action executed. Reassess from the unchanged observation; use a bounded native motion or another suitable Skill."}),
                repairable=True)
        return MotionOutcome(chunk=chunk, tool_result="WAM proposed " + str(len(actions)) +
                             " native actions; execution is confirmed by subsequent observations.")

    def close(self):
        # Diagnostic I/O failure must not skip retirement of model sessions.
        errors = []
        for operation in (self._audit, self.wam.close, self.client.close):
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]


def build_backend(name, spec, deployment):
    if name != "inner_agent" or spec.get("kind") != "agent":
        raise ValueError("Every task must use the common inner Agent")
    return AgentBackend(spec, deployment)

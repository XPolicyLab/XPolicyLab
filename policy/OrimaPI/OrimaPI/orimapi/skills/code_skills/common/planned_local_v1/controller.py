"""GPT task decomposition/semantic guidance; deterministic local execution."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from orimapi.skills.code_skills.align_blocks.generator import Generator
from orimapi.skills.code_skills.classify_objects_by_language.controller import Controller as OpenController
from orimapi.skills.code_skills.common.geometry import rgb

from .adapter import api_class
from .geometry import Measurements, unit, validate_box
from .operations import Operations

# Required and optional fields, not free-form GPT code or low-level robot calls.
CONTRACTS = {
    "pick": ({"arm", "source"}, {"approach", "grasp_fraction"}),
    "place": ({"arm", "destination"}, {"mode", "slot", "front_left"}),
    "transfer": ({"arm", "source", "destination"}, {"mode", "slot", "approach", "grasp_fraction"}),
    "sort_row": ({"arm", "source", "destination"}, {"approach", "grasp_fraction"}),
    "dock": ({"arm", "source", "destination"}, {"entry_sign", "release"}),
    "handover": ({"arm", "source", "receiver"}, set()),
    "stroke": ({"arm", "source", "destination"}, set()),
    "push": ({"arm", "source", "destination"}, set()),
    "intercept": ({"arm", "source"}, set()),
    "inspect": (set(), {"object"}),
    "measure_feature": ({"object", "feature"}, set()),
    "view": ({"arm", "object"}, set()),
    "home": ({"arm"}, set()),
    "wait": ({"seconds"}, set()),
}


class ToolFailure(ValueError):
    def __init__(self, command, result):
        super().__init__(f"{command}: {result}")
        self.result = deepcopy(result)


def validate_arguments(operation, args, allowed):
    if operation not in allowed or operation not in CONTRACTS or not isinstance(args, dict):
        raise ValueError("Choose an actual advertised local composite operation")
    required, optional = CONTRACTS[operation]
    if required-args.keys() or args.keys()-required-optional:
        raise ValueError(f"{operation} requires {sorted(required)}, optional {sorted(optional)}")
    for field in ("arm", "receiver"):
        if field in args and args[field] not in (
            ("left", "right", "both") if operation == "home" else ("left", "right")
        ):
            raise ValueError("Invalid arm")
    for field in ("source", "destination", "object", "feature"):
        if field in args and (not isinstance(args[field], str) or not args[field]):
            raise ValueError(f"Expected semantic {field} identity")
    if "mode" in args and args["mode"] not in {"support", "inside"}:
        raise ValueError("place mode is support or inside")
    if "slot" in args:
        slot = np.asarray(args["slot"], float)
        if slot.shape != (2,) or not np.isfinite(slot).all() or np.max(abs(slot)) > 1:
            raise ValueError("slot is normalized container XY in [-1,1], not metric XYZ")
    if "approach" in args:
        unit(args["approach"])
    if "grasp_fraction" in args and (
        type(args["grasp_fraction"]) not in (int, float) or not 0 < args["grasp_fraction"] < 1
    ):
        raise ValueError("grasp_fraction must lie within the measured body")
    if "entry_sign" in args and (type(args["entry_sign"]) is not int or args["entry_sign"] not in (-1, 1)):
        raise ValueError("entry_sign is -1 or 1 along the measured triangle normal")
    for field in ("front_left", "release"):
        if field in args and type(args[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    if operation == "wait" and (
        type(args["seconds"]) not in (int, float) or not 0 <= args["seconds"] <= 5
    ):
        raise ValueError("Use the existing 0..5 second wait operation")
    return deepcopy(args)


class Controller(OpenController):
    def __init__(self, parameters, directory):
        self.directory = Path(directory)
        self.profile = json.loads((self.directory/"profile.json").read_text())
        self.budget = self.profile["budget"]
        if parameters.get("task_spec", {}).get("budget", self.budget) != self.budget:
            raise ValueError("Frozen task and profile budgets must agree")
        super().__init__(parameters)

    def reset(self):
        super().reset()
        self.plan = None
        self.held = {}
        self.releases = []
        self.ledger = {"phase": None, "reviews": [], "stop_reason": None}
        self.measure = self.operations = None

    def emit(self, event, **fields):
        if self.trace:
            self.trace.write("code_skill_tool_"+event, policy_id=self.policy_id,
                skill_id=self.parameters.get("skill_id", self.profile["task"]),
                episode_id=self.episode_id, **fields)

    def exchange(self, action):
        with self.cv:
            if self.cancelled:
                return False
            if self.pending is not None:
                raise RuntimeError("Only one native action may be outstanding")
            before = self.latest.step
            self.pending = deepcopy(action)
            self.cv.notify_all()
            while not self.cancelled and self.latest.step == before:
                self.cv.wait(timeout=1)
            return not self.cancelled and self.latest.step < self.budget

    def tool(self, command, arguments):
        """Internal tool dispatcher, never an LLM command interface."""
        result, code = OpenController.execute(self, command, arguments)
        self.emit("local_tool", step=self.latest.step, command=command,
                  arguments=arguments, result=result, exit_code=code)
        if code or result.get("plan_ok") is False:
            raise ToolFailure(command, result)
        return result

    def query(self, phase):
        if self.calls >= self.max_model_calls:
            raise RuntimeError("Original skill model-call budget exhausted")
        context = {
            "task": self.latest.instruction, "task_profile": self.profile,
            "experience": self.parameters.get("experience", []),
            "step": self.latest.step, "remaining_native_actions": self.budget-self.latest.step,
            "remaining_model_calls": self.max_model_calls-self.calls,
            "task_plan": self.plan, "ledger": self.ledger, "held_candidates": self.held,
            "release_commands_not_verified_successes": self.releases,
            "previous_operations": self.history[-12:],
            "local_operations": {
                name: {"required": sorted(CONTRACTS[name][0]), "optional": sorted(CONTRACTS[name][1])}
                for name in self.profile["operations"]
            },
            "camera_sizes": {("head" if k == "cam_high" else
                "wrist_l" if k == "cam_left_wrist" else "wrist_r"):
                [rgb(v).shape[1], rgb(v).shape[0]] for k, v in self.latest.images.items()},
            "public_table_z": self.api.robot.table,
            "observed_robot_state": self.latest.robot_state,
            "feedback_scope": "No force or measured finger-stop sensor; native scoring is never an online oracle",
            "review_hold_contract": (
                "Optional review fields arm=left|right and holding=empty|supported|uncertain: "
                "explicit current-RGB empty clears that arm's held HYPOTHESIS so a missed grasp "
                "can be retried in this same stage. Uncertainty never clears it or blocks progress. "
                "Do not infer empty from occlusion or commanded jaw value."
            ),
        }
        self.calls += 1
        images = {("cam_head" if k == "cam_high" else k): v
                  for k, v in self.latest.images.items()}
        return self.generator.query(phase, context, images)

    def accept_plan(self, reply):
        if not isinstance(reply, dict) or set(reply) != {"phases", "targets", "groups", "reason"}:
            raise ValueError("Plan requires phases, targets, groups, reason")
        phases, targets = reply["phases"], reply["targets"]
        if not isinstance(phases, list) or not phases:
            raise ValueError("Decompose into nonempty semantic phases")
        ids = []
        for phase in phases:
            if not isinstance(phase, dict) or set(phase) != {"id", "goal"}:
                raise ValueError("Each phase needs id and goal")
            if any(not isinstance(phase[k], str) or not phase[k] for k in phase):
                raise ValueError("Phase id/goal must be nonempty strings")
            ids.append(phase["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("Distinct phase ids required")
        if (not isinstance(targets, list) or not targets or any(not isinstance(x, str) or not x for x in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("Distinct semantic target identities required")
        expected = self.profile.get("target_count")
        if expected is not None and len(targets) != expected:
            raise ValueError(f"This task requires {expected} targets, not {len(targets)}")
        groups = reply["groups"]
        if not isinstance(groups, list) or not isinstance(reply["reason"], str):
            raise ValueError("groups list and reason text required")
        if self.profile.get("group_counts"):
            if any(not isinstance(group, list) for group in groups):
                raise ValueError("Toy groups contain semantic piece identities")
            flattened = [x for group in groups for x in group]
            if (any(not isinstance(x, str) for x in flattened)
                    or sorted(map(len, groups)) != sorted(self.profile["group_counts"])
                    or sorted(flattened) != sorted(targets)):
                raise ValueError("All ten toy pieces must be assigned once to groups 4/3/2/1")
        self.plan = deepcopy(reply)
        self.emit("task_decomposed", step=self.latest.step, plan=self.plan)

    def dispatch(self, reply):
        required = {"phase", "operation", "arguments", "objects", "review", "reason"}
        if not isinstance(reply, dict) or set(reply) != required:
            raise ValueError("Stage requires phase, operation, arguments, objects, review, reason")
        if reply["phase"] not in {p["id"] for p in self.plan["phases"]}:
            raise ValueError("Stage must refer to the current task plan")
        if not isinstance(reply["reason"], str) or not isinstance(reply["objects"], dict):
            raise ValueError("Current coarse object map and reason required")
        objects = {key: validate_box(value, rgb(self.latest.images["cam_high"]))
                   for key, value in reply["objects"].items()}
        review = reply["review"]
        if review is not None:
            if (not isinstance(review, dict) or not {"status", "evidence"} <= review.keys()
                    or review.keys()-{"status", "evidence", "arm", "holding"}
                    or review["status"] not in {"supported", "uncertain", "failed"}
                    or not isinstance(review["evidence"], str) or not review["evidence"]):
                raise ValueError("Review is null or status/evidence; it is advisory, not a motion veto")
            if ("arm" in review) != ("holding" in review):
                raise ValueError("Held-state review uses arm AND holding together")
            if "arm" in review:
                if review["arm"] not in {"left", "right"} or review["holding"] not in {"empty", "supported", "uncertain"}:
                    raise ValueError("Invalid held-state observation")
                side = review["arm"]
                if review["holding"] == "empty":
                    previous = self.held.pop(side, None)
                    self.emit("held_hypothesis_cleared_by_rgb_review", step=self.latest.step,
                              arm=side, previous=previous, review=review,
                              model_report_not_contact_truth=True)
                elif side in self.held:
                    self.held[side]["retention"] = "model_"+review["holding"]
            self.ledger["reviews"].append({**review, "step": self.latest.step,
                                          "model_report_not_native_verdict": True})
        self.ledger["phase"] = reply["phase"]
        if reply["operation"] == "finish":
            if (set(reply["arguments"]) != {"reason"}
                    or not isinstance(reply["arguments"]["reason"], str)
                    or not reply["arguments"]["reason"]):
                raise ValueError("finish needs an explicit completion/stop reason")
            self.ledger["stop_reason"] = reply["arguments"]["reason"]
            self.finished = True
            return {"policy_stopped": True, "official_success": None}, 0
        args = validate_arguments(reply["operation"], reply["arguments"], self.profile["operations"])
        result = self.operations.run(reply["operation"], args, objects)
        return result, 0

    def run(self):
        try:
            self.api = api_class()(self)
            self.measure, self.operations = Measurements(self), Operations(self)
            prompts = json.loads((Path(__file__).with_name("prompts.json")).read_text())
            self.generator = Generator(self.parameters["llm"],
                self.trace.root/"code_episodes"/self.episode_id/"planned-local",
                prompts=prompts, emit=self.emit)
            # Invalid plan is feedback for a corrected plan, not terminal skill failure.
            while self.plan is None and not self.cancelled and not self.api.over:
                try:
                    self.accept_plan(self.query("plan"))
                except ValueError as error:
                    self.history.append({"phase": "plan", "contract_error": str(error)})
                    self.emit("plan_contract_error", step=self.latest.step, error=str(error))
            while not self.cancelled and not self.finished and not self.api.over:
                reply = self.query("stage")
                start = self.latest.step
                self.operations.stages, self.operations.name = [], None
                try:
                    result, code = self.dispatch(reply)
                except Exception as error:
                    result, code = {
                        "plan_ok": False, "failure_type": type(error).__name__,
                        "error": str(error),
                        "raw_tool_failure": getattr(error, "result", None),
                        "failed_stage": self.operations.name,
                        "executed_local_stages": deepcopy(self.operations.stages),
                        "held_candidates": deepcopy(self.held),
                        "recovery_tools": self.profile["operations"],
                        "new_evidence_required": "Change geometry/arm/view/route based on this failure; do not retry unchanged",
                    }, 1
                row = {"operation": reply, "start_step": start, "end_step": self.latest.step,
                       "result": result, "exit_code": code, "official_success": None}
                self.history.append(deepcopy(row))
                self.emit("planned_operation_result", **row)
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.emit("terminal_failure", step=self.latest.step, error=self.failure)
        finally:
            try:
                if self.api is not None:
                    self.api.close()
            finally:
                with self.cv:
                    self.finished = True
                    self.cv.notify_all()

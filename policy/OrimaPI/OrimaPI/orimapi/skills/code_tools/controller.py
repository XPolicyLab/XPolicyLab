"""Shared tool-execution mode using a selected frozen code skill as reference."""

from copy import deepcopy
import hashlib
import uuid

from orimapi.skills.code_skills.common.geometry import RobotGeometry
from orimapi.skills.code_skills.align_blocks.generator import Generator
from orimapi.skills.common import is_new_observation

from . import depth_availability, identity, settings, snapshot
from .depth import DepthEstimator
from .prompt import prompt_for, catalogue
from .segmentation import SamSegmenter
from .session import ToolSession
from .recipes import RecipeLibrary
from .grasp_preview import recognizes_calibration, validate_resources as validate_finger_resources


class Controller:
    def __init__(self, parameters):
        self.parameters = parameters
        self.settings = settings(parameters)
        self.geometry = RobotGeometry(parameters["geometry"])
        self.nominal_finger_sources = (
            validate_finger_resources(self.geometry)
            if recognizes_calibration(self.geometry)
            else None
        )
        self.trace = self.policy_id = self.episode_id = None
        self.generator = None
        self.closed = False
        self.source_identity = identity()
        self.recipes = RecipeLibrary(parameters.get("tool_recipes"))
        depth = (
            DepthEstimator(self.settings["depth"])
            if depth_availability(self.settings.get("depth"))["available"]
            else None
        )
        self.session = ToolSession(
            self.geometry,
            self.emit,
            depth=depth,
            action_budget=parameters.get("task_spec", {}).get("budget"),
            recipes=self.recipes,
            require_cartesian_completion=self.settings.get("require_cartesian_completion", False),
            camera_pointing=self.settings.get("camera_pointing", False),
            point_pursuit=self.settings.get("point_pursuit", False),
            segmentation=(
                SamSegmenter(self.settings["sam_proposals"])
                if self.settings.get("sam_proposals") else None
            ),
        )
        self.reset()

    def bind_trace(self, trace, *, policy_id):
        self.trace, self.policy_id = trace, policy_id
        if self.session.segmentation is not None:
            self.session.segmentation.bind_trace(trace.root)
        if snapshot(trace) != self.source_identity:
            raise ValueError("Code tools changed before load")
        self.recipes.snapshot(trace.root)
        for key, path in self.geometry.resources.items():
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != self.geometry.hashes[key]:
                raise ValueError("Public calibration changed before snapshotting")
            target = (
                trace.root / "code_assets" / (self.geometry.hashes[key] + path.suffix)
            )
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(content)
            elif target.read_bytes() != content:
                raise ValueError("Public calibration snapshot changed")
        if self.nominal_finger_sources is not None:
            if validate_finger_resources(self.geometry) != self.nominal_finger_sources:
                raise ValueError("Nominal finger resources changed before snapshotting")
            for relative, sha in self.nominal_finger_sources.items():
                source = self.geometry.resources["robot_urdf"].parent / relative
                content = source.read_bytes()
                target = trace.root / "code_assets" / (sha + source.suffix)
                if hashlib.sha256(content).hexdigest() != sha:
                    raise ValueError("Public finger mesh changed before snapshotting")
                if target.exists():
                    if target.read_bytes() != content:
                        raise ValueError("Public finger mesh snapshot changed")
                else:
                    with target.open("xb") as stream:
                        stream.write(content)
        self.emit(
            "loaded",
            identity=self.source_identity,
            depth_model=self.session.depth.identity if self.session.depth else None,
            **(
                {"point_pursuit_enabled": True}
                if self.session.point_pursuit else {}
            ),
            **(
                {"camera_pointing_enabled": True}
                if self.session.camera_pointing else {}
            ),
            **(
                {"segmentation_model": self.session.segmentation.identity}
                if self.session.segmentation is not None else {}
            ),
            **(
                {"nominal_finger_model_sources": self.nominal_finger_sources}
                if self.nominal_finger_sources is not None else {}
            ),
        )

    def emit(self, event, **fields):
        if self.trace is not None:
            self.trace.write(
                "code_skill_tool_" + event,
                policy_id=self.policy_id,
                skill_id=self.parameters.get("skill_id"),
                episode_id=self.episode_id,
                **fields,
            )

    def reset(self):
        self.session.reset()
        self.latest = self.generator = self.episode_id = None
        self.calls = 0
        self.history = []
        self._remaining_tools_in_decision = None

    def begin_episode(self, episode_id):
        if self.latest is not None:
            raise RuntimeError("Reset tools before starting another episode")
        self.episode_id = episode_id

    def observe(self, observation):
        if self.closed:
            raise RuntimeError("Code tools controller closed")
        if not is_new_observation(self.latest, observation):
            return
        self.session.observe(observation)
        self.latest = deepcopy(observation)

    def _query(self):
        if self.calls >= self.settings["max_calls"]:
            self.emit(
                "failure", kind="policy", reason="Tool model-call budget exhausted"
            )
            raise RuntimeError("Code tool model-call budget exhausted")
        if identity() != self.source_identity:
            raise ValueError("Code tools changed during the episode")
        if self.generator is None:
            if self.trace is None:
                raise ValueError("Code tools require a bound run trace")
            self.episode_id = self.episode_id or uuid.uuid4().hex
            self.generator = Generator(
                self.parameters["llm"],
                self.trace.root / "code_episodes" / self.episode_id / "tools",
                prompts={"tool": prompt_for(self.parameters)},
                emit=self.emit,
            )
        context = {
            "task": self.latest.instruction,
            "step": self.latest.step,
            "task_spec": self.parameters.get("task_spec"),
            "skill_id": self.parameters.get("skill_id"),
            "experience": self.parameters.get("experience", []),
            "reference_program": self.parameters.get("best_program"),
            "robot_state": self.latest.robot_state,
            "available_tools": catalogue(self.parameters),
            "available_recipes": self.recipes.describe(),
            "state": self.session.describe(),
            "history": self.history[-self.settings["history_size"] :],
            "remaining_model_calls": self.settings["max_calls"] - self.calls,
        }
        if self.settings.get("report_observation_budget", False):
            remaining = self._remaining_tools_in_decision
            if remaining is None:
                raise ValueError("Observation budget feedback requires an active decision loop")
            episode_remaining = self.settings["max_calls"] - self.calls
            context["observation_tool_budget"] = {
                "limit": self.settings["max_tools_per_turn"],
                "remaining_including_current": remaining,
                "episode_remaining_including_current": episode_remaining,
                "effective_remaining": min(remaining, episode_remaining),
                "plan_and_execute_calls": 2,
                "scope": "Current decision only; includes this query. No guard or limit is changed.",
            }
        self.calls += 1
        return self.generator.query("tool", context, self.session.query_images())

    def predict(self, observation, *, prompt, max_actions):
        if (
            type(max_actions) is not int
            or max_actions < 1
            or prompt != observation.instruction
        ):
            raise ValueError(
                "Code tools require the original task and a positive action budget"
            )
        self.observe(observation)
        action = self.session.action()
        if action is not None:
            return [action]
        if self.session.finished:
            return [self.session.hold()]
        try:
            for index in range(self.settings["max_tools_per_turn"]):
                self._remaining_tools_in_decision = self.settings["max_tools_per_turn"] - index
                reply = self._query()
                try:
                    if (
                        set(reply) != {"tool", "arguments", "reason"}
                        or not isinstance(reply["tool"], str)
                        or not isinstance(reply["reason"], str)
                    ):
                        raise ValueError("Expected exactly tool, arguments and reason")
                    self.emit("call", step=observation.step, request=reply)
                    result = self.session.call(reply["tool"], reply["arguments"])
                except (ValueError, TypeError, KeyError) as error:
                    result = {
                        "status": "rejected",
                        "error": str(error),
                        "official_success": None,
                    }
                self.history.append(
                    {"step": observation.step, "request": reply, "result": result}
                )
                self.history = self.history[-self.settings["history_size"] :]
                self.emit("result", step=observation.step, request=reply, result=result)
                action = self.session.action()
                if action is not None:
                    return [action]
                if self.session.finished:
                    return [self.session.hold()]
            raise RuntimeError(
                "Code tools exceeded bounded non-motion calls at one observation"
            )
        finally:
            self._remaining_tools_in_decision = None

    def close(self):
        self.reset()
        self.closed = True

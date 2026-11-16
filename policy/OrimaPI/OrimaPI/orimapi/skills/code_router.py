"""Per-episode GPT sub-skill selection inside the single code-policy backend."""

from copy import deepcopy
import hashlib
import time
import uuid

from ..evaluation.storage import atomic_json
from ..supervisor import SupervisorEpisodeError
from .code_llm import create_json_client
from .common import action_limit, is_new_observation
from .code_tools.prompt import catalogue


class CodeSkillRouter:
    def __init__(self, config):
        self.config = config
        self.closed = self.loaded = False
        self._trace = self._policy_id = self._settings = self._client = None
        self._children = {}
        self._bound = {}
        self._latest = {}
        self._episodes = {}
        self._failed = {}
        self._catalogue = {
            key: {
                "description": row.description,
                "action_fields": list(config.action_fields),
                "action_dims": list(config.action_dims),
            }
            for key, row in config.code_skills.items()
        }

    def _open(self):
        if self.closed:
            raise RuntimeError("The code-policy router is closed")

    def bind_trace(self, trace, *, policy_id):
        self._open()
        if self.loaded:
            raise RuntimeError("Bind the trace before loading code-policy")
        self._trace, self._policy_id = trace, policy_id

    def configure_generation(self, settings, *, timeout_s):
        self._open()
        if self.loaded:
            raise RuntimeError("Configure code-policy before loading")
        self._settings = {**deepcopy(settings), "timeout_s": timeout_s}

    def load(self):
        self._open()
        if self.loaded:
            return
        self._prompt = self.config.code_selection_prompt.read_text(encoding="utf-8")
        self._prompt_sha256 = hashlib.sha256(self._prompt.encode()).hexdigest()
        if self._trace is not None:
            self._trace.write(
                "code_skill_catalogued",
                policy_id=self._policy_id,
                available_skills=self._catalogue,
                selection_prompt=self._prompt,
                prompt_sha256=self._prompt_sha256,
            )
        self.loaded = True

    def describe(self):
        return {
            "name": "code-policy",
            "loaded": self.loaded and not self.closed,
            "description": "GPT selects a registered sub-skill and decides whether to use shared visual/geometry tools or its existing generated controller, guided by frozen experience and source.",
            "supports_subgoals": False,
            "supported_subgoals": [],
            "sub_skills": deepcopy(self._catalogue),
            "action_fields": list(self.config.action_fields),
            "action_dims": list(self.config.action_dims),
        }

    def prepare_job(self, seed):
        self.load()
        self.reset()
        if (
            self.config.code_selection_prompt.read_text(encoding="utf-8")
            != self._prompt
        ):
            raise ValueError(
                "Code sub-skill selection prompt changed during evaluation"
            )
        for child in self._children.values():
            child.prepare_job(seed)
        return {
            "policy_seed": seed,
            "backend": "code-policy",
            "warmup_s": 0.0,
            "cached_sub_skills": len(self._children),
        }

    def begin_episode(self, env_id, episode_id):
        self._open()
        if env_id in self._bound or env_id in self._latest:
            raise RuntimeError("Reset code-policy before beginning a new episode")
        self._episodes[env_id] = episode_id

    def _emit(self, event, observation, **fields):
        if self._trace is not None:
            self._trace.write(
                event,
                policy_id=self._policy_id,
                episode_id=self._episodes[observation.env_id],
                env_id=observation.env_id,
                step=observation.step,
                **fields,
            )

    def observe(self, observation):
        self.load()
        env_id = observation.env_id
        if env_id in self._failed:
            raise RuntimeError("Code sub-skill failed; reset this environment first")
        self._episodes.setdefault(env_id, uuid.uuid4().hex)
        if is_new_observation(self._latest.get(env_id), observation):
            try:
                if env_id in self._bound:
                    self._children[self._bound[env_id]].observe(deepcopy(observation))
                self._latest[env_id] = deepcopy(observation)
            except Exception as error:
                self._failed[env_id] = str(error)
                raise

    def _select(self, observation):
        available_tools = catalogue(self.config.code_parameters)
        context = {
            "task_goal": observation.instruction,
            "robot_state": observation.robot_state,
            "observation_step": observation.step,
            "available_skills": deepcopy(self._catalogue),
            "available_tools": available_tools,
        }
        directory = None
        if self._trace is not None:
            directory = (
                self._trace.root / "code_episodes" / self._episodes[observation.env_id]
            )
            atomic_json(
                directory / "selection-request.json",
                {
                    "context": context,
                    "current_cameras": list(observation.images),
                    "prompt_sha256": self._prompt_sha256,
                },
            )
        self._emit(
            "code_skill_selection_requested",
            observation,
            available_skills=self._catalogue,
            instruction=observation.instruction,
        )
        started = time.monotonic()
        try:
            if self._settings is None:
                raise SupervisorEpisodeError(
                    "Configure the LLM for code sub-skill selection"
                )
            if self._client is None:
                self._client = create_json_client(self._settings)
            reply = self._client.query_json(context, observation.images, self._prompt)
            if directory is not None:
                atomic_json(directory / "selection-response.json", reply)
            if (
                not isinstance(reply, dict)
                or not {"skill_id", "reason"} <= reply.keys()
                or reply.keys() - {"skill_id", "reason", "tools"}
                or any(
                    not isinstance(reply[key], str) or not reply[key].strip()
                    for key in ("skill_id", "reason")
                )
            ):
                raise ValueError(
                    "Expected skill_id and reason from code sub-skill selection"
                )
            skill_id = reply["skill_id"]
            requested_tools = reply.get("tools", [])
            if (
                not isinstance(requested_tools, list)
                or len(requested_tools) > len(available_tools)
                or any(
                    not isinstance(name, str)
                    or name not in available_tools
                    or not available_tools[name]["available"]
                    for name in requested_tools
                )
                or len(set(requested_tools)) != len(requested_tools)
            ):
                raise ValueError("Select distinct currently available tool names")
            if skill_id not in self.config.code_skills:
                raise ValueError(
                    f"No registered code sub-skill selected: {reply['reason']}"
                )
        except Exception as error:
            self._emit("code_skill_selection_failed", observation, error=str(error))
            raise SupervisorEpisodeError(
                f"Code sub-skill selection failed: {error}"
            ) from error
        self._emit(
            "code_skill_selected",
            observation,
            skill_id=skill_id,
            reason=reply["reason"],
            tools=requested_tools,
            elapsed_s=time.monotonic() - started,
        )
        if skill_id not in self._children:
            from .code_policy import CodePolicy

            child = CodePolicy(
                self.config.code_skills[skill_id].policy_config(self.config)
            )
            try:
                child.configure_generation(
                    self._settings, timeout_s=self._settings["timeout_s"]
                )
                if self._trace is not None:
                    child.bind_trace(self._trace, policy_id=self._policy_id)
                child.load()
            except BaseException:
                child.close()
                raise
            self._children[skill_id] = child
        child = self._children[skill_id]
        if requested_tools:
            child.use_tools(observation.env_id)
        child.begin_episode(observation.env_id, self._episodes[observation.env_id])
        self._bound[observation.env_id] = skill_id
        child.observe(deepcopy(observation))

    def predict(self, observation, *, prompt, max_actions=None):
        self._open()
        action_limit(max_actions)
        if prompt != observation.instruction:
            raise ValueError(
                "Code sub-skills require the original whole-task instruction"
            )
        self.observe(observation)
        try:
            if observation.env_id not in self._bound:
                self._select(observation)
            return self._children[self._bound[observation.env_id]].predict(
                observation, prompt=prompt, max_actions=max_actions
            )
        except Exception as error:
            self._failed[observation.env_id] = str(error)
            raise

    def reset(self, env_id=None):
        self._open()
        if env_id is None:
            for child in self._children.values():
                child.reset()
            self._bound.clear()
            self._latest.clear()
            self._episodes.clear()
            self._failed.clear()
        else:
            self._bound.pop(env_id, None)
            for child in self._children.values():
                child.reset(env_id)
            self._latest.pop(env_id, None)
            self._episodes.pop(env_id, None)
            self._failed.pop(env_id, None)

    def close(self):
        if self.closed:
            return
        failure = None
        for child in self._children.values():
            try:
                child.close()
            except Exception as error:
                failure = failure or error
        self._children.clear()
        self._bound.clear()
        self._latest.clear()
        self._episodes.clear()
        self._failed.clear()
        self._client = None
        self.closed = True
        if failure is not None:
            raise failure

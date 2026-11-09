# coding=utf-8
"""Code-policy backend: route sub-skills or load one trusted local controller.

The plugin loader is not a sandbox and only loads configured trusted controllers
in-process. Generated programs are checked by the selected skill's isolated worker.
"""

from __future__ import annotations

import hashlib
import builtins
import sys
import time
import types
import uuid
from copy import deepcopy

from ..state import Observation
from ..xpolicylab.evaluation import fast_trajectory_chunk_size
from .common import is_new_observation, action_limit, validate_code_actions
from .initial_experience import InitialSkillExperience, initial_experience_scope


def _program_geometry(program):
    for candidate in (
        getattr(program, "geometry", None),
        getattr(getattr(program, "session", None), "geometry", None),
        getattr(getattr(program, "motion", None), "geometry", None),
        getattr(getattr(program, "api", None), "robot", None),
        getattr(getattr(program, "api", None), "robot_geometry", None),
    ):
        if candidate is not None and callable(getattr(candidate, "fk", None)):
            return candidate
    return None


def _advance_synthetic_observation(
    observation: Observation, action: dict, geometry
) -> Observation:
    state = {key: list(value) for key, value in observation.robot_state.items()}
    for side in ("left", "right"):
        arm_key = f"{side}_arm_joint_state"
        ee_key = f"{side}_ee_joint_state"
        if arm_key in action:
            state[arm_key] = [float(x) for x in action[arm_key]]
        if ee_key in action:
            state[ee_key] = [float(x) for x in action[ee_key]]
    if geometry is not None and (
        "left_ee_pose" in state or "right_ee_pose" in state
    ):
        from scipy.spatial.transform import Rotation

        for side in ("left", "right"):
            pose_key = f"{side}_ee_pose"
            arm_key = f"{side}_arm_joint_state"
            if pose_key in state and arm_key in state:
                matrix = geometry.fk(side, state[arm_key], full=True)[6]
                quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
                state[pose_key] = [
                    *matrix[:3, 3].tolist(),
                    *quat[[3, 0, 1, 2]].tolist(),
                ]
    return Observation(
        observation.env_id,
        observation.step + 1,
        observation.instruction,
        observation.images,
        state,
    )


def _motion_has_interior_steps(motion) -> bool:
    if (
        motion is None
        or getattr(motion, "failure", None) is not None
        or getattr(motion, "operation_start", None) is None
    ):
        return False
    plan = getattr(motion, "plan", None)
    index = getattr(motion, "index", 0)
    if not isinstance(plan, list) or not (0 <= index < len(plan)):
        return False
    row = plan[index]
    if not isinstance(row, dict) or row.get("op") not in {"move", "dual_move"}:
        return False
    cursor = getattr(motion, "cursor", 0)
    if row["op"] == "move":
        path = getattr(motion, "path", None)
        return isinstance(path, list) and 0 <= cursor < len(path) - 2
    paths = getattr(motion, "paths", None)
    left = paths.get("left") if isinstance(paths, dict) else None
    return isinstance(left, list) and 0 <= cursor < len(left) - 2


def _worker_in_interior_trajectory(worker) -> bool:
    ident = getattr(worker, "ident", None)
    if ident is None or not worker.is_alive():
        return False
    frame = sys._current_frames().get(ident)
    depth = 0
    while frame is not None and depth < 16:
        name = frame.f_code.co_name
        loc = frame.f_locals
        if name == "execute" and "path" in loc and "target" in loc:
            import numpy as np

            path = loc["path"]
            target = loc["target"]
            try:
                return (
                    len(path) >= 3
                    and not np.array_equal(target, path[-1])
                    and not np.array_equal(target, path[-2])
                )
            except TypeError:
                return False
        if name == "run" and "sequences" in loc and "i" in loc:
            sequences = loc["sequences"]
            i = loc["i"]
            if isinstance(sequences, dict) and sequences and type(i) is int:
                max_len = max((len(seq) for seq in sequences.values()), default=0)
                return 0 <= i < max_len - 2
        if name == "run" and "paths" in loc and "index" in loc:
            paths = loc["paths"]
            index = loc["index"]
            if isinstance(paths, dict) and paths and type(index) is int:
                max_len = max((len(seq) for seq in paths.values()), default=0)
                return 0 <= index < max_len - 2
        frame = frame.f_back
        depth += 1
    return False


def _can_chunk_next_trajectory_step(program) -> bool:
    if (
        getattr(program, "failure", None) is not None
        or getattr(program, "finished", False)
        or getattr(program, "cancelled", False)
    ):
        return False
    if getattr(program, "held_opening", None) is not None:
        return False
    session = getattr(program, "session", None)
    if session is not None:
        if (
            getattr(session, "finished", False)
            or getattr(session, "pursuit", None) is not None
            or getattr(session, "sequence", None) is not None
        ):
            return False
        tracker = getattr(session, "tracker", None)
        points = getattr(tracker, "points", None)
        if isinstance(points, dict) and any(
            isinstance(p, dict) and p.get("status") == "observed"
            for p in points.values()
        ):
            return False
        active = getattr(session, "active", None)
        if isinstance(active, dict):
            plan = active.get("plan")
            if isinstance(plan, dict) and plan.get("evidence"):
                return False
        elif not getattr(program, "active", False) and getattr(program, "phase", None) != "motion":
            return False
        return _motion_has_interior_steps(getattr(session, "motion", None))
    motion = getattr(program, "motion", None)
    if motion is not None:
        if getattr(program, "phase", "motion") != "motion":
            return False
        return _motion_has_interior_steps(motion)
    worker = getattr(program, "worker", None)
    cv = getattr(program, "cv", None)
    if worker is not None and cv is not None:
        return _worker_in_interior_trajectory(worker)
    api = getattr(program, "api", None)
    if api is not None and getattr(api, "worker", None) is not None:
        return _worker_in_interior_trajectory(api.worker)
    return False


class CodePolicy:
    def __init__(self, config):
        self.config = config
        self._router = None
        if config.code_skills:
            from .code_router import CodeSkillRouter

            self._router = CodeSkillRouter(config)
        self.closed = False
        self._module = None
        self._factory = None
        self._source_sha256 = None
        self._programs = {}
        self._initial_experiences = {}
        self._tool_envs = set()
        self._failed = {}
        self._latest = {}
        self._supports_subgoals = False
        self._subgoals = []
        self._trace = None
        self._policy_id = None
        self._bundle = None
        self._generation_settings = None
        self._description = "Caller-supplied local code policy; capabilities must be declared by its program."

    def _open(self):
        if self.closed:
            raise RuntimeError("The code-policy skill is closed")

    def configure_generation(self, settings, *, timeout_s):
        self._open()
        if self._router is not None:
            return self._router.configure_generation(settings, timeout_s=timeout_s)
        if self._factory is not None:
            raise RuntimeError("Configure generation before loading code")
        self._generation_settings = {**deepcopy(settings), "timeout_s": timeout_s}

    def prepare_job(self, seed):
        if self._router is not None:
            return self._router.prepare_job(seed)
        self.load()
        self.reset()
        if self._bundle is not None:
            self._bundle.verify()
        return {"policy_seed": seed, "backend": "code-policy", "warmup_s": 0.0}

    def begin_episode(self, env_id, episode_id):
        if self._router is not None:
            return self._router.begin_episode(env_id, episode_id)
        if self._bundle is not None:
            self._bundle.verify()
        with self._experience_scope(env_id):
            program = self._program(env_id)
            begin = getattr(program, "begin_episode", None)
            if callable(begin):
                begin(episode_id)

    def bind_trace(self, trace, *, policy_id):
        """Attach run evidence for trusted programs; this grants no isolation."""
        self._open()
        if self._router is not None:
            return self._router.bind_trace(trace, policy_id=policy_id)
        if self._factory is not None:
            raise RuntimeError("Bind the trace before loading code")
        self._trace, self._policy_id = trace, policy_id

    def _read_source(self):
        path = self.config.code_path
        if path is None:
            raise ValueError("Provide --code-path or policies.<id>.code_path")
        if path.suffix != ".py" or not path.is_file():
            raise ValueError(f"Expected an existing Python program file: {path}")
        source = path.read_bytes()
        sha = hashlib.sha256(source).hexdigest()
        expected = self._source_sha256 or self.config.code_sha256
        if expected is not None and sha != expected:
            raise ValueError("Code policy source changed or SHA-256 does not match")
        return source, sha

    def load(self):
        self._open()
        if self._router is not None:
            return self._router.load()
        if self._factory is not None:
            return
        source, sha = self._read_source()
        if self.config.code_bundle is not None:
            from .code_bundle import CodeBundle

            self._bundle = CodeBundle(self.config.code_bundle)
            if self._trace is not None:
                snapshot = self._bundle.snapshot(self._trace.root)
                self._trace.write(
                    "code_skill_bundle_loaded",
                    policy_id=self._policy_id,
                    skill_id=self._bundle.skill_id,
                    bundle_sha256=self._bundle.sha256,
                    snapshot=snapshot,
                )
        if self._trace is not None:
            folder = self._trace.root / "code_sources"
            folder.mkdir(exist_ok=True)
            snapshot = folder / (sha + ".py")
            if snapshot.exists():
                if snapshot.read_bytes() != source:
                    raise ValueError("Code source snapshot differs from loaded bytes")
            else:
                with snapshot.open("xb") as stream:
                    stream.write(source)
            self._trace.write(
                "code_source_snapshot",
                policy_id=self._policy_id,
                source_sha256=sha,
                source_path=str(snapshot.relative_to(self._trace.root)),
                original_path=str(self.config.code_path),
            )
        name = "_orimapi_code_" + uuid.uuid4().hex
        module = types.ModuleType(name)
        module.__file__ = str(self.config.code_path)
        if self._bundle is not None and self._bundle.frozen_tools is not None:
            # Only this trusted controller's imports are redirected. The
            # controller file and the qualified tool sources stay byte-exact.
            module.__dict__["__builtins__"] = {
                **vars(builtins),
                "__import__": self._bundle.frozen_tools.importer,
            }
        sys.modules[name] = module
        try:
            # Execute the exact file bytes whose identity was checked above.
            exec(compile(source, str(self.config.code_path), "exec"), module.__dict__)
            factory = getattr(module, self.config.code_entrypoint, None)
            if not callable(factory):
                raise ValueError(
                    f"Program must export {self.config.code_entrypoint}(parameters)"
                )
            supports = getattr(module, "SUPPORTS_SUBGOALS", False)
            subgoals = getattr(module, "SUPPORTED_SUBGOALS", [])
            description = getattr(module, "CAPABILITY_DESCRIPTION", self._description)
            if (
                type(supports) is not bool
                or not isinstance(description, str)
                or not description.strip()
            ):
                raise ValueError("Invalid code-policy capability declaration")
            if (
                not isinstance(subgoals, (list, tuple))
                or len(subgoals) > 128
                or any(
                    not isinstance(goal, str) or not goal.strip() for goal in subgoals
                )
                or len(subgoals) != len(set(subgoals))
            ):
                raise ValueError(
                    "SUPPORTED_SUBGOALS must contain distinct nonempty strings"
                )
            if subgoals and not supports:
                raise ValueError("SUPPORTED_SUBGOALS requires SUPPORTS_SUBGOALS = True")
        except BaseException:
            sys.modules.pop(name, None)
            if self._bundle is not None and self._bundle.frozen_tools is not None:
                self._bundle.frozen_tools.close()
            raise
        self._module, self._factory, self._source_sha256 = module, factory, sha
        self._supports_subgoals, self._subgoals, self._description = (
            supports,
            list(subgoals),
            description,
        )

    def describe(self):
        if self._router is not None:
            return self._router.describe()
        return {
            "name": "code-policy",
            "loaded": self._factory is not None and not self.closed,
            "code_path": str(self.config.code_path) if self.config.code_path else None,
            "entrypoint": self.config.code_entrypoint,
            "source_sha256": self._source_sha256,
            "supports_subgoals": self._supports_subgoals,
            "supported_subgoals": list(self._subgoals),
            "description": self._description,
            "action_fields": list(self.config.action_fields),
            "action_dims": list(self.config.action_dims),
            "execution": "trusted local program in the policy process",
            "code_bundle": (
                str(self.config.code_bundle) if self.config.code_bundle else None
            ),
        }

    def _experience_scope(self, env_id):
        if self._bundle is not None and env_id not in self._initial_experiences:
            self._initial_experiences[env_id] = InitialSkillExperience(
                self._bundle, env_id=env_id, trace=self._trace, policy_id=self._policy_id
            )
        return initial_experience_scope(self._initial_experiences.get(env_id))

    def _program(self, env_id):
        program = self._programs.get(env_id)
        if program is None:
            if self._bundle is not None:
                self._bundle.verify()
            parameters = deepcopy(self.config.code_parameters)
            if self._bundle is not None:
                if self._generation_settings is None:
                    raise ValueError("Configure the LLM for a generated code skill")
                parameters.update(self._bundle.parameters())
                parameters["llm"] = deepcopy(self._generation_settings)
            if env_id in self._tool_envs and not (self._bundle and self._bundle.tool_only):
                from .code_tools.controller import Controller

                program = Controller(parameters)
            else:
                program = self._factory(parameters)
            required = ("observe", "predict", "reset", "close")
            if any(not callable(getattr(program, method, None)) for method in required):
                close = getattr(program, "close", None)
                if callable(close):
                    close()
                raise TypeError(
                    "Code factory must return a program with observe/predict/reset/close"
                )
            if self._trace is not None and callable(
                getattr(program, "bind_trace", None)
            ):
                try:
                    program.bind_trace(self._trace, policy_id=self._policy_id)
                except BaseException:
                    program.close()
                    raise
            self._programs[env_id] = program
        return program

    def use_tools(self, env_id):
        """Router calls this only after GPT requests shared tools for this episode."""
        if self._router is not None or self._bundle is None or env_id in self._latest:
            raise ValueError(
                "Select tools for a loaded task bundle before the episode starts"
            )
        previous = self._programs.pop(env_id, None)
        if previous is not None:
            previous.close()
        self._tool_envs.add(env_id)

    def observe(self, observation: Observation):
        if self._router is not None:
            return self._router.observe(observation)
        self.load()
        if observation.env_id in self._failed:
            raise RuntimeError("Code program failed; reset this environment first")
        if is_new_observation(self._latest.get(observation.env_id), observation):
            if self._bundle is not None and observation.env_id not in self._latest:
                self._bundle.verify()
            self._read_source()
            try:
                with self._experience_scope(observation.env_id):
                    self._program(observation.env_id).observe(deepcopy(observation))
                self._latest[observation.env_id] = deepcopy(observation)
            except Exception as error:
                self._failed[observation.env_id] = str(error)
                raise

    def predict(
        self, observation: Observation, *, prompt: str, max_actions: int | None = None
    ) -> list[dict]:
        if self._router is not None:
            return self._router.predict(
                observation, prompt=prompt, max_actions=max_actions
            )
        if max_actions is None:
            raise ValueError(
                "Code policy requires an explicit max_actions execution budget"
            )
        action_limit(max_actions)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty policy instruction is required")
        self.observe(observation)
        if not self._supports_subgoals and prompt != observation.instruction:
            raise ValueError(
                "This code program only supports the original task instruction; use direct mode or a subgoal-capable program"
            )
        if self._subgoals and prompt not in self._subgoals:
            raise ValueError("Subgoal is not declared by the code program")
        try:
            with self._experience_scope(observation.env_id):
                program = self._programs[observation.env_id]
                result = program.predict(
                    deepcopy(observation), prompt=prompt, max_actions=max_actions
                )
                if (
                    max_actions > 1
                    and fast_trajectory_chunk_size() > 1
                    and isinstance(result, list)
                    and len(result) == 1
                ):
                    chunk = list(result)
                    current_obs = observation
                    geometry = _program_geometry(program)
                    while len(chunk) < max_actions and _can_chunk_next_trajectory_step(
                        program
                    ):
                        next_obs = _advance_synthetic_observation(
                            current_obs, chunk[-1], geometry
                        )
                        self._latest[observation.env_id] = deepcopy(next_obs)
                        step_actions = program.predict(
                            deepcopy(next_obs), prompt=prompt, max_actions=1
                        )
                        if not isinstance(step_actions, list) or len(step_actions) != 1:
                            break
                        chunk.append(step_actions[0])
                        current_obs = next_obs
                    result = chunk
            return validate_code_actions(result, self.config, max_actions)
        except Exception as error:
            self._failed[observation.env_id] = str(error)
            raise

    def reset(self, env_id: int | None = None):
        self._open()
        if self._router is not None:
            return self._router.reset(env_id)
        indices = (
            list(
                set(self._programs)
                | set(self._failed)
                | set(self._latest)
                | set(self._initial_experiences)
                | self._tool_envs
            )
            if env_id is None
            else [env_id]
        )
        for index in indices:
            program = self._programs.get(index)
            if program is not None:
                if index in self._tool_envs:
                    program.close()
                    self._programs.pop(index, None)
                else:
                    program.reset()
            self._tool_envs.discard(index)
            self._latest.pop(index, None)
            self._failed.pop(index, None)
            self._initial_experiences.pop(index, None)

    def close(self):
        if self.closed:
            return
        if self._router is not None:
            try:
                self._router.close()
            finally:
                self.closed = True
            return
        failure = None
        for program in self._programs.values():
            try:
                program.close()
            except Exception as error:
                failure = failure or error
        self._programs.clear()
        self._initial_experiences.clear()
        self._tool_envs.clear()
        self._failed.clear()
        self._latest.clear()
        if self._module is not None:
            sys.modules.pop(self._module.__name__, None)
        if self._bundle is not None and self._bundle.frozen_tools is not None:
            self._bundle.frozen_tools.close()
        self._module = self._factory = None
        self.closed = True
        if failure is not None:
            raise failure

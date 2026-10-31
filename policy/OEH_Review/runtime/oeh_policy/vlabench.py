"""Pinned VLABench/OpenWAM evaluation boundary, separate from RoboDojo.

VLABench cf588fe's ``Evaluator`` (evaluation/evaluator/base.py) runs each episode itself: it builds the scene from the
track's frozen episode config, asks the agent for ``predict(observation)`` once per native step, turns the returned
``(pos, euler, gripper_state)`` into joint targets by inverse kinematics and steps the simulator. OpenWAM's released
client (XPolicyLab fa431ec benchmarks/vlabench/openwam2vlabench_interface.py) is that agent in the direct condition:
it sends three RGB views, the instruction and its EEF10 state to the OpenWAM server and converts the returned EEF10
action with ``eef10_to_vlabench_ee``. In the OEH condition the agent is ``oeh_policy.vlabench_entry.VLABenchEndpoint``,
which runs the OEH runtime in the same process and asks the same released client object for OpenWAM's rows; this
module is the contract between the two: what the policy observes, what a native action is, and the review port's
hooks. It never changes a quaternion order, converts an action itself or reads success.

Review mode only (the registry's ``review``): the teacher reviews one fresh full OpenWAM chunk per decision
(``OpenWAMBackend.review_chunk``) and the official loop still takes one native row per ``predict`` call.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import yaml

from .action_space import vlabench_row
from .contracts import check_review_benchmark, digest, file_digest, observation_digest, plain, validate_review
from .profiles import SINGLE_ARM_EE, check_embodiment
from .vlabench_bounds import PACKAGE, name2config, series_of

VLABENCH_COMMIT = "cf588fe60c0c7282174fe979f5913170cfe69017"
# One native step is one control step of VLABench's dm_control environment: 100 physics substeps
# (tasks/dm_task.py:25 NUM_SUBSTEPS, :62 control_timestep = physics_timestep * NUM_SUBSTEPS) of 1 ms
# (assets/base/default.xml:5 timestep 0.001; configs/task_config.json:6 the same), and OpenWAM's single_eval.py passes
# max_substeps=1, so each predict is one env.step: 0.1 s, 10 Hz.
CONTROL_HZ = 10.0
# The teacher's views and the index of each in the native ``obs["rgb"]`` (the released client's head, left-wrist and
# right-wrist cameras: policy_config.yml head_camera_index 2, left_wrist_camera_index 3, right_wrist_camera_index 0).
# Index 1 (VLABench's "left" camera) is never sent to OpenWAM.
CAMERAS = (("forward", 2), ("wrist", 3), ("right", 0))
NATIVE_CAMERAS = 4
# The released client's settings the camera map and the action contract above depend on (policy_config.yml).
CLIENT_SETTINGS = {"head_camera_index": 2, "left_wrist_camera_index": 3, "right_wrist_camera_index": 0,
                   "send_state": True, "state_dim": 10, "action_dim": 10, "gripper_open_threshold": 0.5,
                   "gripper_open_width": 0.04}
# evaluation/evaluator/base.py:72 (the Evaluator's default) unless the task's series sets
# evaluation.max_episode_length in configs/task_config.json.
DEFAULT_STEP_LIMIT = 200
GRIPPER_SEMANTICS = "measured_finger_gap"
POLICY_REQUIRED = {"schema", "benchmark", "mode", "kernel", "repository", "output", "model_id", "checkpoint_sha256",
                   "normalization_sha256", "identity_files", "max_chunk_steps", "native_step_limit", "wall_seconds",
                   "decision_seconds", "action_limits", "vlabench", "embodiment", "review", "review_workspace",
                   "gripper_state_semantics"}


def package_files(track: str) -> list[str]:
    """The VLABench package files an episode's evaluation reads (paths inside the checkout's ``VLABench/``)."""
    return ["evaluation/evaluator/base.py", "envs/__init__.py", "envs/dm_env.py", "tasks/dm_task.py",
            "configs/__init__.py", "configs/task_config.json", "configs/robot_config.json", "utils/utils.py",
            "robots/base.py", "robots/single_arm/base.py", "robots/single_arm/franka.py", "assets/base/default.xml",
            "assets/robots/franka_emika_panda/panda.xml", f"configs/evaluation/tracks/{track}.json"]


OPENWAM_FILES = ("benchmarks/vlabench/single_eval.py", "benchmarks/vlabench/openwam2vlabench_interface.py",
                 "benchmarks/vlabench/policy_config.yml", "benchmarks/utils/__init__.py",
                 "benchmarks/utils/action_conversion.py", "benchmarks/utils/client.py", "benchmarks/utils/transport.py",
                 "openwam/deploy/server.py", "configs/deploy.yaml", "configs/dataloader/vlabench.yaml",
                 "scripts/deploy.sh")


def required_identity_paths(native) -> list[Path]:
    root = Path(native["root"]).resolve() / PACKAGE
    openwam = Path(native["openwam_root"]).resolve()
    return ([root / name for name in package_files(native["track"])] +
            [openwam / name for name in OPENWAM_FILES] +
            [Path(native["official_client_path"]).resolve(), Path(native["single_eval_path"]).resolve()])


def step_limit(root, task: str) -> int:
    """The Evaluator's ``max_episode_length`` for a task, read as evaluation/evaluator/base.py:72-75 reads it: 200,
    unless the task's series (``find_key_by_value(name2config, task)``) has a truthy entry in configs/task_config.json
    whose ``evaluation.max_episode_length`` is truthy."""
    package = Path(root).resolve() / PACKAGE
    configs = json.loads((package / "configs/task_config.json").read_text(encoding="utf-8"))
    series = series_of(task, name2config(package / "configs/__init__.py"))
    limit = DEFAULT_STEP_LIMIT
    entry = configs.get(series, None)
    if entry:
        value = (entry.get("evaluation", {}) or {}).get("max_episode_length", None)
        if value:
            limit = value
    if type(limit) is not int or limit < 1:
        raise ValueError(f"VLABench step limit for {task} is {limit!r}, not a positive integer")
    return limit


def track_episodes(root, track: str, task: str) -> list:
    """The frozen episode configs a track lists for a task (configs/evaluation/tracks/<track>.json)."""
    path = Path(root).resolve() / PACKAGE / "configs/evaluation/tracks" / f"{track}.json"
    episodes = json.loads(path.read_text(encoding="utf-8")).get(task)
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path.name} lists no episodes for {task}")
    return episodes


def _array(value, shape, name, dtype=np.float64):
    array = np.asarray(value, dtype=dtype).reshape(-1) if shape[0] != -1 else np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be {shape} finite values")
    return array.copy()


def _rgb(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
        raise ValueError(f"{name} must be nonempty uint8 RGB")
    return np.ascontiguousarray(image).copy()


def project_state(state) -> dict:
    """The VLABench state OEH reads: the end-effector site's world pose with upstream's open flag (``ee_state``), the
    two finger joints' positions and the robot base position (``robot_frame``), as finite float64 copies."""
    return {"ee_state": _array(state["ee_state"], (8,), "ee_state"),
            "finger_qpos": _array(state["finger_qpos"], (2,), "finger_qpos"),
            "robot_frame": _array(state["robot_frame"], (3,), "robot_frame")}


def project_observation(obs) -> dict:
    """What the released client reads of one native observation (its three RGB views, ``ee_state``, ``robot_frame``
    and the instruction), plus the measured finger positions the evaluator reads beside it. Same values, dtypes and
    shapes as ``oeh_eval.vlabench_evidence.project``, so the runtime's ``policy_input`` hash and the evaluator's record
    agree when they saw the same observation."""
    native = obs["native"]
    instruction = native.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("native VLABench instruction required")
    frames = np.asarray(native["rgb"])
    if frames.ndim != 4 or frames.shape[0] != NATIVE_CAMERAS or frames.shape[-1] != 3:
        raise ValueError(f"VLABench obs['rgb'] must hold {NATIVE_CAMERAS} RGB views, got {frames.shape}")
    vision = {name: {"color": _rgb(frames[index], name)} for name, index in CAMERAS}
    state = project_state({"ee_state": native["ee_state"], "finger_qpos": obs["finger_qpos"],
                           "robot_frame": native["robot_frame"]})
    return {"vision": vision, "state": state, "instruction": instruction}


class VLABenchContract:
    """Bound to the one frozen configuration it validated (review mode only)."""

    benchmark = "vlabench"
    robot_id = "vlabench_franka_panda"
    action_contract = ("VLABench cf588fe native EE target [xyz, roll/pitch/yaw, finger1, finger2] per predict, "
                       "OpenWAM EEF10 to VLABench EE by the released client's eef10_to_vlabench_ee")
    ledger_kind = "obs/vlabench_ledger"
    # The teacher's views: the three the released client sends OpenWAM, in the profile's order and names.
    review_cameras = SINGLE_ARM_EE.cameras

    def __init__(self):
        self.mode = None
        self.persona_path = None
        self.review_rows = None  # the rows of one OpenWAM chunk (max_chunk_steps)
        self.accepted = None  # digest of the configuration whose files this process already checked

    def validate_config(self, value):
        cfg = self.check_policy(value)
        if self.accepted != digest(cfg):  # the files (the checkpoint among them) are hashed once per process
            self.check_native(cfg)
            self.accepted = digest(cfg)
        return cfg

    def trust(self, cfg):
        """Accept a configuration another contract validated in this process (the runner's, before the loop)."""
        self.accepted = digest(self.check_policy(cfg))

    def check_policy(self, value):
        """Condition, budgets and the review contract; binds this contract to them."""
        cfg = copy.deepcopy(dict(value))
        if POLICY_REQUIRED - cfg.keys():
            raise ValueError(f"missing VLABench policy fields: {sorted(POLICY_REQUIRED - cfg.keys())}")
        if cfg["schema"] != "oeh.policy.v1" or cfg["benchmark"] != "vlabench":
            raise ValueError("expected a pinned VLABench policy configuration")
        check_embodiment(cfg["embodiment"], "vlabench")
        if cfg["mode"] != "review":
            raise ValueError("VLABench runs OEH in review mode only (the hybrid primitives are not ported to it)")
        if cfg.get("brain") or "escalation" in cfg or "primitives" in cfg:
            raise ValueError("VLABench review mode configures its teacher under review.teacher, never brain, "
                             "escalation or primitives")
        check_review_benchmark(cfg["benchmark"], cfg.get("review"))
        cfg["review"] = validate_review(cfg.get("review"))
        if cfg["review"]["control_hz"] != CONTROL_HZ:
            raise ValueError(f"review.control_hz must be VLABench's native step rate {CONTROL_HZ!r} Hz")
        native = cfg["vlabench"]
        if not isinstance(native, dict):
            raise ValueError("vlabench must be an object")
        for key in ("root", "openwam_root", "official_client_path", "single_eval_path", "checkpoint_file",
                    "normalization_file", "track", "task_name"):
            if not isinstance(native.get(key), str) or not native[key]:
                raise ValueError(f"vlabench.{key} required")
        if type(native.get("episode")) is not int or native["episode"] < 0:
            raise ValueError("vlabench.episode must be a track episode index")
        if cfg["review"]["task"] != native["task_name"]:
            raise ValueError("review.task must be the episode's VLABench task (vlabench.task_name)")
        from .vlabench_review import check_review_workspace
        cfg["review_workspace"] = check_review_workspace(cfg["review_workspace"])
        if cfg["gripper_state_semantics"] != GRIPPER_SEMANTICS:
            raise ValueError("VLABench's observed opening is the measured finger gap; declare measured_finger_gap")
        if cfg["action_limits"] != {"native_action": "vlabench_ee", "finite": True}:
            raise ValueError("VLABench action contract must keep the native EE target unclipped")
        # The candidate's length: one full OpenWAM chunk (the template cites its source; the matrix compile checks it
        # against the pinned server config). The official loop still takes one of its rows per predict.
        for key in ("max_chunk_steps", "native_step_limit"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"positive integer {key} required")
        # first_decision_seconds (optional) is the fast path's; a review call is bounded by review.call_seconds.
        if "first_decision_seconds" in cfg:
            raise ValueError("VLABench review mode bounds its first call with review.first_call_extra_seconds")
        for key in ("wall_seconds", "decision_seconds"):
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < math.inf:
                raise ValueError(f"finite positive {key} required")
        for key in ("checkpoint_sha256", "normalization_sha256"):
            value = cfg[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"frozen {key} required")
        self.mode = cfg["mode"]
        self.review_rows = cfg["max_chunk_steps"]
        return cfg

    def check_native(self, cfg):
        """Pinned source, released client settings, step limit, track episode and model identities on disk."""
        for path in review_prompt_paths():
            if str(path) not in cfg["identity_files"]:
                raise ValueError(f"review identity must pin the teacher prompt file {path.name}")
        observed = {}
        for path, expected in cfg["identity_files"].items():
            if not Path(path).is_absolute():
                raise ValueError(f"runtime identity mismatch: {path}")
            key = str(Path(path).resolve())
            observed[key] = file_digest(Path(path))  # each file hashed once, the checkpoint included
            if observed[key] != expected:
                raise ValueError(f"runtime identity mismatch: {path}")
        native = cfg["vlabench"]
        root = Path(native["root"]).resolve()
        if any(str(path) not in cfg["identity_files"] for path in required_identity_paths(native)):
            raise ValueError("official VLABench evaluation and OpenWAM client files must be pinned")
        source_commit = native.get("source_commit", VLABENCH_COMMIT)
        if source_commit != VLABENCH_COMMIT and cfg.get("data_kind", "simulation") != "synthetic":
            raise ValueError(f"VLABench source must be {VLABENCH_COMMIT}")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if revision != source_commit:
            raise ValueError(f"VLABench source must be {source_commit}")
        tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        if hashlib.sha256(tracked_diff).hexdigest() != native.get("tracked_diff_sha256"):
            raise ValueError("VLABench source patch differs from frozen manifest")
        client = yaml.safe_load((Path(native["openwam_root"]) / "benchmarks/vlabench/policy_config.yml")
                                .read_text(encoding="utf-8")) or {}
        differs = {key: client.get(key) for key, value in CLIENT_SETTINGS.items() if client.get(key) != value}
        if differs:
            raise ValueError(f"the released VLABench client config differs from the contract: {differs}")
        task, track = native["task_name"], native["track"]
        if cfg["native_step_limit"] != step_limit(root, task):
            raise ValueError("native step limit differs from the pinned VLABench task")
        if native["episode"] >= len(track_episodes(root, track, task)):
            raise ValueError(f"{track} has no episode {native['episode']} of {task}")

        def observed_digest(path):
            key = str(Path(path).resolve())
            return observed[key] if key in observed else file_digest(Path(path))
        if observed_digest(native["checkpoint_file"]) != cfg["checkpoint_sha256"]:
            raise ValueError("VLABench checkpoint mismatch")
        if observed_digest(native["normalization_file"]) != cfg["normalization_sha256"]:
            raise ValueError("VLABench normalizer mismatch")
        if (cfg["checkpoint_sha256"] != native.get("expected_checkpoint_sha256") or
                cfg["normalization_sha256"] != native.get("expected_normalization_sha256")):
            raise ValueError("VLABench released model identity mismatch")

    def make_method(self, cfg, backend=None):
        from .vlabench_review import VLABenchReviewMethod
        return VLABenchReviewMethod(cfg)

    def review_chunk(self, backend):
        """The review candidate: one fresh full OpenWAM chunk of native rows on the current observation."""
        return backend.review_chunk(self.review_rows)

    def review_views(self, observation):
        """The teacher's views, exactly ``review_cameras`` in order (the three the released client sends)."""
        vision = observation["vision"]
        missing = [name for name in self.review_cameras if name not in vision]
        if missing:
            raise ValueError(f"VLABench review mode needs the {missing} views; the observation has none")
        return {name: vision[name]["color"] for name in self.review_cameras}

    def intent_schema(self, max_steps):
        from .actions import intent_schema
        return intent_schema(max_steps)

    def model_observation(self, runtime, projected):
        return projected

    def subgoal_observation(self, runtime, subgoal):
        obs = copy.deepcopy(runtime.observation)
        obs["instruction"] = subgoal
        return obs

    def escalated(self, runtime):
        pass

    def motion_features(self, state, action):
        """The measured site position after a step and the commanded opening, binarized at the released client's
        threshold (one arm)."""
        native = vlabench_row(action)
        return ([np.asarray(state["ee_state"], dtype=float)[:3]], [float(native[6:8].mean() >= 0.02)])

    def project_state(self, state):
        return project_state(state)

    def project_observation(self, obs):
        return project_observation(obs)

    def validate_action(self, action, limits):
        return vlabench_row(action)

    def camera_payload(self, runtime):
        obs = runtime.observation
        images = [{"camera": name, "sha256": observation_digest(obs["vision"][name])}
                  for name, _ in CAMERAS if name in obs["vision"]]
        return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step, "epoch": 0,
                "task_instruction": obs["instruction"], "robot_state": plain(obs["state"]), "images": images,
                "previous_receipt": runtime.previous_receipt}

    def ledger_payload(self, runtime, projected):
        return {"episode_id": runtime.episode_id, "seq": runtime.gate.step,
                "state_sha256": digest(projected["state"]), "native_step": runtime.gate.step}


def review_prompt_paths() -> list[Path]:
    """The teacher prompt files a review configuration pins (resolved): VLABench's skill, context and task notes and
    the shared gate prompt, the files ``teacher.prompt_identity`` digests."""
    from .teacher import prompt_dir, prompt_files
    paths = (*prompt_files("vlabench").values(), prompt_dir("vlabench") / "task_context.json")
    return [path.resolve() for path in paths]


class OpenWAMBackend:
    """The released client object (``OpenWAMVLABenchPolicy``) as the OEH runtime's backend: its own ``predict``, so
    the request OpenWAM receives and the conversion of its answer are the direct condition's, byte for byte."""

    def __init__(self, official):
        self.official = official
        indices = (official._head_idx, official._left_idx, official._right_idx)
        expected = tuple(CLIENT_SETTINGS[key] for key in ("head_camera_index", "left_wrist_camera_index",
                                                           "right_wrist_camera_index"))
        if indices != expected or official._send_state is not True:
            raise ValueError(f"the released VLABench client reads cameras {indices}; the contract's are {expected}")
        self.observation = None

    def reset(self):
        self.official.reset()

    def update_obs(self, observation):
        self.observation = observation

    def native_observation(self) -> dict:
        """The fields of the native observation the released client reads, rebuilt from the projected one: ``rgb``
        with each view at its native index (index 1, never sent, as black), ``ee_state``, ``robot_frame`` and the
        instruction."""
        obs = self.observation
        if obs is None:
            raise RuntimeError("VLABench observation missing")
        views = {index: obs["vision"][name]["color"] for name, index in CAMERAS}
        blank = np.zeros_like(views[CAMERAS[0][1]])
        frames = np.stack([views.get(index, blank) for index in range(NATIVE_CAMERAS)])
        return {"rgb": frames, "ee_state": obs["state"]["ee_state"].copy(),
                "robot_frame": obs["state"]["robot_frame"].copy(), "instruction": obs["instruction"]}

    def get_action(self):
        pos, euler, gripper = self.official.predict(self.native_observation())
        row = np.concatenate([np.asarray(pos, dtype=np.float64).reshape(-1),
                              np.asarray(euler, dtype=np.float64).reshape(-1),
                              np.asarray(gripper, dtype=np.float64).reshape(-1)])
        return [vlabench_row(row)]

    def review_chunk(self, rows):
        """One fresh full OpenWAM chunk on the current observation, as ``rows`` native rows (review mode).

        The released server answers one action per request: its sync executor generates a chunk only when its buffer
        is empty and pops one row per request, and RESET empties the buffer (openwam/deploy/executors/
        sync_executor.py at fa431ec). So the released client's ``reset`` and then ``rows`` ``predict`` calls on this
        one observation return exactly one chunk generated from it, each row converted as the direct condition's
        are. ``rows`` must be the chunk length, which the frozen max_chunk_steps holds."""
        if self.observation is None:
            raise RuntimeError("VLABench observation missing")
        self.official.reset()
        return [self.get_action()[0] for _ in range(rows)]

"""Pinned EBench (GenManip) / OpenWAM evaluation boundary, separate from RoboDojo.

EBench inverts control: GenManip's evaluation server (Isaac Sim, ``ray_eval_server.py``) serves observations and the
policy side is a client that polls it (``genmanip_client.EvalClient``). In the direct condition that client is
OpenWAM's released EBench bridge (XPolicyLab fa431ec ``benchmarks/ebench/openwam2ebench_interface.py``): per native step
its ``EBenchOpenWAMDriver.act`` turns the observation into OpenWAM's request and the returned raw 23-D action into
GenManip's ``ee_pose`` action dict (``oeh_policy.ebench_boundary``). The OEH review condition runs the same bridge loop
with OEH's driver (``oeh_policy.ebench_entry``) in that driver's place. Its candidate is one fresh full chunk from the
released driver itself (``OpenWAMBackend.review_chunk``), so OpenWAM's rows are the rows the direct condition gets at a
chunk boundary on the same observation, and every native action leaves through OEH's two-phase gate.

The review condition only: OEH's fast passthrough and the slow hybrid are not ported to EBench.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np
import yaml

from .contracts import check_review_benchmark, digest, file_digest, observation_digest, plain, validate_review
from .ebench_boundary import ARMS, native_action
from .profiles import DUAL_ARM_MOBILE_EE, check_embodiment

GENMANIP_COMMIT = "1958e3781e4fb1921d1834b658b295e3660899ab"
# GenManip's observation keys (genmanip/core/evaluator/env.py get_obs) and the names OEH and the teacher use. The
# order is OpenWAM's (head slot, left wrist, right wrist; benchmarks/ebench/openwam2ebench_interface.py HEAD_KEY..).
CAMERAS = (("video.overlook_camera_view", "cam_overlook"), ("video.left_camera_view", "cam_left_wrist"),
           ("video.right_camera_view", "cam_right_wrist"))
STATE_KEYS = ("state.ee_pose", "state.gripper", "state.base")
# The native action contract of every condition: GenManip's ee_pose dict, finite, unchanged (no clipping beyond what
# the released bridge itself does). OpenWAM's own rows leave exactly as the bridge built them.
ACTION_LIMITS = {"native_action": "genmanip_ee_pose", "finite": True}
# GenManip's default physics step when a task config sets none (genmanip/core/scene/scene_config.py, line 235): the
# mobile tasks run at 1/30 s, the table-top configs set 0.0166666667 (1/60 s). One native step is one physics step.
DEFAULT_PHYSICS_DT = 1 / 30
TRACKS = ("mobile_manip", "table_top_manip")
SPLITS = ("val_train", "val_unseen", "test_mini")


def required_identity_paths(native) -> list[Path]:
    """The pinned files a frozen configuration must hash: what the bridge loop, the server and the simulator read."""
    root = Path(native["root"]).resolve()
    openwam = Path(native["openwam_root"]).resolve()
    return [root / "ray_eval_server.py",
            root / "configs/tasks" / (native["config_path"] + ".yml"),
            root / "genmanip/core/evaluator/env.py",
            root / "genmanip/core/evaluator/isaac_worker_pool.py",
            root / "genmanip/core/evaluator/progress_manager.py",
            root / "genmanip/core/robot/dualarm_manip.py",
            root / "genmanip/extensions/robots/default/r5a/lift2.py",
            Path(native["official_bridge_path"]).resolve(),
            openwam / "benchmarks/ebench/prompt_template.py",
            openwam / "benchmarks/ebench/policy_config.yml",
            openwam / "benchmarks/utils/__init__.py",
            openwam / "benchmarks/utils/action_conversion.py",
            openwam / "benchmarks/utils/client.py",
            openwam / "benchmarks/utils/transport.py",
            openwam / "openwam/deploy/server.py",
            openwam / "configs/deploy.yaml",
            openwam / "scripts/deploy.sh"]


def task_settings(root, config_path: str, task_name: str) -> dict:
    """The one evaluation config a frozen case runs, from its pinned GenManip task file: its task name, test count,
    step limit and physics step (GenManip's default when the file sets none)."""
    path = Path(root).resolve() / "configs/tasks" / (config_path + ".yml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    configs = data.get("evaluation_configs") if isinstance(data, dict) else None
    if not isinstance(configs, list) or len(configs) != 1 or not isinstance(configs[0], dict):
        raise ValueError(f"{config_path}.yml must hold exactly one evaluation config")
    config = configs[0]
    if config.get("task_name") != task_name:
        raise ValueError(f"{config_path}.yml evaluates {config.get('task_name')!r}, not {task_name!r}")
    num_test, num_steps = config.get("num_test"), config.get("num_steps")
    if type(num_test) is not int or num_test < 1 or type(num_steps) is not int or num_steps < 1:
        raise ValueError(f"{config_path}.yml needs positive integer num_test and num_steps")
    dt = config.get("physics_dt", DEFAULT_PHYSICS_DT)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not 0 < dt < math.inf:
        raise ValueError(f"{config_path}.yml physics_dt must be a positive number")
    return {"task_name": task_name, "num_test": num_test, "num_steps": num_steps, "physics_dt": float(dt)}


def check_control_hz(control_hz, physics_dt: float) -> None:
    """review.control_hz is the native step rate, 1 / physics_dt (sim_time_s in the packets counts steps at it)."""
    if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)) or \
            abs(1.0 / float(control_hz) - float(physics_dt)) > 1e-6:
        raise ValueError(f"review.control_hz must be EBench's native step rate 1 / physics_dt = "
                         f"{1.0 / float(physics_dt):.6g} Hz for this task")


def _rgb(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
        raise ValueError(f"{name} must be nonempty uint8 RGB")
    return image.copy()


def _floats(value, size: int, name: str) -> np.ndarray:
    def leaves(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                yield from leaves(item)
        else:
            yield np.asarray(node, dtype=np.float64).reshape(-1)
    try:
        array = np.concatenate(list(leaves(value))) if isinstance(value, (list, tuple)) else \
            np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must hold {size} finite numbers") from None
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must hold {size} finite numbers")
    return array.copy()


def project_state(value: Mapping) -> dict:
    """The robot state OpenWAM's bridge reads, as float64 arrays: each arm's measured pose (xyz + wxyz, that arm's own
    base frame), the four measured finger positions [L, L, R, R] (metres) and the base joints [x m, y m, yaw rad].
    Accepts GenManip's observation (``state.*`` keys) or an already projected state."""
    if "state.ee_pose" in value:
        poses = _floats(value["state.ee_pose"], 14, "state.ee_pose")
        return {"left_ee_pose": poses[:7], "right_ee_pose": poses[7:],
                "gripper": _floats(value["state.gripper"], 4, "state.gripper"),
                "base": _floats(value["state.base"], 3, "state.base")}
    return {"left_ee_pose": _floats(value["left_ee_pose"], 7, "left_ee_pose"),
            "right_ee_pose": _floats(value["right_ee_pose"], 7, "right_ee_pose"),
            "gripper": _floats(value["gripper"], 4, "gripper"), "base": _floats(value["base"], 3, "base")}


def project(observation: Mapping) -> dict:
    """What OpenWAM's released bridge reads from one GenManip observation, nothing else: the three RGB views, the
    measured state and the instruction. The evaluator's record (``oeh_eval.ebench_evidence.project``) holds the same
    values, dtypes and shapes, so the runtime's ``policy_input`` hash and the record agree."""
    instruction = observation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("native EBench instruction required")
    vision = {name: {"color": _rgb(observation[key], key)} for key, name in CAMERAS}
    return {"vision": vision, "state": project_state(observation), "instruction": instruction}


def native_view(projected: Mapping) -> dict:
    """A projected observation in GenManip's keys, as the released driver reads it (``EBenchOpenWAMDriver.act``): only
    the projected values, so OpenWAM gets exactly what the direct condition's driver gets from the same observation."""
    state = projected["state"]
    view = {key: projected["vision"][name]["color"] for key, name in CAMERAS}
    view["state.ee_pose"] = [[np.asarray(state[arm + "_ee_pose"])[:3].tolist(),
                              np.asarray(state[arm + "_ee_pose"])[3:].tolist()] for arm in ARMS]
    view["state.gripper"] = np.asarray(state["gripper"]).copy()
    view["state.base"] = np.asarray(state["base"]).copy()
    view["instruction"] = projected["instruction"]
    return view


class EBenchContract:
    """Bound to the one frozen configuration it validated (review mode only)."""

    benchmark = "ebench"
    robot_id = "genmanip_lift2_r5a"
    action_contract = ("GenManip 1958e37 ee_pose action (per arm xyz, wxyz, two finger targets m, in that arm's base "
                       "frame; base_motion [dx m, dy m, dyaw deg] relative), OpenWAM raw23 to ee_pose official conversion")
    ledger_kind = "obs/ebench_ledger"
    persona_path = None  # no slow primitives: review mode only

    # The review port's views (oeh_policy.ebench_review): the three released RGB views in OpenWAM's slot order.
    review_cameras = DUAL_ARM_MOBILE_EE.cameras

    def __init__(self):
        self.mode = None
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
        cfg = copy.deepcopy(dict(value))
        required = {"schema", "benchmark", "mode", "kernel", "repository", "output", "model_id",
                    "checkpoint_sha256", "normalization_sha256", "identity_files", "max_chunk_steps",
                    "native_step_limit", "wall_seconds", "decision_seconds", "action_limits", "ebench", "embodiment",
                    "review", "review_workspace", "gripper_state_semantics"}
        if required - cfg.keys():
            raise ValueError(f"missing EBench policy fields: {sorted(required - cfg.keys())}")
        if cfg["schema"] != "oeh.policy.v1" or cfg["benchmark"] != "ebench":
            raise ValueError("expected pinned EBench policy configuration")
        check_embodiment(cfg["embodiment"], "ebench")
        if cfg["mode"] != "review":
            raise ValueError("EBench runs OEH's review condition only")
        if cfg.get("brain") or "escalation" in cfg or "primitives" in cfg:
            raise ValueError("EBench review mode configures its teacher under review.teacher, never brain, escalation "
                             "or primitives")
        check_review_benchmark(cfg["benchmark"], cfg.get("review"))
        cfg["review"] = validate_review(cfg.get("review"))
        native = cfg["ebench"]
        if not isinstance(native, dict) or isinstance(native.get("physics_dt"), bool) or \
                not isinstance(native.get("physics_dt"), (int, float)):
            raise ValueError("ebench.physics_dt must be the frozen task's physics step")
        check_control_hz(cfg["review"]["control_hz"], native["physics_dt"])
        from .ebench_review import check_review_workspace
        cfg["review_workspace"] = check_review_workspace(cfg.get("review_workspace"))
        if cfg["gripper_state_semantics"] != "measured":
            raise ValueError("EBench observes measured finger positions, not the gripper command; declare "
                             "gripper_state_semantics measured")
        if type(cfg["max_chunk_steps"]) is not int or cfg["max_chunk_steps"] < 1:
            raise ValueError("EBench review mode needs the OpenWAM chunk length as a positive max_chunk_steps")
        if type(cfg["native_step_limit"]) is not int or cfg["native_step_limit"] < 1:
            raise ValueError("positive native step limit required")
        for key in ("wall_seconds", "decision_seconds", "first_decision_seconds"):
            if key == "first_decision_seconds" and key not in cfg:
                continue
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < math.inf:
                raise ValueError(f"finite positive {key} required")
        for key in ("checkpoint_sha256", "normalization_sha256"):
            value_ = cfg[key]
            if not isinstance(value_, str) or len(value_) != 64 or any(c not in "0123456789abcdef" for c in value_):
                raise ValueError(f"frozen {key} required")
        if cfg["action_limits"] != ACTION_LIMITS:
            raise ValueError(f"EBench action contract must preserve GenManip's native action: {ACTION_LIMITS}")
        self.mode = cfg["mode"]
        self.review_rows = cfg["max_chunk_steps"]
        return cfg

    def check_native(self, cfg):
        """Pinned sources, task settings and released model identities on disk."""
        for path in review_prompt_paths():
            if str(path) not in cfg["identity_files"]:
                raise ValueError(f"review identity must pin the teacher prompt file {path.name}")
        observed = {}
        for path, expected in cfg["identity_files"].items():
            if not Path(path).is_absolute():
                raise ValueError(f"runtime identity mismatch: {path}")
            key = str(Path(path).resolve())
            observed[key] = file_digest(Path(path))
            if observed[key] != expected:
                raise ValueError(f"runtime identity mismatch: {path}")
        native = cfg["ebench"]
        if any(str(path) not in cfg["identity_files"] for path in required_identity_paths(native)):
            raise ValueError("GenManip's evaluation files and OpenWAM's EBench bridge must be pinned")
        root = Path(native["root"]).resolve()
        source_commit = native.get("source_commit", GENMANIP_COMMIT)
        if source_commit != GENMANIP_COMMIT and cfg.get("data_kind", "simulation") != "synthetic":
            raise ValueError(f"GenManip source must be {GENMANIP_COMMIT}")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if revision != source_commit:
            raise ValueError(f"GenManip source must be {source_commit}")
        tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        if hashlib.sha256(tracked_diff).hexdigest() != native["tracked_diff_sha256"]:
            raise ValueError("GenManip source patch differs from frozen manifest")
        if native.get("send_state") is not True:
            raise ValueError("the released EBench checkpoint is proprio-conditioned: send_state must be true")
        settings = task_settings(root, native["config_path"], native["task_name"])
        seed = native.get("seed")
        if not isinstance(seed, str) or seed not in [str(i).zfill(3) for i in range(settings["num_test"])]:
            raise ValueError(f"seed {seed!r} is not one of the task's {settings['num_test']} GenManip seeds")
        if cfg["native_step_limit"] != settings["num_steps"]:
            raise ValueError("native step limit differs from the pinned GenManip task's num_steps")
        if abs(float(native["physics_dt"]) - settings["physics_dt"]) > 1e-9:
            raise ValueError("ebench.physics_dt differs from the pinned GenManip task")

        def observed_digest(path):
            key = str(Path(path).resolve())
            return observed[key] if key in observed else file_digest(Path(path))
        if observed_digest(native["checkpoint_file"]) != cfg["checkpoint_sha256"]:
            raise ValueError("EBench checkpoint mismatch")
        if observed_digest(native["normalization_file"]) != cfg["normalization_sha256"]:
            raise ValueError("EBench normalizer mismatch")
        if (cfg["checkpoint_sha256"] != native["expected_checkpoint_sha256"] or
                cfg["normalization_sha256"] != native["expected_normalization_sha256"]):
            raise ValueError("EBench released model identity mismatch")

    def make_method(self, cfg, backend=None):
        from .ebench_review import EBenchReviewMethod
        return EBenchReviewMethod(cfg)

    def review_chunk(self, backend):
        """The review candidate: one fresh full OpenWAM chunk of native actions on the current observation."""
        return backend.review_chunk(self.review_rows)

    def review_views(self, observation):
        vision = observation["vision"]
        return {name: vision[name]["color"] for name in self.review_cameras}

    def intent_schema(self, max_steps):
        from .actions import intent_schema
        return intent_schema(max_steps)

    def model_observation(self, runtime, projected):
        return projected

    def subgoal_observation(self, runtime, subgoal):
        raise RuntimeError("EBench runs review mode only; no slow subgoals")

    def escalated(self, runtime):
        pass

    def motion_features(self, state, action):
        """Measured arm positions after a step and the openings it commanded, binarized at 0.5 (1 open, 0 closed)."""
        native = native_action(action)
        from .ebench_boundary import opening
        return ([np.asarray(state[arm + "_ee_pose"], dtype=float)[:3] for arm in ARMS],
                [float(opening(fingers) > 0.5) for _, _, fingers in native["action"]])

    def project_state(self, state):
        return project_state(state)

    def project_observation(self, obs):
        return project(obs["native"])

    def validate_action(self, action, limits):
        """GenManip's ee_pose dict, checked (finite, unit quaternions, fingers in range) and copied unchanged."""
        return native_action(action)

    def camera_payload(self, runtime):
        obs = runtime.observation
        images = [{"camera": name, "sha256": observation_digest(obs["vision"][name])} for name in self.review_cameras]
        return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step,
                "epoch": 0, "task_instruction": obs["instruction"],
                "robot_state": plain(obs["state"]), "images": images,
                "previous_receipt": runtime.previous_receipt}

    def ledger_payload(self, runtime, projected):
        return {"episode_id": runtime.episode_id, "seq": runtime.gate.step,
                "state_sha256": digest(projected["state"]), "native_step": runtime.gate.step}


def review_prompt_paths() -> list[Path]:
    """The teacher prompt files a review configuration pins (resolved): EBench's skill, context and task notes and the
    shared gate prompt, the files ``teacher.prompt_identity`` digests."""
    from .teacher import prompt_dir, prompt_files
    paths = (*prompt_files("ebench").values(), prompt_dir("ebench") / "task_context.json")
    return [path.resolve() for path in paths]


def import_official_bridge(path: Path):
    """Load the exact released EBench bridge file after identity preflight (it puts its OpenWAM root on sys.path)."""
    spec = importlib.util.spec_from_file_location("oeh_pinned_openwam2ebench", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned OpenWAM EBench bridge {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenWAMBackend:
    """OpenWAM's released EBench driver as OEH's student (review mode).

    ``driver`` is the released ``EBenchOpenWAMDriver`` on the released ``WSPolicyClient``. The released server answers
    one action per request: its sync executor generates a chunk only when its buffer is empty and pops one row per
    request, and RESET empties the buffer (openwam/deploy/executors/sync_executor.py at fa431ec). So a RESET, then
    ``rows`` ``act`` calls on one observation, return exactly one chunk generated from that observation: the rows the
    direct condition gets at a chunk boundary on it. The request that generates the chunk carries the base history
    the direct driver would have at that step (``_prev_base``: the base one native step earlier; none at the episode's
    first observation, where the released driver starts its episode exactly as in the direct condition)."""

    def __init__(self, driver):
        self.driver = driver
        self.observation = None
        self.previous_base = None
        self.started = False

    def reset(self):
        """Nothing before the first observation: the released driver starts its episode (OpenWAM RESET, the prompt,
        no base history) at the first candidate, as it does at the direct condition's first step."""

    def update_obs(self, observation):
        """Called once per native observation (``Runtime.update_obs``), like the released driver's ``act``."""
        if self.observation is not None:
            self.previous_base = np.asarray(self.observation["state"]["base"], dtype=np.float64).copy()
        self.observation = observation

    def review_chunk(self, rows):
        if self.observation is None:
            raise RuntimeError("EBench observation missing")
        inner = native_view(self.observation)
        driver = self.driver
        if not self.started:
            self.started = True
            first = driver.act({**inner, "reset": True})  # on_episode_start: RESET, prompt, no base history
        else:
            reply = driver._south.reset()
            if not isinstance(reply, Mapping) or reply.get("type") != "reset_ack":
                raise RuntimeError(f"OpenWAM server reset not acknowledged: {reply}")
            driver._prev_base = None if self.previous_base is None else self.previous_base.copy()
            first = driver.act({**inner, "reset": False})
        return [first] + [driver.act({**inner, "reset": False}) for _ in range(rows - 1)]

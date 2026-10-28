"""Pinned RoboTwin/OpenWAM evaluation boundary, separate from RoboDojo.

The released RoboTwin client supplies RGB, endpose and instruction only. Its
model action is EEF20 and its native action is EE16. Conversion is delegated to
the imported, pinned OpenWAM client; this module never adjusts its quaternion
order or substitutes a success predicate. Native EE16 quaternions are wxyz
(transforms3d); OpenWAM labels the same numbers xyzw consistently, so the fast
path round-trips them. The hybrid condition adds the dual_arm_ee16 slow method
(``robotwin_hybrid``), which treats native as wxyz. The review condition runs OEH review mode
(``robotwin_review``): a teacher reviews one fresh full OpenWAM chunk per decision, and the official
loop still submits one EE16 row per call.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess

import numpy as np
import yaml

from .contracts import (check_review_benchmark, digest, file_digest, observation_digest, plain,
                        validate_review)
from .escalation import validate_escalation
from .profiles import DUAL_ARM_EE16, check_embodiment, check_primitive_limits

ROBOTWIN_COMMIT = "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
CAMERAS = (("head_camera", "cam_head"), ("left_camera", "cam_left_wrist"),
           ("right_camera", "cam_right_wrist"))


def required_identity_paths(native):
    root = Path(native["root"]).resolve()
    openwam = Path(native["openwam_root"]).resolve()
    return [root / "script/eval_policy.py",
            root / "task_config" / (native["task_config"] + ".yml"),
            root / "task_config/_eval_step_limit.yml",
            Path(native["official_client_path"]).resolve(),
            Path(native["official_wrapper_path"]).resolve(),
            openwam / "benchmarks/utils/action_conversion.py",
            openwam / "benchmarks/robotwin/prompt_template.py",
            openwam / "benchmarks/robotwin/policy_config.yml",
            openwam / "benchmarks/robotwin/step_limits.yml",
            openwam / "openwam/deploy/server.py",
            openwam / "configs/deploy.yaml",
            openwam / "scripts/deploy.sh"]


def _rgb(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
        raise ValueError(f"{name} must be nonempty uint8 RGB")
    return image.copy()


def _pose(value, name):
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be finite xyz+quaternion")
    return pose.copy()


class RoboTwinFastMethod:
    """Fast mode offers one VLA action; slow geometry is not yet verified."""

    phase = "grasp"
    target = "full task"

    def __init__(self, cfg):
        self.gripper_commands = {}

    def seed(self, state):
        pass

    def acknowledged(self, action):
        pass

    def check_intent(self, params):
        if params.get("mode") != "vla" or params.get("execute_steps") != 1:
            raise ValueError("RoboTwin fast mode requires one unmodified VLA action")


class RoboTwinContract:
    """Bound to the one frozen configuration it validated (mode decides the method)."""

    benchmark = "robotwin"
    robot_id = "robotwin_aloha_agilex"
    action_contract = "RoboTwin 0aeea2d native EE16, OpenWAM EEF20 to EE16 official conversion"
    ledger_kind = "obs/robotwin_ledger"

    # The review port (review mode, oeh_policy.robotwin_review): the three released RGB views, in the order and with
    # the names the dual_arm_ee16 profile gives them (head, left wrist, right wrist).
    review_cameras = DUAL_ARM_EE16.cameras

    def __init__(self):
        self.mode = None
        self.hybrid = False
        self.persona_path = None
        self.chunk_rows = None
        self.review_rows = None  # review mode: the rows of one OpenWAM chunk (max_chunk_steps)
        self.accepted = None  # digest of the configuration whose files this process already checked

    def validate_config(self, value):
        cfg = self.check_policy(value)
        if self.accepted != digest(cfg):  # the files (a 24 GB checkpoint among them) are hashed once per process
            self.check_native(cfg)
            self.accepted = digest(cfg)
        return cfg

    def trust(self, cfg):
        """Accept a configuration another contract validated in this process (the runner's, before the loop)."""
        self.accepted = digest(self.check_policy(cfg))

    def check_policy(self, value):
        """Condition, budgets and slow-method contract; binds this contract to them."""
        cfg = copy.deepcopy(dict(value))
        required = {"schema", "benchmark", "mode", "kernel", "repository", "output",
                    "model_id", "checkpoint_sha256", "normalization_sha256", "identity_files",
                    "max_chunk_steps", "native_step_limit", "wall_seconds", "decision_seconds",
                    "action_limits", "robotwin", "embodiment"}
        if required - cfg.keys():
            raise ValueError(f"missing RoboTwin policy fields: {sorted(required - cfg.keys())}")
        if cfg["schema"] != "oeh.policy.v1" or cfg["benchmark"] != "robotwin":
            raise ValueError("expected pinned RoboTwin policy configuration")
        check_embodiment(cfg["embodiment"], "robotwin")
        if cfg["mode"] not in ("fast", "hybrid", "review"):
            raise ValueError("RoboTwin runs the fast, hybrid and review conditions only")
        if cfg["mode"] != "review" and ("review" in cfg or "review_workspace" in cfg):
            raise ValueError("only RoboTwin's review mode configures review and review_workspace")
        if cfg["mode"] == "fast":
            if cfg.get("brain") or "escalation" in cfg or "primitives" in cfg:
                raise ValueError("RoboTwin fast mode cannot configure GPT, escalation or primitives")
        elif cfg["mode"] == "review":
            self._check_review(cfg)
        else:
            brain = cfg.get("brain") or {}
            if not all(isinstance(brain.get(k), str) and brain[k] for k in ("provider", "model", "providers")):
                raise ValueError("hybrid mode requires a configured internal model")
            if type(brain.get("max_calls")) is not int or brain["max_calls"] < 1:
                raise ValueError("hybrid mode requires a model call budget")
            cfg["escalation"] = validate_escalation(cfg.get("escalation"))
            cfg["primitives"] = check_primitive_limits(DUAL_ARM_EE16, cfg.get("primitives"))
            if cfg.get("gripper_state_semantics") != "previous_command":
                raise ValueError("RoboTwin endpose grippers are the previous command; declare it")
        if cfg["mode"] == "review":
            # The candidate's length: one full OpenWAM chunk (the template cites its source; the matrix compile checks
            # it against the pinned server config). The official loop still takes one of its rows per call.
            if type(cfg["max_chunk_steps"]) is not int or cfg["max_chunk_steps"] < 1:
                raise ValueError("RoboTwin review mode needs the OpenWAM chunk length as a positive max_chunk_steps")
        elif cfg["max_chunk_steps"] != 1 or type(cfg["max_chunk_steps"]) is not int:
            raise ValueError("official RoboTwin client requests one action per observation")
        if type(cfg["native_step_limit"]) is not int or cfg["native_step_limit"] < 1:
            raise ValueError("positive native step limit required")
        # first_decision_seconds (optional, default decision_seconds) bounds the episode's first decision, which
        # pays the model's warm-up; the runtime and the endpoint's call bound read it (oeh_policy.runtime, .model).
        for key in ("wall_seconds", "decision_seconds", "first_decision_seconds"):
            if key == "first_decision_seconds" and key not in cfg:
                continue
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < math.inf:
                raise ValueError(f"finite positive {key} required")
        for key in ("checkpoint_sha256", "normalization_sha256"):
            digest_value = cfg[key]
            if not isinstance(digest_value, str) or len(digest_value) != 64 or any(c not in "0123456789abcdef" for c in digest_value):
                raise ValueError(f"frozen {key} required")
        self.mode = cfg["mode"]
        self.hybrid = cfg["mode"] == "hybrid"
        self.persona_path = DUAL_ARM_EE16.persona_path if self.hybrid else None
        self.chunk_rows = cfg["primitives"]["chunk_rows"] if self.hybrid else None
        self.review_rows = cfg["max_chunk_steps"] if cfg["mode"] == "review" else None
        return cfg

    def _check_review(self, cfg):
        """Review mode: the teacher under ``review`` (the RoboDojo review block, RoboTwin's control rate and task),
        the workspace its rows are checked against under ``review_workspace``, and nothing of the hybrid's."""
        if cfg.get("brain") or "escalation" in cfg or "primitives" in cfg:
            raise ValueError("RoboTwin review mode configures its teacher under review.teacher, never brain, "
                             "escalation or primitives")
        check_review_benchmark(cfg["benchmark"], cfg.get("review"))
        cfg["review"] = validate_review(cfg.get("review"))
        from .robotwin_hybrid import CONTROL_HZ
        from .robotwin_review import check_review_workspace
        if cfg["review"]["control_hz"] != CONTROL_HZ:
            raise ValueError(f"review.control_hz must be RoboTwin's native EE step rate 250/15 = {CONTROL_HZ!r} Hz")
        cfg["review_workspace"] = check_review_workspace(cfg.get("review_workspace"))
        if cfg.get("gripper_state_semantics") != "previous_command":
            raise ValueError("RoboTwin endpose grippers are the previous command; declare it")

    def check_native(self, cfg):
        """Pinned source, step limit and released model identities on disk."""
        if self.hybrid:
            for path in (Path(cfg["brain"]["providers"]).resolve(), DUAL_ARM_EE16.persona_path):
                if str(path) not in cfg["identity_files"]:
                    raise ValueError(f"hybrid identity must pin {path.name}")
        if self.mode == "review":
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
        if self.hybrid:
            from .brain_preflight import _providers
            brain = cfg["brain"]
            if brain.get("provider_declaration") != _providers(Path(brain["providers"])).get(brain["provider"]):
                raise ValueError("brain.provider_declaration differs from the pinned providers file")
        native = cfg["robotwin"]
        root = Path(native["root"]).resolve()
        if any(str(path) not in cfg["identity_files"] for path in required_identity_paths(native)):
            raise ValueError("official RoboTwin evaluation and OpenWAM client files must be pinned")
        source_commit = native.get("source_commit", ROBOTWIN_COMMIT)
        if source_commit != ROBOTWIN_COMMIT and cfg.get("data_kind", "simulation") != "synthetic":
            raise ValueError(f"RoboTwin source must be {ROBOTWIN_COMMIT}")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if revision != source_commit:
            raise ValueError(f"RoboTwin source must be {source_commit}")
        tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        if hashlib.sha256(tracked_diff).hexdigest() != native["tracked_diff_sha256"]:
            raise ValueError("RoboTwin source patch differs from frozen manifest")
        if native.get("action_type") != "ee" or native.get("send_state") is not True or native.get("state_dim") != 20:
            raise ValueError("released RoboTwin-Full client requires EE mode and EEF20 state")
        if cfg["action_limits"] != {"native_action": "ee16", "finite": True}:
            raise ValueError("RoboTwin action contract must preserve native EE16 without clipping")
        with (root / "task_config/_eval_step_limit.yml").open(encoding="utf-8") as stream:
            official_limits = yaml.safe_load(stream) or {}
        with (Path(native["openwam_root"]) / "benchmarks/robotwin/step_limits.yml").open(encoding="utf-8") as stream:
            client_limits = yaml.safe_load(stream) or {}
        task = native["task_name"]
        expected_limit = client_limits.get(task, official_limits.get(task, 1000))
        if cfg["native_step_limit"] != expected_limit:
            raise ValueError("native step limit differs from pinned RoboTwin/OpenWAM task")
        def observed_digest(path):
            key = str(Path(path).resolve())
            return observed[key] if key in observed else file_digest(Path(path))
        if observed_digest(native["checkpoint_file"]) != cfg["checkpoint_sha256"]:
            raise ValueError("RoboTwin checkpoint mismatch")
        if observed_digest(native["normalization_file"]) != cfg["normalization_sha256"]:
            raise ValueError("RoboTwin normalizer mismatch")
        if (cfg["checkpoint_sha256"] != native["expected_checkpoint_sha256"] or
                cfg["normalization_sha256"] != native["expected_normalization_sha256"]):
            raise ValueError("RoboTwin released model identity mismatch")

    def make_method(self, cfg, backend=None):
        if cfg.get("mode") == "review":
            from .robotwin_review import RoboTwinReviewMethod
            return RoboTwinReviewMethod(cfg)
        if self.hybrid:
            from .robotwin_hybrid import RoboTwinDualArmMethod
            return RoboTwinDualArmMethod(cfg)
        return RoboTwinFastMethod(cfg)

    def review_chunk(self, backend):
        """The review candidate: one fresh full OpenWAM chunk of native EE16 rows on the current observation."""
        return backend.review_chunk(self.review_rows)

    def review_views(self, observation):
        """The teacher's views, exactly ``review_cameras`` in order. RoboTwin's task configs collect both wrist
        cameras; a configuration that does not cannot run review mode, whose packets name three views."""
        vision = observation["vision"]
        missing = [name for name in self.review_cameras if name not in vision]
        if missing:
            raise ValueError(f"RoboTwin review mode needs the {missing} views; the native observation has none")
        return {name: vision[name]["color"] for name in self.review_cameras}

    def intent_schema(self, max_steps):
        from .actions import intent_schema
        # One VLA action per official callback; slow primitives may chain up to chunk_rows.
        return intent_schema(self.chunk_rows if self.hybrid else max_steps)

    def model_observation(self, runtime, projected):
        return projected

    def subgoal_observation(self, runtime, subgoal):
        """RoboDojo semantics: the subgoal drives this slow decision's VLA step only."""
        obs = copy.deepcopy(runtime.observation)
        obs["instruction"] = subgoal
        return obs

    def escalated(self, runtime):
        pass

    def motion_features(self, state, action):
        """Observed arm positions after a step and the grippers it commanded, binarized at 0.5 (1 open,
        0 closed), so only open/closed crossings count as reversals."""
        native = np.asarray(action, dtype=float).reshape(-1)
        return ([np.asarray(state[arm + "_endpose"], dtype=float)[:3] for arm in ("left", "right")],
                [float(native[7] > 0.5), float(native[15] > 0.5)])

    def project_state(self, state):
        endpose = state["endpose"] if "endpose" in state else state
        result = {}
        for arm in ("left", "right"):
            result[arm + "_endpose"] = _pose(endpose[arm + "_endpose"], arm + "_endpose")
            grip = np.asarray(endpose[arm + "_gripper"], dtype=np.float32).reshape(-1)
            if grip.shape != (1,) or not np.isfinite(grip).all():
                raise ValueError(f"{arm} gripper must be one finite scalar")
            result[arm + "_gripper"] = grip.copy()
        return result

    def project_observation(self, obs):
        source = obs["native"]
        instruction = obs["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("native RoboTwin instruction required")
        cameras = source["observation"]
        vision = {}
        for native_name, policy_name in CAMERAS:
            if native_name == "head_camera" or native_name in cameras:
                vision[policy_name] = {"color": _rgb(cameras[native_name]["rgb"], native_name)}
        return {"vision": vision, "state": self.project_state(source["endpose"]),
                "instruction": instruction}

    def validate_action(self, action, limits):
        native = np.asarray(action, dtype=np.float32).reshape(-1)
        if native.shape != (16,) or not np.isfinite(native).all():
            raise ValueError("RoboTwin native EE action must contain 16 finite scalars")
        return native.copy()

    def camera_payload(self, runtime):
        if self.hybrid:
            from .robotwin_hybrid import camera_payload
            return camera_payload(runtime)
        # Fast brain ignores camera events. Keep an explicit whitelist even here.
        obs = runtime.observation
        images = []
        for _, name in CAMERAS:
            if name in obs["vision"]:
                images.append({"camera": name, "sha256": observation_digest(obs["vision"][name])})
        return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step,
                "epoch": 0, "task_instruction": obs["instruction"],
                "robot_state": plain(obs["state"]), "images": images,
                "previous_receipt": runtime.previous_receipt}

    def ledger_payload(self, runtime, projected):
        return {"episode_id": runtime.episode_id, "seq": runtime.gate.step,
                "state_sha256": digest(projected["state"]), "native_step": runtime.gate.step}


def review_prompt_paths() -> list[Path]:
    """The teacher prompt files a review configuration pins (resolved): RoboTwin's skill, context and task notes and
    the shared gate prompt, the files ``teacher.prompt_identity`` digests."""
    from .teacher import prompt_dir, prompt_files
    paths = (*prompt_files("robotwin").values(), prompt_dir("robotwin") / "task_context.json")
    return [path.resolve() for path in paths]


def import_official_client(path: Path):
    """Load the exact released client file after identity preflight."""
    spec = importlib.util.spec_from_file_location("oeh_pinned_robotwin_openwam", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned RoboTwin OpenWAM client {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenWAMBackend:
    """Use the released client/server boundary and its exact action conversion."""

    def __init__(self, client, official):
        self.client, self.official = client, official
        self.observation = None

    def reset(self):
        self.client.reset(task_description="")

    def update_obs(self, observation):
        self.observation = observation

    def get_action(self):
        obs = self.observation
        if obs is None:
            raise RuntimeError("RoboTwin observation missing")
        vision = obs["vision"]
        state = self.official._extract_eef_proprio({"endpose": obs["state"]})
        example = {"cams": {"head": vision["cam_head"]["color"],
                            "left": vision.get("cam_left_wrist", {}).get("color"),
                            "right": vision.get("cam_right_wrist", {}).get("color")},
                   "lang": obs["instruction"], "state": state}
        action = self.client.step(example)
        if np.asarray(action).shape != (20,):
            raise ValueError("released RoboTwin-Full server must return EEF20")
        return [self.official.action_conversion.eef20d_to_ee16d(action)]

    def review_chunk(self, rows):
        """One fresh full OpenWAM chunk on the current observation, as ``rows`` native EE16 rows (review mode).

        The released server answers one action per request: its sync executor generates a chunk only when its buffer
        is empty and pops one row per request, and RESET empties the buffer (openwam/deploy/executors/
        sync_executor.py at fa431ec). So the released client's ``reset`` with the instruction, then ``rows`` requests
        on this one observation, return exactly one chunk generated from it: the rows the direct condition would get
        at a chunk boundary on this observation (the model conditions on the current observation only). ``rows`` must
        be the chunk length, which the frozen max_chunk_steps holds."""
        if self.observation is None:
            raise RuntimeError("RoboTwin observation missing")
        self.client.reset(task_description=str(self.observation["instruction"]))
        return [self.get_action()[0] for _ in range(rows)]

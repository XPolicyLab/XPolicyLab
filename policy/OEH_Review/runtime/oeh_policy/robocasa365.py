"""Pinned RoboCasa365/OpenWAM evaluation boundary, separate from RoboDojo.

RoboCasa365 (robocasa a07e365 on robosuite 5ce6643) runs through XPolicyLab fa431ec's released OpenWAM client: the
official loop ``benchmarks/robocasa365/single_eval.py`` drives RoboCasa's gym wrapper (one PandaOmron, a Panda arm on
the Omron mobile base) and asks its policy for one native action per observation, ``policy.act(obs, prompt)``. The
released client (``openwam2robocasa365_interface.OpenWAMRoboCasa365Policy``) reads the three 256x256 RGB views, the
base's world pose, the gripper's pose relative to the base and the finger joints, and turns the server's compact
action15 into RoboCasa's native action12 (``ACTION_SLICES``); this module never adjusts that conversion.

Only OEH review mode runs here (``robocasa365_review``): a teacher reviews one fresh full OpenWAM chunk per decision,
and the official loop still submits one action12 per call. The OEH runtime runs inside the official client process
(``robocasa365_entry``), as on RoboTwin; the direct condition is the official client itself.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from .action_space import (RC365_FINGER_GAP_M, RC365_OSC_SCALE, RC365_STATE_KEYS, robocasa365_state,
                           validate_robocasa365_action)
from .contracts import (check_review_benchmark, digest, file_digest, observation_digest, plain,
                        validate_review)
from .profiles import MOBILE_ARM_EEF12, check_embodiment

ROBOCASA_COMMIT = "a07e365c958c4216cd6bbd5f30b47f09a65c6f00"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
# RoboCasa365's benchmark split (OpenWAM's published numbers): the pretrain kitchens, layouts and objects.
SPLIT = "pretrain"
# One native step: robocasa a07e365 environments/kitchen/kitchen.py Kitchen(control_freq=20); robocasa's create_env
# and the gym wrapper pass no other rate. check_pinned_robot reads it again from the pinned file.
CONTROL_HZ = 20.0
INSTRUCTION_KEY = "annotation.human.task_description"
# The gym wrapper's views (PandaOmronKeyConverter, 256x256, flipped upright) -> the review cameras, in the order of
# OpenWAM's slots (head, left wrist, right): the released client fills the right slot with robot0_agentview_right.
CAMERAS = (("video.robot0_agentview_left", "robot0_agentview_left"),
           ("video.robot0_eye_in_hand", "robot0_eye_in_hand"),
           ("video.robot0_agentview_right", "robot0_agentview_right"))
# The released client's state keys (openwam2robocasa365_interface.DEFAULT_STATE_KEYS) -> the projected state.
STATE_KEYS = tuple(("state." + key, key) for key in RC365_STATE_KEYS)
# What the projected state says about the gripper: the two finger joints as measured, never the last command (the
# review method's ledger keeps that).
GRIPPER_STATE_SEMANTICS = "measured_finger_joints"
ACTION_LIMITS = {"native_action": "robocasa365_action12", "finite": True}
# The pinned upstream files the configuration's identity must hold (beside the released client and the loop).
OPENWAM_FILES = ("benchmarks/robocasa365/prompt_template.py", "benchmarks/robocasa365/policy_config.yml",
                 "benchmarks/utils/__init__.py", "benchmarks/utils/client.py", "benchmarks/utils/transport.py",
                 "benchmarks/utils/action_conversion.py", "openwam/deploy/server.py", "configs/deploy.yaml",
                 "scripts/deploy.py")
ROBOCASA_FILES = ("robocasa/utils/dataset_registry.py", "robocasa/utils/dataset_registry_utils.py",
                  "robocasa/environments/kitchen/kitchen.py", "robocasa/wrappers/gym_wrapper.py",
                  "robocasa/utils/env_utils.py")
ROBOSUITE_FILES = ("robosuite/controllers/config/robots/default_pandaomron.json",
                   "robosuite/controllers/parts/generic/joint_pos.py",
                   "robosuite/models/assets/grippers/panda_gripper.xml",
                   "robosuite/models/assets/bases/omron_mobile_base.xml")


def required_identity_paths(native) -> list[Path]:
    """The pinned files a RoboCasa365 configuration's identity must hold: the official loop and released client,
    OpenWAM's utilities and server, RoboCasa's registry, kitchen and gym wrapper, robosuite's PandaOmron controller,
    gripper and base."""
    openwam = Path(native["openwam_root"]).resolve()
    root, robosuite = Path(native["root"]).resolve(), Path(native["robosuite_root"]).resolve()
    return [Path(native["official_eval_path"]).resolve(), Path(native["official_client_path"]).resolve(),
            *(openwam / name for name in OPENWAM_FILES), *(root / name for name in ROBOCASA_FILES),
            *(robosuite / name for name in ROBOSUITE_FILES)]


def _registry_tables(path: Path) -> dict:
    """``ATOMIC_TASK_DATASETS``, ``COMPOSITE_TASK_DATASETS`` and ``TARGET_TASKS`` of robocasa's dataset_registry.py,
    read without importing robocasa: {task: horizon} per dataset table, {group: [task]} for the targets."""
    tables = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)):
            continue
        name = node.targets[0].id
        if name in ("ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"):
            entries = {}
            for keyword in node.value.keywords:
                horizons = [item.value for item in getattr(keyword.value, "keywords", []) if item.arg == "horizon"]
                entries[keyword.arg] = ast.literal_eval(horizons[0]) if len(horizons) == 1 else None
            tables[name] = entries
        elif name == "TARGET_TASKS":
            tables[name] = {keyword.arg: [ast.literal_eval(item) for item in keyword.value.elts]
                            for keyword in node.value.keywords}
    missing = {"ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS", "TARGET_TASKS"} - set(tables)
    if missing:
        raise ValueError(f"{path} defines no {sorted(missing)}")
    return tables


def task_horizon(registry: Path, task: str) -> int:
    """robocasa's ``get_task_horizon(task)`` (utils/dataset_registry_utils.py), which the official loop's
    ``_resolve_max_steps`` returns when ``max_steps_override`` is null: the task's ``horizon`` in the atomic table,
    else in the composite one."""
    tables = _registry_tables(registry)
    for name in ("ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"):
        if task in tables[name]:
            horizon = tables[name][task]
            if type(horizon) is not int or horizon < 1:
                raise ValueError(f"RoboCasa task {task} has no positive horizon in {registry}")
            return horizon
    raise ValueError(f"RoboCasa task {task} is in neither dataset table of {registry}")


def _one(pattern: str, text: str, where: str) -> str:
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise ValueError(f"{where}: expected one match of {pattern!r}, found {len(found)}")
    return found[0]


def pinned_robot(root: Path, robosuite_root: Path) -> dict:
    """The PandaOmron facts this port relies on, read from the pinned files: the control rate, the arm's OSC_POSE
    delta scale and reference frame, the finger joints' ranges, the base's velocity ranges and the torso's step."""
    root, robosuite = Path(root), Path(robosuite_root) / "robosuite"
    kitchen = (root / "robocasa/environments/kitchen/kitchen.py").read_text(encoding="utf-8")
    rate = float(_one(r"(?m)^\s+control_freq=(\d+(?:\.\d+)?),\s*$", kitchen, "kitchen.py"))
    controller = json.loads((robosuite / "controllers/config/robots/default_pandaomron.json").read_text(
        encoding="utf-8"))
    parts = controller["body_parts"]
    arm, torso, base = parts["arms"]["right"], parts["torso"], parts["base"]
    joint_pos = (robosuite / "controllers/parts/generic/joint_pos.py").read_text(encoding="utf-8")
    torso_step = float(_one(r"(?m)^\s+output_max=([0-9.]+),\s*$", joint_pos, "joint_pos.py"))
    gripper = ET.parse(robosuite / "models/assets/grippers/panda_gripper.xml").getroot()
    ranges = {joint.get("name"): [float(v) for v in joint.get("range").split()]
              for joint in gripper.iter("joint") if joint.get("name") in ("finger_joint1", "finger_joint2")}
    omron = ET.parse(robosuite / "models/assets/bases/omron_mobile_base.xml").getroot()
    actuators = {item.get("name"): [float(v) for v in item.get("ctrlrange").split()] for item in omron.iter("velocity")}
    lift = [float(v) for joint in omron.iter("joint") if joint.get("name") == "joint_torso_height"
            for v in joint.get("range").split()]
    return {"control_hz": rate, "composite": controller.get("type"), "arm_type": arm.get("type"),
            "arm_input": [arm.get("input_type"), arm.get("input_ref_frame"), arm.get("input_min"), arm.get("input_max")],
            "arm_output_max": arm.get("output_max"), "arm_output_min": arm.get("output_min"),
            "gripper": (arm.get("gripper") or {}).get("type"), "torso_type": torso.get("type"),
            "torso_overrides": sorted(set(torso) - {"type", "interpolation", "kp"}), "torso_step_m": torso_step,
            "base_type": base.get("type"),
            "finger_gap_m": round(ranges["finger_joint1"][1] - ranges["finger_joint2"][0], 6),
            "finger_closed": [ranges["finger_joint1"][0], ranges["finger_joint2"][1]],
            "base_velocity": [actuators.get(f"actuator_mobile_{name}") for name in ("forward", "side", "yaw")],
            "torso_range_m": lift}


def check_pinned_robot(root: Path, robosuite_root: Path) -> dict:
    """``pinned_robot``, refused unless it is what oeh_policy.action_space (RC365_*) and robocasa365_review
    (STEP_SECONDS, BASE_SCALE) assume."""
    from .robocasa365_review import BASE_SCALE, STEP_SECONDS
    facts = pinned_robot(root, robosuite_root)
    translation, turn = RC365_OSC_SCALE
    expected = {"control_hz": CONTROL_HZ, "composite": "HYBRID_MOBILE_BASE", "arm_type": "OSC_POSE",
                "arm_input": ["delta", "base", -1, 1], "arm_output_max": [translation] * 3 + [turn] * 3,
                "arm_output_min": [-translation] * 3 + [-turn] * 3, "gripper": "GRIP", "torso_type": "JOINT_POSITION",
                "torso_overrides": [], "torso_step_m": BASE_SCALE[3], "base_type": "JOINT_VELOCITY",
                "finger_gap_m": RC365_FINGER_GAP_M, "finger_closed": [0.0, 0.0],
                "base_velocity": [[-scale, scale] for scale in BASE_SCALE[:3]], "torso_range_m": [0.0, 0.34]}
    differences = [f"{key}: pinned {facts.get(key)!r}, assumed {value!r}" for key, value in expected.items()
                   if facts.get(key) != value]
    if differences or 1.0 / facts["control_hz"] != STEP_SECONDS:
        raise ValueError("the pinned PandaOmron differs from what the RoboCasa365 port assumes: "
                         + "; ".join(differences or ["step seconds"]))
    return facts


def _rgb(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
        raise ValueError(f"{name} must be nonempty uint8 RGB")
    return image.copy()


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def flat_action(interface, sliced) -> np.ndarray:
    """The released client's action dict (``slice_action``) as the 12-vector it sliced, in ``ACTION_SLICES`` order."""
    slices = interface.ACTION_SLICES
    if sorted(slices.values()) != [(0, 3), (3, 6), (6, 7), (7, 11), (11, 12)] or set(sliced) != set(slices):
        raise ValueError("the RoboCasa365 action dict is not the released client's ACTION_SLICES")
    flat = np.zeros(12, dtype=np.float32)
    for key, (start, end) in slices.items():
        part = np.asarray(sliced[key], dtype=np.float32).reshape(-1)
        if part.shape != (end - start,):
            raise ValueError(f"RoboCasa365 {key} needs {end - start} values")
        flat[start:end] = part
    return flat


def sliced_action(interface, action) -> dict:
    """A native action12 as the dict the official loop hands ``env.step`` (the released client's ``slice_action``)."""
    return interface.slice_action(validate_robocasa365_action(action))


class RoboCasa365Contract:
    """Review mode on RoboCasa365, bound to the one frozen configuration it validated."""

    benchmark = "robocasa365"
    robot_id = "robocasa365_panda_omron"
    action_contract = "RoboCasa365 a07e365 native action12 (PandaOmron), OpenWAM action15 to action12 official conversion"
    ledger_kind = "obs/robocasa365_ledger"

    # The review port (oeh_policy.robocasa365_review): the gym wrapper's three views in OpenWAM's slot order, with the
    # names the mobile_arm_eef12 profile gives them.
    review_cameras = MOBILE_ARM_EEF12.cameras

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
        """The review condition, its budgets and teacher; binds this contract to them."""
        cfg = copy.deepcopy(dict(value))
        required = {"schema", "benchmark", "mode", "kernel", "repository", "output", "model_id", "checkpoint_sha256",
                    "normalization_sha256", "identity_files", "max_chunk_steps", "native_step_limit", "wall_seconds",
                    "decision_seconds", "action_limits", "robocasa365", "embodiment"}
        if required - cfg.keys():
            raise ValueError(f"missing RoboCasa365 policy fields: {sorted(required - cfg.keys())}")
        if cfg["schema"] != "oeh.policy.v1" or cfg["benchmark"] != "robocasa365":
            raise ValueError("expected pinned RoboCasa365 policy configuration")
        check_embodiment(cfg["embodiment"], "robocasa365")
        if cfg["mode"] != "review":
            raise ValueError("RoboCasa365 runs OEH's review condition only; the direct condition is OpenWAM's own "
                             "released client")
        self._check_review(cfg)
        # The candidate's length: one full OpenWAM chunk (the template cites its source; the matrix compile checks it
        # against the pinned server config). The official loop still takes one of its rows per call.
        if type(cfg["max_chunk_steps"]) is not int or cfg["max_chunk_steps"] < 1:
            raise ValueError("RoboCasa365 review mode needs the OpenWAM chunk length as a positive max_chunk_steps")
        if type(cfg["native_step_limit"]) is not int or cfg["native_step_limit"] < 1:
            raise ValueError("positive native step limit required")
        for key in ("wall_seconds", "decision_seconds"):
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < math.inf:
                raise ValueError(f"finite positive {key} required")
        if "first_decision_seconds" in cfg:
            raise ValueError("a review call is bounded by review.call_seconds and first_call_extra_seconds, not "
                             "first_decision_seconds")
        for key in ("checkpoint_sha256", "normalization_sha256"):
            if not _hex64(cfg[key]):
                raise ValueError(f"frozen {key} required")
        self.mode = cfg["mode"]
        self.review_rows = cfg["max_chunk_steps"]
        return cfg

    def _check_review(self, cfg):
        """The teacher under ``review`` (the RoboDojo review block, RoboCasa365's control rate and task), the
        workspace its rows are checked against under ``review_workspace``, and no slow-brain block."""
        if cfg.get("brain") or "escalation" in cfg or "primitives" in cfg:
            raise ValueError("RoboCasa365 review mode configures its teacher under review.teacher, never brain, "
                             "escalation or primitives")
        check_review_benchmark(cfg["benchmark"], cfg.get("review"))
        cfg["review"] = validate_review(cfg.get("review"))
        from .robocasa365_review import check_review_workspace
        from .teacher import task_context
        if cfg["review"]["control_hz"] != CONTROL_HZ:
            raise ValueError(f"review.control_hz must be RoboCasa365's control rate {CONTROL_HZ!r} Hz")
        task = cfg["review"]["task"]
        native = cfg.get("robocasa365") if isinstance(cfg.get("robocasa365"), dict) else {}
        if task != native.get("task"):
            raise ValueError(f"review.task {task!r} must be the episode's task, robocasa365.task "
                             f"{native.get('task')!r}")
        if task_context(task, cfg["benchmark"]) is None:
            # A task without a note would reach the teacher with no task context and no error.
            raise ValueError(f"review.task {task!r} has no note in the robocasa365 teacher's task_context.json")
        cfg["review_workspace"] = check_review_workspace(cfg.get("review_workspace"))
        if cfg.get("gripper_state_semantics") != GRIPPER_STATE_SEMANTICS:
            raise ValueError("RoboCasa365's state carries the measured finger joints, not the gripper command; "
                             f"declare gripper_state_semantics {GRIPPER_STATE_SEMANTICS}")

    def check_native(self, cfg):
        """Pinned sources, robot facts, step limit and released model identities on disk."""
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
        native = cfg["robocasa365"]
        root, robosuite = Path(native["root"]).resolve(), Path(native["robosuite_root"]).resolve()
        folder = Path(native["openwam_root"]).resolve() / "benchmarks/robocasa365"
        if (Path(native["official_client_path"]).resolve() != folder / "openwam2robocasa365_interface.py"
                or Path(native["official_eval_path"]).resolve() != folder / "single_eval.py"):
            raise ValueError("the released RoboCasa365 client and loop are OpenWAM's benchmarks/robocasa365 files")
        if any(str(path) not in cfg["identity_files"] for path in required_identity_paths(native)):
            raise ValueError("official RoboCasa365 loop, OpenWAM client and pinned robocasa/robosuite files must be "
                             "pinned")
        synthetic = cfg.get("data_kind", "simulation") == "synthetic"
        for label, checkout, commit_key, pin, diff_key in (
                ("robocasa", root, "source_commit", ROBOCASA_COMMIT, "tracked_diff_sha256"),
                ("robosuite", robosuite, "robosuite_commit", ROBOSUITE_COMMIT, "robosuite_tracked_diff_sha256")):
            commit = native.get(commit_key, pin)
            if commit != pin and not synthetic:
                raise ValueError(f"{label} source must be {pin}")
            if _git(checkout, "rev-parse", "HEAD").decode().strip() != commit:
                raise ValueError(f"{label} source must be {commit}")
            if hashlib.sha256(_git(checkout, "diff", "--binary", "HEAD")).hexdigest() != native.get(diff_key):
                raise ValueError(f"{label} source patch differs from frozen manifest")
        if native.get("split") != SPLIT:
            raise ValueError(f"RoboCasa365's benchmark split is {SPLIT}")
        if cfg["action_limits"] != ACTION_LIMITS:
            raise ValueError("RoboCasa365 action contract must preserve native action12 without clipping")
        if cfg["native_step_limit"] != task_horizon(root / "robocasa/utils/dataset_registry.py", native["task"]):
            raise ValueError("native step limit differs from the pinned RoboCasa task horizon")
        check_pinned_robot(root, robosuite)

        def observed_digest(path):
            key = str(Path(path).resolve())
            return observed[key] if key in observed else file_digest(Path(path))
        if observed_digest(native["checkpoint_file"]) != cfg["checkpoint_sha256"]:
            raise ValueError("RoboCasa365 checkpoint mismatch")
        if observed_digest(native["normalization_file"]) != cfg["normalization_sha256"]:
            raise ValueError("RoboCasa365 normalizer mismatch")
        if (cfg["checkpoint_sha256"] != native["expected_checkpoint_sha256"] or
                cfg["normalization_sha256"] != native["expected_normalization_sha256"]):
            raise ValueError("RoboCasa365 released model identity mismatch")

    def make_method(self, cfg, backend=None):
        from .robocasa365_review import RoboCasa365ReviewMethod
        return RoboCasa365ReviewMethod(cfg)

    def review_chunk(self, backend):
        """The review candidate: one fresh full OpenWAM chunk of native action12 rows on the current observation."""
        return backend.review_chunk(self.review_rows)

    def review_views(self, observation):
        """The teacher's views, exactly ``review_cameras`` in order."""
        vision = observation["vision"]
        missing = [name for name in self.review_cameras if name not in vision]
        if missing:
            raise ValueError(f"RoboCasa365 review mode needs the {missing} views; the native observation has none")
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
        """The gripper's world position after a step and the opening it commanded (1 open, 0 closed)."""
        from .action_space import RoboCasa365Action12
        space = RoboCasa365Action12()
        pose = space.review_state(state)[space.arm + "_ee_pose"]
        return [np.asarray(pose[:3], dtype=float)], [space.openings(action)[space.arm]]

    def project_state(self, state):
        """The five state arrays the released client reads, from a native observation (``state.<key>``) or a projected
        state (``<key>``), as float32 copies (the gym wrapper hands float32; the client reads float32)."""
        result = {}
        for native_key, key in STATE_KEYS:
            value = state[native_key] if native_key in state else state[key]
            result[key] = np.asarray(value, dtype=np.float32).reshape(-1).copy()
        robocasa365_state(result)  # sizes, finite values, rotations
        return result

    def project_observation(self, obs):
        source, instruction = obs["native"], obs["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("native RoboCasa365 instruction required")
        vision = {name: {"color": _rgb(source[key], key)} for key, name in CAMERAS}
        return {"vision": vision, "state": self.project_state(source), "instruction": instruction}

    def validate_action(self, action, limits):
        return validate_robocasa365_action(action)

    def camera_payload(self, runtime):
        # Review mode reads no camera event; keep an explicit whitelist even so.
        obs = runtime.observation
        images = [{"camera": name, "sha256": observation_digest(obs["vision"][name])}
                  for _, name in CAMERAS if name in obs["vision"]]
        return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step,
                "epoch": 0, "task_instruction": obs["instruction"],
                "robot_state": plain(obs["state"]), "images": images,
                "previous_receipt": runtime.previous_receipt}

    def ledger_payload(self, runtime, projected):
        return {"episode_id": runtime.episode_id, "seq": runtime.gate.step,
                "state_sha256": digest(projected["state"]), "native_step": runtime.gate.step}


def review_prompt_paths() -> list[Path]:
    """The teacher prompt files a review configuration pins (resolved): RoboCasa365's skill, context and task notes
    and the shared gate prompt, the files ``teacher.prompt_identity`` digests."""
    from .teacher import prompt_dir, prompt_files
    paths = (*prompt_files("robocasa365").values(), prompt_dir("robocasa365") / "task_context.json")
    return [path.resolve() for path in paths]


class OpenWAMBackend:
    """The released client (``OpenWAMRoboCasa365Policy``) and its exact conversion, on the projected observation."""

    def __init__(self, policy, interface):
        self.policy, self.interface = policy, interface
        self.observation = None

    def reset(self):
        """The episode's start, as the direct condition's ``policy.reset()``: the server's RESET."""
        self.policy.reset()

    def update_obs(self, observation):
        self.observation = observation

    def native_observation(self) -> dict:
        """The observation the released client reads, rebuilt from the projected one: the three views under the gym
        wrapper's keys and the five state arrays (``build_obs_payload`` reads nothing else)."""
        obs = self.observation
        if obs is None:
            raise RuntimeError("RoboCasa365 observation missing")
        native = {key: obs["vision"][name]["color"] for key, name in CAMERAS}
        native.update({key: obs["state"][name] for key, name in STATE_KEYS})
        return native

    def get_action(self):
        raise RuntimeError("RoboCasa365 runs review mode only: its candidate is a whole chunk (review_chunk)")

    def review_chunk(self, rows):
        """One fresh full OpenWAM chunk on the current observation, as ``rows`` native action12 rows.

        The released server answers one action per request: its sync executor generates a chunk only when its buffer
        is empty and pops one row per request, and RESET empties the buffer (openwam/deploy/executors/
        sync_executor.py at fa431ec). So the released client's ``reset`` and then ``rows`` of its ``act`` on this one
        observation return exactly one chunk generated from it, bridged and binarized by the released client: the
        rows the direct condition would get at a chunk boundary on this observation. ``rows`` must be the chunk
        length, which the frozen max_chunk_steps holds."""
        native = self.native_observation()
        instruction = str(self.observation["instruction"])
        self.policy.reset()
        return [flat_action(self.interface, self.policy.act(native, instruction)) for _ in range(rows)]

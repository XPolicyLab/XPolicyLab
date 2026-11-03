"""Pinned RoboCasa GR1 tabletop / OpenWAM evaluation boundary, separate from RoboDojo.

The released GR1 client (OpenWAM ``benchmarks/robocasa_gr1/openwam2robocasa_gr1_interface.py``, run by its
``single_eval.py``) sends OpenWAM the ego view, the prompt and the EEF33 state it reads from the live simulation
(``gr1_kinematics``), receives one EEF33 row per request and solves it with ``gr1_kinematics``' IK into absolute arm
joint targets, passing the hands and the waist through. OEH runs one condition here, review mode
(``robocasa_gr1_review``): a teacher reviews one fresh full OpenWAM chunk per decision, and the official loop still
gets one env action per ``act``, the released client's own conversion of one admitted row (``robocasa_gr1_entry``).
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import subprocess
import threading

import numpy as np

from .contracts import (check_review_benchmark, digest, file_digest, observation_digest, plain,
                        validate_review)
from .profiles import GR1_EEF33, check_embodiment

ROBOCASA_GR1_COMMIT = "4840e671596f93ca03651524b9f72ffb1aadfeff"
# robocasa 4840e67 robocasa/environments/tabletop/tabletop.py:278 `control_freq=20`: one native step is 0.05 s.
CONTROL_HZ = 20.0
STATE_DIM = 33
CAMERA = "ego_view"  # the policy-side name of the released client's head camera (its only view here)
ROBOT = "GR1ArmsAndWaistFourierHands"
# The released client's constructor settings single_eval.py passes from policy_config.yml; the frozen configuration
# names each, and they must be the pinned file's (oeh_policy.robocasa_gr1_entry checks the ones the loop passes).
CLIENT_KEYS = ("action_mode", "request_timeout", "head_camera_key", "left_wrist_camera_key", "right_wrist_camera_key",
               "prompt_key", "fallback_prompt_key", "send_state", "state_dim")
NATIVE_KEYS = ("root", "robosuite_root", "openwam_root", "official_client_path", "official_runner_path",
               "official_config_path", "checkpoint_file", "normalization_file", "expected_checkpoint_sha256",
               "expected_normalization_sha256", "task", "env_id", "seed", "tracked_diff_sha256", *CLIENT_KEYS)
ACTION_LIMITS = {"native_action": "eef33", "finite": True}
# The checkout's tracked edits (the asset download writes beside tracked files), hashed by the git invocation the
# matrix compile's pin uses (oeh_eval.matrix_emit.tracked_diff): the frozen digest is compared with that one.
TRACKED_DIFF = ("-c", "core.quotePath=false", "diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules=dirty",
                "--binary", "HEAD")


def env_id(task: str) -> str:
    """The gym id the released README and single_eval.sh run a task under."""
    return f"gr1_unified/{task}_{ROBOT}_Env"


def required_identity_paths(native) -> list[Path]:
    root = Path(native["root"]).resolve()
    openwam = Path(native["openwam_root"]).resolve()
    return [root / "robocasa/__init__.py",
            root / "robocasa/utils/gym_utils/__init__.py",
            root / "robocasa/utils/gym_utils/gymnasium_groot.py",
            root / "robocasa/utils/gym_utils/gymnasium_basic.py",
            root / "robocasa/environments/tabletop/tabletop.py",
            root / "robocasa/models/robots/__init__.py",
            root / "robocasa/models/robots/manipulators/gr1_robot.py",
            Path(native["official_client_path"]).resolve(),
            Path(native["official_runner_path"]).resolve(),
            Path(native["official_config_path"]).resolve(),
            openwam / "benchmarks/__init__.py",
            openwam / "benchmarks/utils/__init__.py",
            openwam / "benchmarks/utils/action_conversion.py",
            openwam / "benchmarks/utils/client.py",
            openwam / "benchmarks/utils/transport.py",
            openwam / "openwam/__init__.py",
            openwam / "openwam/dataloader/utils/__init__.py",
            openwam / "openwam/dataloader/utils/gr1_kinematics.py",
            openwam / "openwam/deploy/server.py",
            openwam / "configs/deploy.yaml",
            openwam / "scripts/deploy.sh",
            Path(native["robosuite_root"]).resolve() / "__init__.py"]


def tracked_diff_sha256(root) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "-C", str(root), *TRACKED_DIFF])).hexdigest()


def official_settings(path: Path) -> dict:
    """The pinned ``policy_config.yml`` the released ``single_eval.py`` reads its settings from."""
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def _rgb(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
        raise ValueError(f"{name} must be nonempty uint8 RGB")
    return image.copy()


def eef33(value) -> np.ndarray:
    row = np.asarray(value, dtype=np.float32).reshape(-1)
    if row.shape != (STATE_DIM,) or not np.isfinite(row).all():
        raise ValueError("RoboCasa GR1 EEF33 must contain 33 finite scalars")
    return row.copy()


class RoboCasaGR1Contract:
    """Bound to the one frozen review configuration it validated."""

    benchmark = "robocasa_gr1"
    robot_id = "robocasa_gr1_fourier_hands"
    action_contract = ("RoboCasa GR1 4840e67 native EEF33 (robot-base frame, rot6d, Fourier hands, waist), solved by "
                       "OpenWAM's released GR1 client (gr1_kinematics IK) into absolute arm-joint targets")
    ledger_kind = "obs/robocasa_gr1_ledger"
    # The review port (oeh_policy.robocasa_gr1_review): the released client's one view, the ego (head) camera.
    review_cameras = GR1_EEF33.cameras

    def __init__(self):
        self.mode = None
        self.persona_path = None  # review mode only: no slow persona
        self.review_rows = None  # the rows of one OpenWAM chunk (max_chunk_steps)
        self.head_camera_key = None
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
        """Condition, budgets and the review contract; binds this contract to them."""
        cfg = copy.deepcopy(dict(value))
        required = {"schema", "benchmark", "mode", "kernel", "repository", "output", "model_id", "checkpoint_sha256",
                    "normalization_sha256", "identity_files", "max_chunk_steps", "native_step_limit", "wall_seconds",
                    "decision_seconds", "action_limits", "robocasa_gr1", "embodiment", "review", "review_workspace"}
        if required - cfg.keys():
            raise ValueError(f"missing RoboCasa GR1 policy fields: {sorted(required - cfg.keys())}")
        if cfg["schema"] != "oeh.policy.v1" or cfg["benchmark"] != self.benchmark:
            raise ValueError("expected pinned RoboCasa GR1 policy configuration")
        check_embodiment(cfg["embodiment"], self.benchmark)
        if cfg["mode"] != "review":
            raise ValueError("RoboCasa GR1 runs the direct and review conditions; its OEH policy runs review mode only")
        if cfg.get("brain") or "escalation" in cfg or "primitives" in cfg:
            raise ValueError("RoboCasa GR1 review mode configures its teacher under review.teacher, never brain, "
                             "escalation or primitives")
        check_review_benchmark(cfg["benchmark"], cfg.get("review"))
        cfg["review"] = validate_review(cfg.get("review"))
        if cfg["review"]["control_hz"] != CONTROL_HZ:
            raise ValueError(f"review.control_hz must be RoboCasa GR1's control rate {CONTROL_HZ!r} Hz")
        from .robocasa_gr1_review import check_review_workspace
        cfg["review_workspace"] = check_review_workspace(cfg.get("review_workspace"))
        # The candidate's length: one full OpenWAM chunk (the matrix compile derives it from the pinned server config).
        if type(cfg["max_chunk_steps"]) is not int or cfg["max_chunk_steps"] < 1:
            raise ValueError("RoboCasa GR1 review mode needs the OpenWAM chunk length as a positive max_chunk_steps")
        if type(cfg["native_step_limit"]) is not int or cfg["native_step_limit"] < 1:
            raise ValueError("positive native step limit required")
        for key in ("wall_seconds", "decision_seconds"):
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < math.inf:
                raise ValueError(f"finite positive {key} required")
        if "first_decision_seconds" in cfg:
            raise ValueError("a review call is bounded by review.call_seconds and first_call_extra_seconds")
        for key in ("checkpoint_sha256", "normalization_sha256"):
            value = cfg[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"frozen {key} required")
        if cfg["action_limits"] != ACTION_LIMITS:
            raise ValueError("RoboCasa GR1 action contract must preserve native EEF33 without clipping")
        native = cfg["robocasa_gr1"]
        if not isinstance(native, dict) or set(NATIVE_KEYS) - set(native):
            raise ValueError(f"robocasa_gr1 must name {sorted(set(NATIVE_KEYS) - set(native or {}))}")
        if native["env_id"] != env_id(native["task"]):
            raise ValueError(f"robocasa_gr1.env_id must be {env_id(native['task'])!r}, the task's gr1_unified id")
        if type(native["seed"]) is not int or native["seed"] < 0:
            raise ValueError("robocasa_gr1.seed must be a nonnegative integer")
        if native["action_mode"] != "eef" or native["send_state"] is not True or native["state_dim"] != STATE_DIM:
            raise ValueError("the released GR1 client runs action_mode eef with the 33-dimensional EEF33 state")
        if not isinstance(native["head_camera_key"], str) or not native["head_camera_key"]:
            raise ValueError("robocasa_gr1.head_camera_key must name the released client's head camera")
        self.mode = "review"
        self.review_rows = cfg["max_chunk_steps"]
        self.head_camera_key = native["head_camera_key"]
        return cfg

    def check_native(self, cfg):
        """Pinned source, client settings, step limit and released model identities on disk."""
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
        native = cfg["robocasa_gr1"]
        root = Path(native["root"]).resolve()
        if any(str(path) not in cfg["identity_files"] for path in required_identity_paths(native)):
            raise ValueError("official RoboCasa GR1 and OpenWAM client files must be pinned")
        source_commit = native.get("source_commit", ROBOCASA_GR1_COMMIT)
        if source_commit != ROBOCASA_GR1_COMMIT and cfg.get("data_kind", "simulation") != "synthetic":
            raise ValueError(f"RoboCasa GR1 source must be {ROBOCASA_GR1_COMMIT}")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if revision != source_commit:
            raise ValueError(f"RoboCasa GR1 source must be {source_commit}")
        if tracked_diff_sha256(root) != native["tracked_diff_sha256"]:
            raise ValueError("RoboCasa GR1 source patch differs from frozen manifest")
        official = official_settings(native["official_config_path"])
        for key in CLIENT_KEYS:
            if native[key] != official.get(key):
                raise ValueError(f"robocasa_gr1.{key} {native[key]!r} is not the pinned policy_config.yml's "
                                 f"{official.get(key)!r}")
        if cfg["native_step_limit"] != official.get("max_steps"):
            raise ValueError("native step limit differs from the pinned single_eval policy_config.yml max_steps")

        def observed_digest(path):
            key = str(Path(path).resolve())
            return observed[key] if key in observed else file_digest(Path(path))
        if observed_digest(native["checkpoint_file"]) != cfg["checkpoint_sha256"]:
            raise ValueError("RoboCasa GR1 checkpoint mismatch")
        if observed_digest(native["normalization_file"]) != cfg["normalization_sha256"]:
            raise ValueError("RoboCasa GR1 normalizer mismatch")
        if (cfg["checkpoint_sha256"] != native["expected_checkpoint_sha256"] or
                cfg["normalization_sha256"] != native["expected_normalization_sha256"]):
            raise ValueError("RoboCasa GR1 released model identity mismatch")

    def make_method(self, cfg, backend=None):
        from .robocasa_gr1_review import RoboCasaGR1ReviewMethod
        return RoboCasaGR1ReviewMethod(cfg)

    def review_chunk(self, backend):
        """The review candidate: one fresh full OpenWAM chunk of native EEF33 rows on the current observation."""
        return backend.review_chunk(self.review_rows)

    def review_views(self, observation):
        """The teacher's view, exactly ``review_cameras``: the ego view the released client sends OpenWAM."""
        return {name: observation["vision"][name]["color"] for name in self.review_cameras}

    def intent_schema(self, max_steps):
        raise ValueError("RoboCasa GR1 runs review mode only")

    def model_observation(self, runtime, projected):
        return projected

    def subgoal_observation(self, runtime, subgoal):
        raise ValueError("RoboCasa GR1 runs review mode only")

    def escalated(self, runtime):
        pass

    def motion_features(self, state, action):
        raise ValueError("RoboCasa GR1 runs review mode only")

    def project_state(self, state):
        """The measured EEF33 the released client sends as its state (robot-base frame, rot6d, hands, waist)."""
        return {"eef33": eef33(state["eef33"])}

    def project_observation(self, obs):
        """What the released client gives OpenWAM: the head camera image, the EEF33 state and the prompt."""
        instruction = obs["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("native RoboCasa GR1 instruction required")
        if self.head_camera_key is None:
            raise RuntimeError("the contract has validated no configuration")
        return {"vision": {CAMERA: {"color": _rgb(obs["native"][self.head_camera_key], self.head_camera_key)}},
                "state": self.project_state(obs), "instruction": instruction}

    def validate_action(self, action, limits):
        return eef33(action)

    def camera_payload(self, runtime):
        obs = runtime.observation
        return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step, "epoch": 0,
                "task_instruction": obs["instruction"], "robot_state": plain(obs["state"]),
                "images": [{"camera": CAMERA, "sha256": observation_digest(obs["vision"][CAMERA])}],
                "previous_receipt": runtime.previous_receipt}

    def ledger_payload(self, runtime, projected):
        return {"episode_id": runtime.episode_id, "seq": runtime.gate.step,
                "state_sha256": digest(projected["state"]), "native_step": runtime.gate.step}


def review_prompt_paths() -> list[Path]:
    """The teacher prompt files a review configuration pins (resolved): GR1's skill, context and task notes and the
    shared gate prompt, the files ``teacher.prompt_identity`` digests."""
    from .teacher import prompt_dir, prompt_files
    paths = (*prompt_files("robocasa_gr1").values(), prompt_dir("robocasa_gr1") / "task_context.json")
    return [path.resolve() for path in paths]


class _Captured(Exception):
    def __init__(self, payload):
        super().__init__("payload captured")
        self.payload = payload


class _Capture:
    """Stands in for the released client's WebSocket client for one ``act`` call: takes its payload and stops it."""

    def predict(self, payload):
        raise _Captured(copy.deepcopy(payload))


class _Row:
    """Stands in for it while the released client converts one admitted row: answers with that row."""

    def __init__(self, row):
        self.row = row

    def predict(self, payload):
        return {"type": "action", "action": self.row.tolist()}


class OpenWAMBackend:
    """The released GR1 client's own boundary: its payload, its WebSocket client and its EEF33-to-joint conversion.

    ``official`` is the ``OpenWAMRoboCasaGR1Policy`` the official ``single_eval.py`` loop's arguments built. Its
    ``_client`` is the WebSocket client every OpenWAM request goes through. While the released ``act`` builds the
    payload for an observation (``capture``) a stand-in takes the payload instead of sending it; while one admitted row
    is converted (``convert``) a stand-in answers with that row. So OpenWAM gets the released client's own payload, and
    the joint targets the simulator gets are the released client's own conversion of exactly the admitted row."""

    def __init__(self, official):
        self.official = official
        self.observation = None
        self.native = self.request = None
        self.lock = threading.Lock()

    def reset(self):
        """The released client's episode reset: kinematics rebuilt on the reset simulation, the server's buffer
        emptied (RESET)."""
        with self.lock:
            self.official.reset()

    def update_obs(self, observation):
        self.observation = observation

    def capture(self, native):
        """The payload the released ``act`` builds for ``native``, taken where it would be sent: the client's own FK
        on the live simulation (the EEF33 state), the resized and encoded head image and the prompt. Nothing is sent,
        and the client counts no step (``act`` stops before its IK and step count)."""
        with self.lock:
            client, self.official._client = self.official._client, _Capture()
            try:
                self.official.act(native)
            except _Captured as captured:
                return captured.payload
            finally:
                self.official._client = client
        raise RuntimeError("the released GR1 client's act() sent no request")

    def bind(self, native, request):
        """The native observation the next requests are about and the payload ``capture`` took for it: the policy
        input is that payload's state and prompt, and the review chunk is requested with the payload itself."""
        self.native, self.request = native, request

    def measured(self, native):
        """The EEF33 the released client would send as the state for ``native`` (its FK on the live simulation)."""
        with self.lock:
            return eef33(self.official._kinematics.observation_to_eef33(native))

    def review_chunk(self, rows):
        """One fresh full OpenWAM chunk on the current observation, as ``rows`` native EEF33 rows (review mode).

        The released server answers one action per request: its sync executor generates a chunk only when its buffer
        is empty and pops one row per request, and RESET empties the buffer (openwam/deploy/executors/
        sync_executor.py at fa431ec). So the WebSocket client's ``reset`` (RESET only: the released client's own
        ``reset``, which starts an episode, is not called), then ``rows`` requests with this observation's payload,
        return exactly one chunk generated from it: the rows the direct condition gets at a chunk boundary on this
        observation. ``rows`` must be the chunk length, which the frozen max_chunk_steps holds."""
        if self.request is None or self.observation is None:
            raise RuntimeError("RoboCasa GR1 observation missing")
        with self.lock:
            client = self.official._client
            interface = _interface(self.official)
            ack = client.reset()
            if ack.get("type") != interface.transport.RESET_ACK:
                raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")
            chunk = []
            for _ in range(rows):
                raw = np.asarray(client.predict(self.request)["action"], dtype=np.float32).reshape(-1)
                if raw.shape != (interface.EEF33_DIM,):
                    raise ValueError(f"OpenWAM GR1 server must return EEF33, got {raw.shape}")
                chunk.append(raw)
            return chunk

    def convert(self, native, row):
        """The released client's ``act`` on ``native`` with ``row`` as OpenWAM's answer: the env action it returns
        (its IK solve of the arms; the hands and the waist passed through), with its own step and IK-failure counts."""
        with self.lock:
            client, self.official._client = self.official._client, _Row(eef33(row))
            try:
                return self.official.act(native)
            finally:
                self.official._client = client


def _interface(official):
    import sys
    return sys.modules[type(official).__module__]

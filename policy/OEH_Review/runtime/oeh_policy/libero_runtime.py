"""LIBERO-family contract (single_arm_eef profile) for the shared OEH kernel runtime.

Fast mode preserves the released model prompt and 7D OSC conversion. Hybrid mode
keeps that fast path and adds the single-arm slow capabilities: a VLA subgoal
that replaces the prompt until the next escalation, and closed-loop OSC
primitives whose rows are admitted and executed one per official step. Review
mode is the single-arm review port (``oeh_policy.libero_review``): OpenWAM's
fresh executed chunk is the teacher's candidate on every decision, and the
teacher sees the released client's two views.
"""
from __future__ import annotations

import base64
import copy
import io
from typing import Any

import numpy as np

from .contracts import check_review_benchmark, digest, validate_review
from .libero_boundary import (OfficialModelBoundary, project_wire_observation,
                              validate_native_action, validate_raw_action)
from .libero_config import check as check_frozen
from .profiles import SINGLE_ARM_EEF, check_embodiment, normalize_benchmark
from .single_arm import SingleArmMethod, intent_schema as single_arm_schema, raw_eef10

# Wire slots of the released LIBERO client -> slow-brain camera names.
SLOW_CAMERAS = (("head_camera", "agentview"), ("left_wrist_camera", "eye_in_hand"))


def decode_view(value: str) -> np.ndarray:
    """One released-client wire image (base64 PNG) as the teacher sees it: HxWx3 uint8 RGB, mirrored left-right.

    robosuite renders with ``IMAGE_CONVENTION`` "opengl" by default (LIBERO and LIBERO-Plus leave the "opencv" switch
    commented out), so a raw frame is upside down; the released client's ``rotate_180`` (``image[::-1, ::-1]``) makes
    it upright and mirrored. Undoing the mirror shows the scene as a camera image: on the agent view the robot stands
    at the far edge of the table, facing the camera. Nothing else changes: no crop and no aspect correction (the
    client resizes 256x256 frames to its 320x256 and 160x128 slots), so the teacher sees what OpenWAM is given.
    """
    from PIL import PngImagePlugin
    try:
        # The PNG decoder itself, not Image.open's format sniffing: the client sends PNG and nothing else.
        image = PngImagePlugin.PngImageFile(io.BytesIO(base64.b64decode(value, validate=True)))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any undecodable payload is the same refusal
        raise ValueError(f"LIBERO view is not a base64 PNG: {type(exc).__name__}") from None
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)[:, ::-1])


class LiberoMethod:
    """One official VLA action per fast decision; no invented analytic motion."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = "grasp"
        self.target = "full task"
        self.gripper_commands = {}

    def seed(self, state):
        project_state(state)

    def check_intent(self, params):
        required = {"observation_id", "mode", "phase", "target", "execute_steps", "subgoal"}
        if set(params) != required or params["mode"] != "vla" or params["execute_steps"] != 1:
            raise ValueError("LIBERO fast path permits only the official one-step VLA action")
        if params["phase"] != "grasp" or params["target"] != "full task" or \
                not isinstance(params["observation_id"], str) or \
                not isinstance(params["subgoal"], str) or not params["subgoal"].strip():
            raise ValueError("invalid LIBERO fast intent")

    def primitives(self, params, state):
        raise ValueError("LIBERO fast condition has no analytic primitives")

    def acknowledged(self, action):
        validate_native_action(action)


def project_state(value: Any) -> list[float]:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("LIBERO achieved EEF10 state required")
    return state.tolist()


class LiberoContract:
    """Bound to the one frozen configuration it validated (mode decides the method)."""

    action_contract = "OpenWAM native EEF10 converted by pinned official client to 7D OSC delta"
    ledger_kind = "obs/sim_policy_ready"
    robot_id = "libero_panda"

    # Review mode (oeh_policy.libero_review): the released client's wire slots, as SLOW_CAMERAS names them.
    review_cameras = tuple(name for _, name in SLOW_CAMERAS)

    def __init__(self, benchmark: str):
        if not isinstance(benchmark, str) or normalize_benchmark(benchmark) not in SINGLE_ARM_EEF.benchmarks:
            raise ValueError("LIBERO, LIBERO-Plus or LIBERO-Pro required")
        self.benchmark = normalize_benchmark(benchmark)
        self.hybrid = False
        self.review = False
        self.persona_path = None
        self.chunk_rows = None
        self.candidate_rows = None

    def validate_config(self, value):
        check_frozen(value)
        if normalize_benchmark(value["benchmark"]) != self.benchmark:
            raise ValueError("benchmark identity mismatch")
        check_embodiment(value.get("embodiment"), self.benchmark)
        if value["mode"] not in ("fast", "hybrid", "review"):
            raise ValueError("LIBERO runs the fast, hybrid and review conditions only")
        if value["mode"] == "review":
            # Also part of the freeze `check_frozen` repeats; named here as every review contract names them.
            check_review_benchmark(self.benchmark, value.get("review"))
            validate_review(value.get("review"))
        self.hybrid = value["mode"] == "hybrid"
        self.review = value["mode"] == "review"
        self.persona_path = SINGLE_ARM_EEF.persona_path if self.hybrid else None
        self.chunk_rows = value["primitives"]["chunk_rows"] if self.hybrid else None
        # libero_config's freeze ties a review candidate's rows to the official server's inference horizon.
        self.candidate_rows = value["max_chunk_steps"] if self.review else None
        return copy.deepcopy(value)

    def make_method(self, cfg, backend=None):
        if self.review:
            from .libero_review import LiberoReviewMethod
            return LiberoReviewMethod(cfg, backend)
        return SingleArmMethod(cfg, backend) if self.hybrid else LiberoMethod(cfg)

    def review_chunk(self, backend):
        """The review candidate: OpenWAM's fresh executed chunk on the current observation (``LiberoBackend.chunk``),
        as many rows as the frozen ``max_chunk_steps`` (the official server's inference horizon)."""
        return backend.chunk(self.candidate_rows)

    def review_views(self, observation):
        """The released client's own encoded views (``SLOW_CAMERAS``), decoded upright (``decode_view``)."""
        images = observation["images"]
        views = {}
        for slot, name in SLOW_CAMERAS:
            if not images.get(slot):
                raise ValueError(f"the released client sent no {slot} image; review shows both LIBERO views")
            views[name] = decode_view(images[slot])
        return views

    def intent_schema(self, max_steps):
        if max_steps != 1:
            raise ValueError("LIBERO fast endpoint returns one official action")
        if self.hybrid:
            return single_arm_schema(self.chunk_rows)
        return {"type": "object", "additionalProperties": False,
                "required": ["observation_id", "mode", "phase", "target", "execute_steps", "subgoal"],
                "properties": {"observation_id": {"type": "string", "minLength": 1},
                               "mode": {"type": "string", "enum": ["vla"]},
                               "phase": {"type": "string", "enum": ["grasp"]},
                               "target": {"type": "string", "enum": ["full task"]},
                               "execute_steps": {"type": "integer", "minimum": 1, "maximum": 1},
                               "subgoal": {"type": "string", "minLength": 1, "maxLength": 512}}}

    def project_observation(self, obs):
        wire = project_wire_observation(obs)
        # Runtime/FastBrain names the task string `instruction`; the actual
        # OpenWAM call goes through OfficialModelBoundary and uses `prompt`.
        return {**wire, "instruction": wire["prompt"]}

    def model_observation(self, runtime, projected):
        """A slow VLA subgoal replaces the prompt until the next escalation."""
        subgoal = getattr(runtime.method, "subgoal", None)
        return {**projected, "prompt": subgoal} if subgoal else projected

    def subgoal_observation(self, runtime, subgoal):
        runtime.method.subgoal = subgoal
        return self.model_observation(runtime, runtime.observation)

    def escalated(self, runtime):
        if self.hybrid:
            runtime.method.subgoal = None

    def project_state(self, state):
        return project_state(state)

    def validate_action(self, action, limits):
        if limits != {"native": "libero_osc7", "low": -1.0, "high": 1.0}:
            raise ValueError("LIBERO OSC7 frozen action limits mismatch")
        return validate_native_action(action)

    def motion_features(self, state, action):
        """Achieved EE position and the commanded gripper, binarized at 0 (native -1 open, +1 close; returned
        1 open, 0 closed as Escalation counts flips), so a VLA's jitter around one side is no reversal."""
        return [state[:3]], [0.0 if float(action[6]) > 0 else 1.0]

    def camera_payload(self, runtime):
        observation = runtime.observation
        if not self.hybrid:
            images = [{"camera": camera, "data_url": "data:image/png;base64," + value}
                      for camera, value in observation["images"].items() if value is not None]
            return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step,
                    "epoch": 0, "task_instruction": observation["prompt"],
                    "robot_state": {"achieved_eef10": observation["state"]},
                    "phase": runtime.method.phase, "target": runtime.method.target,
                    "images": images, "previous_receipt": runtime.previous_receipt}
        # Only the released client's own encoded views, task text, achieved state and
        # the policy's own commands, phase and receipt; never outcome, reward or layout.
        method = runtime.method
        payload = {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step,
                   "epoch": 0, "task_instruction": observation["prompt"],
                   "robot_state": {"achieved_eef10": list(observation["state"]),
                                   "gripper_command": [method.gripper_command]},
                   "phase": method.phase, "target": method.target,
                   "images": [{"camera": name, "data_url": "data:image/png;base64," + observation["images"][slot]}
                              for slot, name in SLOW_CAMERAS if observation["images"].get(slot)],
                   "previous_receipt": runtime.previous_receipt}
        if method.subgoal:
            payload["subgoal"] = method.subgoal
        return payload

    def ledger_payload(self, runtime, projected):
        return None  # distill-sim-ready has a distinct benchmark-neutral event


class LiberoBackend:
    """One live official PolicyServer prediction per admitted native OSC step.

    Every admitted row keeps the raw EEF10 the released client will convert:
    the model's own for VLA rows, a synthesized one for primitive rows.
    """

    def __init__(self, official_server, official_conversion):
        self.boundary = OfficialModelBoundary(official_server)
        self.official_server = official_server
        self.convert = official_conversion
        self.observation = None
        self.raw_action = None
        self.native_action = None
        self.issued = {}

    def take_serving_timing(self):
        """The shared OpenWAM session's time for the predicts since the last take, or None in process."""
        take = getattr(self.official_server, "take_timing", None)
        return take() if callable(take) else None

    def reset(self):
        self.boundary.reset()
        self.observation = None
        self.raw_action = None
        self.native_action = None
        self.issued = {}

    def update_obs(self, obs):
        self.observation = project_wire_observation(obs)

    def get_action(self):
        if self.observation is None:
            raise RuntimeError("LIBERO observation required before prediction")
        result = self.boundary.predict(self.observation)
        self.raw_action = validate_raw_action(result["action"])
        native = self.convert(np.asarray(self.raw_action, dtype=np.float32))
        self.native_action = validate_native_action(native)
        self.issued[digest(self.native_action)] = self.raw_action
        return [self.native_action]

    def chunk(self, rows):
        """OpenWAM's fresh executed chunk on the current observation (review mode), each row through the official
        conversion and recorded for ``raw_for`` like ``get_action``'s.

        The released server's sync executor generates a 32-step horizon when its buffer is empty, keeps the first
        ``inference_horizon`` (10) actions and pops one per predict, ignoring the observations of the calls that pop.
        A reset empties that buffer (and nothing else: the executor keeps no observation history), so ``rows`` predicts
        on this observation return exactly the chunk the direct condition would run open loop from it. The 22
        generated actions past the horizon never leave the server's public API, so they cannot be shown."""
        if self.observation is None:
            raise RuntimeError("LIBERO observation required before prediction")
        if type(rows) is not int or rows < 1:
            raise ValueError("a review candidate needs its frozen row count")
        self.boundary.reset()
        natives = []
        for _ in range(rows):
            raw = validate_raw_action(self.boundary.predict(self.observation)["action"])
            native = validate_native_action(self.convert(np.asarray(raw, dtype=np.float32)))
            self.issued[digest(native)] = raw
            natives.append(native)
        return natives

    def encode(self, row):
        """Admit a harness-made OSC row exactly as the released client will convert it."""
        intended = np.asarray(row, dtype=float)
        if intended.shape != (7,) or not np.isfinite(intended).all() or np.any(np.abs(intended) > 1):
            raise ValueError("primitive row must be finite 7D OSC within [-1, 1]")
        raw = validate_raw_action(raw_eef10(intended))
        native = validate_native_action(self.convert(np.asarray(raw, dtype=np.float32)))
        if np.max(np.abs(np.asarray(native) - intended)) > 1e-5:
            raise ValueError("official conversion does not reproduce the primitive row")
        self.issued[digest(native)] = raw
        return native

    def raw_for(self, row):
        raw = self.issued.get(digest(validate_native_action(row)))
        if raw is None:
            raise RuntimeError("admitted action differs from official OSC conversion")
        return raw


async def serve_step(runtime, backend, payload):
    """One released-client predict: observe, take the next admitted row, authorize it.

    The row is a fresh fast decision, a slow decision, or the next row of an
    admitted primitive chunk; the client receives the raw EEF10 it converts.
    """
    await runtime.update_obs(payload)
    row = await runtime.next_action()
    raw = backend.raw_for(row)
    ticket = await runtime.before_action(row)
    return {"type": "action", "action": raw, "oeh_ticket": ticket}

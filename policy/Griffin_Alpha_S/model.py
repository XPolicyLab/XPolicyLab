"""XPolicyLab adapter for Griffin Alpha-S (griffinlabs-ai/alpha-s), a LeRobot policy plugin.

The plugin registers two lerobot policy types selected by a checkpoint's ``config.json``:
``griffin_alpha`` (flow-matching action expert) and ``griffin_alpha_fast`` (FAST action tokens). Both
share the same prompt, image pipeline and action-space transform, so one adapter serves both: the
checkpoint decides.

Observation -> lerobot batch: ``observation.state`` is the XPolicyLab state packed by
``pack_robot_state`` (arm then end-effector, per arm, in the same order the official LeRobot
converter writes ``observation.state``); each configured camera becomes a float CHW image in [0, 1]
under the image key the checkpoint was trained with; the instruction becomes ``task``. The
checkpoint's own pre/post-processors (normalization, relative -> absolute actions, prompt building)
run unchanged. The predicted chunk is executed for ``n_action_steps`` steps, then the policy replans.

The policy server decodes camera colors before ``update_obs`` / ``update_obs_batch``, so this file
never decodes images.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_POLICY_DIR = Path(__file__).resolve().parent
# The importable root is the parent of the XPolicyLab checkout: the server imports
# ``XPolicyLab.policy.Griffin_Alpha_S.model``.
_REPO_ROOT = _POLICY_DIR.parents[2]
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

import lerobot_policy_griffin_alpha  # noqa: F401  registers griffin_alpha / griffin_alpha_fast
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot_policy_griffin_alpha import GriffinAlphaPolicy, reconnect_se3_steps

logger = logging.getLogger("Griffin_Alpha_S")

# XPolicyLab camera name -> the names it may appear under in an observation's ``vision`` dict.
CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "cam_head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
    "cam_wrist": ("cam_wrist", "wrist_camera", "wrist"),
    "cam_third_view": ("cam_third_view", "third_view", "side_camera"),
}
DEFAULT_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

_HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+$")


# ---- observation helpers -----------------------------------------------------------------------
def extract_image(observation: dict, camera: str) -> np.ndarray:
    """The decoded color image of ``camera`` (or one of its aliases) from ``observation["vision"]``."""
    vision = observation.get("vision", {})
    for name in CAMERA_ALIASES.get(camera, (camera,)):
        if name not in vision:
            continue
        entry = vision[name]
        if isinstance(entry, dict):
            for key in ("color", "rgb"):
                if key in entry:
                    return entry[key]
            raise KeyError(f"camera {name!r} has no 'color' entry (keys: {list(entry)})")
        return entry
    raise KeyError(f"camera {camera!r} not found in observation (available: {list(vision)})")


def image_to_tensor(image: Any) -> torch.Tensor:
    """(H, W, 3) or (3, H, W) uint8/float RGB array -> float32 (3, H, W) in [0, 1].

    Decoded observation arrays can be read-only views, so the array is copied before conversion.
    """
    array = np.array(image, copy=True)
    if array.ndim != 3:
        raise ValueError(f"expected an image with 3 dims, got shape {array.shape}")
    if array.shape[-1] in (1, 3) and array.shape[0] not in (1, 3):
        array = np.transpose(array, (2, 0, 1))
    elif array.shape[-1] in (1, 3) and array.shape[0] in (1, 3):
        # Ambiguous tiny image: XPolicyLab observations are HWC, so treat it as HWC.
        array = np.transpose(array, (2, 0, 1))
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    if array.shape[0] != 3:
        raise ValueError(f"expected an RGB image, got shape {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float().clamp_(0.0, 1.0)


def _first_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_string(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def resolve_instruction(observation: dict, fallback: str | None) -> str:
    for key in ("instruction", "instructions", "prompt", "task", "language_instruction"):
        text = _first_string(observation.get(key))
        if text:
            return text
    if fallback:
        return fallback
    raise ValueError("observation carries no instruction and deploy.yml sets no `prompt` fallback")


# ---- checkpoint helpers --------------------------------------------------------------------------
def _is_policy_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def _step_number(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def find_policy_dir(root: Path, checkpoint_num: Any = None) -> Path | None:
    """Locate a lerobot policy directory under ``root``.

    Accepts the directory itself, ``<root>/pretrained_model`` and a lerobot-train run directory
    (``<root>/checkpoints/<step>/pretrained_model``, with ``last`` preferred unless ``checkpoint_num``
    names a step).
    """
    if not root.is_dir():
        return None
    if _is_policy_dir(root):
        return root
    if _is_policy_dir(root / "pretrained_model"):
        return root / "pretrained_model"

    steps_dir = root / "checkpoints" if (root / "checkpoints").is_dir() else root
    candidates = [child for child in steps_dir.iterdir() if child.is_dir()]
    if not candidates:
        return None

    def as_policy(child: Path) -> Path | None:
        for candidate in (child / "pretrained_model", child):
            if _is_policy_dir(candidate):
                return candidate
        return None

    wanted = _step_number(str(checkpoint_num)) if checkpoint_num not in (None, "", "last") else None
    if wanted is not None:
        for child in candidates:
            if _step_number(child.name) == wanted and as_policy(child):
                return as_policy(child)
        raise FileNotFoundError(f"no checkpoint step {wanted} under {steps_dir} (have: {sorted(c.name for c in candidates)})")

    if (steps_dir / "last").is_dir() and as_policy(steps_dir / "last"):
        return as_policy(steps_dir / "last")
    numbered = [c for c in candidates if _step_number(c.name) is not None and as_policy(c)]
    if numbered:
        return as_policy(max(numbered, key=lambda c: _step_number(c.name)))
    return None


def resolve_pretrained(model_cfg: dict[str, Any]) -> tuple[str, str | None]:
    """``(pretrained_path, revision)`` for ``from_pretrained``.

    Local directories win, following the shared precedence in ``XPolicyLab.utils.checkpoint_resolver``
    (``pretrained_path`` / ``model_path`` / ``checkpoint_path``, then ``ckpt_name`` as a path, then
    ``checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>``, then
    ``checkpoints/<ckpt_name>``). When nothing exists locally and ``pretrained_path`` or ``ckpt_name``
    looks like a Hugging Face repo id, that id is returned with ``revision`` from the config.
    """
    explicit_keys = ("pretrained_path", "model_path", "checkpoint_path")
    candidates = candidate_checkpoint_roots(
        model_cfg, _CHECKPOINTS_DIR, policy_dir=_POLICY_DIR, explicit_keys=explicit_keys
    )
    checkpoint_num = model_cfg.get("checkpoint_num")
    for root in candidates:
        found = find_policy_dir(root, checkpoint_num)
        if found is not None:
            return str(found), None

    revision = model_cfg.get("revision") or None
    for key in (*explicit_keys, "ckpt_name"):
        value = model_cfg.get(key)
        if value and _HUB_ID_RE.match(str(value).strip()) and not Path(str(value)).exists():
            return str(value).strip(), revision

    checked = "\n  ".join(str(path) for path in candidates) or "  <none>"
    raise FileNotFoundError(
        "Could not resolve a Griffin Alpha-S checkpoint. Pass ckpt_name as a run name or a path, set "
        "`pretrained_path` in deploy.yml to a lerobot policy directory or a Hugging Face repo id "
        f"(e.g. griffinlabs/Griffin-Alpha-S-LIBERO).\nChecked:\n  {checked}"
    )


# ---- the adapter -----------------------------------------------------------------------------------
class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name")
        self.action_type = self.model_cfg.get("action_type") or "joint"
        if self.action_type != "joint":
            # XPolicyLab's pack_robot_state validates `*_ee_pose` against the arm dimension, so the
            # shared state packing cannot carry 7-D poses yet; the model itself is control-mode agnostic.
            raise ValueError("Griffin_Alpha_S in XPolicyLab currently supports only action_type='joint'.")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("env_cfg_type is required to pack observations and unpack actions")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        self.default_prompt = _first_string(self.model_cfg.get("prompt")) or self.task_name

        self.device = self._resolve_device(self.model_cfg.get("device", "cuda"))
        self.pretrained_path, self.revision = resolve_pretrained(self.model_cfg)
        self.policy = self._load_policy()
        self.model = self.policy
        self.preprocessor, self.postprocessor = self._build_processors()

        self.cameras = tuple(self.model_cfg.get("cameras") or DEFAULT_CAMERAS)
        self.image_keys = self._resolve_image_keys()
        self.n_action_steps = int(self.model_cfg.get("n_action_steps") or self.policy.config.n_action_steps)
        self.n_action_steps = max(1, min(self.n_action_steps, int(self.policy.config.horizon)))
        self.chunk_kwargs = self._resolve_chunk_kwargs()

        self._payloads: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list: list[int] = [0]
        self._warned_width = False

        head = type(self.policy.config).get_choice_name(type(self.policy.config))
        print(
            f"[Griffin_Alpha_S] loaded {head} from {self.pretrained_path}"
            f"{'@' + self.revision if self.revision else ''} on {self.device}; cameras={self.cameras} -> "
            f"{self.image_keys}; n_action_steps={self.n_action_steps}; action_dim={self.action_dim} "
            f"({self.action_type}, {self.env_cfg_type})"
        )

    # -- construction -------------------------------------------------------------------------------
    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; running on CPU")
            return torch.device("cpu")
        return device

    def _hub_kwargs(self) -> dict[str, Any]:
        return {"revision": self.revision} if self.revision else {}

    def _load_policy(self):
        config = PreTrainedConfig.from_pretrained(self.pretrained_path, **self._hub_kwargs())
        config.device = str(self.device)
        policy_cls = get_policy_class(type(config).get_choice_name(type(config)))
        policy = policy_cls.from_pretrained(self.pretrained_path, config=config, **self._hub_kwargs())
        policy.to(self.device)
        policy.eval()
        return policy

    def _build_processors(self):
        pre, post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.pretrained_path,
            pretrained_revision=self.revision,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        # lerobot re-pairs its own relative/absolute steps after loading; the plugin's optional SE(3)
        # pair needs this call (a no-op for checkpoints that do not use it).
        reconnect_se3_steps(pre, post)
        return pre, post

    def _resolve_image_keys(self) -> tuple[str, ...]:
        override = self.model_cfg.get("image_keys")
        keys = tuple(override) if override else tuple(self.policy.config.resolved_image_keys)
        if not keys:
            # A base checkpoint with an open camera list: name the keys after the cameras.
            keys = tuple(f"{OBS_IMAGES}.{camera}" for camera in self.cameras)
        if len(keys) != len(self.cameras):
            raise ValueError(
                f"deploy.yml lists {len(self.cameras)} cameras {self.cameras} but the checkpoint expects "
                f"{len(keys)} images {keys}; set `cameras` (in prompt order) to match"
            )
        return keys

    def _resolve_chunk_kwargs(self) -> dict[str, Any]:
        num_steps = self.model_cfg.get("num_inference_steps")
        if num_steps and isinstance(self.policy, GriffinAlphaPolicy):
            return {"num_steps": int(num_steps)}
        if num_steps:
            logger.warning("num_inference_steps only applies to the flow head; ignored for %s", type(self.policy).__name__)
        return {}

    # -- observations -----------------------------------------------------------------------------------
    def _encode_obs(self, obs: dict) -> dict[str, Any]:
        state = pack_robot_state(obs, self.action_type, self.robot_action_dim_info, source_type="obs")
        images = [image_to_tensor(extract_image(obs, camera)) for camera in self.cameras]
        return {
            "state": torch.as_tensor(np.asarray(state, dtype=np.float32)),
            "images": images,
            "task": resolve_instruction(obs, self.default_prompt),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        self._payloads = {
            env_idx: self._encode_obs(obs) for env_idx, obs in zip(self._latest_env_idx_list, obs_list)
        }

    def _to_lerobot_batch(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            OBS_STATE: torch.stack([payload["state"] for payload in payloads], dim=0),
            "task": [payload["task"] for payload in payloads],
        }
        for index, key in enumerate(self.image_keys):
            batch[key] = torch.stack([payload["images"][index] for payload in payloads], dim=0)
        return batch

    # -- actions --------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def _predict(self, payloads: list[dict[str, Any]]) -> np.ndarray:
        """``(B, n_action_steps, action_dim)`` environment actions for the given observations."""
        batch = self.preprocessor(self._to_lerobot_batch(payloads))
        chunk = self.policy.predict_action_chunk(batch, **self.chunk_kwargs)  # (B, horizon, dim)
        chunk = chunk[:, : self.n_action_steps]
        actions = self.postprocessor(chunk)  # unnormalize + relative -> absolute, against the cached state
        actions = actions.detach().to("cpu", torch.float32).numpy()
        if actions.ndim == 2:
            actions = actions[None]

        width = actions.shape[-1]
        if width < self.action_dim:
            raise ValueError(
                f"checkpoint predicts {width}-D actions but {self.env_cfg_type} needs {self.action_dim} "
                f"({self.action_type}); fine-tune on this robot first (see README)"
            )
        if width > self.action_dim:
            if not self._warned_width:
                self._warned_width = True
                logger.warning(
                    "checkpoint predicts %d-D actions, keeping the leading %d for %s",
                    width, self.action_dim, self.env_cfg_type,
                )
            actions = actions[..., : self.action_dim]
        return actions

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if not self._payloads:
            raise AssertionError("update_obs / update_obs_batch must be called before get_action")
        env_idx_list = self._latest_env_idx_list if env_idx_list is None else [int(i) for i in env_idx_list]
        missing = [env_idx for env_idx in env_idx_list if env_idx not in self._payloads]
        if missing:
            raise KeyError(f"no observation for env_idx {missing}; have {sorted(self._payloads)}")

        actions = self._predict([self._payloads[env_idx] for env_idx in env_idx_list])
        return [
            unpack_robot_state(per_env, self.action_type, self.robot_action_dim_info, source_type="obs")
            for per_env in actions
        ]

    def reset(self):
        self.policy.reset()
        self._payloads = {}
        self._latest_env_idx_list = [0]

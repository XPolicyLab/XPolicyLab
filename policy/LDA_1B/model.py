from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate


CURRENT_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = CURRENT_DIR / "LDA-1B"

if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))


def _require_path(path: str | None, name: str) -> str:
    if not path:
        raise ValueError(f"{name} is required for LDA_1B deployment.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _process_data_helpers():
    from XPolicyLab.utils.process_data import (
        get_robot_action_dim_info,
        pack_robot_state,
        unpack_robot_state,
    )

    return get_robot_action_dim_info, pack_robot_state, unpack_robot_state


def _extract_camera(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    try:
        camera_obs = observation["vision"][camera_name]
    except KeyError as exc:
        raise KeyError(f"Missing observation['vision']['{camera_name}'].") from exc

    if isinstance(camera_obs, dict):
        if "color" not in camera_obs:
            raise KeyError(f"Missing observation['vision']['{camera_name}']['color'].")
        camera_obs = camera_obs["color"]

    return _standardize_rgb_image(camera_obs)


# Match gr00t_lerobot/datasets.py:IMG_MEAN — used for the square padding color so
# the padded border lands on the ImageNet mean tone the DINOv3/QwenVL backbones
# were pretrained on (otherwise zero/black borders themselves are OOD).
_GR00T_IMG_MEAN_U8 = (
    int(0.485 * 255),
    int(0.456 * 255),
    int(0.406 * 255),
)
# Final square side fed to the model. Must match the size the dataloader resizes
# to AFTER expand2square (gr00t_lerobot/datasets.py: image.resize((224, 224))).
_MODEL_INPUT_SIZE = 224
# Native simulator/dataset frame size before square-padding. Cosmetic only — we
# resize-then-pad, so any landscape resolution would work, but matching the
# parquet/MP4 size (process_data.py: 240x320) keeps the resize step a no-op for
# pixels coming straight out of the simulator.
_NATIVE_HEIGHT = 240
_NATIVE_WIDTH = 320


def _expand2square_uint8(image: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    """Center-pad an HWC uint8 image to a square with `background` color.

    Mirrors `lda.dataloader.gr00t_lerobot.datasets.expand2square` (which uses
    PIL.Image.new + paste). Implemented in numpy to avoid an extra PIL round-trip.
    """
    h, w = image.shape[:2]
    if h == w:
        return image
    side = max(h, w)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[..., 0] = background[0]
    canvas[..., 1] = background[1]
    canvas[..., 2] = background[2]
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top : top + h, left : left + w, :] = image
    return canvas


def _standardize_rgb_image(image: Any) -> np.ndarray:
    """Reproduce the training-time visual preprocessing exactly.

    Training pipeline (see `gr00t_lerobot/datasets.py:962-966` /
    `:2122-2125`) for every video frame is:

        frame_240x320 -> Image.fromarray
                      -> expand2square(mean_color)   # 320x320 with mean borders
                      -> resize((224, 224))           # final input to model

    The previous implementation skipped expand2square and only resized to
    (240, 320). The downstream DINOv3/QwenVL processors then resized to
    (224, 224) by *stretching* the aspect ratio, which is a different image
    distribution than training and pushed the diffusion head into an OOD
    regime where it collapses to ~mean output (visible as the arm "twitching
    in place"). We now match training byte-for-byte: square-pad with the
    ImageNet mean color, then bilinear-resize to 224x224.
    """
    import cv2
    from PIL import Image

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape {image.shape}.")

    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Step 1: align to the native simulator/dataset size (cv2 resize is (W, H)).
    image = cv2.resize(image, (_NATIVE_WIDTH, _NATIVE_HEIGHT), interpolation=cv2.INTER_AREA)

    # Step 2: square-pad with mean color (PIL Image.new + paste-equivalent).
    image = _expand2square_uint8(image, _GR00T_IMG_MEAN_U8)

    # Step 3: resize to the model's expected input. The dataloader resizes with
    # PIL's Image.resize (default BICUBIC); cv2.INTER_CUBIC uses a different cubic
    # kernel and drifts by a few LSBs at edges, so use PIL here to stay byte-exact
    # with training (gr00t_lerobot/datasets.py: image.resize((224, 224))).
    image = np.asarray(Image.fromarray(image).resize((_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE)))
    if image.shape != (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE, 3):
        raise ValueError(
            f"Expected RGB image shape ({_MODEL_INPUT_SIZE}, {_MODEL_INPUT_SIZE}, 3), got {image.shape}."
        )
    return image


def _normalize_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any] | None) -> np.ndarray:
    """Map model-space normalized actions in [-1, 1] back to robot-space actions.

    The XPolicyLab arx_x5 path trains with `q99` mode in
    `lda.dataloader.gr00t_lerobot.transform.state_action.Normalizer` (see
    `ArxX5DataConfig.transform`):

        forward (training):
            norm = clamp(2*(x - q01)/(q99 - q01) - 1, -1, 1)   if q01 != q99
            norm = x (passthrough)                              if q01 == q99
        inverse (deployment, this function):
            x = (norm + 1)/2 * (q99 - q01) + q01                if q01 != q99
            x = norm (passthrough)                              if q01 == q99

    Earlier this function preferred `min`/`max` (the inverse for `min_max` mode,
    used by Robocasa's PolicyWarper) which is wrong for arx_x5 — the min/max
    range is ~2x wider than q01/q99, so unnormalized arm joints over-extended
    by up to ~70 degrees and grippers came out outside [0, 1]. We now prefer
    q01/q99 to match arx_x5 training, falling back to min/max only when q01/q99
    are unavailable (e.g. checkpoints from min_max-mode configs).

    The `mask` field that LDA writes into dataset_statistics.json is generated
    from key names ("gripper" -> False) by `generate_action_mask_for_used_keys`,
    NOT from the per-element `q01 != q99` check the Normalizer actually uses at
    training time. For arx_x5 the gripper has q01=0, q99=1, so it IS q99-
    normalized at training and must be inverted here (norm in [-1, 1] -> raw
    in [0, 1]). We therefore IGNORE the saved mask and replicate the
    Normalizer's internal `mask = q01 != q99` so the inverse exactly mirrors
    the training-time forward.
    """
    if action_stats is None:
        return normalized_actions

    if "q01" in action_stats and "q99" in action_stats:
        low = np.asarray(action_stats["q01"], dtype=np.float64)
        high = np.asarray(action_stats["q99"], dtype=np.float64)
    elif "min" in action_stats and "max" in action_stats:
        low = np.asarray(action_stats["min"], dtype=np.float64)
        high = np.asarray(action_stats["max"], dtype=np.float64)
    else:
        return normalized_actions

    clipped = np.clip(normalized_actions, -1.0, 1.0).astype(np.float64)
    inv_mask = high != low
    out = np.where(inv_mask, 0.5 * (clipped + 1.0) * (high - low) + low, clipped)
    return out.astype(normalized_actions.dtype, copy=False)


def _normalize_state(state: np.ndarray, state_stats: dict[str, Any] | None) -> np.ndarray:
    """Apply the same forward normalization used by LDA's StateActionTransform."""
    if state_stats is None:
        return state

    state_f64 = state.astype(np.float64, copy=False)
    if "q01" in state_stats and "q99" in state_stats:
        low = np.asarray(state_stats["q01"], dtype=np.float64)
        high = np.asarray(state_stats["q99"], dtype=np.float64)
        if low.shape[-1] != state_f64.shape[-1] or high.shape[-1] != state_f64.shape[-1]:
            return state
        mask = high != low
        out = np.zeros_like(state_f64)
        out[..., mask] = 2.0 * (state_f64[..., mask] - low[..., mask]) / (high[..., mask] - low[..., mask]) - 1.0
        out[..., ~mask] = state_f64[..., ~mask]
        out = np.clip(out, -1.0, 1.0)
    elif "mean" in state_stats and "std" in state_stats:
        mean = np.asarray(state_stats["mean"], dtype=np.float64)
        std = np.asarray(state_stats["std"], dtype=np.float64)
        if mean.shape[-1] != state_f64.shape[-1] or std.shape[-1] != state_f64.shape[-1]:
            return state
        mask = std != 0
        out = np.zeros_like(state_f64)
        out[..., mask] = (state_f64[..., mask] - mean[..., mask]) / std[..., mask]
        out[..., ~mask] = state_f64[..., ~mask]
    elif "min" in state_stats and "max" in state_stats:
        low = np.asarray(state_stats["min"], dtype=np.float64)
        high = np.asarray(state_stats["max"], dtype=np.float64)
        if low.shape[-1] != state_f64.shape[-1] or high.shape[-1] != state_f64.shape[-1]:
            return state
        mask = high != low
        out = np.zeros_like(state_f64)
        out[..., mask] = 2.0 * (state_f64[..., mask] - low[..., mask]) / (high[..., mask] - low[..., mask]) - 1.0
    else:
        return state

    return out.astype(state.dtype, copy=False)


def _get_instruction(obs: dict[str, Any], fallback: str) -> str:
    """Resolve the language prompt from a RoboDojo v1.0 observation.

    Reference obs layout (``/mnt/xspark-data/zijian/test.pkl``):
        {
            "vision": {"cam_head": ..., "cam_left_wrist": ..., ...},
            "state": {...},
            "instruction": "Pick up two slices of bread, ...",  # full English
            "env_idx": 0,
            ...
        }

    Prefer the env-provided ``instruction`` string over the deploy ``task_name``
    slug fallback so language matches training ``tasks.jsonl`` entries.
    """
    value = obs.get("instruction")
    if value is None:
        value = obs.get("task_instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text else fallback


def _lookup_config(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    cursor = config
    for key in path:
        if isinstance(cursor, dict):
            cursor = cursor.get(key, default)
        else:
            cursor = getattr(cursor, key, default)
        if cursor is default:
            return default
    return cursor


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg["action_type"]
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        get_robot_action_dim_info, _, _ = _process_data_helpers()
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.expected_action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        # Older arx_x5 checkpoints were trained before the dataloader stopped
        # padding 6-D arm keys to 7-D. Keep the layout only for compatibility
        # with those 16-D checkpoints; new arx_x5 checkpoints emit raw 14-D actions.
        self.action_key_layout = self._build_action_key_layout()

        self.camera_names = [
            "cam_head",
            "cam_left_wrist",
            "cam_right_wrist",
        ]
        self.task_instruction = (
            self.model_cfg.get("prompt")
            or self.model_cfg.get("task_name")
            or "follow the robot instruction"
        )
        self.device = self.model_cfg.get("device", "cuda")

        self.model = self.get_model(self.model_cfg)
        self.robot_type = (
            _lookup_config(self.model.config, ("datasets", "vla_data", "robot_type"))
            or self.env_cfg_type
        )
        self.embodiment_id = self._resolve_embodiment_id()
        self.state_encoding = self._resolve_state_encoding()

        # Pull the shape contract from the loaded checkpoint's config so encode_obs
        # matches what the checkpoint was actually trained on (e.g. num_views=1 means
        # send only one camera per example; state_dim=None means do not pack state).
        action_model_cfg = self.model.config.framework.action_model
        self.num_views = int(getattr(action_model_cfg, "num_views", 1))
        self.model_state_dim = getattr(action_model_cfg, "state_dim", None)
        self.model_state_dim = int(self.model_state_dim) if self.model_state_dim is not None else None
        if self.num_views > len(self.camera_names):
            raise ValueError(
                f"LDA model expects num_views={self.num_views} cameras, but only "
                f"{len(self.camera_names)} were configured in camera_names={self.camera_names}."
            )
        # Truncate to the first num_views cameras (cam_head first by default). The
        # remaining cameras are intentionally dropped to honor the trained model
        # contract; passing extras would be silently reinterpreted as time steps by
        # the upstream `predict_action` rearrange and produce shape-mismatch errors.
        self.active_camera_names = self.camera_names[: self.num_views]
        self.use_state = self.model_state_dim is not None

        # === Observation horizon contract ===========================================
        # The trained model stores its visual history length as `obs_horizon`. The
        # `MMDiT_ActionHeader.obs_merger` linear is shaped (hidden, num_chans *
        # (obs_horizon + 1)), so a deploy frame count != obs_horizon will silently
        # send the wrong number of channels into obs_merger and the model collapses.
        # We mirror the dataloader's `observation_indices` behavior here: maintain
        # a per-env per-view rolling image buffer and pull frames at the same
        # relative offsets the trainer used. ArxX5DataConfig uses `[-5, 0]` for
        # obs_horizon=2. The offsets themselves come from the checkpoint robot's
        # DataConfig, not from deploy-time overrides.
        self.obs_horizon = int(getattr(action_model_cfg, "obs_horizon", 1))
        if self.obs_horizon == 1:
            self.obs_indices: list[int] = [0]
        else:
            self.obs_indices = self._resolve_observation_indices()
        if len(self.obs_indices) != self.obs_horizon:
            raise ValueError(
                f"obs_indices {self.obs_indices} length must match obs_horizon={self.obs_horizon}. "
                "Check the checkpoint config and robot DataConfig."
            )
        if self.obs_indices[-1] != 0:
            raise ValueError(
                f"obs_indices {self.obs_indices} must end with 0 (the current frame); "
                "history offsets are negative."
            )
        self._obs_history_keep = abs(min(self.obs_indices)) + 1
        # (env_idx, camera_name) -> list[np.ndarray HWC uint8] in time order, oldest first
        self._image_history: dict[tuple[int, str], list[np.ndarray]] = {}
        # env_idx -> list[np.ndarray raw state] in time order, oldest first
        self._state_history: dict[int, list[np.ndarray]] = {}

        self.action_stats = self._get_action_stats()
        self.state_stats = self._get_state_stats()
        self._last_example = None
        self._last_examples = None

    def get_model(self, model_cfg):
        checkpoint_path = _require_path(model_cfg.get("checkpoint_path"), "checkpoint_path")

        # Load exactly the checkpoint-side config.yaml and dataset_statistics.json.
        # Deployment config must not override model shape, backbone paths, or
        # preprocessing behavior.
        from pathlib import Path

        import torch as _torch
        from lda.model.framework import build_framework
        from lda.model.framework.share_tools import dict_to_namespace, read_mode_config

        config_dict, norm_stats = read_mode_config(Path(checkpoint_path))

        config_ns = dict_to_namespace(config_dict)
        config_ns.trainer.pretrained_checkpoint = None

        policy = build_framework(cfg=config_ns)
        policy.norm_stats = norm_stats
        state_dict = _torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(state_dict, strict=True)

        policy.eval()
        policy.to(self.device)
        return policy

    def _resolve_state_encoding(self) -> str:
        from lda.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
        from lda.dataloader.gr00t_lerobot.transform.state_action import StateActionSinCosTransform

        data_config = ROBOT_TYPE_CONFIG_MAP.get(self.robot_type)
        if data_config is None:
            return "raw"

        transforms = getattr(data_config.transform(), "transforms", [])
        state_keys = set(getattr(data_config, "state_keys", []))
        for transform in transforms:
            if not isinstance(transform, StateActionSinCosTransform):
                continue
            apply_to = set(getattr(transform, "apply_to", []))
            if not apply_to or apply_to & state_keys:
                return "sin_cos"
        return "raw"

    def _resolve_observation_indices(self) -> list[int]:
        from lda.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP

        data_config = ROBOT_TYPE_CONFIG_MAP.get(self.robot_type)
        if data_config is None:
            return [0] * self.obs_horizon
        return [int(idx) for idx in getattr(data_config, "observation_indices", [0] * self.obs_horizon)]

    def _resolve_embodiment_id(self) -> int:
        from lda.dataloader.gr00t_lerobot.embodiment_tags import (
            EMBODIMENT_TAG_MAPPING,
            ROBOT_TYPE_TO_EMBODIMENT_TAG,
        )

        tag = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(self.robot_type)
        if tag is None:
            return int(EMBODIMENT_TAG_MAPPING["new_embodiment"])
        return int(EMBODIMENT_TAG_MAPPING[tag.value])

    def _build_action_key_layout(self):
        """Return [(action_key, raw_width, padded_width), ...] in gr00t concat order.

        Reads `cfg.action_keys` from upstream `ROBOT_TYPE_CONFIG_MAP` and pairs each
        key with:
          - raw_width:    the robot's physical sub-dim (from `robot_action_dim_info`)
          - padded_width: the per-key pad width that the gr00t loader applied at
                          training time (via `pad_action_state_with_key`).

        Used by `_unpad_actions` to drop the padded slots from the model output
        before unnormalization / `unpack_robot_state`.
        """
        from lda.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
        from lda.dataloader.gr00t_lerobot.datasets import pad_action_state_with_key

        action_keys = list(ROBOT_TYPE_CONFIG_MAP[self.env_cfg_type].action_keys)

        arm_dims = list(self.robot_action_dim_info["arm_dim"])
        ee_dims = list(self.robot_action_dim_info["ee_dim"])
        num_arms = len(arm_dims)

        if num_arms == 1:
            arm_prefixes = [""]
        elif num_arms == 2:
            arm_prefixes = ["left_", "right_"]
        else:
            raise ValueError(f"Unsupported arm count: {num_arms}")

        def raw_width_for(key: str) -> int:
            suffix = key.split(".", 1)[-1]
            for i, prefix in enumerate(arm_prefixes):
                if suffix == f"{prefix}arm" or (prefix == "" and suffix == "arm"):
                    return arm_dims[i]
                if suffix.startswith(f"{prefix}gripper") or suffix.startswith(f"{prefix}ee"):
                    return ee_dims[i]
            raise ValueError(
                f"Cannot map action_key '{key}' to robot_action_dim_info for "
                f"env_cfg_type={self.env_cfg_type!r} (arm_prefixes={arm_prefixes})."
            )

        layout = []
        for key in action_keys:
            raw = raw_width_for(key)
            padded = int(pad_action_state_with_key(np.zeros((1, 1)), key)[0].shape[1])
            if raw > padded:
                raise ValueError(
                    f"action_key '{key}' raw_width={raw} > padded_width={padded}."
                )
            layout.append((key, raw, padded))
        return layout

    def _unpad_actions(self, padded_actions: np.ndarray) -> np.ndarray:
        """Drop gr00t per-key padding columns from a [..., padded_total] action tensor.

        e.g. arx_x5: keys=[left_arm(6/7), left_gripper(1/1), right_arm(6/7), right_gripper(1/1)]
             -> drops the trailing pad slot in each arm chunk
             -> output last-dim is sum(raw_width) = 14.
        """
        slices = []
        offset = 0
        for _key, raw, padded in self.action_key_layout:
            slices.append(padded_actions[..., offset : offset + raw])
            offset += padded
        if offset != padded_actions.shape[-1]:
            raise ValueError(
                f"action_key_layout total padded_width={offset} does not match "
                f"model output last-dim {padded_actions.shape[-1]}."
            )
        return np.concatenate(slices, axis=-1)

    def _get_action_stats(self):
        unnorm_key = self.model_cfg.get("unnorm_key")
        if not hasattr(self.model, "norm_stats"):
            return None

        if unnorm_key is None:
            if len(self.model.norm_stats) != 1:
                raise ValueError(
                    "unnorm_key is required because the LDA checkpoint contains "
                    f"multiple normalization keys: {list(self.model.norm_stats.keys())}"
                )
            unnorm_key = next(iter(self.model.norm_stats.keys()))

        if unnorm_key not in self.model.norm_stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not found in checkpoint stats.")
        return self.model.norm_stats[unnorm_key]["action"]

    def _get_state_stats(self):
        unnorm_key = self.model_cfg.get("unnorm_key")
        if not hasattr(self.model, "norm_stats"):
            return None

        if unnorm_key is None:
            if len(self.model.norm_stats) != 1:
                raise ValueError(
                    "unnorm_key is required because the LDA checkpoint contains "
                    f"multiple normalization keys: {list(self.model.norm_stats.keys())}"
                )
            unnorm_key = next(iter(self.model.norm_stats.keys()))

        if unnorm_key not in self.model.norm_stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not found in checkpoint stats.")
        return self.model.norm_stats[unnorm_key].get("state")

    def update_obs(self, obs):
        self._last_example = self.encode_obs(obs)
        self._last_examples = [self._last_example]

    def update_obs_batch(self, obs_list):
        self._last_examples = [self.encode_obs(obs) for obs in obs_list]
        self._last_example = self._last_examples[0] if self._last_examples else None

    def encode_obs(self, observation):
        env_idx = int(observation.get("env_idx", 0))
        # Build the flat (V*T) image list in the order `predict_action` expects:
        # `rearrange(curr_imgs, "b (v t) c h w -> b v t c h w", v=num_views)` consumes
        # frames in (view-major, time-minor) order, i.e.
        # [v0_t0, v0_t1, ..., v0_t(T-1), v1_t0, ..., v(V-1)_t(T-1)]. T==obs_horizon.
        flat_images: list[np.ndarray] = []
        for cam_name in self.active_camera_names:
            buf = self._image_history.setdefault((env_idx, cam_name), [])
            buf.append(_extract_camera(observation, cam_name))
            # Bound the rolling buffer; older frames beyond what obs_indices needs are dropped.
            if len(buf) > self._obs_history_keep:
                del buf[: len(buf) - self._obs_history_keep]
            cur_idx = len(buf) - 1
            for offset in self.obs_indices:
                # Clamp negative offsets to the first available frame, mirroring the
                # gr00t dataloader's behavior at episode start when t + offset < 0.
                target = cur_idx + offset
                if target < 0:
                    target = 0
                flat_images.append(buf[target])

        example = {
            "image": flat_images,
            "lang": _get_instruction(observation, self.task_instruction),
            "embodiment_id": self.embodiment_id,
        }

        if self.use_state:
            _, pack_robot_state, _ = _process_data_helpers()
            raw_state = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)

            state_buf = self._state_history.setdefault(env_idx, [])
            state_buf.append(raw_state)
            if len(state_buf) > self._obs_history_keep:
                del state_buf[: len(state_buf) - self._obs_history_keep]
            cur_idx = len(state_buf) - 1

            state_frames = []
            for offset in self.obs_indices:
                target = cur_idx + offset
                if target < 0:
                    target = 0
                state_frames.append(state_buf[target])
            state_sequence = np.stack(state_frames, axis=0).astype(np.float32)

            if self.state_encoding == "sin_cos":
                state_sequence = np.concatenate([np.sin(state_sequence), np.cos(state_sequence)], axis=-1).astype(np.float32)
            elif self.state_encoding != "raw":
                raise ValueError("state_encoding must be either 'raw' or 'sin_cos'.")

            state_sequence = _normalize_state(state_sequence, self.state_stats)
            flat_state = state_sequence.reshape(-1)
            if self.model_state_dim == state_sequence.shape[-1]:
                state = state_sequence
            elif self.model_state_dim == flat_state.shape[-1]:
                state = flat_state[None, :]
            else:
                raise ValueError(
                    "LDA state dimension mismatch: encoded current state has "
                    f"{state_sequence.shape[-1]} dims and obs-history state has "
                    f"{flat_state.shape[-1]} dims, but checkpoint expects "
                    f"state_dim={self.model_state_dim}."
                )
            example["state"] = state

        return example

    def _predict(self, examples):
        if not examples:
            raise RuntimeError("No observation has been provided. Call update_obs() first.")

        output = self.model.predict_action(examples=examples)
        normalized = output.get("normalized_actions") if isinstance(output, dict) else output
        normalized = np.asarray(normalized)

        if normalized.shape[-1] == self.expected_action_dim:
            normalized_robot = normalized
        else:
            # Compatibility path for older checkpoints trained with padded arm keys.
            normalized_robot = self._unpad_actions(normalized)
        actions = _normalize_actions(normalized_robot, self.action_stats)
        if actions.shape[-1] != self.expected_action_dim:
            raise ValueError(
                "LDA action dimension mismatch: after unpadding got "
                f"{actions.shape[-1]}, but env/action_type expects {self.expected_action_dim} "
                f"(env_cfg_type={self.env_cfg_type!r}, action_type={self.action_type!r})."
            )
        return actions

    def get_action(self):
        actions = self._predict([self._last_example])[0]
        _, _, unpack_robot_state = _process_data_helpers()
        return unpack_robot_state(
            actions,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )

    def get_action_batch(self, env_idx_list):
        if self._last_examples is None:
            raise RuntimeError("No batch observation has been provided. Call update_obs_batch() first.")

        actions = self._predict(self._last_examples)
        _, _, unpack_robot_state = _process_data_helpers()
        return [
            unpack_robot_state(env_actions, self.action_type, self.robot_action_dim_info, source_type="obs")
            for env_actions in actions
        ]

    def reset(self):
        self._last_example = None
        self._last_examples = None
        # Drop the per-env image history so the next episode does not leak frames
        # or states from the previous one through obs_indices=[-5, 0].
        self._image_history.clear()
        self._state_history.clear()

"""Minimal local inference policy for Focus-VLWA."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from focus_vlwa.configs.model import FocusVLWAConfig, load_model_config
from focus_vlwa.data import head_history
from focus_vlwa.data.dataset import _image, _prompt
from focus_vlwa.data.normalization import load_norm_stats, normalize_quantile, unnormalize_quantile
from focus_vlwa.inference.tokenizer import FocusVLWATokenizer
from focus_vlwa.model.checkpoint import load_checkpoint_weights
from focus_vlwa.model.focus_vlwa import FocusVLWA
from focus_vlwa.model.observation import FocusVLWAObservation

_CAMERA_MAPPING = {
    "base_0_rgb": "cam_high",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}
_DELTA_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False])


def _pad_last_dim(array: np.ndarray, dimension: int) -> np.ndarray:
    if array.shape[-1] >= dimension:
        return array[..., :dimension]
    return np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, dimension - array.shape[-1])])


class FocusVLWAPolicy:
    """Load a released checkpoint and predict a fifty-step dual-arm action chunk."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str | None = None,
        tokenizer_path: str | Path | None = None,
        config: FocusVLWAConfig | None = None,
        asset_id: str = "arx_x5_sim",
        num_steps: int = 10,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or load_model_config(self.checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_steps = num_steps
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.tokenizer = FocusVLWATokenizer(self.config.max_token_len, tokenizer_path)
        self.norm_stats = load_norm_stats(self.checkpoint_dir, asset_id)
        self.model = FocusVLWA(self.config)
        load_checkpoint_weights(self.model, self.checkpoint_dir)
        # The reference inference loader casts after loading, including an FP32-to-BF16
        # roundtrip for selected FP32 vision and norm parameters. Training does not.
        self.model.joint_experts.to_bfloat16_for_selected_params(self.config.dtype)
        self.model.to(self.device).eval()

    def _prepare_observation(self, observation: dict[str, Any]) -> tuple[FocusVLWAObservation, np.ndarray]:
        raw_state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        if raw_state.shape != (14,) or not np.isfinite(raw_state).all():
            raise ValueError("state must contain 14 finite joint values")
        normalized_state = normalize_quantile(raw_state, self.norm_stats["state"])
        tokens, token_mask = self.tokenizer.tokenize(_prompt(observation), normalized_state)
        source_images = observation["images"]
        images: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for target_key, source_key in _CAMERA_MAPPING.items():
            present = source_key in source_images
            if not present:
                raise ValueError(f"Missing required camera: {source_key}")
            image = np.array(_image(source_images[source_key], pad="wrist" in source_key,
                                    head=source_key == "cam_high"), copy=True)
            tensor = torch.from_numpy(image).to(self.device)[None].to(torch.float32)
            images[target_key] = tensor.permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            masks[target_key] = torch.tensor([present], dtype=torch.bool, device=self.device)

        if "hist_cells" in observation:
            raise ValueError("History cells are not supported; provide full head frames")
        history = head_history.pack_history(observation.get("hist_images", ()), observation.get("hist_mask"))
        history_tensor = torch.from_numpy(history["hist_images"]).to(self.device)[None]
        history_tensor = history_tensor.to(torch.float32) / 255.0 * 2.0 - 1.0
        history_mask = torch.from_numpy(history["hist_mask"]).to(self.device)[None]

        model_observation = FocusVLWAObservation(
            images=images,
            image_masks=masks,
            state=torch.from_numpy(_pad_last_dim(normalized_state, self.config.action_dim)).to(self.device)[None],
            tokenized_prompt=torch.from_numpy(tokens).to(self.device)[None],
            tokenized_prompt_mask=torch.from_numpy(token_mask).to(self.device)[None],
            history_images=history_tensor,
            history_mask=history_mask,
        )
        return model_observation, raw_state

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any], *, noise=None, event_noise=None, world_noise=None) -> dict[str, Any]:
        """Run deterministic inference when all three noise tensors are supplied."""
        model_observation, _ = self._prepare_observation(observation)
        model_noise = None
        if noise is not None:
            model_noise = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
            if model_noise.ndim == 2:
                model_noise = model_noise[None]
        started = time.monotonic()

        def prepare_noise(value):
            if value is None:
                return None
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            return tensor[None] if tensor.ndim == 2 else tensor

        sampled = self.model.sample_actions(
            self.device,
            model_observation,
            noise=model_noise,
            num_steps=self.num_steps,
            event_noise=prepare_noise(event_noise),
            world_noise=prepare_noise(world_noise),
        )
        if isinstance(sampled, torch.Tensor):
            sampled = {"actions": sampled}
        actions = sampled["actions"][0].float().cpu().numpy()
        actions = unnormalize_quantile(actions, self.norm_stats["actions"])
        restored_state = unnormalize_quantile(model_observation.state[0].cpu().numpy(), self.norm_stats["state"])
        actions[..., :14] += np.where(_DELTA_MASK, restored_state[:14], 0.0)
        result: dict[str, Any] = {
            "actions": actions[..., :14],
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000.0},
        }
        if "event_action" in sampled:
            event_action = sampled["event_action"][0].float().cpu().numpy()
            event_action = unnormalize_quantile(event_action, self.norm_stats["actions"]).astype(np.float32)
            event_action[..., :14] += np.where(_DELTA_MASK, restored_state[:14], 0.0)
            result["event_action"] = event_action[..., :14]
        if "world_state" in sampled:
            result["world_state"] = sampled["world_state"][0].float().cpu().numpy()
        return result

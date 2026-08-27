"""
Qwen-MMDiT Framework
A VLA implementation using Qwen-VL backbone + MM-DiT (Ψ₀-style) flow-matching action head.

Key difference from QwenGR00T:
  - Replaces cross-attention DiT with MM-DiT (joint attention between action and VL tokens)
  - Enables bidirectional information flow for better vision-action fusion
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.image_tools import resize_with_pad, to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.MMDiT_ActionHeader import (
    MMDiTFlowmatchingActionHead,
    get_mmdit_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


def resize_images(images, target_size=(224, 224)):
    """Recursively apply the training path's legacy PIL stretch resize."""
    if isinstance(images, Image.Image):
        return images.resize(tuple(target_size), Image.Resampling.BILINEAR)
    if isinstance(images, list):
        return [resize_images(image, target_size) for image in images]
    raise ValueError("Unsupported image type or structure.")


def _resize_images_with_pad(images, target_size=(224, 224)):
    """Letterbox counterpart of trainer_tools.resize_images (openpi resize_with_pad).

    Recursively walks the same nested PIL-image structure resize_images expects and
    returns the same structure, but resizes aspect-preserving with black padding
    instead of stretching. target_size is (width, height), matching PIL convention.
    """
    if isinstance(images, Image.Image):
        w, h = target_size
        padded = resize_with_pad(np.array(images), h, w)  # resize_with_pad takes (height, width)
        return Image.fromarray(padded)
    elif isinstance(images, list):
        return [_resize_images_with_pad(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


@dataclass
class QwenMMDiTDefaultConfig:
    """QwenMMDiT framework default parameters."""

    name: str = "QwenMMDiT"

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "Qwen/Qwen3.5-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
        "vl_hidden_dim": 2048,
    })

    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MMDiT-Psi0",
        "action_dim": 14,
        "state_dim": 14,
        "future_action_window_size": 49,
        "action_horizon": 50,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "vl_feature_dim": 2048,
        "num_target_vision_tokens": 32,
        "add_pos_embed": True,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 10,
        "state_placement": "condition",
        "state_dropout_ratio": 0.0,
    })

    obs_image_size: Optional[list] = None
    advantage_conditioning: bool = False
    positive_only_conditional: bool = True
    unconditional_prob: float = 0.3


def route_recap_prompts(
    instructions: List[str],
    advantages: List[bool],
    *,
    positive_only_conditional: bool = True,
    unconditional_prob: float = 0.3,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[str], dict]:
    """Route RECAP prompts with independent per-sample CFG dropout.

    With positive-only conditioning, a positive-labeled sample receives the
    positive suffix with probability ``1 - unconditional_prob``; dropped
    positives and all negative-labeled samples keep the original instruction.
    This probability is unrelated to an episode position or frame prefix.
    """
    if len(instructions) != len(advantages):
        raise ValueError(
            f"instructions and advantages must have the same length, got {len(instructions)} and {len(advantages)}"
        )
    if not 0.0 <= unconditional_prob <= 1.0:
        raise ValueError(f"unconditional_prob must be in [0, 1], got {unconditional_prob}")
    if any(type(advantage) is not bool for advantage in advantages):
        raise TypeError("RECAP advantages must be bool values (true=positive, false=negative)")

    dropout = torch.rand(len(instructions), generator=generator).lt(unconditional_prob).tolist()
    routed_instructions = []
    metrics = {
        "positive_count": 0,
        "negative_count": 0,
        "positive_conditional_count": 0,
        "positive_unconditional_count": 0,
        "negative_conditional_count": 0,
        "negative_unconditional_count": 0,
    }
    for instruction, advantage, drop_condition in zip(instructions, advantages, dropout, strict=True):
        label = "positive" if advantage else "negative"
        metrics[f"{label}_count"] += 1
        conditional = not drop_condition and (advantage or not positive_only_conditional)
        metrics[f"{label}_{'conditional' if conditional else 'unconditional'}_count"] += 1
        suffix = f"\nAdvantage: {label}" if conditional else ""
        routed_instructions.append(instruction + suffix)

    return routed_instructions, metrics


@FRAMEWORK_REGISTRY.register("QwenMMDiT")
class Qwen_MMDiT(baseframework):
    """
    VLA model using Qwen-VL backbone + MM-DiT (Ψ₀-style) action head.

    The MM-DiT action head uses joint attention between action tokens and VL
    feature tokens, enabling bidirectional information flow (unlike the standard
    cross-attention DiT where only actions attend to conditions).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenMMDiTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align VL hidden dim for the action model
        vl_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.vl_feature_dim = vl_hidden_dim

        self.action_model: MMDiTFlowmatchingActionHead = get_mmdit_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        routing_metrics = {}
        if self.config.framework.advantage_conditioning:
            missing = [index for index, example in enumerate(examples) if "advantage" not in example]
            if missing:
                raise KeyError(f"advantage_conditioning requires a boolean advantage for every sample; missing={missing}")
            instructions, routing_metrics = route_recap_prompts(
                instructions,
                [example["advantage"] for example in examples],
                positive_only_conditional=self.config.framework.positive_only_conditional,
                unconditional_prob=self.config.framework.unconditional_prob,
            )
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        action_mask = (
            [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        )
        action_loss_mask = (
            [example["action_loss_mask"] for example in examples]
            if "action_loss_mask" in examples[0]
            else None
        )

        # Step 1: QWenVL encode
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # (B, L, H)
            encoder_attention_mask = qwen_inputs.get("attention_mask", None)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(device=last_hidden.device)

        # Step 2: Action loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = getattr(
                self.config.framework.action_model, "repeated_diffusion_steps", 8
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            encoder_attention_mask_repeated = (
                encoder_attention_mask.repeat(repeated_diffusion_steps, 1)
                if encoder_attention_mask is not None
                else None
            )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(np.array(action_mask), device=last_hidden.device, dtype=torch.bool)
                action_mask_repeated = action_mask_tensor.repeat(repeated_diffusion_steps, 1)
            if action_loss_mask is not None:
                action_loss_mask_tensor = torch.tensor(
                    np.array(action_loss_mask), device=last_hidden.device, dtype=torch.bool
                )
                action_loss_mask_tensor = action_loss_mask_tensor[:, -(self.future_action_window_size + 1) :, :]
                action_loss_mask_repeated = action_loss_mask_tensor.repeat(repeated_diffusion_steps, 1, 1)
                action_mask_repeated = (
                    action_loss_mask_repeated
                    if action_mask_repeated is None
                    else action_mask_repeated[:, None, :] & action_loss_mask_repeated
                )

            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=encoder_attention_mask_repeated,
                action_mask=action_mask_repeated,
            )

        return {"action_loss": action_loss, **routing_metrics}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        # Same flag the dataloader uses (single source of truth): "stretch" (legacy PIL
        # resize) vs "pad" (openpi-style letterbox). Keeps train/inference preprocessing aligned.
        resize_mode = "stretch"
        vla_data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        if vla_data_cfg is not None and hasattr(vla_data_cfg, "get"):
            resize_mode = vla_data_cfg.get("image_resize_mode", "stretch")
        if not hasattr(self, "_logged_predict"):
            incoming_size = batch_images[0][0].size if hasattr(batch_images[0][0], "size") else "unknown"
            logger.warning(
                f"[PREDICT] Incoming image size: {incoming_size}, config obs_image_size: "
                f"{train_obs_image_size}, resize_mode: {resize_mode}"
            )
            self._logged_predict = True
        if train_obs_image_size:
            if resize_mode == "pad":
                batch_images = _resize_images_with_pad(batch_images, target_size=train_obs_image_size)
            else:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # QWenVL encode
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            encoder_attention_mask = qwen_inputs.get("attention_mask", None)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(device=last_hidden.device)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Action prediction via MM-DiT
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=encoder_attention_mask,
                noise=noise,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_cfg(
        self,
        examples: List[dict],
        beta: float = 2.0,
        suffix: str = "Advantage: positive",
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run final-action CFG with positive/base prompts and one shared initial noise."""
        if type(examples) is not list:
            examples = [examples]
        if not suffix or not isinstance(suffix, str):
            raise ValueError("suffix must be a non-empty string")
        beta = float(beta)
        if not np.isfinite(beta):
            raise ValueError(f"beta must be finite, got {beta}")

        batch_size = len(examples)
        device = self.action_model.device
        dtype = torch.bfloat16 if device.type == "cuda" else self.action_model.dtype
        expected_shape = (batch_size, self.action_model.action_horizon, self.action_model.action_dim)
        if noise is None:
            noise = torch.randn(expected_shape, device=device, dtype=dtype)
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
            if noise.device != device or noise.dtype != dtype:
                raise ValueError(f"noise must be on {device} with dtype {dtype}, got {noise.device}/{noise.dtype}")

        conditional_examples = [{**example, "lang": f"{example['lang']}\n{suffix}"} for example in examples]
        unconditional_examples = [{**example, "lang": example["lang"]} for example in examples]
        conditional = self.predict_action(conditional_examples, noise=noise)["normalized_actions"]
        unconditional = self.predict_action(unconditional_examples, noise=noise)["normalized_actions"]
        if beta == 0.0:
            normalized_actions = unconditional
        elif beta == 1.0:
            normalized_actions = conditional
        else:
            normalized_actions = unconditional + beta * (conditional - unconditional)
        return {
            "normalized_actions": normalized_actions,
            "conditional_actions": conditional,
            "unconditional_actions": unconditional,
            "cfg_beta": beta,
            "cfg_suffix": suffix,
        }

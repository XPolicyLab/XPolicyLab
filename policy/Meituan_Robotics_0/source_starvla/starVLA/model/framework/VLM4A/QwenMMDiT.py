"""
Qwen-MMDiT Framework
A VLA implementation using Qwen-VL backbone + MM-DiT (Ψ₀-style) flow-matching action head.

Key difference from QwenGR00T:
  - Replaces cross-attention DiT with MM-DiT (joint attention between action and VL tokens)
  - Enables bidirectional information flow for better vision-action fusion
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.MMDiT_ActionHeader import (
    MMDiTFlowmatchingActionHead,
    get_mmdit_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenMMDiTDefaultConfig:
    """QwenMMDiT framework default parameters."""

    name: str = "QwenMMDiT"

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
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
        "compute_dtype": "float32",
        "vl_feature_dim": 2048,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 10,
    })

    obs_image_size: Optional[list] = None


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
        self.action_model.history_position_ids = bool(
            self.config.datasets.vla_data.get("visual_history", {}).get("enabled", False))

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size


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

        # QWenVL encode
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions,
            image_metas=[e.get("image_metas") for e in examples], history_configs=[e.get("visual_history") for e in examples],
            lam_memories=[e.get("lam_memory") for e in examples])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                **({"logits_to_keep": 1, "use_cache": False}
                   if self.config.datasets.vla_data.get("visual_history", {}).get("enabled", False) else {}),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            encoder_attention_mask = qwen_inputs.get("attention_mask", None)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(device=last_hidden.device)

        input_dtype = last_hidden.dtype
        last_hidden = last_hidden.to(dtype=input_dtype)
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=input_dtype)
            if state is not None
            else None
        )

        # Action prediction via MM-DiT
        with self.action_model.compute_autocast(last_hidden.device):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=encoder_attention_mask,
                noise=noise,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

"""
Pi0-FAST Policy wrapper.

This implementation is based on the OpenPI project:
https://github.com/Physical-Intelligence/openpi

Original code: openpi/src/openpi/models/pi0_fast.py
Original implementation: JAX/Flax
Licensed under Apache License 2.0

This file contains a PyTorch policy wrapper for the Pi0-FAST model.
Pi0-FAST uses a single LLM (Gemma 2B) with autoregressive action generation.
Action tokens are mapped to the PaliGemma vocabulary using the FAST tokenizer.

Key differences from Pi0:
- Single LLM instead of joint model with action expert
- Action tokens in LLM vocabulary (via FAST tokenizer mapping)
- Standard next-token prediction instead of flow matching

Adapted to PyTorch by Galaxea AI.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import numpy as np
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf

from galaxea_fm.models.base_policy import BasePolicy
from .pi0_fast import Pi0Fast

logger = get_logger(__name__)


class Pi0FastPolicy(BasePolicy):
    """Policy wrapper for Pi0-FAST model with autoregressive action generation."""

    def __init__(
        self,
        **model_cfg: DictConfig,
    ) -> None:
        super().__init__()
        model_cfg = OmegaConf.create(model_cfg)
        self.model = Pi0Fast(model_cfg)
        self.backbone_lr_multiplier = model_cfg.get("backbone_lr_multiplier", 1.0)

        self.model_config = model_cfg
        self.action_horizon = model_cfg.get("horizon_steps", 10)
        self.action_dim = model_cfg.get("action_dim", 7)

        # Tokenizer for inference decoding (set via set_tokenizer before inference)
        self._tokenizer = None

        if self.model_config.get("pretrained_model_path", None):
            self.model.load_pretrained_weights()

    def set_tokenizer(self, tokenizer):
        """Set tokenizer for inference decoding. Must be called before predict_action."""
        self._tokenizer = tokenizer

    def get_optim_param_groups(self, lr, weight_decay):
        """Returns parameter groups with separate learning rates for backbone and LLM."""
        # Separate vision backbone and LLM parameters
        vision_params = list(self.model.vision_tower.parameters()) + \
                       list(self.model.multi_modal_projector.parameters())
        vision_param_ids = set(id(p) for p in vision_params)

        llm_params = [p for p in self.model.parameters() if id(p) not in vision_param_ids]

        param_groups = [
            {
                "params": [p for p in vision_params if p.requires_grad],
                "lr": lr * self.backbone_lr_multiplier,
                "weight_decay": weight_decay,
                "name": "vision_backbone",
            },
            {
                "params": [p for p in llm_params if p.requires_grad],
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "llm",
            },
        ]

        all_requires_grad_params = [p for p in self.parameters() if p.requires_grad]
        assert len(all_requires_grad_params) == sum([len(g['params']) for g in param_groups])

        return param_groups

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        inference_mode=False,
    ):
        # Batch is already tokenized by processor (input_ids, attention_mask, ar_mask, loss_mask)
        if inference_mode:
            was_training = self.training
            self.model.eval()
            normalized_action = self.forward_inference(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                ar_mask=batch["ar_mask"],
                pixel_values=batch["pixel_values"],
            )
            batch["action"] = normalized_action
            self.model.train(was_training)
            return batch
        else:
            return self.forward_train(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                ar_mask=batch["ar_mask"],
                loss_mask=batch["loss_mask"],
                pixel_values=batch["pixel_values"],
            )

    def forward_train(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        ar_mask: torch.LongTensor,
        loss_mask: torch.BoolTensor,
        pixel_values: torch.FloatTensor,
    ):
        loss_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ar_mask=ar_mask,
            loss_mask=loss_mask,
            pixel_values=pixel_values,
        )
        loss = sum(loss_dict.values())
        loss_value_dict = {key: val.detach() for key, val in loss_dict.items()}

        return loss, loss_value_dict

    @torch.no_grad()
    def forward_inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        ar_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() before inference.")

        # Generate action tokens autoregressively
        output_tokens = self.model.infer_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ar_mask=ar_mask,
            pixel_values=pixel_values,
            max_new_tokens=self.model.max_token_len,
        )

        # Extract actions from generated tokens
        batch_size = output_tokens.shape[0]
        actions_list = []

        for i in range(batch_size):
            tokens = output_tokens[i].cpu().numpy()
            actions = self._tokenizer.extract_actions(
                tokens,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
            )
            actions_list.append(actions)

        # Stack and convert to tensor
        actions = np.stack(actions_list, axis=0)
        actions = torch.from_numpy(actions).to(pixel_values.device, dtype=pixel_values.dtype)

        return actions

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(batch, inference_mode=True)

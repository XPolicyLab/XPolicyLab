"""Thin wrapper around Dexbotic DM0 inference for XPolicyLab deployment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from dexbotic.exp.dm0_exp import DM0InferenceConfig as _BaseDM0InferenceConfig


@dataclass
class DM0InferenceConfig(_BaseDM0InferenceConfig):
    """RoboDojo DM0 inference settings (matches robodojo_dm0.py)."""

    num_images: int = 3
    non_delta_mask: list[int] = field(default_factory=lambda: [6, 20])
    action_dim: int = 32


class DM0Infer:
    def __init__(self, model_path: str, *, norm_stats_path: Optional[str] = None):
        self.config = DM0InferenceConfig()
        self.config.model_name_or_path = model_path
        if norm_stats_path is not None:
            self.config.norm_stats = self.config.read_normalization_stats(norm_stats_path)
        self.config._initialize_inference()

    def predict(
        self,
        *,
        prompt: str,
        images_rgb: list[np.ndarray],
        state: np.ndarray,
    ) -> np.ndarray:
        """Run DM0 inference.

        Returns:
            np.ndarray with shape (chunk_size, 32).
        """
        cfg = self.config
        batch_size = 1
        pil_images = [Image.fromarray(np.asarray(img, dtype=np.uint8)) for img in images_rgb]
        num_images = len(pil_images)

        batch_images_tensor = [
            cfg.model.process_images(pil_images).to(dtype=cfg.model.dtype)
        ]
        if num_images != cfg.num_images:
            batch_images_tensor = [
                torch.cat(
                    [
                        image_tensor,
                        torch.zeros_like(image_tensor[0:1]).repeat(
                            cfg.num_images - num_images, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
                if len(image_tensor) < cfg.num_images
                else image_tensor[: cfg.num_images]
                for image_tensor in batch_images_tensor
            ]

        batch_image_masks = [
            torch.tensor(
                [True for _ in range(num_images)]
                + [False for _ in range(cfg.num_images - num_images)],
                device=image_tensor.device,
            )
            for image_tensor in batch_images_tensor
        ]
        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)
        batch_image_masks = torch.stack(batch_image_masks, dim=0)

        batch_input_ids = np.array(
            [cfg.tokenization_func([{"from": "human", "value": prompt}])["input_ids"]]
        )
        batch_attention_mask = np.array(
            [np.array(ids != cfg.tokenizer.pad_token_id) for ids in batch_input_ids]
        )

        batch_states = np.asarray(state, dtype=np.float32).reshape(1, -1)

        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(cfg.non_delta_mask),
            },
        }

        inputs = cfg.input_transform(inference_args)
        inputs["states"] = inputs["state"]
        inputs = {
            k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        actions = cfg.model.inference_action(**inputs)
        outputs = {
            k: v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = actions.detach().cpu().numpy()
        outputs = cfg.output_transform(outputs)
        action_arr = np.asarray(outputs["action"][..., : cfg.action_dim], dtype=np.float32)
        return action_arr.reshape(-1, cfg.action_dim)


def load_dm0_infer(model_path: str, norm_stats_path: Optional[str] = None) -> DM0Infer:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"DM0 checkpoint directory not found: {model_path}")
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"DM0 checkpoint missing config.json under {model_path}. "
            "Point model_path to a checkpoint-* directory."
        )
    logger.info(f"[DM0Infer] Loading from {model_path}")
    t0 = time.monotonic()
    infer = DM0Infer(model_path, norm_stats_path=norm_stats_path)
    logger.info(f"[DM0Infer] Ready in {time.monotonic() - t0:.1f}s")
    return infer

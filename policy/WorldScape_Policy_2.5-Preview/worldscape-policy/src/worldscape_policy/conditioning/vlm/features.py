"""Frozen Qwen+LoRA prefix features for native WSP Auto.

WSP observations are RGB. Only this encoder adapts current/history pixels to
the trained encoder's BGR-valued PIL inputs, before bilinear resize.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .encoder_core.encoder import VLPromptEncoder
from .protocol import AutoPlanningFeatures


def prepare_vlm_images(images, *, input_range, size, teacher_color_order="bgr"):
    """Convert [B,T,3,H,W] RGB tensors without modifying shared observations."""
    if teacher_color_order != "bgr":
        raise ValueError("Released vlm weights require teacher_color_order=bgr")
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("vlm images must have shape [B,T,3,H,W]")
    pixels = images.detach().cpu()
    if input_range == "uint8":
        if pixels.dtype != torch.uint8:
            raise TypeError("uint8 input_range requires uint8 images")
    elif input_range in {"zero_one", "minus_one_one"}:
        low = 0 if input_range == "zero_one" else -1
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
            raise ValueError("Normalized vlm images must be finite floating tensors")
        if (pixels < low).any() or (pixels > 1).any():
            raise ValueError(f"vlm images are outside {input_range}")
        pixels = pixels.float()
        if input_range == "minus_one_one":
            pixels = (pixels + 1) / 2
        pixels = (pixels * 255).round().to(torch.uint8)
    else:
        raise ValueError(f"Unknown visual_input_range: {input_range}")
    # Match the trained color contract: one RGB-to-BGR conversion in this
    # private input copy, then wrap the values in PIL without another swap.
    pixels = pixels[:, :, [2, 1, 0]].permute(0, 1, 3, 4, 2).numpy()
    return [[Image.fromarray(np.ascontiguousarray(frame)).resize(
        tuple(size), Image.Resampling.BILINEAR) for frame in row] for row in pixels]


class VLMFeatureEncoder(nn.Module):
    """PlanningVLM adapter; one joint prefill, with no subtask generation.

    Histories must be explicit past-only frames, oldest first, at 25-frame
    offsets (25 Hz). Session history stays with the caller, never this module.
    """

    def __init__(self, *, assets_dir, teacher_color_order="bgr", device="cpu",
                 attn_implementation="sdpa", vlm_base=None,
                 config_filename="encoder_config.json"):
        super().__init__()
        from safetensors.torch import load_file

        if teacher_color_order != "bgr":
            raise ValueError("Released vlm weights require teacher_color_order=bgr")
        root = Path(assets_dir).expanduser().resolve()
        config = json.loads((root / config_filename).read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if config.get("text_layout") != "prefix_v2":
            raise ValueError("vlm bundle must use prefix_v2")
        if config.get("teacher_input_color_order") != teacher_color_order:
            raise ValueError("vlm bundle color contract does not match official BGR")
        parameters = config["encoder"]
        expected = dict(qwen_layers=[9, 18, 27], summary_layers=[18, 27, 36],
                        history_num_slots=20, history_period_s=1.0, history_fps=25.0,
                        history_pool=[4, 4])
        if any(parameters.get(key) != value for key, value in expected.items()):
            raise ValueError("vlm bundle does not match the released feature/history contract")
        if not parameters.get("lora", {}).get("enabled"):
            raise ValueError("vlm features require the released LoRA")
        for relative, record in manifest["files"].items():
            path = root / relative
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Incomplete vlm asset: {path}")
        lora_path = root / config["lora_file"]
        digest = hashlib.sha256()
        with lora_path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != manifest["files"][config["lora_file"]]["sha256"]:
            raise ValueError("vlm LoRA checksum mismatch")
        self.encoder = VLPromptEncoder(
            str(vlm_base or root / config["base_directory"]), device=device,
            torch_dtype=torch.bfloat16,
            **dict(parameters, attn_implementation=attn_implementation))
        state = load_file(str(lora_path), device="cpu")
        result = self.encoder.load_lora_state_dict(state)
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError(f"Incomplete vlm LoRA load: {result}")
        self.encoder.loaded_text_layout = config["text_layout"]
        self.output_dim = self.encoder.dim
        if self.output_dim != 7680:
            raise ValueError("Released vlm features must have width 7680")
        self.teacher_color_order = teacher_color_order
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent policy.train() must not reactivate frozen LoRA dropout.
        super().train(False)
        return self

    def _apply(self, fn, recurse=True):
        # Preserve ALL pretrained dtypes, including FP32 rotary buffers and
        # PEFT matrices, when DeepSpeed calls policy.bfloat16(). Device moves
        # still apply. Casting and then promoting back would lose precision.
        def preserve_dtype(tensor):
            converted = fn(tensor)
            if converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(preserve_dtype, recurse=recurse)

    def encode_cached(self, features, mask):
        """Use immutable prefix features while keeping the normal inference VLM."""
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("Cached vlm features require a frozen VLM")
        if features.ndim != 3 or features.shape[-1] != self.output_dim:
            raise ValueError("Cached vlm features must have shape [B,L,7680]")
        return AutoPlanningFeatures(perception_features=features.detach(),
                                    direct_condition_mask=mask, semantic_supervision=False)

    def encode_planning(self, *, images, planning_text, negative_text, training,
                        visual_input_range="zero_one", history_images=None,
                        history_mask=None, planning_supervision=False,
                        planning_labels_text=None):
        if planning_supervision:
            raise ValueError("vlm feature mode does not generate planning logits")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("vlm feature mode requires frozen Qwen and LoRA")
        if len(planning_text) != images.shape[0] or not all(
            isinstance(text, str) and text.strip() for text in planning_text
        ):
            raise ValueError("vlm requires one nonempty high-level instruction per sample")
        if images.ndim != 5 or images.shape[1] != 1:
            raise ValueError("vlm requires exactly one current head frame per sample")
        current = prepare_vlm_images(images, input_range=visual_input_range,
                                       size=self.encoder.image_size,
                                       teacher_color_order=self.teacher_color_order)
        if history_images is None:
            if history_mask is not None:
                raise ValueError("History mask requires history images")
            histories = [[] for _ in current]
        else:
            if (history_images.ndim != 5 or history_images.shape[0] != len(current) or history_mask is None
                    or history_mask.shape != history_images.shape[:2]
                    or history_mask.dtype != torch.bool):
                raise ValueError("History requires a matching boolean [B,T] mask")
            if (history_mask.sum(-1) > self.encoder.history_num_slots).any():
                raise ValueError("vlm history exceeds 20 past frames")
            # Ignore padded history before conversion; invalid slots need not
            # contain valid pixels, and must never enter the prompt.
            histories = [prepare_vlm_images(row[mask.to(row.device)][None],
                input_range=visual_input_range, size=self.encoder.history_image_size,
                teacher_color_order=self.teacher_color_order)[0]
                for row, mask in zip(history_images, history_mask, strict=True)]
        # Do not let the trainer's BF16 autocast alter the official FP32 LoRA
        # computation. Base Qwen weights and feature outputs remain BF16.
        device_type = next(self.encoder.parameters()).device.type
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            self.encoder.eval()
            batch = self.encoder.build_infer_batch(
                planning_text, [row[0] for row in current], histories)
            tokens, valid = self.encoder.infer_vlm(batch)
            # Native vlm appends its summary AFTER padded instruction slots.
            # Compact each row so WSP receives [instruction, summary, padding].
            rows = [row[mask] for row, mask in zip(tokens, valid, strict=True)]
            features = nn.utils.rnn.pad_sequence(rows, batch_first=True)
            lengths = torch.tensor([len(row) for row in rows], device=features.device)
            mask = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        return AutoPlanningFeatures(perception_features=features,
                                    direct_condition_mask=mask, semantic_supervision=False)

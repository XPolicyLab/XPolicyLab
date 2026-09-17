"""Qwen3-VL conditioning with one learned arm-query bank and no base branch.

Native Qwen3-VL handles image feature insertion and DeepStack visual features;
its language model is never called in isolation.
"""

from __future__ import annotations

import torch
from torch import nn


def build_query_attention_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Causal [B,1,L,L] SDPA visibility for [prompt, arm queries]."""
    if padding_mask.ndim != 2 or padding_mask.shape[1] == 0:
        raise ValueError("padding_mask must have nonempty shape [B,L]")
    valid = padding_mask.bool()
    positions = torch.arange(valid.shape[1], device=valid.device)
    causal = positions[:, None] >= positions[None, :]
    return causal[None, None] & valid[:, None, :, None] & valid[:, None, None, :]


class QwenArmQueryBackbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Load transformers only when constructing the Qwen backbone.
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_id = config.get("base_vlm")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("base_vlm must name the Qwen3-VL pretrained model directory or identifier")
        dtype_name = str(config.get("backbone_dtype", config.get("dtype", "bfloat16")))
        dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
        if dtype_name not in dtypes:
            raise ValueError("backbone_dtype must be bfloat16, float32 or float16")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", dtype=dtypes[dtype_name],
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.image_token_id = int(self.model.config.image_token_id)
        if bool(config.get("gradient_checkpointing", False)):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.config.use_cache = False

    def encode(self, images, instructions, query_tokens: torch.Tensor) -> torch.Tensor:
        from PIL import Image
        import numpy as np

        if not images or len(images) != len(instructions):
            raise ValueError("Provide a nonempty image batch and one instruction per sample")
        messages = []
        for views, instruction in zip(images, instructions):
            if not isinstance(views, (list, tuple)) or not views:
                raise ValueError("Each sample must contain a nonempty ordered list of RGB views")
            content = []
            for frame in views:
                if not isinstance(frame, Image.Image):
                    frame = np.asarray(frame)
                    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                        raise ValueError("Camera frames must be RGB uint8 [H,W,3] arrays")
                    frame = Image.fromarray(frame)
                if frame.mode != "RGB":
                    raise ValueError("PIL camera frames must already be RGB")
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": str(instruction)})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        input_ids = inputs.pop("input_ids")
        padding = inputs.pop("attention_mask", torch.ones_like(input_ids))
        if not ((input_ids == self.image_token_id) & padding.bool()).any(dim=1).all():
            raise ValueError("Qwen prompt must contain image tokens for every sample")
        count = query_tokens.shape[0]
        if count <= 0 or query_tokens.shape != (count, self.hidden_size):
            raise ValueError("query_tokens must have shape [positive Q, Qwen hidden_size]")
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("The Qwen processor must define a padding token")
        placeholders = torch.full((input_ids.shape[0], count), int(pad_id),
                                  device=input_ids.device, dtype=input_ids.dtype)
        expanded_ids = torch.cat((input_ids, placeholders), dim=1)
        expanded_padding = torch.cat((padding, torch.ones_like(placeholders)), dim=1)
        native_backbone = self.model.model
        positions, _ = native_backbone.get_rope_index(
            input_ids=expanded_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=expanded_padding,
        )
        prompt_embeddings = self.model.get_input_embeddings()(input_ids)
        queries = query_tokens.to(device=prompt_embeddings.device, dtype=prompt_embeddings.dtype)
        embeddings = torch.cat((prompt_embeddings,
                                queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)), dim=1)
        outputs = native_backbone(
            **inputs, inputs_embeds=embeddings,
            attention_mask=build_query_attention_mask(expanded_padding),
            position_ids=positions, use_cache=False,
            output_attentions=False, return_dict=True,
        )
        return outputs.last_hidden_state[:, -count:]

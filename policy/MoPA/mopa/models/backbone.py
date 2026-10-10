"""Qwen3-VL conditioning with isolated learned arm and base query banks.

Native Qwen3-VL handles image feature insertion and DeepStack visual features;
its language model is never called in isolation.
"""

from __future__ import annotations

import torch
from torch import nn


def build_query_attention_mask(
    padding_mask: torch.Tensor,
    *,
    arm_query_tokens: int | None = None,
    base_query_tokens: int = 0,
) -> torch.Tensor:
    """Causal prompt mask plus isolated arm/base query suffixes.

    Source Query-DMoT appends two learned banks to the Qwen prompt.  Prompt
    tokens retain causal visibility; each query bank may attend to the prompt
    and to its own causal prefix, while the arm and base banks cannot read one
    another.  The old one-bank call remains valid when ``base_query_tokens`` is
    zero.
    """
    if padding_mask.ndim != 2 or padding_mask.shape[1] == 0:
        raise ValueError("padding_mask must have nonempty shape [B,L]")
    valid = padding_mask.bool()
    length = valid.shape[1]
    if arm_query_tokens is None:
        return (torch.arange(length, device=valid.device)[:, None]
                >= torch.arange(length, device=valid.device)[None, :])[None, None] \
            & valid[:, None, :, None] & valid[:, None, None, :]
    arm_query_tokens = int(arm_query_tokens)
    base_query_tokens = int(base_query_tokens)
    if arm_query_tokens <= 0 or base_query_tokens < 0:
        raise ValueError("arm_query_tokens must be positive and base_query_tokens non-negative")
    prompt_length = length - arm_query_tokens - base_query_tokens
    if prompt_length <= 0:
        raise ValueError("Query suffix must leave at least one prompt token")
    allowed = torch.zeros((length, length), device=valid.device, dtype=torch.bool)
    prompt_positions = torch.arange(prompt_length, device=valid.device)
    allowed[:prompt_length, :prompt_length] = (
        prompt_positions[:, None] >= prompt_positions[None, :]
    )
    # Appended query rows can inspect the complete prompt, then their own
    # query group's causal prefix.  No arm/base cross-bank edges are present.
    for start, count in ((prompt_length, arm_query_tokens),
                         (prompt_length + arm_query_tokens, base_query_tokens)):
        if count:
            rows = torch.arange(start, start + count, device=valid.device)
            allowed[rows, :prompt_length] = True
            allowed[rows[:, None], rows[None, :]] = rows[:, None] >= rows[None, :]
    return allowed[None, None] & valid[:, None, :, None] & valid[:, None, None, :]


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
        self.arm_num_query_tokens = int(config.get("arm_num_query_tokens", 8))
        self.base_num_query_tokens = int(config.get("base_num_query_tokens", 0))
        if self.arm_num_query_tokens <= 0 or self.base_num_query_tokens < 0:
            raise ValueError("Invalid arm/base query counts")
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
        if count != self.arm_num_query_tokens + self.base_num_query_tokens:
            raise ValueError("query_tokens count does not match configured arm/base banks")
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
            attention_mask=build_query_attention_mask(
                expanded_padding,
                arm_query_tokens=self.arm_num_query_tokens,
                base_query_tokens=self.base_num_query_tokens,
            ),
            position_ids=positions, use_cache=False,
            output_attentions=False, return_dict=True,
        )
        return outputs.last_hidden_state[:, -count:]

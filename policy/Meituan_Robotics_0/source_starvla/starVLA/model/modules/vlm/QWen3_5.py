# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoConfig, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # FAST-tokenizer action-token range.
_ACTION_TOKEN_COUNT = 2048
_MMDIT_CHECKPOINT_VOCAB_SIZE = 250125


def _resolve_robot_action_token_range(tokenizer, count: int = _ACTION_TOKEN_COUNT) -> tuple[int, int]:
    token_ids = []
    missing = []
    for idx in range(count):
        token = f"<robot_action_{idx}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            missing.append(token)
        else:
            token_ids.append(int(token_id))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"Tokenizer is missing {len(missing)} FAST action tokens, e.g. {preview}")

    start = token_ids[0]
    expected = list(range(start, start + count))
    if token_ids != expected:
        raise ValueError(
            "FAST action token ids must be contiguous and ordered: "
            f"got first={token_ids[0]}, last={token_ids[-1]}"
        )
    return start, token_ids[-1]


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        if qwenvl_config.get("random_init", False):
            # random_init=true: keep the base_vlm ARCHITECTURE but randomly initialize ALL
            # backbone weights from the HF config (full from-scratch incl. the Qwen base).
            # The processor/tokenizer below still loads from the checkpoint dir — it carries
            # vocab/preprocessing config, not learned weights.
            hf_config = AutoConfig.from_pretrained(model_id)
            model = Qwen3_5ForConditionalGeneration._from_config(
                hf_config,
                dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            logger.info(f"qwenvl.random_init=true: built {model_id} architecture with randomly initialized weights.")
        else:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        # The RoboDojo MMDiT checkpoint stores 250125 embedding/head rows. The
        # public Qwen3.5-4B base has fewer rows; strict checkpoint loading will
        # replace the expanded tensors in full before inference.
        if (
            config.framework.name == "QwenMMDiT"
            and model.get_input_embeddings().num_embeddings != _MMDIT_CHECKPOINT_VOCAB_SIZE
        ):
            model.resize_token_embeddings(_MMDIT_CHECKPOINT_VOCAB_SIZE)

        # VLA observation images: control resolution via Qwen smart_resize (preserves
        # aspect ratio) instead of a hard square resize in the dataloader. When
        # vla_data sets max_pixels/min_pixels, push them onto this processor's image
        # processor so each frame is resized to ~max_pixels keeping its native ratio.
        # This processor is independent of the VQA path's processor (vlm_datasets.py),
        # so VLA and VLM resolutions are configured separately.
        vla_data_cfg = config.datasets.get("vla_data", {}) if config is not None else {}
        vla_max_pixels = vla_data_cfg.get("max_pixels", None)
        vla_min_pixels = vla_data_cfg.get("min_pixels", None)
        if vla_max_pixels is not None:
            processor.image_processor.max_pixels = int(vla_max_pixels)
            processor.image_processor.size["longest_edge"] = int(vla_max_pixels)
        if vla_min_pixels is not None:
            processor.image_processor.min_pixels = int(vla_min_pixels)
            processor.image_processor.size["shortest_edge"] = int(vla_min_pixels)

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        self._ACTION_TOKEN_MIN = None
        self._ACTION_TOKEN_MAX = None
        try:
            self._ACTION_TOKEN_MIN, self._ACTION_TOKEN_MAX = _resolve_robot_action_token_range(processor.tokenizer)
        except ValueError as exc:
            if "-Action" in str(model_id):
                raise
            logger.info(f"Qwen3.5 tokenizer has no FAST action-token range: {exc}")

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        chat_template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": solutions is None,
            "return_dict": True,
            "return_tensors": "pt",
            "processor_kwargs": {"padding": True},
        }
        if solutions is None:
            # Qwen3.5's template opens a thinking block by default. FAST training
            # sees an empty closed block before action tokens, so inference must
            # use the same prefix and start generation at <robot_action_*>
            # instead of inside <think>.
            chat_template_kwargs["enable_thinking"] = False

        batch_inputs = self.processor.apply_chat_template(messages, **chat_template_kwargs)

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            if self._ACTION_TOKEN_MIN is None or self._ACTION_TOKEN_MAX is None:
                raise ValueError(
                    "QwenFast requires <robot_action_0> ... <robot_action_2047> in the Qwen tokenizer. "
                    "Create/use a Qwen3.5-Action checkpoint first."
                )
            action_token_min = self._ACTION_TOKEN_MIN
            action_token_max = self._ACTION_TOKEN_MAX
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; check that the tokenizer contains the FAST action tokens."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)

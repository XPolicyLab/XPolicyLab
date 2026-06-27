# Copyright 2025 eventvla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
_ACTION_TOKEN_MIN = 151669
_ACTION_TOKEN_MAX = 153716


class _QWen3_VL_Interface(nn.Module):
    """Lightweight wrapper around Qwen3-VL."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # Align with existing Qwen2.5 wrapper expectation.
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(**kwargs)
        return generation_output

    @staticmethod
    def _build_image_content(imgs, metas=None, use_image_role_text: bool = False):
        if not use_image_role_text or metas is None or len(metas) != len(imgs):
            return [{"type": "image", "image": img} for img in imgs]

        content = []
        last_group = None
        for img, meta in zip(imgs, metas):
            role = str(meta.get("role", "")).lower() if isinstance(meta, dict) else ""
            group = "memory" if role == "memory_keyframe" else "anchor"
            if group != last_group:
                label = "Past keyframe images:" if group == "memory" else "Temporal observation images:"
                content.append({"type": "text", "text": label})
                last_group = group
            content.append({"type": "image", "image": img})
        return content

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        image_metas=None,
        use_image_role_text: bool = False,
        **kwargs,
    ):
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        for sample_idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            metas = None
            if image_metas is not None and sample_idx < len(image_metas):
                metas = image_metas[sample_idx]
            content = self._build_image_content(
                imgs,
                metas=metas,
                use_image_role_text=use_image_role_text,
            )

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token not found in tokenizer; check "
                        "eventvla/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    import debugpy

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./eventvla/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    print(qwen_vl)

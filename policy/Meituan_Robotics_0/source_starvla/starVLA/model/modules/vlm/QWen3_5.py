# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from typing import Optional

import torch
from starVLA.visual_inputs import first_role_mask, detach_first_vision
from starVLA.lam_memory import limit_memory_window, MEMORY_MARKER, LAMMemoryProjection, validate_memory, inject_memory_embeddings
from starVLA.visual_history import VisualHistoryConfig, HistoryContractError, history_label, prepare_history_images
from transformers import AutoConfig, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Keep preprocessing and causal memory aligned with the checkpoint.

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

        prefix = "qwen_vl_interface.model."
        vlm_state = {
            key.removeprefix(prefix): value
            for key, value in torch.load(
                qwenvl_config.checkpoint_path, map_location="cpu", mmap=True, weights_only=True,
            ).items()
            if key.startswith(prefix)
        }
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            None,
            config=AutoConfig.from_pretrained(model_id),
            state_dict=vlm_state,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        self._native_image_size = dict(processor.image_processor.size)

        # VLA observation images: control resolution via Qwen smart_resize (preserves
        # aspect ratio) instead of a hard square resize in the dataloader. When
        # vla_data sets max_pixels/min_pixels, push them onto this processor's image
        # processor so each frame is resized to ~max_pixels keeping its native ratio.
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
        self.first_frame_encoder_detach = bool(vla_data_cfg.get("first_frame_encoder_detach", False))

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        self.lam_memory_enabled = bool(vla_data_cfg.get("lam_memory", {}).get("enabled", False))
        if self.lam_memory_enabled:
            marker_ids = processor.tokenizer.encode(MEMORY_MARKER, add_special_tokens=False)
            if len(marker_ids) != 1 or marker_ids[0] == processor.tokenizer.unk_token_id:
                raise ValueError("LAM requires an existing single reserved placeholder token")
            self.lam_marker_id = marker_ids[0]
            # Preserve the baseline RNG stream for action-head initialization.
            with torch.random.fork_rng(devices=[]):
                self.lam_memory_projection = LAMMemoryProjection(self.model.config.hidden_size)

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        first_mask = kwargs.pop("_first_frame_detach_mask", None)
        with detach_first_vision(self.model, kwargs.get("image_grid_thw"), first_mask), \
             torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)

        return outputs


    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Build inference inputs from images, instructions and causal memory.
        Follow Oficial Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct
        """

        # History geometry is role-specific; avoid the current-frame processor
        # budget upscaling low-resolution historical observations again.
        lam_memories = kwargs.pop("lam_memories", None)
        if self.lam_memory_enabled:
            if lam_memories is None or len(lam_memories) != len(images):
                raise ValueError("LAM memory is required for every example")
            lam_memories = [limit_memory_window(memory, self.config.datasets.vla_data.lam_memory.get("window_sec"))
                            for memory in lam_memories]
            for memory in lam_memories:
                validate_memory(memory)
        image_metas = kwargs.pop("image_metas", None)
        history_configs = kwargs.pop("history_configs", None)
        default_spec = VisualHistoryConfig.from_data_config(self.config.datasets.vla_data)
        memory_enabled = default_spec is not None
        if memory_enabled:
            if image_metas is None or len(image_metas) != len(images) or any(m is None for m in image_metas):
                raise HistoryContractError("memory-enabled inputs require image_metas")
            if history_configs is not None and len(history_configs) != len(images):
                raise HistoryContractError("history config batch size differs from image batch size")
            resized, resolved_specs = [], []
            for i, (imgs, metas) in enumerate(zip(images, image_metas)):
                spec = (VisualHistoryConfig.from_data_config({"visual_history": {"enabled": True, **history_configs[i]}})
                        if history_configs and history_configs[i] else default_spec)
                resolved_specs.append(spec)
                resized.append(prepare_history_images(imgs, metas, spec, self.processor,
                    self._native_image_size, self.config.datasets.vla_data.get("image_size")))
            images = resized
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            if memory_enabled:
                metas = image_metas[len(messages)]
                content = []
                inserted_memory = False
                for img, meta in zip(imgs, metas):
                    if self.lam_memory_enabled and meta["role"] == "current" and not inserted_memory:
                        count = len(lam_memories[len(messages)]["features"])
                        content.append({"type": "text", "text": "Past visual changes, oldest to newest: " + MEMORY_MARKER * count})
                        inserted_memory = True
                    content.extend([{"type": "text", "text": history_label(meta, resolved_specs[len(messages)].main_camera)},
                                    {"type": "image", "image": img}])
            else:
                content = [{"type": "image", "image": img} for img in imgs]

            prompt = instruction

            if self.lam_memory_enabled and MEMORY_MARKER in prompt:
                raise ValueError("Instruction contains reserved LAM marker")
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            messages.append(msg)

        # Preparation for inference

        chat_template_kwargs = {
            "tokenize": True,
            "padding": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        chat_template_kwargs["enable_thinking"] = False

        if memory_enabled:
            # Transformers replaces processor_kwargs when flat processor flags
            # are also passed. Keep padding and image flags together.
            chat_template_kwargs["processor_kwargs"] = {
                "padding": chat_template_kwargs.pop("padding"),
                "images_kwargs": {"do_resize": False},
            }
        batch_inputs = self.processor.apply_chat_template(messages, **chat_template_kwargs)

        batch_inputs = batch_inputs.to(self.model.device)
        if self.lam_memory_enabled:
            batch_inputs = inject_memory_embeddings(self.model, self.lam_memory_projection,
                                                    batch_inputs, lam_memories, self.lam_marker_id)
        if self.first_frame_encoder_detach:
            if not memory_enabled:
                raise HistoryContractError("First-frame detach requires camera/role metadata")
            mask = first_role_mask(image_metas)
            if len(mask) != len(batch_inputs["image_grid_thw"]):
                raise HistoryContractError("Processor image order/count differs from metadata")
            batch_inputs["_first_frame_detach_mask"] = mask
        return batch_inputs

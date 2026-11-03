"""WorldScape-owned Qwen3-VL planning encoder.

Adapted from DreamZero's ``qwen3_vl_interface``,
``qwen3vl_qformer_backbone``, and ``layerwise_qformer``. Parameter names,
shared-prefill generation, label construction, and state-dict layout are
preserved so existing checkpoints load without remapping changes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from worldscape_policy.conditioning.prompt_format import render_instruction_template
from worldscape_policy.conditioning.projectors import LayerwiseQFormer
from PIL import Image
from torch import nn
from torch.nn import functional as F

from worldscape_policy.conditioning.vlm.protocol import AutoPlanningFeatures
from worldscape_policy.memory.visual.normalization import VisualInputRange

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 151669
_ACTION_TOKEN_MAX = 153716


def _transformers():
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except (ImportError, AttributeError) as exc:  # pragma: no cover - install dependent
        raise ImportError(
            "Auto conditioning with Qwen3-VL requires a transformers build "
            "that provides AutoProcessor and Qwen3VLForConditionalGeneration."
        ) from exc
    return AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VLInterface(nn.Module):
    """Lightweight Qwen3-VL wrapper with the legacy state layout."""

    def __init__(
        self,
        config: Any = None,
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.bfloat16,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        AutoProcessor, Qwen3VLForConditionalGeneration = _transformers()
        self.config = config
        processor_kwargs = processor_kwargs or {}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_vlm,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        self.processor = AutoProcessor.from_pretrained(
            config.base_vlm, **processor_kwargs
        )
        self.processor.tokenizer.padding_side = "left"
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self.model_id = config.base_vlm

    def forward(self, **kwargs: Any):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def generate(self, **kwargs: Any):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(
        self,
        images: list[list[object]],
        instructions: list[str],
        solutions: list[str] | None = None,
        add_generation_prompt: bool = True,
        apply_cot_prompt: bool = True,
    ):
        if len(images) != len(instructions):
            raise ValueError("Images and instructions must have the same length")
        messages = []
        for index, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": image} for image in imgs]
            cot_prompt = (
                getattr(self.config, "CoT_prompt", None)
                if self.config is not None and apply_cot_prompt
                else None
            )
            prompt = (
                render_instruction_template(cot_prompt, instruction)
                if cot_prompt
                else instruction
            )
            content.append({"type": "text", "text": prompt})
            message = [{"role": "user", "content": content}]
            if solutions is not None:
                message.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": solutions[index]}],
                    }
                )
            messages.append(message)
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        if solutions is not None and "-Action" in self.model_id:
            labels = batch_inputs["input_ids"].clone()
            for sequence in labels:
                action_mask = (sequence >= _ACTION_TOKEN_MIN) & (
                    sequence <= _ACTION_TOKEN_MAX
                )
                indices = torch.nonzero(action_mask, as_tuple=False)
                if indices.numel() > 0:
                    sequence[: indices[0].item()] = IGNORE_INDEX
                else:
                    sequence[:] = IGNORE_INDEX
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
        return batch_inputs.to(self.model.device)


class QwenPlanningEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_base: str | None = None,
        vlm_cot_prompt: str | None = None,
        enable_vlm_tokens: bool = True,
        vlm_token_mode: str = "last",
        vlm_tokens_key: str = "vlm_tokens",
        vlm_tokens_negative_key: str = "vlm_tokens_negative",
        vlm_text_emb_key: str = "vlm_text_emb",
        enable_planning_branch: bool = True,
        planning_tokens_key: str = "vlm_planning_tokens",
        planning_tokens_negative_key: str = "vlm_planning_tokens_negative",
        planning_logits_key: str = "vlm_planning_logits",
        planning_labels_key: str = "planning_labels",
        planning_num_tokens: int = 4,
        vlm_history_tokens_key: str = "vlm_history_tokens",
        vlm_history_planning_tokens_key: str = "vlm_history_planning_tokens",
        vlm_history_mask_key: str = "vlm_history_mask",
        attn_implementation: str = "flash_attention_2",
        dtype: str = "bfloat16",
        qformer_start_layer: int = 36,
        qformer_end_layer: int = 37,
        qformer_num_query_tokens: int = 64,
        qformer_num_heads: int = 8,
        qformer_input_dim: int | None = None,
        qformer_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        mode = (vlm_token_mode or "last").lower()
        if mode not in {"last", "qformer"}:
            raise ValueError(f"Unknown vlm_token_mode: {vlm_token_mode}")
        self.vlm_tokens_key = vlm_tokens_key
        self.vlm_tokens_negative_key = vlm_tokens_negative_key
        self.vlm_text_emb_key = vlm_text_emb_key
        self.enable_planning_branch = bool(enable_planning_branch)
        self.planning_tokens_key = planning_tokens_key
        self.planning_tokens_negative_key = planning_tokens_negative_key
        self.planning_logits_key = planning_logits_key
        self.planning_labels_key = planning_labels_key
        self.planning_num_tokens = int(planning_num_tokens)
        self.vlm_history_tokens_key = vlm_history_tokens_key
        self.vlm_history_planning_tokens_key = vlm_history_planning_tokens_key
        self.vlm_history_mask_key = vlm_history_mask_key
        self.enable_vlm_tokens = enable_vlm_tokens
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.vlm = Qwen3VLInterface(
            config=SimpleNamespace(
                base_vlm=vlm_base or model_id,
                CoT_prompt=vlm_cot_prompt,
            ),
            attn_implementation=attn_implementation,
            dtype=torch_dtype,
        )
        self.qformer_start_layer = qformer_start_layer
        self.qformer_end_layer = qformer_end_layer
        self.qformer_num_layers = qformer_end_layer - qformer_start_layer
        self.vlm_token_mode = mode
        self.qformer: LayerwiseQFormer | None = None
        if mode == "qformer":
            hidden_dim = qformer_input_dim or self.vlm.model.config.hidden_size
            qformer_output_dim = (
                hidden_dim if qformer_output_dim is None else qformer_output_dim
            )
            self.qformer = LayerwiseQFormer(
                input_hidden_dim=hidden_dim,
                output_hidden_dim=qformer_output_dim,
                num_query_tokens=qformer_num_query_tokens,
                num_layers=self.qformer_num_layers,
                num_heads=qformer_num_heads,
            )
    def unused_parameter_module_paths(self) -> tuple[str, ...]:
        return ()

    def set_trainable_parameters(self, **kwargs: Any) -> None:
        del kwargs

    def _to_pil_images(
        self,
        batch_images: Any,
        *,
        input_range: VisualInputRange = "uint8",
    ) -> list[list[Image.Image]]:
        if input_range not in {"uint8", "zero_one", "minus_one_one"}:
            raise ValueError(f"Unknown visual input range: {input_range!r}")
        output = []
        for sample in batch_images:
            if torch.is_tensor(sample):
                sample = sample.detach().cpu()
            else:
                sample = torch.as_tensor(sample)
            if input_range == "uint8":
                if sample.dtype != torch.uint8:
                    raise TypeError(
                        "visual_input_range='uint8' requires torch.uint8 images"
                    )
                uint8_sample = sample
            else:
                if not sample.is_floating_point():
                    raise TypeError(
                        f"visual_input_range={input_range!r} requires floating-point images"
                    )
                low = -1.0 if input_range == "minus_one_one" else 0.0
                if sample.numel() and (
                    bool((sample < low).any()) or bool((sample > 1.0).any())
                ):
                    raise ValueError(
                        f"Images must be in [{low}, 1] for "
                        f"visual_input_range={input_range!r}"
                    )
                normalized = (
                    (sample + 1.0) * 0.5
                    if input_range == "minus_one_one"
                    else sample
                )
                uint8_sample = (normalized * 255.0).round().to(torch.uint8)
            if uint8_sample.ndim == 3:
                uint8_sample = uint8_sample.unsqueeze(0)
            if uint8_sample.ndim != 4 or uint8_sample.shape[-1] != 3:
                raise ValueError(
                    "VLM images must have shape [T,H,W,3] or [H,W,3]"
                )
            # Match the original DreamTransform preprocessing exactly: Qwen
            # sees head-camera frames letterboxed to 224x224, while the WAM
            # continues to consume its native-resolution observation tensor.
            frames = uint8_sample.permute(0, 3, 1, 2).float()
            source_h, source_w = int(frames.shape[-2]), int(frames.shape[-1])
            scale = min(224.0 / source_h, 224.0 / source_w)
            resized_h = max(1, int(round(source_h * scale)))
            resized_w = max(1, int(round(source_w * scale)))
            resized = torch.nn.functional.interpolate(
                frames,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
            canvas = torch.zeros(
                (resized.shape[0], 3, 224, 224), dtype=resized.dtype
            )
            top = (224 - resized_h) // 2
            left = (224 - resized_w) // 2
            canvas[:, :, top : top + resized_h, left : left + resized_w] = resized
            sample = (
                canvas.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .numpy()
            )
            output.append(
                [Image.fromarray(frame) for frame in sample]
            )
        return output

    def _pool_vlm_hidden_states(self, hidden_states: Any) -> torch.Tensor:
        if self.vlm_token_mode == "last":
            return hidden_states[-1]
        if self.qformer is None:
            raise RuntimeError("qformer token mode requires a constructed QFormer")
        selected = hidden_states[self.qformer_start_layer : self.qformer_end_layer]
        return self.qformer(list(selected))

    def _encode_vlm_last(self, inputs: dict[str, Any]) -> torch.Tensor:
        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _encode_vlm_qformer(self, inputs: dict[str, Any]) -> torch.Tensor:
        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return self._pool_vlm_hidden_states(outputs.hidden_states)

    def _encode_vlm_tokens(self, inputs: dict[str, Any]) -> torch.Tensor:
        if self.vlm_token_mode == "last":
            return self._encode_vlm_last(inputs)
        return self._encode_vlm_qformer(inputs)

    def _extract_text_embedding(
        self,
        inputs: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        attention_mask = inputs.get("attention_mask")
        embed_layer = self.vlm.model.get_input_embeddings()
        text_tokens = embed_layer(input_ids.to(embed_layer.weight.device))
        if attention_mask is None:
            pooled = text_tokens.mean(dim=1)
        else:
            mask = (
                attention_mask.to(text_tokens.device)
                .unsqueeze(-1)
                .to(text_tokens.dtype)
            )
            pooled = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.to(device=device, dtype=dtype)

    def _normalize_instructions(
        self, instructions: Any, batch_size: int
    ) -> list[str]:
        if isinstance(instructions, str):
            return [instructions] * batch_size
        if isinstance(instructions, (list, tuple)):
            return [str(value) for value in instructions]
        return [str(instructions)] * batch_size

    def _encode_history_tokens(
        self,
        history_images: Any,
        history_mask: torch.Tensor | None,
        instructions: Any,
        device: torch.device,
        dtype: torch.dtype,
        *,
        input_range: VisualInputRange = "uint8",
    ):
        pil_history = self._to_pil_images(
            history_images,
            input_range=input_range,
        )
        if not pil_history:
            return None, None, None
        instructions = self._normalize_instructions(
            instructions, len(pil_history)
        )
        flat_images: list[list[Image.Image]] = []
        flat_text: list[str] = []
        counts: list[int] = []
        for index, (sample_images, instruction) in enumerate(
            zip(pil_history, instructions, strict=True)
        ):
            if history_mask is not None:
                valid = history_mask[index].to(dtype=torch.bool).tolist()
                if len(valid) != len(sample_images):
                    raise ValueError(
                        "history mask length must match history image length"
                    )
                sample_images = [
                    image
                    for image, keep in zip(sample_images, valid, strict=True)
                    if keep
                ]
            counts.append(len(sample_images))
            flat_images.extend([[image] for image in sample_images])
            flat_text.extend([instruction] * len(sample_images))
        if not flat_images:
            return None, None, None
        inputs = self.vlm.build_qwenvl_inputs(
            images=flat_images,
            instructions=flat_text,
            apply_cot_prompt=True,
        )
        if self.enable_planning_branch:
            tokens, planning, _, _ = self._generate_planning_features(
                vlm_inputs=inputs,
                planning_labels_text=None,
                device=device,
            )
        else:
            tokens = self._encode_vlm_tokens(inputs)
            planning = None
        max_history = max(counts)
        padded = torch.zeros(
            (len(counts), max_history, tokens.shape[1], tokens.shape[-1]),
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros(
            (len(counts), max_history), device=device, dtype=torch.bool
        )
        planning_padded = (
            torch.zeros(
                (
                    len(counts),
                    max_history,
                    planning.shape[1],
                    planning.shape[-1],
                ),
                device=device,
                dtype=dtype,
            )
            if planning is not None
            else None
        )
        cursor = 0
        for index, count in enumerate(counts):
            if count:
                padded[index, :count] = tokens[cursor : cursor + count].to(
                    device=device, dtype=dtype
                )
                if planning_padded is not None:
                    planning_padded[index, :count] = planning[
                        cursor : cursor + count
                    ].to(device=device, dtype=dtype)
                mask[index, :count] = True
                cursor += count
        return padded, planning_padded, mask

    def _build_planning_labels(
        self,
        target_texts: Any,
        device: torch.device,
        num_tokens: int,
        batch_size: int,
    ) -> torch.Tensor:
        texts = [target_texts] if isinstance(target_texts, str) else target_texts
        if texts is None:
            texts = [None] * batch_size
        elif len(texts) != batch_size:
            raise ValueError(
                "Planning label batch size must match generated planning batch: "
                f"{len(texts)} != {batch_size}"
            )
        labels = torch.full(
            (batch_size, max(1, num_tokens)),
            IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )
        for index, text in enumerate(texts):
            if text is None:
                continue
            ids = self.vlm.processor.tokenizer.encode(
                str(text), add_special_tokens=False
            )
            take = min(num_tokens, len(ids))
            if take:
                labels[index, :take] = torch.as_tensor(
                    ids[:take], dtype=torch.long, device=device
                )
        return labels

    def _teacher_force_planning_features(
        self,
        *,
        vlm_inputs: dict[str, Any],
        planning_labels_text: Any,
        device: torch.device,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one differentiable multimodal forward over ground-truth planning tokens."""

        if not any(parameter.requires_grad for parameter in self.vlm.parameters()):
            raise ValueError(
                "planning supervision requires an unfrozen VLM; set freeze.vlm=false"
            )
        input_ids = vlm_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("planning supervision requires VLM input_ids")
        labels = self._build_planning_labels(
            planning_labels_text,
            device=input_ids.device,
            num_tokens=num_tokens,
            batch_size=input_ids.shape[0],
        )
        if not bool((labels != IGNORE_INDEX).any()):
            raise ValueError(
                "planning supervision requires at least one ground-truth subtask token"
            )

        tokenizer = self.vlm.processor.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        teacher_tokens = labels.masked_fill(labels == IGNORE_INDEX, int(pad_token_id))
        teacher_inputs = dict(vlm_inputs)
        teacher_inputs.pop("labels", None)
        teacher_inputs["input_ids"] = torch.cat((input_ids, teacher_tokens), dim=1)
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        teacher_mask = (labels != IGNORE_INDEX).to(dtype=attention_mask.dtype)
        teacher_inputs["attention_mask"] = torch.cat(
            (attention_mask, teacher_mask), dim=1
        )
        # Let Qwen recompute sequence-position metadata for the extended tokens.
        teacher_inputs.pop("position_ids", None)
        teacher_inputs.pop("cache_position", None)

        output = self.vlm(
            **teacher_inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = output.hidden_states
        prompt_length = input_ids.shape[1]
        pooled = self._pool_vlm_hidden_states(hidden_states)
        perception = (
            pooled[:, :prompt_length]
            if self.vlm_token_mode == "last"
            else pooled
        ).to(device=device)
        planning = hidden_states[-1][
            :, prompt_length : prompt_length + num_tokens
        ].to(device=device, dtype=perception.dtype)
        logits = output.logits[
            :, prompt_length - 1 : prompt_length + num_tokens - 1
        ].to(device=device)
        return perception, planning, logits, labels.to(device=device)

    def _generate_planning_features(
        self,
        vlm_inputs: dict[str, Any],
        planning_labels_text: Any,
        device: torch.device,
        planning_supervision: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Use one multimodal prefill and collect exact generated-token states."""
        num_tokens = max(1, int(self.planning_num_tokens))
        if planning_supervision:
            return self._teacher_force_planning_features(
                vlm_inputs=vlm_inputs,
                planning_labels_text=planning_labels_text,
                device=device,
                num_tokens=num_tokens,
            )
        with torch.no_grad():
            output = self.vlm.generate(
                **vlm_inputs,
                max_new_tokens=num_tokens + 1,
                min_new_tokens=num_tokens + 1,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        expected_steps = num_tokens + 1
        if hidden_states is None or len(hidden_states) < expected_steps:
            actual = 0 if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                "VLM generation did not return enough hidden-state steps for "
                f"shared prefill: expected at least {expected_steps}, got {actual}."
            )
        perception = self._pool_vlm_hidden_states(hidden_states[0]).to(device=device)
        planning = torch.cat(
            [
                step_hidden_states[-1][:, -1:, :]
                for step_hidden_states in hidden_states[1 : num_tokens + 1]
            ],
            dim=1,
        ).to(device=device, dtype=perception.dtype)
        logits = None
        if getattr(output, "scores", None):
            logits = torch.stack(list(output.scores)[:num_tokens], dim=1).to(
                device=device, dtype=perception.dtype
            )
            if logits.shape[1] < num_tokens:
                logits = F.pad(logits, (0, 0, 0, num_tokens - logits.shape[1]))
            else:
                logits = logits[:, :num_tokens]
        labels = self._build_planning_labels(
            planning_labels_text,
            device=device,
            num_tokens=num_tokens,
            batch_size=planning.shape[0],
        )
        return perception, planning, logits, labels

    def encode_planning(
        self,
        *,
        images: torch.Tensor,
        planning_text: list[str],
        negative_text: list[str] | None,
        training: bool,
        planning_labels_text: list[str | None] | None = None,
        visual_input_range: VisualInputRange = "zero_one",
        planning_supervision: bool = False,
        history_images: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> AutoPlanningFeatures:
        if images.ndim != 5:
            raise ValueError("Qwen images must have shape [B, T, C, H, W]")
        if images.shape[0] != len(planning_text):
            raise ValueError("Qwen image and planning-text batch sizes must match")
        pil_images = self._to_pil_images(
            images.permute(0, 1, 3, 4, 2).contiguous(),
            input_range=visual_input_range,
        )
        # Goal task embedding uses raw high-level text. Perception and planning
        # share one CoT multimodal prefill + cached autoregressive decode.
        planning_inputs = self.vlm.build_qwenvl_inputs(
            images=pil_images, instructions=planning_text, apply_cot_prompt=True
        )
        task_inputs = self.vlm.build_qwenvl_inputs(
            images=pil_images, instructions=planning_text, apply_cot_prompt=False
        )
        if self.enable_planning_branch:
            perception, planning, logits, labels = self._generate_planning_features(
                vlm_inputs=planning_inputs,
                planning_labels_text=planning_labels_text if training else None,
                device=images.device,
                planning_supervision=training and planning_supervision,
            )
        else:
            perception = self._encode_vlm_tokens(planning_inputs).to(
                device=images.device
            )
            planning = logits = labels = None
        task = self._extract_text_embedding(
            task_inputs, device=images.device, dtype=perception.dtype
        )
        negative = None
        if negative_text is not None:
            negative_inputs = self.vlm.build_qwenvl_inputs(
                images=pil_images, instructions=negative_text
            )
            negative = self._encode_vlm_tokens(negative_inputs).to(
                device=images.device
            )
        history = history_planning = encoded_history_mask = None
        if history_images is not None:
            history, history_planning, encoded_history_mask = (
                self._encode_history_tokens(
                    history_images.permute(0, 1, 3, 4, 2).contiguous(),
                    history_mask,
                    planning_text,
                    images.device,
                    perception.dtype,
                    input_range=visual_input_range,
                )
            )
        return AutoPlanningFeatures(
            perception_features=perception,
            planning_features=planning,
            negative_perception_features=negative,
            negative_planning_features=planning if negative is not None else None,
            task_embedding=task,
            planning_logits=logits,
            planning_labels=labels,
            history_perception_features=history,
            history_planning_features=history_planning,
            history_mask=encoded_history_mask,
        )

    def forward(self, backbone_input: Any):
        from transformers.feature_extraction_utils import BatchFeature

        first_value = next(iter(backbone_input.values()))
        batch = first_value.shape[0]
        device = first_value.device
        if not self.enable_vlm_tokens:
            return BatchFeature(
                data={
                    "backbone_features": torch.empty(
                        batch, 1, 0, dtype=torch.float32, device=device
                    )
                }
            )
        get = backbone_input.get
        inputs = get("vlm_inputs")
        images = get("vlm_images")
        text = get("vlm_text")
        negative_text = get("vlm_text_negative")
        planning_text = get("vlm_planning_text")
        planning_labels = get("vlm_planning_label_text")
        history_images = get("vlm_history_images")
        shared_text = planning_text if planning_text is not None else text
        task_text = text if text is not None else shared_text
        if images is not None and shared_text is not None:
            planning_inputs = self.vlm.build_qwenvl_inputs(
                images=self._to_pil_images(images),
                instructions=shared_text,
                apply_cot_prompt=True,
            )
            task_inputs = self.vlm.build_qwenvl_inputs(
                images=self._to_pil_images(images),
                instructions=task_text,
                apply_cot_prompt=False,
            )
        elif inputs is None:
            raise ValueError(
                "QwenPlanningEncoder requires vlm_inputs or "
                "vlm_images+vlm_planning_text"
            )
        else:
            planning_inputs = inputs
            task_inputs = inputs
        planning = logits = labels = None
        if self.enable_planning_branch:
            tokens, planning, logits, labels = self._generate_planning_features(
                vlm_inputs=planning_inputs,
                planning_labels_text=planning_labels,
                device=device,
            )
        else:
            tokens = self._encode_vlm_tokens(planning_inputs)
        output = {
            "backbone_features": torch.empty(
                batch, 1, 0, dtype=torch.float32, device=device
            ),
            self.vlm_tokens_key: tokens,
        }
        text_embedding = self._extract_text_embedding(
            task_inputs, device=device, dtype=tokens.dtype
        )
        if text_embedding is not None:
            output[self.vlm_text_emb_key] = text_embedding
        if planning is not None:
            output[self.planning_tokens_key] = planning
            output[self.planning_tokens_negative_key] = planning
        if logits is not None:
            output[self.planning_logits_key] = logits
        if labels is not None:
            output[self.planning_labels_key] = labels
        if negative_text is not None:
            negative_inputs = self.vlm.build_qwenvl_inputs(
                images=self._to_pil_images(images),
                instructions=negative_text,
            )
            output[self.vlm_tokens_negative_key] = self._encode_vlm_tokens(
                negative_inputs
            )
        elif get("vlm_inputs_negative") is not None:
            output[self.vlm_tokens_negative_key] = self._encode_vlm_tokens(
                get("vlm_inputs_negative")
            )
        if history_images is not None:
            history, history_planning, history_mask = self._encode_history_tokens(
                history_images,
                get("vlm_history_mask"),
                shared_text,
                device,
                tokens.dtype,
            )
            if history is not None:
                output[self.vlm_history_tokens_key] = history
                output[self.vlm_history_mask_key] = history_mask
                if history_planning is not None:
                    output[self.vlm_history_planning_tokens_key] = history_planning
        return BatchFeature(data=output)

    def prepare_input(self, batch: dict[str, Any]):
        from transformers.feature_extraction_utils import BatchFeature

        if not self.enable_vlm_tokens:
            for key in ("action", "state", "images"):
                if key in batch:
                    return BatchFeature(data={key: batch[key]})
            return BatchFeature(data=batch)
        keys = (
            "vlm_inputs",
            "vlm_inputs_negative",
            "vlm_images",
            "vlm_text",
            "vlm_text_negative",
            "vlm_planning_text",
            "vlm_planning_label_text",
            "vlm_history_images",
            "vlm_history_mask",
        )
        data = {key: batch[key] for key in keys if key in batch}
        if not data:
            for key in ("action", "state"):
                if key in batch:
                    data[key] = batch[key]
                    break
            else:
                data = batch
        return BatchFeature(data=data)


# Compatibility name used by saved Hydra configs and oracle tests.
Qwen3VLQFormerBackbone = QwenPlanningEncoder

__all__ = [
    "LayerwiseQFormer",
    "Qwen3VLInterface",
    "Qwen3VLQFormerBackbone",
    "QwenPlanningEncoder",
]

"""Qwen3.x VL text encoder adapter for Wan video pipelines.

Security
--------
``AutoProcessor`` / ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
runs Python shipped with the model repo (custom modeling code). If the checkpoint
path points to an untrusted or maliciously modified tree, that can lead to
arbitrary code execution. Mitigations:

- Use only **trusted** sources (e.g. Hugging Face org you control, internal mirrors,
  or paths verified by your org).
- Prefer loading from **immutable snapshots** or verify **checksums** of weights and
  Python files before first use in production.
- Treat ``pretrained_path`` like any other executable supply chain input.
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional, Union, List
from PIL import Image


class Qwen35TextEncoderAdapter(nn.Module):
    """Multimodal Qwen adapter with a trainable projection to DiT ``text_dim``.

    ``trust_remote_code=True`` is required for many VL checkpoints. See module docstring
    for security implications of loading from ``pretrained_path``.

    The ``device`` argument is passed to ``from_pretrained(..., device_map=...)`` and to
    ``project_out`` placement. Actual backbone tensor placement may differ (e.g.
    ``device_map=\"auto\"`` splits layers). Use ``self.model.device`` for inputs and
    device checks, not the constructor string.
    """

    def __init__(
        self,
        pretrained_path: str,
        proj_out_dim: int = 4096,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        system_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.system_prompt = system_prompt

        # trust_remote_code: required for some Qwen-VL repos; only use trusted pretrained_path
        self.processor = AutoProcessor.from_pretrained(
            pretrained_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        # Freeze backbone: no weight updates. Do not wrap encode() in no_grad — graph must reach project_out.
        for _p in self.model.parameters():
            _p.requires_grad = False

        # Marker for vision end
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        )
        self.eot_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Hidden size from config
        # For Qwen3.5, the text hidden size is in model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size

        # Projection layer to match DiT expected dimension
        if self.hidden_size != proj_out_dim:
            self.project_out = nn.Linear(
                self.hidden_size,
                proj_out_dim,
                bias=False,
                dtype=torch_dtype,
                device=device,
            )
        else:
            # No projection needed; avoid nn.Identity() (no parameters, misleading .device checks)
            self.project_out = None

    def _use_system_role(self) -> bool:
        return bool(self.system_prompt and str(self.system_prompt).strip())

    def _build_text_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Keep user-side text tokens; robust to system <|im_end|> before vision and to no-image prompts."""
        # pad_id: used in text-only no-eot fallback row, and once at end for all rows
        pad_id = self.processor.tokenizer.pad_token_id
        is_vision_end = input_ids == self.vision_end_id
        is_eot = input_ids == self.eot_id
        batch_size, seq_len = input_ids.shape
        text_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=input_ids.device
        )
        has_vision = is_vision_end.any(dim=1)

        for i in range(batch_size):
            if has_vision[i]:
                row_ve = is_vision_end[i]
                row_eot = is_eot[i]
                after_last_ve = (row_ve.cumsum(0) == row_ve.sum()) & (~row_ve)
                eot_after = row_eot & after_last_ve
                text_mask[i] = after_last_ve & (eot_after.cumsum(0) == 0) & (~row_eot)
            else:
                # Text-only row: no <|vision_end|> — mask user/chat text tokens (not implicit fallthrough)
                eot_pos = torch.where(input_ids[i] == self.eot_id)[0]
                if eot_pos.numel() >= 2:
                    s, e = int(eot_pos[0]), int(eot_pos[1])
                    text_mask[i, s + 1 : e] = True
                elif eot_pos.numel() == 1:
                    s = int(eot_pos[0])
                    text_mask[i, s + 1 :] = True
                else:
                    text_mask[i] = input_ids[i] != pad_id

        if pad_id is not None:
            text_mask &= input_ids != pad_id
        return text_mask

    def encode(
        self,
        prompt: Union[str, List[str]],
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        print_generated: bool = False,
    ) -> torch.Tensor:
        """Encode prompt(s) + optional image(s) into hidden-state embeddings.

        During training the caller must ensure this runs inside a grad-enabled context
        so that project_out.weight receives gradients. During inference wrap the call
        site with torch.no_grad() to avoid building an unnecessary autograd graph.

        **Batch vs images**

        - **One prompt** (``len(prompt)==1``): ``image`` may be a single ``PIL.Image`` or a
          list of images; all are attached to that prompt (multi-image conditioning).
        - **Multiple prompts** (``len(prompt) > 1``):

          - If ``len(image) == len(prompt)``, images are paired **one-to-one** with prompts
            (``prompt[i]`` uses only ``image[i]``). Processor order matches ``texts``.
          - Otherwise (**broadcast**): the same image list is repeated for **every** prompt
            (``len(prompt)`` copies of the full list, flattened in order). Use this when each
            prompt should see the **same** set of reference images. A ``logging.warning`` is
            emitted when ``len(image) > 1`` and lengths differ so accidental misuse is visible.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        if image is not None:
            # Match HF + Qwen-VL expectations: only wrap PIL as a one-element list.
            # Do NOT wrap torch.Tensor in [tensor]: iterating a Tensor / list.extend(tensor)
            # splits dim 0 into per-frame images; a single 5D tensor as one "image" breaks
            # transformers.image_utils.make_flat_list_of_images (video / batched frames).
            if isinstance(image, Image.Image):
                image = [image]
            elif isinstance(image, (list, tuple)):
                image = list(image)
            # else: Tensor / ndarray — leave as-is so dim-0 is frames or batch

            # One image per prompt when batch>1 and counts match; else broadcast same set to each prompt
            pairwise = len(prompt) > 1 and len(image) == len(prompt)
            if len(prompt) > 1 and not pairwise and len(image) > 1:
                logging.warning(
                    "[Qwen35Adapter] len(prompt)=%d but len(image)=%d: broadcasting the "
                    "same %d image(s) to every prompt. For one image per prompt, pass "
                    "len(image)==len(prompt).",
                    len(prompt),
                    len(image),
                    len(image),
                )

            # Prepare multimodal messages: optional system + user (images + instruction)
            messages = []
            for i, p in enumerate(prompt):
                imgs_for_prompt = [image[i]] if pairwise else image
                content = []
                for img in imgs_for_prompt:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": p})

                if self._use_system_role():
                    messages.append(
                        [
                            {
                                "role": "system",
                                "content": [
                                    {"type": "text", "text": self.system_prompt}
                                ],
                            },
                            {"role": "user", "content": content},
                        ]
                    )
                else:
                    messages.append([{"role": "user", "content": content}])

            # Set add_generation_prompt=False to make it easier to find the end of the user message
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]

            if pairwise:
                all_images = list(image)
            else:
                all_images = []
                for _ in range(len(prompt)):
                    all_images.extend(image)

            inputs = self.processor(
                text=texts, images=all_images, padding=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            messages = []
            for p in prompt:
                if self._use_system_role():
                    messages.append(
                        [
                            {
                                "role": "system",
                                "content": [
                                    {"type": "text", "text": self.system_prompt}
                                ],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": p}],
                            },
                        ]
                    )
                else:
                    messages.append(
                        [{"role": "user", "content": [{"type": "text", "text": p}]}]
                    )
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]
            inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(
                self.model.device
            )

        # Backbone forward must run with grad enabled when training project_out: wrapping in no_grad()
        # would detach hidden states and break gradients to project_out.weight. Backbone weights stay
        # frozen via requires_grad=False above.
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        # last_hidden_state shape: [B, L, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        input_ids = inputs.input_ids
        batch_size = input_ids.size(0)
        text_mask = self._build_text_mask(input_ids)

        # Extract and pad text embeddings
        max_text_len = text_mask.sum(dim=1).max().item()
        if max_text_len == 0:
            raise ValueError(
                "[Qwen35Adapter] text_mask is empty for all batches; cannot build text "
                "embeddings. Check chat template / special tokens / image alignment."
            )
        filtered_hidden_states = []

        for i in range(batch_size):
            text_embeddings = last_hidden_state[i][text_mask[i]]
            if text_embeddings.size(0) < max_text_len:
                padding = torch.zeros(
                    (max_text_len - text_embeddings.size(0), text_embeddings.size(1)),
                    dtype=text_embeddings.dtype,
                    device=text_embeddings.device,
                )
                text_embeddings = torch.cat([text_embeddings, padding], dim=0)
            filtered_hidden_states.append(text_embeddings)

        last_hidden_state = torch.stack(filtered_hidden_states, dim=0)

        # project_out: trainable Linear when dims differ; else pass-through (same dtype)
        h = last_hidden_state.to(self.torch_dtype)
        if self.project_out is not None:
            projected_state = self.project_out(h)
        else:
            projected_state = h

        return projected_state

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 128,
    ) -> str:
        if image is not None:
            if self._use_system_role():
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.model.device)
        else:
            if self._use_system_role():
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                ]
            else:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], return_tensors="pt").to(
                self.model.device
            )

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return generated_text

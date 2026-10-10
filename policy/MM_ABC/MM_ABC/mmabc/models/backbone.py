"""Qwen3-VL backbone with intermediate context taps for the action expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

# Which canonical view goes into which prompt slot, for readable prompts.
VIEW_LABELS = {"primary": "scene", "wrist_left": "left wrist", "wrist_right": "right wrist"}


def _assert_pretrained(model: nn.Module, path: str) -> None:
    """Fail loudly if the trunk looks randomly initialised.

    A silent random init costs a full training run to notice, so it is checked
    rather than trusted. Pretrained transformer norm weights sit near 1.0; a
    fresh init leaves them exactly at 1.0 with zero spread across layers.
    """
    norms = [
        p.detach().float()
        for name, p in model.named_parameters()
        if name.endswith("input_layernorm.weight")
    ]
    if not norms:
        return
    spread = torch.stack([n.mean() for n in norms]).std()
    if not torch.isfinite(spread) or spread.item() < 1e-6:
        raise RuntimeError(
            f"backbone at {path} appears randomly initialised (layernorm spread "
            f"{spread.item():.2e}); check that checkpoint keys match the model class"
        )


@dataclass
class BackboneOutput:
    memory: list[torch.Tensor]  # len(memory_layers) x (B, S, d_vlm)
    attention_mask: torch.Tensor  # (B, S) 1 = real token
    image_token_mask: torch.Tensor  # (B, S) 1 = visual token


class QwenVLBackbone(nn.Module):
    """Vision-language trunk producing multi-layer context for the action expert."""

    def __init__(
        self,
        path: str,
        *,
        memory_layers: tuple[int, ...],
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = True,
        # fp32 master weights. FSDP2's MixedPrecisionPolicy casts to bf16 for
        # compute, and it requires a single dtype across all parameters, so the
        # trunk must match the freshly built expert rather than loading in bf16.
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(path)
        # Keep the vision tower in fp32 with SDPA; the language tower uses the requested backend.
        attn_impl_arg: str | dict = attn_implementation
        if isinstance(attn_implementation, str) and attn_implementation != "sdpa":
            attn_impl_arg = {
                "text_config": attn_implementation,
                "vision_config": "sdpa",
            }
        # Load the generation wrapper to match checkpoint model.* keys, then retain its trunk.
        full = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            config=config,
            dtype=dtype,
            attn_implementation=attn_impl_arg,
        )
        self.model = full.model
        del full
        _assert_pretrained(self.model, path)
        self.processor = AutoProcessor.from_pretrained(path, use_fast=True)
        self.hidden_size = int(getattr(config, "text_config", config).hidden_size)
        self.num_layers = int(getattr(config, "text_config", config).num_hidden_layers)

        for layer in memory_layers:
            if not 0 <= layer < self.num_layers:
                raise ValueError(
                    f"memory layer {layer} outside trunk of {self.num_layers} layers"
                )
        self.memory_layers = tuple(int(x) for x in memory_layers)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Checkpointing requires differentiable embedding outputs because token IDs cannot carry gradients.
            self.model.enable_input_require_grads()

        self._image_token_id = int(getattr(config, "image_token_id", -1))

    @property
    def image_token_id(self) -> int:
        return self._image_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> BackboneOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states[0] is the embedding output, so layer i is at index i+1.
        hs = out.hidden_states
        memory = [hs[i + 1] for i in self.memory_layers]
        image_token_mask = (
            (input_ids == self._image_token_id)
            if self._image_token_id >= 0
            else torch.zeros_like(attention_mask, dtype=torch.bool)
        )
        return BackboneOutput(
            memory=memory,
            attention_mask=attention_mask,
            image_token_mask=image_token_mask,
        )


def build_inputs(
    processor,
    prompts: list[str],
    images,
    view_mask,
    *,
    device: torch.device | None,
) -> dict:
    """Tokenise prompts and pack the present views into backbone inputs.

    ``images`` is (B, V, H, W, 3) uint8 for a single timestep and ``view_mask``
    is (B, V), as tensors or numpy arrays. Absent views contribute no image
    tokens at all rather than a black frame, so the sequence length reflects
    what the platform actually has. ``device=None`` leaves the result on the CPU,
    which is how dataloader workers call it.
    """
    from PIL import Image

    def _np(x):
        # FSDP2 casts floating forward inputs to bf16, which numpy cannot hold.
        if torch.is_tensor(x):
            return (x.float() if x.is_floating_point() else x).cpu().numpy()
        return x

    images, view_mask = _np(images), _np(view_mask)
    batch_images: list[list] = []
    texts: list[str] = []
    for b, prompt in enumerate(prompts):
        present = []
        labels = []
        for v, slot in enumerate(VIEW_LABELS):
            if bool(view_mask[b, v]):
                present.append(Image.fromarray(images[b, v]))
                labels.append(VIEW_LABELS[slot])
        batch_images.append(present)
        header = "".join(f"<|vision_start|><|image_pad|><|vision_end|>" for _ in present)
        view_note = ", ".join(labels)
        content = f"{header}views: {view_note}\n{prompt}"
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    encoded = processor(
        text=texts,
        images=batch_images if any(batch_images) else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    out = {k: encoded[k] for k in keep if k in encoded}
    if device is None:
        return out
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}

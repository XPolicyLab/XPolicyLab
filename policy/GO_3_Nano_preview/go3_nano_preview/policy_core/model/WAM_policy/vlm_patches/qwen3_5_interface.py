import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


def _patch_single_batch_padding_mask(q35mod):
    original = q35mod.apply_mask_to_padding_states
    if getattr(original, "_forge_masks_single_batch", False):
        return

    def apply_mask_to_padding_states(hidden_states, attention_mask):
        masked = original(hidden_states, attention_mask)
        if attention_mask is not None and attention_mask.shape[0] == 1:
            masked = (masked * attention_mask[:, :, None]).to(masked.dtype)
        return masked

    apply_mask_to_padding_states._forge_masks_single_batch = True
    q35mod.apply_mask_to_padding_states = apply_mask_to_padding_states


class _QWen3_5_Interface(nn.Module):
    def __init__(self, model_id, dtype=torch.bfloat16, fixed_sequence_length=None, **kwargs):
        super().__init__()

        import transformers.models.qwen3_5.modeling_qwen3_5 as q35mod
        _patch_single_batch_padding_mask(q35mod)

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.truncation_side = "right"

        self.model = model
        self.processor = processor
        self.fixed_sequence_length = (
            None if fixed_sequence_length is None else int(fixed_sequence_length)
        )
        if self.fixed_sequence_length is not None and self.fixed_sequence_length <= 0:
            raise ValueError("fixed_sequence_length must be positive when specified.")

        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def forward_features(self, **kwargs):
        if kwargs.get("labels") is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("VGen features do not accept labels or a generation cache")
        kwargs["use_cache"] = False
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.model(**kwargs)

    def generate(
        self,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, drop_images=False, **kwargs):
        if solutions is not None:
            raise NotImplementedError(
                "The action-token label masking from qwen3_interface.py uses a "
                "Qwen3-VL-specific fast-tokenizer id range and has not been ported "
                "to the Qwen3.5 tokenizer."
            )

        messages = []
        if drop_images:
            iterator = ((None, ins) for ins in instructions)
        else:
            assert len(images) == len(instructions), "Images and instructions must have the same length"
            iterator = zip(images, instructions)
        for imgs, instruction in iterator:
            content = [] if drop_images else [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])

        length_kwargs = {"padding": True}
        if self.fixed_sequence_length is not None:
            length_kwargs = {
                "padding": "max_length",
                "max_length": self.fixed_sequence_length,
                "truncation": True,
            }
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **length_kwargs,
        )

        return batch_inputs.to(self.model.device)

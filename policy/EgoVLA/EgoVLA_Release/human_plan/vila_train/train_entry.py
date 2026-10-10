"""Shared compatibility hooks for the SparkArena EgoVLA trainer."""

from __future__ import annotations

import os


def _patch_optional_attention() -> None:
    """Use eager attention unless the user explicitly opts into FlashAttention."""

    if os.environ.get("EGOVLA_USE_FLASH_ATTN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    from llava.model.language_model.llava_llama import LlavaLlamaModel

    original_init = getattr(LlavaLlamaModel, "_egovla_original_init", None)
    if original_init is None:
        original_init = LlavaLlamaModel.__init__

        def safe_init(self, *args, **kwargs):
            kwargs["attn_implementation"] = "eager"
            return original_init(self, *args, **kwargs)

        LlavaLlamaModel._egovla_original_init = original_init
        LlavaLlamaModel.__init__ = safe_init

    for module_name in (
        "llava.model.multimodal_encoder.siglip.siglip",
        "llava.model.multimodal_encoder.siglip.modeling_siglip",
    ):
        try:
            module = __import__(module_name, fromlist=["SiglipVisionModel"])
            vision_model = module.SiglipVisionModel
            original = getattr(vision_model, "_egovla_original_from_pretrained", None)
            if original is None:
                original = vision_model.from_pretrained.__func__

                @classmethod
                def safe_from_pretrained(cls, *args, _original=original, **kwargs):
                    kwargs["attn_implementation"] = "eager"
                    return _original(cls, *args, **kwargs)

                vision_model._egovla_original_from_pretrained = original
                vision_model.from_pretrained = safe_from_pretrained
            break
        except (ImportError, AttributeError):
            continue

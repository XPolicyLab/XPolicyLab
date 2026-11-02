# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

__all__ = [
    "Wan2_1_VAE",
    "Wan2_2_VAE",
    "WanModel",
    "T5Model",
    "T5Encoder",
    "T5Decoder",
    "T5EncoderModel",
    "HuggingfaceTokenizer",
    "flash_attention",
]


def __getattr__(name):
    if name == "flash_attention":
        from .attention import flash_attention

        return flash_attention
    if name == "WanModel":
        from .model import WanModel

        return WanModel
    if name == "Wan2_1_VAE":
        from .vae2_1 import Wan2_1_VAE

        return Wan2_1_VAE
    if name == "Wan2_2_VAE":
        from .vae2_2 import Wan2_2_VAE

        return Wan2_2_VAE
    if name == "T5Model":
        from .t5 import T5Model

        return T5Model
    if name == "T5Encoder":
        from .t5 import T5Encoder

        return T5Encoder
    if name == "T5Decoder":
        from .t5 import T5Decoder

        return T5Decoder
    if name == "T5EncoderModel":
        from .t5 import T5EncoderModel

        return T5EncoderModel
    if name == "HuggingfaceTokenizer":
        from .tokenizers import HuggingfaceTokenizer

        return HuggingfaceTokenizer
    raise AttributeError(name)

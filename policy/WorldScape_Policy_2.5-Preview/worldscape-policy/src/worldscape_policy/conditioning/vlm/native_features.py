"""Frozen VLM whose base and LoRA parameters live in the native policy shards."""
from torch import nn
import torch

from .cached_features import validate_bundled_assets
from .encoder_core.encoder import VLPromptEncoder
from .features import VLMFeatureEncoder


class NativeVLMFeatureEncoder(VLMFeatureEncoder):
    """Same feature computation as the bundled encoder, restored by the policy.

    No pretrained weight file is opened here. The native builder strictly loads
    encoder.* together with the WAM, T5 and fusion parameters before inference.
    The inherited _apply preserves BF16 base weights and FP32 LoRA/rotary values.
    """

    def __init__(self, *, assets_dir, bundle_identity, feature_source_identity,
                 teacher_color_order="bgr", attn_implementation="sdpa", cache_only=False):
        nn.Module.__init__(self)
        if cache_only:
            raise ValueError("NativeVLMFeatureEncoder is an inference-only deployment component")
        if teacher_color_order != "bgr" or attn_implementation != "sdpa":
            raise ValueError("VLM feature contract requires BGR and SDPA")
        checked = validate_bundled_assets(
            assets_dir, expected_identity=bundle_identity,
            feature_source_identity=feature_source_identity,
            weight_storage="native_model_shards")
        self.assets_dir = str(checked["root"])
        self.bundle_identity = checked["bundle_identity"]
        self.feature_source_identity = checked["feature_source_identity"]
        config = checked["config"]
        self.encoder = VLPromptEncoder(
            str(checked["root"] / config["base_directory"]), device="cpu",
            torch_dtype=torch.bfloat16, load_pretrained=False,
            **dict(config["encoder"], attn_implementation=attn_implementation))
        self.encoder.loaded_text_layout = config["text_layout"]
        self.output_dim = self.encoder.dim
        if self.output_dim != 7680:
            raise ValueError("VLM features must have width 7680")
        self.teacher_color_order = teacher_color_order
        self.requires_grad_(False)
        self.eval()

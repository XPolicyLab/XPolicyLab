"""Checkpoint-local frozen VLM with the original feature computation."""
from .features import VLMFeatureEncoder
from .cached_features import validate_bundled_assets


class BundledVLMFeatureEncoder(VLMFeatureEncoder):
    """Load verified assets without changing any parameter or preprocessing."""

    def __init__(self, *, assets_dir, bundle_identity, feature_source_identity,
                 teacher_color_order="bgr", device="cpu", attn_implementation="sdpa"):
        if teacher_color_order != "bgr" or attn_implementation != "sdpa":
            raise ValueError("VLM feature contract requires BGR and SDPA")
        validate_bundled_assets(assets_dir, expected_identity=bundle_identity,
                                feature_source_identity=feature_source_identity)
        super().__init__(assets_dir=assets_dir, teacher_color_order=teacher_color_order,
                         device=device, attn_implementation=attn_implementation,
                         config_filename="encoder_config.json")
